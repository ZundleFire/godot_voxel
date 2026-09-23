#include "voxel_memory_pool.h"
#include "../util/macros.h"
#include "../util/memory/memory.h"
#include "../util/profiling.h"
#include "../util/string/format.h"
#include "../util/string/std_string.h"

namespace zylann::voxel {

namespace {
VoxelMemoryPool *g_memory_pool = nullptr;
} // namespace

// Thread-local cache instance – one per thread, lazily initialized.
VoxelMemoryPool::TLSCache &VoxelMemoryPool::get_tls_cache() {
	thread_local TLSCache tls_cache;
	return tls_cache;
}

void VoxelMemoryPool::create_singleton() {
	ZN_ASSERT(g_memory_pool == nullptr);
	g_memory_pool = ZN_NEW(VoxelMemoryPool);
}

void VoxelMemoryPool::destroy_singleton() {
	const unsigned int used_blocks = VoxelMemoryPool::get_singleton().debug_get_used_blocks();
	if (used_blocks > 0) {
		ZN_PRINT_ERROR(
				format("VoxelMemoryPool: "
					   "{} memory blocks are still used when unregistering the module. Recycling leak?",
					   used_blocks)
		);
#ifdef DEBUG_ENABLED
		VoxelMemoryPool::get_singleton().debug_print_used_blocks(10);
#endif
	}

	ZN_ASSERT(g_memory_pool != nullptr);
	VoxelMemoryPool *pool = g_memory_pool;
	g_memory_pool = nullptr;
	ZN_DELETE(pool);
}

#ifdef DEBUG_ENABLED
void VoxelMemoryPool::debug_print_used_blocks(unsigned int max_count) {
	struct L {
		static void debug_print_used_blocks(
				const VoxelMemoryPool::DebugUsedBlocks &debug_used_blocks,
				unsigned int &count,
				unsigned int max_count,
				size_t mem_size
		) {
			if (count > max_count) {
				count += debug_used_blocks.blocks.size();
				return;
			}
			const unsigned int initial_count = count;
			for (auto it = debug_used_blocks.blocks.begin(); it != debug_used_blocks.blocks.end(); ++it) {
				if (count > max_count) {
					break;
				}
				StdString s;
				const dstack::Info &info = it->second;
				info.to_string(s);
				if (mem_size == 0) {
					print_line(format("--- Alloc {}:", count));
				} else {
					print_line(format("--- Alloc {}, size {}:", count, mem_size));
				}
				print_line(s);
				++count;
			}
			count = initial_count + debug_used_blocks.blocks.size();
		}
	};

	unsigned int count = 0;
	for (unsigned int pool_index = 0; pool_index < _pot_pools.size(); ++pool_index) {
		const Pool &pool = _pot_pools[pool_index];
		L::debug_print_used_blocks(pool.debug_used_blocks, count, max_count, get_size_from_pool_index(pool_index));
	}
	L::debug_print_used_blocks(_debug_nonpooled_used_blocks, count, max_count, 0);
	if (count > 0 && count > max_count) {
		print_line(format("[...] and {} more allocs.", max_count - count));
	}
}
#endif

VoxelMemoryPool &VoxelMemoryPool::get_singleton() {
	ZN_ASSERT(g_memory_pool != nullptr);
	return *g_memory_pool;
}

VoxelMemoryPool::VoxelMemoryPool() {}

VoxelMemoryPool::~VoxelMemoryPool() {
#ifdef TOOLS_ENABLED
	if (is_verbose_output_enabled()) {
		debug_print();
	}
#endif
	clear();
}

uint8_t *VoxelMemoryPool::allocate(size_t size) {
	ZN_DSTACK();
	ZN_PROFILE_SCOPE();

	// In practice this is not supposed to happen, it might hide a mistake.
	// We should be able to keep running with this only guarantee: the returned value should be able to be "freed"
	// using the same pool.
	ZN_ASSERT_RETURN_V(size != 0, nullptr);

	uint8_t *block = nullptr;
	// Not calculating `pot` immediately because the function we use to calculate it uses 32 bits,
	// while `size_t` can be larger than that.
	if (size > get_highest_supported_size()) {
		// Sorry, memory is not pooled past this size
		block = (uint8_t *)ZN_ALLOC(size * sizeof(uint8_t));
		_total_memory += size;
#ifdef DEBUG_ENABLED
		if (block != nullptr) {
			_debug_nonpooled_used_blocks.add(block);
		}
#endif
	} else {
		const unsigned int pot = get_pool_index_from_size(size);

		// --- Try thread-local cache first (no lock) ---
		TLSCache &tls = get_tls_cache();
		TLSPoolBucket &tls_bucket = tls.buckets[pot];
		if (tls_bucket.count > 0) {
			block = tls_bucket.blocks[--tls_bucket.count];
		} else {
			// TLS cache empty – try to refill from the shared pool in one lock
			Pool &pool = _pot_pools[pot];
			pool.mutex.lock();
			const unsigned int available = static_cast<unsigned int>(pool.blocks.size());
			if (available > 0) {
				const unsigned int grab = (available < TLS_REFILL_COUNT) ? available : TLS_REFILL_COUNT;
				// Take the first block for the caller
				block = pool.blocks.back();
				pool.blocks.pop_back();
				// Fill TLS cache with the rest
				for (unsigned int i = 1; i < grab; ++i) {
					tls_bucket.blocks[tls_bucket.count++] = pool.blocks.back();
					pool.blocks.pop_back();
				}
				pool.mutex.unlock();
			} else {
				pool.mutex.unlock();
				ZN_PROFILE_SCOPE_NAMED("new alloc");
				// All allocations done in this pool have the same size,
				// which must be greater or equal to `size`
				const size_t capacity = get_size_from_pool_index(pot);
#ifdef DEBUG_ENABLED
				ZN_ASSERT(capacity >= size);
#endif
				block = (uint8_t *)ZN_ALLOC(capacity * sizeof(uint8_t));
				// The whole bucket is allocated, and clear_unused_blocks subtracts it that way
				_total_memory += capacity;
			}
		}
#ifdef DEBUG_ENABLED
		if (block != nullptr) {
			Pool &pool = _pot_pools[pot];
			pool.debug_used_blocks.add(block);
		}
#endif
	}
	if (block == nullptr) {
		ZN_PRINT_ERROR("Out of memory");
	} else {
		++_used_blocks;
		_used_memory += size;
	}
	return block;
}

void VoxelMemoryPool::recycle(uint8_t *block, size_t size) {
	// In case we have done empty allocations (we prefer not to do that, but it shouldn't warrant a crash)
	if (block == nullptr && size == 0) {
		return;
	}
	ZN_ASSERT(size != 0);
	ZN_ASSERT(block != nullptr);
	// Not calculating `pot` immediately because the function we use to calculate it uses 32 bits,
	// while `size_t` can be larger than that.
	if (size > get_highest_supported_size()) {
#ifdef DEBUG_ENABLED
		// Make sure this allocation was done by this pool in this scenario
		_debug_nonpooled_used_blocks.remove(block);
#endif
		ZN_FREE(block);
		_total_memory -= size;
	} else {
		const unsigned int pot = get_pool_index_from_size(size);
#ifdef DEBUG_ENABLED
		{
			Pool &pool = _pot_pools[pot];
			// Make sure this allocation was done by this pool in this scenario
			pool.debug_used_blocks.remove(block);
		}
#endif
		// --- Try thread-local cache first (no lock) ---
		TLSCache &tls = get_tls_cache();
		TLSPoolBucket &tls_bucket = tls.buckets[pot];
		if (tls_bucket.count < TLS_CACHE_CAPACITY) {
			tls_bucket.blocks[tls_bucket.count++] = block;
		} else {
			// TLS cache full – flush half of the cache (including this block) back to the shared pool
			Pool &pool = _pot_pools[pot];
			MutexLock lock(pool.mutex);
			pool.blocks.push_back(block);
			const unsigned int flush_count = TLS_CACHE_CAPACITY / 2;
			for (unsigned int i = 0; i < flush_count; ++i) {
				pool.blocks.push_back(tls_bucket.blocks[--tls_bucket.count]);
			}
		}
	}
	--_used_blocks;
	_used_memory -= size;
}

void VoxelMemoryPool::flush_tls_caches() {
	// Flush the calling thread's TLS cache back to the shared pool.
	// NOTE: This only flushes the current thread's cache. For a full flush across all threads,
	// each thread must call this (e.g., before a thread exits).
	TLSCache &tls = get_tls_cache();
	for (unsigned int pot = 0; pot < _pot_pools.size(); ++pot) {
		TLSPoolBucket &tls_bucket = tls.buckets[pot];
		if (tls_bucket.count > 0) {
			Pool &pool = _pot_pools[pot];
			MutexLock lock(pool.mutex);
			for (unsigned int i = 0; i < tls_bucket.count; ++i) {
				pool.blocks.push_back(tls_bucket.blocks[i]);
			}
			tls_bucket.count = 0;
		}
	}
}

void VoxelMemoryPool::clear_unused_blocks() {
	for (unsigned int pot = 0; pot < _pot_pools.size(); ++pot) {
		Pool &pool = _pot_pools[pot];
		MutexLock lock(pool.mutex);
		for (unsigned int i = 0; i < pool.blocks.size(); ++i) {
			void *block = pool.blocks[i];
			ZN_FREE(block);
		}
		_total_memory -= get_size_from_pool_index(pot) * pool.blocks.size();
		pool.blocks.clear();
	}
}

void VoxelMemoryPool::clear() {
	for (unsigned int pot = 0; pot < _pot_pools.size(); ++pot) {
		Pool &pool = _pot_pools[pot];
		MutexLock lock(pool.mutex);
		for (unsigned int i = 0; i < pool.blocks.size(); ++i) {
			void *block = pool.blocks[i];
			ZN_FREE(block);
		}
		pool.blocks.clear();
	}
	_used_memory = 0;
	_total_memory = 0;
	_used_blocks = 0;
}

void VoxelMemoryPool::debug_print() {
	print_line("-------- VoxelMemoryPool ----------");
	for (unsigned int pot = 0; pot < _pot_pools.size(); ++pot) {
		Pool &pool = _pot_pools[pot];
		MutexLock lock(pool.mutex);
		print_line(format("Pool {}: {} blocks (capacity {})", pot, pool.blocks.size(), pool.blocks.capacity()));
	}
}

unsigned int VoxelMemoryPool::debug_get_used_blocks() const {
	return _used_blocks;
}

size_t VoxelMemoryPool::debug_get_used_memory() const {
	return _used_memory;
}

size_t VoxelMemoryPool::debug_get_total_memory() const {
	return _total_memory;
}

} // namespace zylann::voxel
