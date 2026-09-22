#include "gpu_range_allocator.h"

namespace zylann::voxel::gpu_driven {

uint32_t RangeAllocator::allocate(uint32_t size) {
	if (size == 0) {
		return INVALID;
	}
	for (auto it = _free.begin(); it != _free.end(); ++it) {
		if (it->second < size) {
			continue;
		}
		const uint32_t offset = it->first;
		const uint32_t remaining = it->second - size;
		_free.erase(it);
		if (remaining > 0) {
			_free.emplace(offset + size, remaining);
		}
		_used += size;
		return offset;
	}
	return INVALID;
}

void RangeAllocator::free(uint32_t offset, uint32_t size) {
	if (size == 0) {
		return;
	}
	_used -= size;
	insert_free(offset, size);
}

void RangeAllocator::grow(uint32_t new_capacity) {
	if (new_capacity <= _capacity) {
		return;
	}
	const uint32_t old_capacity = _capacity;
	_capacity = new_capacity;
	insert_free(old_capacity, new_capacity - old_capacity);
}

void RangeAllocator::insert_free(uint32_t offset, uint32_t size) {
	auto next = _free.lower_bound(offset);

	// Merge with the previous range if adjacent
	if (next != _free.begin()) {
		auto prev = std::prev(next);
		if (prev->first + prev->second == offset) {
			offset = prev->first;
			size += prev->second;
			_free.erase(prev);
		}
	}

	// Merge with the next range if adjacent
	if (next != _free.end() && offset + size == next->first) {
		size += next->second;
		_free.erase(next);
	}

	_free.emplace(offset, size);
}

} // namespace zylann::voxel::gpu_driven
