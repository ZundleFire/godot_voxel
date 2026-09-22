#include "far_cache.h"

#include "../../util/godot/classes/directory.h"
#include "../../util/godot/classes/file_access.h"
#include "../../util/godot/core/string.h"
#include "../../util/io/log.h"
#include "../../util/profiling.h"
#include "../../util/string/format.h"

namespace zylann::voxel::far {

using namespace zylann::godot;

namespace {

// Header layout:
//   uint32 magic
//   uint16 version
//   uint8  lod
//   uint8  reserved
//   { uint32 offset, uint32 size } * SECTORS_PER_REGION
constexpr uint64_t HEADER_FIXED_BYTES = 4 + 2 + 1 + 1;
constexpr uint64_t HEADER_BYTES = HEADER_FIXED_BYTES + FarCache::SECTORS_PER_REGION * 8;

} // namespace

FarCache::~FarCache() {
	close();
}

FarCache::RegionKey FarCache::to_region_key(const SectorId &id) {
	RegionKey key;
	key.x = floordiv(id.x, static_cast<int32_t>(REGION_SIZE));
	key.z = floordiv(id.z, static_cast<int32_t>(REGION_SIZE));
	key.lod = id.lod;
	key.face = id.face;
	return key;
}

uint32_t FarCache::to_slot(const SectorId &id) {
	const uint32_t lx = static_cast<uint32_t>(floormod(id.x, static_cast<int32_t>(REGION_SIZE)));
	const uint32_t lz = static_cast<uint32_t>(floormod(id.z, static_cast<int32_t>(REGION_SIZE)));
	return lz * REGION_SIZE + lx;
}

String FarCache::get_region_path(const RegionKey &key) const {
	// Face is in the name too, or the six faces of a planet would all share one
	// region file per (lod, x, z) and overwrite each other.
	return _directory + String("/far_l") + String::num_int64(key.lod) + String("_f") +
			String::num_int64(key.face) + String("_") + String::num_int64(key.x) + String("_") +
			String::num_int64(key.z) + String(".bin");
}

bool FarCache::open(const String &directory) {
	MutexLock lock(_mutex);

	close();

	if (!directory_exists(directory)) {
		Error err = OK;
		Ref<DirAccess> dir = open_directory(String("user://"), &err);
		if (dir.is_null()) {
			ZN_PRINT_ERROR("FarCache: could not open user:// to create the cache directory");
			return false;
		}
		if (dir->make_dir_recursive(directory) != OK) {
			ZN_PRINT_ERROR(format("FarCache: could not create directory {}", directory));
			return false;
		}
	}

	_directory = directory;
	_open = true;
	_stats = Stats();
	return true;
}

void FarCache::close() {
	// Not taking the lock here: `close` is called from `open` (which holds it)
	// and from the destructor. Both are single-threaded points in the lifetime.
	for (auto &pair : _regions) {
		Region &region = pair.second;
		if (region.valid && region.file.is_valid()) {
			write_header(region);
			region.file->flush();
		}
		region.file.unref();
	}
	_regions.clear();
	_open = false;
}

bool FarCache::write_header(Region &region) {
	if (!region.file.is_valid()) {
		return false;
	}
	region.file->seek(HEADER_FIXED_BYTES);
	for (const IndexEntry &entry : region.index) {
		region.file->store_32(entry.offset);
		region.file->store_32(entry.size);
	}
	return true;
}

FarCache::Region *FarCache::get_or_open_region(const RegionKey &key) {
	auto it = _regions.find(key);
	if (it != _regions.end()) {
		return it->second.valid ? &it->second : nullptr;
	}

	Region region;
	region.index.assign(SECTORS_PER_REGION, IndexEntry());

	const String path = get_region_path(key);
	const bool existed = FileAccess::exists(path);

	Error err = OK;
	// READ_WRITE does not create the file, so a new region needs WRITE_READ
	// first. Doing it the other way round truncates an existing region, which
	// would silently discard the whole cache on every run.
	region.file = open_file(path, existed ? FileAccess::READ_WRITE : FileAccess::WRITE_READ, err);
	if (region.file.is_null()) {
		ZN_PRINT_ERROR(format("FarCache: could not open region file {}", path));
		Region &stored = _regions[key];
		stored.valid = false;
		return nullptr;
	}

	bool header_ok = false;
	if (existed && region.file->get_length() >= HEADER_BYTES) {
		region.file->seek(0);
		const uint32_t magic = region.file->get_32();
		const uint16_t version = static_cast<uint16_t>(region.file->get_16());
		const uint8_t lod = region.file->get_8();
		region.file->get_8(); // reserved

		if (magic == FILE_MAGIC && version == FILE_VERSION && lod == key.lod) {
			for (uint32_t i = 0; i < SECTORS_PER_REGION; ++i) {
				region.index[i].offset = region.file->get_32();
				region.index[i].size = region.file->get_32();
			}
			region.end_of_data = region.file->get_length();
			header_ok = true;

			// Recover the wasted-byte estimate, so a long-lived cache still
			// compacts after a restart instead of growing without bound.
			uint64_t live = 0;
			for (const IndexEntry &entry : region.index) {
				live += entry.size;
			}
			const uint64_t total = (region.end_of_data > HEADER_BYTES) ? (region.end_of_data - HEADER_BYTES) : 0;
			region.wasted_bytes = (total > live) ? (total - live) : 0;
		}
	}

	if (!header_ok) {
		// New, or unreadable. Either way, start clean: a cache that cannot be
		// parsed is worth exactly as much as no cache, and regenerating is
		// always correct.
		region.file->seek(0);
		region.file->store_32(FILE_MAGIC);
		region.file->store_16(FILE_VERSION);
		region.file->store_8(key.lod);
		region.file->store_8(0);
		for (uint32_t i = 0; i < SECTORS_PER_REGION; ++i) {
			region.file->store_32(0);
			region.file->store_32(0);
		}
		region.end_of_data = HEADER_BYTES;
		region.wasted_bytes = 0;
	}

	region.valid = true;
	Region &stored = _regions[key];
	stored = std::move(region);
	return &stored;
}

bool FarCache::load(const SectorId &id, std::vector<uint8_t> &out_bytes) {
	if (!_open) {
		return false;
	}

	ZN_PROFILE_SCOPE();
	MutexLock lock(_mutex);

	++_stats.loads;

	Region *region = get_or_open_region(to_region_key(id));
	if (region == nullptr) {
		return false;
	}

	const IndexEntry entry = region->index[to_slot(id)];
	if (entry.size == 0) {
		return false;
	}
	// Guard against a truncated file claiming a blob past its end.
	if (entry.offset + entry.size > region->file->get_length()) {
		return false;
	}

	out_bytes.resize(entry.size);
	region->file->seek(entry.offset);
	const uint64_t read = get_buffer(**region->file, Span<uint8_t>(out_bytes.data(), out_bytes.size()));
	if (read != entry.size) {
		out_bytes.clear();
		return false;
	}

	++_stats.hits;
	_stats.bytes_read += read;
	return true;
}

bool FarCache::save(const SectorId &id, const std::vector<uint8_t> &bytes) {
	if (!_open || bytes.empty()) {
		return false;
	}

	ZN_PROFILE_SCOPE();
	MutexLock lock(_mutex);

	const RegionKey key = to_region_key(id);
	Region *region = get_or_open_region(key);
	if (region == nullptr) {
		return false;
	}

	const uint32_t slot = to_slot(id);
	IndexEntry &entry = region->index[slot];

	// Rewriting in place only when the new blob fits exactly. Allowing it to
	// fit "within" the old one would leave unreclaimable holes that the waste
	// counter cannot track.
	const bool in_place = (entry.size == bytes.size() && entry.offset >= HEADER_BYTES);

	if (in_place) {
		region->file->seek(entry.offset);
		store_buffer(**region->file, Span<const uint8_t>(bytes.data(), bytes.size()));
	} else {
		if (entry.size > 0) {
			region->wasted_bytes += entry.size;
		}
		const uint64_t offset = region->end_of_data;
		region->file->seek(offset);
		store_buffer(**region->file, Span<const uint8_t>(bytes.data(), bytes.size()));
		entry.offset = static_cast<uint32_t>(offset);
		entry.size = static_cast<uint32_t>(bytes.size());
		region->end_of_data = offset + bytes.size();

		// The index lives in the header, so it has to be pushed out too. Only
		// this slot's 8 bytes, not the whole table.
		region->file->seek(HEADER_FIXED_BYTES + slot * 8);
		region->file->store_32(entry.offset);
		region->file->store_32(entry.size);
	}

	++_stats.saves;
	_stats.bytes_written += bytes.size();

	const uint64_t payload = (region->end_of_data > HEADER_BYTES) ? (region->end_of_data - HEADER_BYTES) : 1;
	if (region->wasted_bytes > 65536 &&
			static_cast<float>(region->wasted_bytes) / static_cast<float>(payload) > COMPACT_WASTE_RATIO) {
		compact_region(key, *region);
	}

	return true;
}

void FarCache::compact_region(const RegionKey &key, Region &region) {
	ZN_PROFILE_SCOPE();

	// Read every live blob, then rewrite the file with them packed together.
	// Regions are a few megabytes at most, so buffering the live set in memory
	// is simpler and faster than an in-place shuffle, and it means a failure
	// part way through leaves the original file untouched.
	std::vector<std::vector<uint8_t>> blobs(SECTORS_PER_REGION);
	for (uint32_t i = 0; i < SECTORS_PER_REGION; ++i) {
		const IndexEntry &entry = region.index[i];
		if (entry.size == 0) {
			continue;
		}
		if (entry.offset + entry.size > region.file->get_length()) {
			continue;
		}
		blobs[i].resize(entry.size);
		region.file->seek(entry.offset);
		if (get_buffer(**region.file, Span<uint8_t>(blobs[i].data(), blobs[i].size())) != entry.size) {
			blobs[i].clear();
		}
	}

	region.file.unref();

	const String path = get_region_path(key);
	Error err = OK;
	Ref<FileAccess> file = open_file(path, FileAccess::WRITE_READ, err);
	if (file.is_null()) {
		ZN_PRINT_ERROR(format("FarCache: compaction could not reopen {}", path));
		region.valid = false;
		return;
	}

	file->seek(0);
	file->store_32(FILE_MAGIC);
	file->store_16(FILE_VERSION);
	file->store_8(key.lod);
	file->store_8(0);
	for (uint32_t i = 0; i < SECTORS_PER_REGION; ++i) {
		file->store_32(0);
		file->store_32(0);
	}

	uint64_t cursor = HEADER_BYTES;
	for (uint32_t i = 0; i < SECTORS_PER_REGION; ++i) {
		if (blobs[i].empty()) {
			region.index[i] = IndexEntry();
			continue;
		}
		file->seek(cursor);
		store_buffer(**file, Span<const uint8_t>(blobs[i].data(), blobs[i].size()));
		region.index[i].offset = static_cast<uint32_t>(cursor);
		region.index[i].size = static_cast<uint32_t>(blobs[i].size());
		cursor += blobs[i].size();
	}

	region.file = file;
	region.end_of_data = cursor;
	region.wasted_bytes = 0;
	write_header(region);
	file->flush();
}

bool FarCache::clear_all() {
	MutexLock lock(_mutex);

	if (!_open) {
		return false;
	}

	const String directory = _directory;
	close();

	Error err = OK;
	Ref<DirAccess> dir = open_directory(directory, &err);
	if (dir.is_null()) {
		return false;
	}
	dir->erase_contents_recursive();

	_directory = directory;
	_open = true;
	_stats = Stats();
	return true;
}

FarCache::Stats FarCache::get_stats() const {
	MutexLock lock(_mutex);
	Stats stats = _stats;
	stats.open_regions = static_cast<uint32_t>(_regions.size());
	return stats;
}

} // namespace zylann::voxel::far
