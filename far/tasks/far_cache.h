#ifndef VOXEL_FAR_CACHE_H
#define VOXEL_FAR_CACHE_H

#include "../core/far_lod_tree.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "../../util/godot/classes/file_access.h"
#include "../../util/thread/mutex.h"

namespace zylann::voxel::far {

// ---------------------------------------------------------------------------
// Persistence for generated column data.
//
// Why cache at all, when the generator can always be re-run?
//
// Because at the outer LOD levels it cannot, cheaply. A level-10 sector covers
// 32,768 world units per side. Building it by evaluating the generator across
// that span is enormously more expensive than building a level-0 sector, and
// the result never changes. Worse, coarse levels are built by reducing finer
// ones (see `downsample_field`), so producing one level-10 sector from scratch
// means producing the entire pyramid beneath it. Without a cache, every flight
// across the world re-does that work continuously.
//
// Once cached, revisiting an area is a disk read of a few kilobytes.
//
// Format: region files, one per (LOD, region), each holding a 32x32 block of
// sectors. A fixed header table of (offset, size) is followed by the blobs.
//
// Region files rather than one file per sector: a level-0 ring alone is a few
// hundred sectors, and filesystems handle a few large files far better than a
// few hundred thousand small ones. Rather than SQLite -- which is what a mature
// version of this should probably use, and which godot_voxel already bundles --
// this keeps the module free of that build dependency. The trade is that
// rewrites leave dead blobs behind, which is what `_wasted_bytes` and the
// compaction pass exist to handle.
//
// Concurrency: writes come from mesh/generate threads and reads from IO
// threads, so every file handle is behind a mutex. Contention is low because
// the payloads are small and the work either side of the IO dwarfs it.
// ---------------------------------------------------------------------------

class FarCache {
public:
	static constexpr uint32_t REGION_SIZE = 32;
	static constexpr uint32_t SECTORS_PER_REGION = REGION_SIZE * REGION_SIZE;
	static constexpr uint32_t FILE_MAGIC = 0x47524146u; // 'FARG'
	static constexpr uint16_t FILE_VERSION = 1;

	// Compact a region once dead blobs exceed this share of the file.
	static constexpr float COMPACT_WASTE_RATIO = 0.4f;

	~FarCache();

	// `directory` is a Godot path such as "user://far_lod". Created if missing.
	// Returns false if the directory could not be prepared, in which case the
	// cache disables itself and the system runs generate-only.
	bool open(const String &directory);
	void close();

	inline bool is_open() const {
		return _open;
	}

	// Returns false on a miss. A corrupt or truncated entry also reads as a
	// miss: regenerating is always correct, so there is never a reason to
	// surface a cache error to the caller.
	bool load(const SectorId &id, std::vector<uint8_t> &out_bytes);

	bool save(const SectorId &id, const std::vector<uint8_t> &bytes);

	// Drops every cached sector. Used when the generator changes, since cached
	// data no longer describes the world it claims to.
	bool clear_all();

	struct Stats {
		uint64_t loads = 0;
		uint64_t hits = 0;
		uint64_t saves = 0;
		uint64_t bytes_written = 0;
		uint64_t bytes_read = 0;
		uint32_t open_regions = 0;
	};

	Stats get_stats() const;

private:
	struct RegionKey {
		int32_t x;
		int32_t z;
		uint8_t lod;
		uint8_t face;

		inline bool operator==(const RegionKey &o) const {
			return x == o.x && z == o.z && lod == o.lod && face == o.face;
		}
	};

	struct RegionKeyHash {
		inline size_t operator()(const RegionKey &k) const {
			uint64_t h = static_cast<uint64_t>(static_cast<uint32_t>(k.x));
			h = h * 0x9e3779b97f4a7c15ull + static_cast<uint64_t>(static_cast<uint32_t>(k.z));
			h = h * 0x9e3779b97f4a7c15ull + k.lod;
			h = h * 0x9e3779b97f4a7c15ull + k.face;
			h ^= h >> 31;
			return static_cast<size_t>(h);
		}
	};

	struct IndexEntry {
		uint32_t offset = 0;
		uint32_t size = 0;
	};

	struct Region {
		Ref<FileAccess> file;
		std::vector<IndexEntry> index;
		uint64_t end_of_data = 0;
		uint64_t wasted_bytes = 0;
		bool valid = false;
	};

	static RegionKey to_region_key(const SectorId &id);
	static uint32_t to_slot(const SectorId &id);
	String get_region_path(const RegionKey &key) const;

	// Opens or creates a region, writing a fresh header if the file is new or
	// unreadable. Returns null if the file could not be opened at all.
	Region *get_or_open_region(const RegionKey &key);

	bool write_header(Region &region);
	void compact_region(const RegionKey &key, Region &region);

	mutable Mutex _mutex;
	String _directory;
	bool _open = false;
	std::unordered_map<RegionKey, Region, RegionKeyHash> _regions;
	Stats _stats;
};

} // namespace zylann::voxel::far

#endif // VOXEL_FAR_CACHE_H
