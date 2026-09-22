#ifndef VOXEL_FAR_LOD_TREE_H
#define VOXEL_FAR_LOD_TREE_H

#include "far_column_data.h"
#include "far_sphere.h"
#include "far_math.h"

#include <cstdint>
#include <unordered_set>
#include <vector>

namespace zylann::voxel::far {

// ---------------------------------------------------------------------------
// Which sectors should exist, at which LOD, given where the viewer is.
//
// This is a clipmap rather than a quadtree, and that is a deliberate
// simplification worth explaining.
//
// A quadtree is the obvious structure: one root, subdivide toward the viewer,
// merge away from it. It is also a stateful graph that has to be kept
// consistent while nodes are asynchronously loading, and that is where far-LOD
// systems usually go wrong -- a node that split while its children were still
// generating leaves a hole, and a node that merged while its parent was still
// generating leaves a hole too. Handling that correctly means every split and
// merge becomes a small transaction with a pending state.
//
// Concentric rings get the same result with no state machine at all. For every
// level, keep the sectors within a fixed radius of the viewer, minus whatever
// the finer level already covers. The required set is a pure function of the
// viewer position, so it can be recomputed from scratch on any frame and diffed
// against what is currently resident. Nothing can be inconsistent, because
// there is no incremental state to become inconsistent.
//
// The cost is that each level keeps a fixed ring of sectors instead of adapting
// its shape to the terrain. At a few hundred sectors per level that is a price
// worth paying, and it makes the memory ceiling exactly predictable, which a
// quadtree's is not.
//
// Total coverage: with `ring_radius_sectors = R` and `lod_count = N`, the
// outermost level reaches R * SECTOR_CELLS * 2^(N-1) world units. At R = 4,
// SECTOR_CELLS = 32 and N = 10 that is 65,536 units in every direction, for a
// bounded ~(2R+1)^2 * N sectors resident.
// ---------------------------------------------------------------------------

struct SectorId {
	int32_t x = 0;
	int32_t z = 0;
	uint8_t lod = 0;
	// Which cube face this sector lives on, for a planet. Always 0 on a flat
	// world, where there is a single unbounded grid and `x`/`z` may be negative.
	// On a sphere each face has its own grid with x, z in [0, sectors_per_side).
	uint8_t face = 0;

	SectorId() = default;
	SectorId(int32_t p_x, int32_t p_z, uint8_t p_lod) : x(p_x), z(p_z), lod(p_lod) {}
	SectorId(int32_t p_x, int32_t p_z, uint8_t p_lod, uint8_t p_face) :
			x(p_x), z(p_z), lod(p_lod), face(p_face) {}

	inline bool operator==(const SectorId &o) const {
		return x == o.x && z == o.z && lod == o.lod && face == o.face;
	}
	inline bool operator!=(const SectorId &o) const {
		return !(*this == o);
	}

	// The four sectors one level finer that tile this one exactly.
	inline SectorId get_child(unsigned int index) const {
		return SectorId(
				x * 2 + static_cast<int32_t>(index & 1u), z * 2 + static_cast<int32_t>(index >> 1), lod - 1, face
		);
	}

	inline SectorId get_parent() const {
		return SectorId(floordiv(x, 2), floordiv(z, 2), lod + 1, face);
	}

	// World-space size of one side of this sector, in units.
	inline float get_world_size() const {
		return static_cast<float>(SECTOR_CELLS) * static_cast<float>(1u << lod);
	}

	// World-space X/Z of the sector's minimum corner.
	inline float get_world_x() const {
		return static_cast<float>(x) * get_world_size();
	}
	inline float get_world_z() const {
		return static_cast<float>(z) * get_world_size();
	}

	// Distance in world units between adjacent columns inside this sector.
	inline float get_cell_size() const {
		return static_cast<float>(1u << lod);
	}
};

struct SectorIdHash {
	inline size_t operator()(const SectorId &id) const {
		// Mixing rather than concatenating: sector coordinates near the origin
		// are small and highly correlated, which a naive shift-and-xor turns
		// into heavy bucket clustering exactly where the viewer usually is.
		uint64_t h = static_cast<uint64_t>(static_cast<uint32_t>(id.x));
		h = h * 0x9e3779b97f4a7c15ull + static_cast<uint64_t>(static_cast<uint32_t>(id.z));
		h = h * 0x9e3779b97f4a7c15ull + id.lod;
		h = h * 0x9e3779b97f4a7c15ull + id.face;
		h ^= h >> 29;
		h *= 0xbf58476d1ce4e5b9ull;
		h ^= h >> 32;
		return static_cast<size_t>(h);
	}
};

struct FarLodSettings {
	// Finest level this system will produce. Levels run
	// [first_lod, first_lod + lod_count).
	//
	// This exists because the fine levels are usually pure waste. The far field
	// is a companion to a near terrain that already draws everything within its
	// own view distance, so every level whose ring fits inside that radius is
	// generated, stored and then not drawn. The fine levels are also by far the
	// most expensive: a level-0 sector spanning the full vertical range of the
	// world needs (range) Y samples per column, where a level-6 sector needs
	// 1/64th as many.
	//
	// Set it so that the innermost ring lands just inside the near terrain's
	// edge: first_lod such that ring_radius_sectors * SECTOR_CELLS * 2^first_lod
	// is a little under the near view distance.
	uint8_t first_lod = 0;

	// Number of detail levels, starting at `first_lod`. Cell size at level N is
	// 2^N world units.
	unsigned int lod_count = 8;

	// How many sectors to keep on each side of the viewer, per level. The ring
	// is (2R+1)^2 sectors minus the hole covered by the finer level.
	unsigned int ring_radius_sectors = 4;

	// Sectors entirely inside this radius are not rendered: the near terrain is
	// already drawing that ground, and drawing it twice costs fill rate and
	// produces z-fighting wherever the two surfaces nearly coincide.
	//
	// Set this from the near terrain's own view distance. Slightly less, so the
	// far field starts underneath the near field rather than beyond it --
	// a gap between the two is a visible ring of missing world, whereas an
	// overlap is hidden by the near terrain drawing on top.
	float near_clip_radius = 0.f;

	// -- Planet mode --------------------------------------------------------
	// When set, sectors live on the six faces of a cubesphere around
	// `planet_centre` rather than on one unbounded XZ grid. `first_lod` keeps
	// its meaning: cells stay about 2^first_lod units across.
	bool spherical = false;
	Vec3f planet_centre;
	// LOD at which one sector covers a whole cube face. Derive it with
	// `choose_sphere_root_lod()`.
	uint8_t sphere_root_lod = 1;
	// Radius the sector footprints are measured at, for distance and clipping.
	float sphere_reference_radius = 0.f;

	// Sectors are kept until they exceed the load radius by this many sectors.
	// Without the band, a viewer standing on a boundary loads and unloads the
	// same ring every time they step back and forth.
	unsigned int hysteresis_sectors = 1;

	unsigned int get_max_resident_sectors() const;
};

// One entry of the required set.
struct SectorRequest {
	SectorId id;
	// Distance from the viewer to the sector's centre, used to prioritise
	// generation so the ground the viewer is looking at fills in first.
	float distance = 0.f;
	// False when the sector lies entirely within `near_clip_radius`. Rings tile
	// rather than nest, so nothing is built from such a sector either: it is
	// deferred rather than generated, and picked up if the viewer moves and it
	// comes out from under the near terrain.
	bool visible = true;
};

class FarLodTree {
public:
	void configure(const FarLodSettings &settings);

	inline const FarLodSettings &get_settings() const {
		return _settings;
	}

	// Recomputes the required set for `viewer_pos` and diffs it against what is
	// currently resident.
	//
	// `out_to_load` comes back sorted nearest-first, so the caller can feed it
	// straight into a priority queue without re-sorting.
	void update(
			const Vec3f &viewer_pos,
			std::vector<SectorRequest> &out_to_load,
			std::vector<SectorId> &out_to_unload
	);

	// Drops all residency state. The next update reloads everything.
	void clear();

	inline const std::unordered_set<SectorId, SectorIdHash> &get_resident() const {
		return _resident;
	}

	inline size_t get_resident_count() const {
		return _resident.size();
	}

	// Whether a sector is inside the near terrain's coverage and so must not be
	// drawn. Exposed because the renderer re-checks it as the viewer moves,
	// without waiting for a reload.
	bool is_clipped(const SectorId &id, const Vec3f &viewer_pos) const;

private:
	void gather_required(const Vec3f &viewer_pos, std::vector<SectorRequest> &out_required) const;
	void gather_required_flat(const Vec3f &viewer_pos, std::vector<SectorRequest> &out_required) const;
	void gather_required_spherical(const Vec3f &viewer_pos, std::vector<SectorRequest> &out_required) const;
	bool is_covered_by_finer(const SectorId &id, const Vec3f &viewer_pos, int radius) const;
	SectorRequest make_request(const SectorId &id, const Vec3f &viewer_pos) const;
	// Distance from the viewer to the sector's centre, in world units.
	float sector_distance(const SectorId &id, const Vec3f &viewer_pos) const;

	FarLodSettings _settings;
	// Residency means "built or building", never merely "asked for".
	std::unordered_set<SectorId, SectorIdHash> _resident;
	// Required but fully hidden under the near terrain, so deliberately not
	// built. Held separately from `_resident` so that it is re-offered for
	// loading the moment the viewer moves and it becomes visible.
	std::unordered_set<SectorId, SectorIdHash> _deferred;
	// Scratch, kept across calls so a per-frame update does not reallocate.
	mutable std::unordered_set<SectorId, SectorIdHash> _required_scratch;
	mutable std::vector<SectorRequest> _required_list_scratch;
};

} // namespace zylann::voxel::far

#endif // VOXEL_FAR_LOD_TREE_H
