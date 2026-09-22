#include "far_lod_tree.h"
#include "far_sphere.h"

#include <algorithm>

namespace zylann::voxel::far {

namespace {

// Widens [centre - radius, centre + radius] so that it begins on an even
// sector coordinate and ends on an odd one.
//
// The point is that such a range always contains a whole number of parent
// sectors, which is what lets a level's coverage be subtracted exactly from
// its parent's. See the hole computation in `gather_required` for why an
// unsnapped range cannot tile.
//
// The range grows by at most one sector on each side, so the ring is between
// 2R+1 and 2R+2 sectors across.
inline void snapped_ring(int centre, int radius, int &out_min, int &out_max) {
	int lo = centre - radius;
	int hi = centre + radius;
	lo -= floormod(lo, 2); // down to even
	hi += 1 - floormod(hi, 2); // up to odd
	out_min = lo;
	out_max = hi;
}

} // namespace

unsigned int FarLodSettings::get_max_resident_sectors() const {
	// Each level keeps a snapped ring (at most 2R+2 sectors per side) minus the
	// hole covered by the finer level. Level 0 has no hole, so ignoring the
	// holes gives a safe upper bound -- which is what a memory budget wants.
	// The +1 mirrors the one in the unload band: snapping can widen a ring by
	// one sector per side, and sectors inside the band stay resident.
	const unsigned int r = ring_radius_sectors + hysteresis_sectors + 1;
	const unsigned int side = 2 * r + 1;
	return side * side * std::max(lod_count, 1u);
}

void FarLodTree::configure(const FarLodSettings &settings) {
	_settings = settings;
	if (_settings.lod_count == 0) {
		_settings.lod_count = 1;
	}
	if (_settings.lod_count > MAX_LOD_COUNT) {
		_settings.lod_count = MAX_LOD_COUNT;
	}
	if (_settings.ring_radius_sectors == 0) {
		_settings.ring_radius_sectors = 1;
	}
	// A configuration change invalidates residency: sectors from the old
	// layout may not exist in the new one.
	clear();
}

void FarLodTree::clear() {
	_resident.clear();
	_deferred.clear();
}

bool FarLodTree::is_clipped(const SectorId &id, const Vec3f &viewer_pos) const {
	const float clip = _settings.near_clip_radius;
	if (clip <= 0.f) {
		return false;
	}

	if (_settings.spherical) {
		// Same rule as below -- clipped only when the whole footprint is inside
		// the radius -- measured as distance *along the surface*, not through
		// space.
		//
		// The flat version below deliberately ignores Y and compares in XZ only,
		// and this is the same choice one dimension up. Measuring in 3D instead
		// looks more correct and is not: the corner positions have to be placed
		// at some assumed radius, and the only one available is the middle of
		// the terrain's radial shell. On a shell a few thousand units thick the
		// difference between that guess and the real ground swamps a clip radius
		// of a few hundred, so nothing is ever clipped and the far field quietly
		// draws underneath the near terrain everywhere.
		//
		// Distance along the surface has no such dependency: it is an angle,
		// scaled by the reference radius only to put it in world units.
		const int32_t n = sectors_per_face_side(id.lod, _settings.sphere_root_lod);
		const float inv_n = 1.f / static_cast<float>(n);
		const float u0 = static_cast<float>(id.x) * 2.f * inv_n - 1.f;
		const float u1 = static_cast<float>(id.x + 1) * 2.f * inv_n - 1.f;
		const float v0 = static_cast<float>(id.z) * 2.f * inv_n - 1.f;
		const float v1 = static_cast<float>(id.z + 1) * 2.f * inv_n - 1.f;

		const Vec3f viewer_dir = (viewer_pos - _settings.planet_centre).normalized();

		const float corners[4][2] = { { u0, v0 }, { u1, v0 }, { u0, v1 }, { u1, v1 } };
		float furthest = 0.f;
		for (const auto &c : corners) {
			const Vec3f d = face_uv_to_dir(id.face, c[0], c[1]);
			// Via the chord: acos(dot) throws away most of its precision at the
			// small angles a clip radius covers on a planet.
			const float chord = (d - viewer_dir).length();
			const float angle = 2.f * std::asin(std::min(1.f, chord * 0.5f));
			furthest = std::max(furthest, angle * _settings.sphere_reference_radius);
		}
		return furthest <= clip;
	}

	// Clipped only when the whole footprint is inside the radius. A sector that
	// merely overlaps must still be drawn, or the boundary between near and far
	// terrain becomes a ring of missing ground.
	const float size = id.get_world_size();
	const float min_x = id.get_world_x();
	const float min_z = id.get_world_z();
	const float max_x = min_x + size;
	const float max_z = min_z + size;

	// Furthest corner from the viewer in XZ.
	const float dx = std::max(std::abs(min_x - viewer_pos.x), std::abs(max_x - viewer_pos.x));
	const float dz = std::max(std::abs(min_z - viewer_pos.z), std::abs(max_z - viewer_pos.z));

	return (dx * dx + dz * dz) <= clip * clip;
}


// Centre direction of a sector on the sphere.
static Vec3f sector_centre_dir(const SectorId &id, uint8_t root_lod) {
	const int32_t n = sectors_per_face_side(id.lod, root_lod);
	const float inv_n = 1.f / static_cast<float>(n);
	const float u = (static_cast<float>(id.x) + 0.5f) * 2.f * inv_n - 1.f;
	const float v = (static_cast<float>(id.z) + 0.5f) * 2.f * inv_n - 1.f;
	return face_uv_to_dir(id.face, u, v);
}

float FarLodTree::sector_distance(const SectorId &id, const Vec3f &viewer_pos) const {
	if (!_settings.spherical) {
		const float sector_size = id.get_world_size();
		const float dx = id.get_world_x() + sector_size * 0.5f - viewer_pos.x;
		const float dz = id.get_world_z() + sector_size * 0.5f - viewer_pos.z;
		return std::sqrt(dx * dx + dz * dz);
	}
	const Vec3f centre = _settings.planet_centre +
			sector_centre_dir(id, _settings.sphere_root_lod) * _settings.sphere_reference_radius;
	return (centre - viewer_pos).length();
}

// Ring selection on a planet.
//
// Same idea as the flat rings and for the same reason (README 2.3): the
// required set stays a pure function of viewer position, recomputable from
// scratch every frame, so nothing can drift out of sync. What changes is the
// coordinate system it is expressed in.
//
// Per level, take the square ring of sectors around the one the viewer is over,
// in that face's own grid. Coordinates that run off an edge are moved onto the
// neighbouring face by `remap_sector_across_faces()`. Two things differ from
// the flat case:
//
//   * A face is finite. Once the ring is as wide as the face, it wraps around
//     and starts naming the same sectors twice, so at that point the answer is
//     simply "every sector on all six faces" -- which is at most 6 * n^2 and
//     only happens at the coarsest levels, where n is tiny.
//
//   * A cube corner joins three faces, not four, so the ring's corner cells
//     sometimes name nothing. Those are skipped.
//
// The hole a level leaves for the finer one is computed the same way as the
// flat case: the finer level's range mapped up one level. Ring snapping
// (README 3.2) still applies and still matters -- a face grid is a power of two
// on a side, so an unsnapped ring tiles no better here than it does on a plane.
void FarLodTree::gather_required_spherical(const Vec3f &viewer_pos, std::vector<SectorRequest> &out_required) const {
	const Vec3f from_centre = viewer_pos - _settings.planet_centre;
	unsigned int viewer_face = 0;
	float viewer_u = 0.f;
	float viewer_v = 0.f;
	dir_to_face_uv(from_centre, viewer_face, viewer_u, viewer_v);

	const int radius = static_cast<int>(_settings.ring_radius_sectors);
	const unsigned int first = _settings.first_lod;
	const unsigned int last = first + _settings.lod_count;

	// A square ring around a cube corner names some sectors twice: only three
	// faces meet there, so the ring's four corner quadrants fold onto three.
	// Cheaper and clearer to drop the repeats than to special-case the fold.
	std::unordered_set<SectorId, SectorIdHash> emitted;

	for (unsigned int lod = first; lod < last; ++lod) {
		const int32_t n = sectors_per_face_side(static_cast<uint8_t>(lod), _settings.sphere_root_lod);

		// Extent of the finer level, so this one can leave it a hole.
		const bool has_finer = (lod > first);
		const int32_t finer_n = has_finer
				? sectors_per_face_side(static_cast<uint8_t>(lod - 1), _settings.sphere_root_lod)
				: 0;
		const bool finer_covers_everything = has_finer && (finer_n <= 2 * radius + 1);

		if (n <= 2 * radius + 1) {
			// The ring is at least as wide as a face. Enumerate the whole
			// sphere rather than wrapping a square ring around it repeatedly.
			if (finer_covers_everything) {
				// The finer level already draws every sector there is.
				continue;
			}
			for (unsigned int face = 0; face < CUBE_FACE_COUNT; ++face) {
				for (int32_t sz = 0; sz < n; ++sz) {
					for (int32_t sx = 0; sx < n; ++sx) {
						const SectorId id(sx, sz, static_cast<uint8_t>(lod), static_cast<uint8_t>(face));
						if (has_finer && is_covered_by_finer(id, viewer_pos, radius)) {
							continue;
						}
						if (!emitted.insert(id).second) {
							continue;
						}
						out_required.push_back(make_request(id, viewer_pos));
					}
				}
			}
			continue;
		}

		// Sector the viewer is over, at this level.
		const int32_t centre_x = static_cast<int32_t>(std::floor((viewer_u + 1.f) * 0.5f * static_cast<float>(n)));
		const int32_t centre_z = static_cast<int32_t>(std::floor((viewer_v + 1.f) * 0.5f * static_cast<float>(n)));

		int min_x = 0;
		int max_x = 0;
		int min_z = 0;
		int max_z = 0;
		snapped_ring(static_cast<int>(centre_x), radius, min_x, max_x);
		snapped_ring(static_cast<int>(centre_z), radius, min_z, max_z);

		for (int sz = min_z; sz <= max_z; ++sz) {
			for (int sx = min_x; sx <= max_x; ++sx) {
				unsigned int face = 0;
				int32_t fx = 0;
				int32_t fz = 0;
				if (!remap_sector_across_faces(viewer_face, sx, sz, n, face, fx, fz)) {
					// Diagonal past a cube corner: no such sector.
					continue;
				}

				const SectorId id(fx, fz, static_cast<uint8_t>(lod), static_cast<uint8_t>(face));
				if (has_finer && is_covered_by_finer(id, viewer_pos, radius)) {
					continue;
				}
				if (!emitted.insert(id).second) {
					continue;
				}
				out_required.push_back(make_request(id, viewer_pos));
			}
		}
	}
}

// Whether the level one finer already covers this sector, so it must leave a
// hole. Works by asking where the sector's four children fall in the finer
// level's ring rather than by mapping integer ranges: across a face boundary the
// two levels' grids do not share an origin, so range arithmetic would be wrong.
bool FarLodTree::is_covered_by_finer(const SectorId &id, const Vec3f &viewer_pos, int radius) const {
	const uint8_t finer_lod = static_cast<uint8_t>(id.lod - 1);
	const int32_t n = sectors_per_face_side(finer_lod, _settings.sphere_root_lod);

	const Vec3f from_centre = viewer_pos - _settings.planet_centre;
	unsigned int viewer_face = 0;
	float viewer_u = 0.f;
	float viewer_v = 0.f;
	dir_to_face_uv(from_centre, viewer_face, viewer_u, viewer_v);

	if (n <= 2 * radius + 1) {
		// The finer level covers the whole sphere.
		return true;
	}

	const int32_t centre_x = static_cast<int32_t>(std::floor((viewer_u + 1.f) * 0.5f * static_cast<float>(n)));
	const int32_t centre_z = static_cast<int32_t>(std::floor((viewer_v + 1.f) * 0.5f * static_cast<float>(n)));

	int min_x = 0;
	int max_x = 0;
	int min_z = 0;
	int max_z = 0;
	snapped_ring(static_cast<int>(centre_x), radius, min_x, max_x);
	snapped_ring(static_cast<int>(centre_z), radius, min_z, max_z);

	// All four children must be inside the finer ring, or this sector still has
	// ground the finer level is not drawing.
	for (unsigned int c = 0; c < 4; ++c) {
		const SectorId child = id.get_child(c);

		bool inside = false;
		if (child.face == viewer_face) {
			inside = child.x >= min_x && child.x <= max_x && child.z >= min_z && child.z <= max_z;
		} else {
			// Off the viewer's face: check whether any ring coordinate maps to
			// it. Cheaper than it looks -- the ring is at most (2r+2)^2 and this
			// only runs for sectors on a face boundary.
			for (int sz = min_z; sz <= max_z && !inside; ++sz) {
				for (int sx = min_x; sx <= max_x && !inside; ++sx) {
					unsigned int f = 0;
					int32_t fx = 0;
					int32_t fz = 0;
					if (!remap_sector_across_faces(viewer_face, sx, sz, n, f, fx, fz)) {
						continue;
					}
					inside = (f == child.face && fx == child.x && fz == child.z);
				}
			}
		}

		if (!inside) {
			return false;
		}
	}
	return true;
}

SectorRequest FarLodTree::make_request(const SectorId &id, const Vec3f &viewer_pos) const {
	SectorRequest request;
	request.id = id;
	request.distance = sector_distance(id, viewer_pos);
	request.visible = !is_clipped(id, viewer_pos);
	return request;
}

void FarLodTree::gather_required(const Vec3f &viewer_pos, std::vector<SectorRequest> &out_required) const {
	out_required.clear();
	if (_settings.spherical) {
		gather_required_spherical(viewer_pos, out_required);
	} else {
		gather_required_flat(viewer_pos, out_required);
	}
}

void FarLodTree::gather_required_flat(const Vec3f &viewer_pos, std::vector<SectorRequest> &out_required) const {
	out_required.clear();

	const int radius = static_cast<int>(_settings.ring_radius_sectors);

	const unsigned int first = _settings.first_lod;
	const unsigned int last = first + _settings.lod_count;

	for (unsigned int lod = first; lod < last; ++lod) {
		const SectorId probe(0, 0, static_cast<uint8_t>(lod));
		const float sector_size = probe.get_world_size();

		// Sector the viewer is standing in, at this level.
		const int centre_x = static_cast<int>(std::floor(viewer_pos.x / sector_size));
		const int centre_z = static_cast<int>(std::floor(viewer_pos.z / sector_size));

		int min_x = 0;
		int max_x = 0;
		int min_z = 0;
		int max_z = 0;
		snapped_ring(centre_x, radius, min_x, max_x);
		snapped_ring(centre_z, radius, min_z, max_z);

		// The hole this level leaves for the finer one.
		//
		// This is the arithmetic that decides whether the rings tile. An
		// unsnapped ring cannot: a range of 2R+1 sectors has odd length, so its
		// edges fall in the middle of the parent's sectors, and whichever way
		// the conversion rounds you get either a gap (nothing drawn) or a
		// double-covered band (two levels drawing the same ground, z-fighting
		// along every boundary). Rounding conservatively does not fix it, it
		// just picks overlap over gaps.
		//
		// `snapped_ring` sidesteps it by widening each level's range to start
		// on an even coordinate and end on an odd one. Such a range always maps
		// to a whole number of parent sectors, so the hole is exact and the two
		// levels meet edge to edge.
		// The innermost level leaves no hole: nothing finer exists to fill it.
		const bool has_finer = (lod > first);
		int hole_min_x = 0;
		int hole_max_x = -1;
		int hole_min_z = 0;
		int hole_max_z = -1;

		if (has_finer) {
			const SectorId finer_probe(0, 0, static_cast<uint8_t>(lod - 1));
			const float finer_size = finer_probe.get_world_size();
			const int finer_centre_x = static_cast<int>(std::floor(viewer_pos.x / finer_size));
			const int finer_centre_z = static_cast<int>(std::floor(viewer_pos.z / finer_size));

			int finer_min_x = 0;
			int finer_max_x = 0;
			int finer_min_z = 0;
			int finer_max_z = 0;
			snapped_ring(finer_centre_x, radius, finer_min_x, finer_max_x);
			snapped_ring(finer_centre_z, radius, finer_min_z, finer_max_z);

			// finer_min is even and finer_max is odd, so both divisions are
			// exact and the hole covers precisely the finer level's extent.
			hole_min_x = floordiv(finer_min_x, 2);
			hole_max_x = floordiv(finer_max_x - 1, 2);
			hole_min_z = floordiv(finer_min_z, 2);
			hole_max_z = floordiv(finer_max_z - 1, 2);
		}

		for (int sz = min_z; sz <= max_z; ++sz) {
			for (int sx = min_x; sx <= max_x; ++sx) {
				if (has_finer && sx >= hole_min_x && sx <= hole_max_x && sz >= hole_min_z && sz <= hole_max_z) {
					continue;
				}

				SectorRequest request;
				request.id = SectorId(sx, sz, static_cast<uint8_t>(lod));

				const float centre_world_x = (static_cast<float>(sx) + 0.5f) * sector_size;
				const float centre_world_z = (static_cast<float>(sz) + 0.5f) * sector_size;
				const float dx = centre_world_x - viewer_pos.x;
				const float dz = centre_world_z - viewer_pos.z;
				request.distance = std::sqrt(dx * dx + dz * dz);
				request.visible = !is_clipped(request.id, viewer_pos);

				out_required.push_back(request);
			}
		}
	}
}

void FarLodTree::update(
		const Vec3f &viewer_pos,
		std::vector<SectorRequest> &out_to_load,
		std::vector<SectorId> &out_to_unload
) {
	out_to_load.clear();
	out_to_unload.clear();

	gather_required(viewer_pos, _required_list_scratch);

	_required_scratch.clear();
	for (const SectorRequest &request : _required_list_scratch) {
		_required_scratch.insert(request.id);
	}

	// Anything required but not resident has to be loaded.
	for (const SectorRequest &request : _required_list_scratch) {
		if (_resident.find(request.id) != _resident.end()) {
			continue;
		}
		if (!request.visible) {
			// Fully under the near terrain. Generating it would be pure waste,
			// and the most expensive kind -- these are the finest levels. Park
			// it instead of marking it resident: an unbuilt sector recorded as
			// resident is never offered for loading again, so it stays missing
			// for good once the viewer moves and it comes out from under the
			// near terrain.
			_deferred.insert(request.id);
			continue;
		}
		_deferred.erase(request.id);
		out_to_load.push_back(request);
	}

	// Anything resident but no longer required is a candidate for unloading --
	// subject to the hysteresis band, so a viewer pacing across a boundary does
	// not thrash the same ring in and out.
	//
	// The +1 is not slack, it is required. Ring snapping means a level's range
	// shifts by two sectors when the viewer crosses one sector boundary, so the
	// sector that drops out of the required set can be `radius + 1` away from
	// the new centre. Without it, a single step across a boundary unloads two
	// full edges of every level, which is exactly the thrashing the band exists
	// to prevent.
	const int radius = static_cast<int>(_settings.ring_radius_sectors + _settings.hysteresis_sectors) + 1;

	for (const SectorId &id : _resident) {
		if (_required_scratch.find(id) != _required_scratch.end()) {
			continue;
		}

		// A sector from a level that is no longer configured is dropped
		// immediately -- the band only protects levels that still exist.
		const bool level_still_active =
				id.lod >= _settings.first_lod && id.lod < _settings.first_lod + _settings.lod_count;

		bool within_band = false;
		if (_settings.spherical) {
			// Grid coordinates are per face here, so a coordinate difference
			// means nothing across a face boundary. Measure the band in world
			// units instead, which is the same quantity the flat check
			// approximates.
			constexpr float HALF_PI = 1.57079632679f;
			const int32_t n = sectors_per_face_side(id.lod, _settings.sphere_root_lod);
			const float sector_size = (HALF_PI / static_cast<float>(n)) * _settings.sphere_reference_radius;
			within_band = level_still_active &&
					sector_distance(id, viewer_pos) <= sector_size * static_cast<float>(radius);
		} else {
			const float sector_size = id.get_world_size();
			const int centre_x = static_cast<int>(std::floor(viewer_pos.x / sector_size));
			const int centre_z = static_cast<int>(std::floor(viewer_pos.z / sector_size));
			within_band = level_still_active && std::abs(id.x - centre_x) <= radius &&
					std::abs(id.z - centre_z) <= radius;
		}

		if (!within_band) {
			out_to_unload.push_back(id);
		}
	}

	for (const SectorId &id : out_to_unload) {
		_resident.erase(id);
	}

	// Deferred sectors that are no longer required stop being tracked, or the
	// set grows without bound as the viewer travels.
	for (auto it = _deferred.begin(); it != _deferred.end();) {
		if (_required_scratch.find(*it) == _required_scratch.end()) {
			it = _deferred.erase(it);
		} else {
			++it;
		}
	}

	for (const SectorRequest &request : out_to_load) {
		_resident.insert(request.id);
	}

	// Nearest first: the ground the viewer is standing on should resolve before
	// the horizon does.
	std::sort(out_to_load.begin(), out_to_load.end(), [](const SectorRequest &a, const SectorRequest &b) {
		if (a.id.lod != b.id.lod) {
			return a.id.lod < b.id.lod;
		}
		return a.distance < b.distance;
	});
}

} // namespace zylann::voxel::far
