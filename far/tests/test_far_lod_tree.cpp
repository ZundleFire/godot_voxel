// Host-compilable tests for the LOD ring selection.
//
// The property that matters most here is coverage: every world position inside
// the outermost ring must be covered by exactly one level. A gap is a hole in
// the horizon; an overlap is double-drawn geometry that z-fights. The hole
// arithmetic that produces this is the easiest thing in the module to get
// subtly wrong, and the hardest to notice by eye, so it is tested directly.

#include "../core/far_lod_tree.h"
#include "../core/far_sphere.h"

#include <unordered_set>

#include <cstdio>
#include <map>
#include <vector>

using namespace zylann::voxel::far;

static int g_failures = 0;
static int g_checks = 0;
static const char *g_current_test = "";

#define CHECK(cond)                                                                                                    \
	do {                                                                                                               \
		++g_checks;                                                                                                    \
		if (!(cond)) {                                                                                                 \
			++g_failures;                                                                                              \
			std::printf("  FAIL [%s] %s:%d: %s\n", g_current_test, __FILE__, __LINE__, #cond);                         \
		}                                                                                                              \
	} while (0)

#define TEST(name)                                                                                                     \
	g_current_test = name;                                                                                             \
	std::printf("- %s\n", name);

namespace {

struct Box2 {
	float min_x, min_z, max_x, max_z;
	inline bool contains(float x, float z) const {
		return x >= min_x && x < max_x && z >= min_z && z < max_z;
	}
};

Box2 sector_box(const SectorId &id) {
	const float size = id.get_world_size();
	return Box2{ id.get_world_x(), id.get_world_z(), id.get_world_x() + size, id.get_world_z() + size };
}

// Returns the full required set for a viewer position, as a fresh tree so the
// result is independent of residency history.
std::vector<SectorRequest> required_set(const FarLodSettings &settings, const Vec3f &viewer) {
	FarLodTree tree;
	tree.configure(settings);
	std::vector<SectorRequest> to_load;
	std::vector<SectorId> to_unload;
	tree.update(viewer, to_load, to_unload);
	return to_load;
}

} // namespace

// ---------------------------------------------------------------------------

static void test_coverage_has_no_gaps_or_overlaps() {
	TEST("rings tile the covered area exactly once");

	FarLodSettings settings;
	settings.lod_count = 6;
	settings.ring_radius_sectors = 3;

	// Several viewer positions, including ones deliberately off-grid and
	// negative, since floor-division around the origin is where this kind of
	// arithmetic usually breaks.
	const Vec3f VIEWERS[] = {
		Vec3f(0.f, 0.f, 0.f),
		Vec3f(17.f, 0.f, -43.f),
		Vec3f(-1.f, 0.f, -1.f),
		Vec3f(-5000.f, 0.f, 3333.f),
		Vec3f(129.5f, 0.f, -64.25f),
	};

	for (const Vec3f &viewer : VIEWERS) {
		const std::vector<SectorRequest> required = required_set(settings, viewer);
		CHECK(!required.empty());

		// The innermost level's ring defines the area that must be covered
		// without gaps all the way out to the coarsest ring.
		std::vector<Box2> boxes;
		boxes.reserve(required.size());
		for (const SectorRequest &r : required) {
			boxes.push_back(sector_box(r.id));
		}

		// Sample the area covered by level 0 and by each ring, and count how
		// many sectors contain each sample.
		const SectorId coarsest(0, 0, static_cast<uint8_t>(settings.lod_count - 1));
		const float reach = static_cast<float>(settings.ring_radius_sectors) * coarsest.get_world_size();

		unsigned int gaps = 0;
		unsigned int overlaps = 0;
		unsigned int sampled = 0;

		const int STEPS = 61;
		for (int iz = 0; iz < STEPS; ++iz) {
			for (int ix = 0; ix < STEPS; ++ix) {
				// Sample within the guaranteed-covered core: half the coarsest
				// reach, so edge effects of the outermost ring do not count as
				// gaps.
				const float t_x = (static_cast<float>(ix) / (STEPS - 1)) * 2.f - 1.f;
				const float t_z = (static_cast<float>(iz) / (STEPS - 1)) * 2.f - 1.f;
				const float px = viewer.x + t_x * reach * 0.45f;
				const float pz = viewer.z + t_z * reach * 0.45f;

				unsigned int hits = 0;
				for (const Box2 &box : boxes) {
					if (box.contains(px, pz)) {
						++hits;
					}
				}
				++sampled;
				if (hits == 0) {
					++gaps;
				} else if (hits > 1) {
					++overlaps;
				}
			}
		}

		CHECK(sampled > 0);
		CHECK(gaps == 0);
		CHECK(overlaps == 0);
		if (gaps != 0 || overlaps != 0) {
			std::printf(
					"    viewer (%.1f, %.1f): %u gaps, %u overlaps out of %u samples\n",
					viewer.x,
					viewer.z,
					gaps,
					overlaps,
					sampled
			);
		}
	}
}

static void test_detail_increases_toward_viewer() {
	TEST("finer levels sit closer to the viewer");

	FarLodSettings settings;
	settings.lod_count = 5;
	settings.ring_radius_sectors = 3;

	const Vec3f viewer(500.f, 0.f, -250.f);
	const std::vector<SectorRequest> required = required_set(settings, viewer);

	// For each level, the furthest sector at that level should be closer than
	// the furthest at the next coarser one.
	std::map<unsigned int, float> max_distance;
	for (const SectorRequest &r : required) {
		float &d = max_distance[r.id.lod];
		d = std::max(d, r.distance);
	}

	CHECK(max_distance.size() == settings.lod_count);
	float previous = 0.f;
	for (const auto &entry : max_distance) {
		CHECK(entry.second > previous);
		previous = entry.second;
	}

	// And the viewer's own position must be covered by level 0.
	bool covered_by_lod0 = false;
	for (const SectorRequest &r : required) {
		if (r.id.lod == 0 && sector_box(r.id).contains(viewer.x, viewer.z)) {
			covered_by_lod0 = true;
			break;
		}
	}
	CHECK(covered_by_lod0);
}

static void test_parent_child_relationship() {
	TEST("a sector's four children tile it exactly");

	const SectorId ids[] = { SectorId(0, 0, 3), SectorId(-7, 5, 1), SectorId(13, -2, 6) };

	for (const SectorId &parent : ids) {
		const Box2 pbox = sector_box(parent);
		float area = 0.f;
		for (unsigned int i = 0; i < 4; ++i) {
			const SectorId child = parent.get_child(i);
			CHECK(child.get_parent() == parent);
			const Box2 cbox = sector_box(child);
			CHECK(cbox.min_x >= pbox.min_x - 1e-3f);
			CHECK(cbox.min_z >= pbox.min_z - 1e-3f);
			CHECK(cbox.max_x <= pbox.max_x + 1e-3f);
			CHECK(cbox.max_z <= pbox.max_z + 1e-3f);
			area += (cbox.max_x - cbox.min_x) * (cbox.max_z - cbox.min_z);
		}
		const float parent_area = (pbox.max_x - pbox.min_x) * (pbox.max_z - pbox.min_z);
		CHECK(std::abs(area - parent_area) < 1e-2f);
	}
}

static void test_hysteresis_prevents_thrashing() {
	TEST("pacing across a boundary does not thrash");

	FarLodSettings settings;
	settings.lod_count = 4;
	settings.ring_radius_sectors = 3;
	settings.hysteresis_sectors = 1;

	FarLodTree tree;
	tree.configure(settings);

	std::vector<SectorRequest> to_load;
	std::vector<SectorId> to_unload;

	const float cell = SectorId(0, 0, 0).get_world_size();

	// Settle at a sector boundary.
	tree.update(Vec3f(cell, 0.f, 0.f), to_load, to_unload);
	const size_t settled = tree.get_resident_count();
	CHECK(settled > 0);

	// Step back and forth across it several times.
	unsigned int total_churn = 0;
	for (int i = 0; i < 6; ++i) {
		const float x = (i % 2 == 0) ? (cell - 0.5f) : (cell + 0.5f);
		tree.update(Vec3f(x, 0.f, 0.f), to_load, to_unload);
		total_churn += static_cast<unsigned int>(to_unload.size());
	}

	// With the hysteresis band, crossing a single boundary should never unload
	// anything: the sectors that drop out of the required set are all still
	// inside the band.
	CHECK(total_churn == 0);

	// A long move must still release memory.
	tree.update(Vec3f(cell * 1000.f, 0.f, 0.f), to_load, to_unload);
	CHECK(to_unload.size() > 0);
}

static void test_near_clip_withholds_sectors() {
	TEST("sectors under the near terrain are withheld, not offered for loading");

	FarLodSettings settings;
	settings.lod_count = 5;
	settings.ring_radius_sectors = 4;
	settings.near_clip_radius = 200.f;

	FarLodSettings unclipped = settings;
	unclipped.near_clip_radius = 0.f;

	const Vec3f viewer(0.f, 0.f, 0.f);

	FarLodTree tree;
	tree.configure(settings);

	const std::vector<SectorRequest> offered = required_set(settings, viewer);
	const std::vector<SectorRequest> all = required_set(unclipped, viewer);

	// The caller builds everything it is handed, so nothing handed out may be
	// hidden.
	std::unordered_set<SectorId, SectorIdHash> offered_ids;
	for (const SectorRequest &r : offered) {
		CHECK(r.visible);
		CHECK(!tree.is_clipped(r.id, viewer));
		offered_ids.insert(r.id);
	}

	unsigned int withheld = 0;
	for (const SectorRequest &r : all) {
		if (offered_ids.find(r.id) != offered_ids.end()) {
			continue;
		}
		++withheld;
		// A withheld sector must be entirely inside the clip radius.
		const Box2 box = sector_box(r.id);
		const float far_x = std::max(std::abs(box.min_x), std::abs(box.max_x));
		const float far_z = std::max(std::abs(box.min_z), std::abs(box.max_z));
		CHECK(far_x * far_x + far_z * far_z <= settings.near_clip_radius * settings.near_clip_radius + 1e-2f);
	}
	// With a 200-unit radius and 32-unit level-0 sectors, several are enclosed.
	CHECK(withheld > 0);
	CHECK(all.size() == offered.size() + withheld);

	// With clipping off, nothing is withheld.
	for (const SectorRequest &r : all) {
		CHECK(r.visible);
	}
}

static void test_withheld_sectors_are_built_when_they_come_into_view() {
	TEST("a sector hidden under the near terrain is built once it comes into view");

	// The bug this pins: a withheld sector used to be recorded as resident even
	// though nothing built it. Residency is what suppresses a reload request, so
	// the moment the viewer moved and the sector came out from under the near
	// terrain, it was already "resident" and was never built -- a hole in the
	// ground that only appeared after moving, and never healed.

	FarLodSettings settings;
	settings.lod_count = 4;
	settings.ring_radius_sectors = 8;
	settings.near_clip_radius = 300.f;

	FarLodSettings unclipped = settings;
	unclipped.near_clip_radius = 0.f;

	FarLodTree tree;
	tree.configure(settings);

	// Stands in for the renderer: it builds exactly what the tree hands it.
	std::unordered_set<SectorId, SectorIdHash> built;

	std::vector<SectorRequest> to_load;
	std::vector<SectorId> to_unload;
	unsigned int withheld_seen = 0;

	for (int step = 0; step <= 8; ++step) {
		const Vec3f viewer(static_cast<float>(step) * 90.f, 0.f, 0.f);

		tree.update(viewer, to_load, to_unload);
		for (const SectorId &id : to_unload) {
			built.erase(id);
		}
		for (const SectorRequest &r : to_load) {
			CHECK(r.visible);
			built.insert(r.id);
		}

		// Everything required and not hidden must be drawable right now.
		for (const SectorRequest &r : required_set(unclipped, viewer)) {
			if (tree.is_clipped(r.id, viewer)) {
				++withheld_seen;
				continue;
			}
			CHECK(built.find(r.id) != built.end());
		}
	}

	// The test is worthless if the clip radius never actually hid anything.
	CHECK(withheld_seen > 0);
}

static void test_resident_count_is_bounded() {
	TEST("resident sector count stays under the stated bound");

	FarLodSettings settings;
	settings.lod_count = 10;
	settings.ring_radius_sectors = 4;

	FarLodTree tree;
	tree.configure(settings);

	std::vector<SectorRequest> to_load;
	std::vector<SectorId> to_unload;

	// Wander, and confirm residency never exceeds the advertised ceiling. This
	// bound is what makes the memory budget predictable; a quadtree has no
	// equivalent guarantee.
	const unsigned int bound = settings.get_max_resident_sectors();
	size_t peak = 0;
	for (int i = 0; i < 40; ++i) {
		const float t = static_cast<float>(i);
		tree.update(Vec3f(t * 137.f, 0.f, t * -91.f), to_load, to_unload);
		peak = std::max(peak, tree.get_resident_count());
		CHECK(tree.get_resident_count() <= bound);
	}
	std::printf("    peak resident = %zu sectors (bound %u)\n", peak, bound);
	CHECK(peak > 0);
}

static void test_stationary_viewer_is_stable() {
	TEST("a stationary viewer requests nothing after the first update");

	FarLodSettings settings;
	settings.lod_count = 6;
	settings.ring_radius_sectors = 3;

	FarLodTree tree;
	tree.configure(settings);

	std::vector<SectorRequest> to_load;
	std::vector<SectorId> to_unload;

	const Vec3f viewer(321.f, 12.f, -654.f);
	tree.update(viewer, to_load, to_unload);
	CHECK(to_load.size() > 0);

	for (int i = 0; i < 5; ++i) {
		tree.update(viewer, to_load, to_unload);
		CHECK(to_load.empty());
		CHECK(to_unload.empty());
	}
}

static void test_load_order_is_nearest_first() {
	TEST("load order is finest and nearest first");

	FarLodSettings settings;
	settings.lod_count = 5;
	settings.ring_radius_sectors = 3;

	const std::vector<SectorRequest> required = required_set(settings, Vec3f(0.f, 0.f, 0.f));
	CHECK(required.size() > 1);

	for (size_t i = 1; i < required.size(); ++i) {
		const SectorRequest &a = required[i - 1];
		const SectorRequest &b = required[i];
		if (a.id.lod == b.id.lod) {
			CHECK(a.distance <= b.distance + 1e-3f);
		} else {
			CHECK(a.id.lod < b.id.lod);
		}
	}
}

static void test_first_lod_shifts_the_whole_pyramid() {
	TEST("first_lod skips the fine levels without opening a hole");

	FarLodSettings settings;
	settings.first_lod = 4;
	settings.lod_count = 5;
	settings.ring_radius_sectors = 3;

	const Vec3f viewer(1234.f, 0.f, -567.f);
	const std::vector<SectorRequest> required = required_set(settings, viewer);
	CHECK(!required.empty());

	// No level below first_lod may be requested. Generating those is the waste
	// this setting exists to avoid.
	unsigned int min_lod = 255;
	unsigned int max_lod = 0;
	for (const SectorRequest &r : required) {
		CHECK(r.id.lod >= settings.first_lod);
		min_lod = std::min<unsigned int>(min_lod, r.id.lod);
		max_lod = std::max<unsigned int>(max_lod, r.id.lod);
	}
	CHECK(min_lod == settings.first_lod);
	CHECK(max_lod == settings.first_lod + settings.lod_count - 1);

	// The innermost level must leave no hole: nothing finer exists to fill it,
	// so it has to cover its whole ring including the viewer's own position.
	bool covered = false;
	for (const SectorRequest &r : required) {
		if (r.id.lod == settings.first_lod && sector_box(r.id).contains(viewer.x, viewer.z)) {
			covered = true;
			break;
		}
	}
	CHECK(covered);

	// And coverage overall must still tile exactly.
	std::vector<Box2> boxes;
	for (const SectorRequest &r : required) {
		boxes.push_back(sector_box(r.id));
	}
	const SectorId coarsest(0, 0, static_cast<uint8_t>(settings.first_lod + settings.lod_count - 1));
	const float reach = static_cast<float>(settings.ring_radius_sectors) * coarsest.get_world_size();

	unsigned int gaps = 0;
	unsigned int overlaps = 0;
	const int STEPS = 41;
	for (int iz = 0; iz < STEPS; ++iz) {
		for (int ix = 0; ix < STEPS; ++ix) {
			const float t_x = (static_cast<float>(ix) / (STEPS - 1)) * 2.f - 1.f;
			const float t_z = (static_cast<float>(iz) / (STEPS - 1)) * 2.f - 1.f;
			const float px = viewer.x + t_x * reach * 0.45f;
			const float pz = viewer.z + t_z * reach * 0.45f;
			unsigned int hits = 0;
			for (const Box2 &box : boxes) {
				if (box.contains(px, pz)) {
					++hits;
				}
			}
			if (hits == 0) {
				++gaps;
			} else if (hits > 1) {
				++overlaps;
			}
		}
	}
	CHECK(gaps == 0);
	CHECK(overlaps == 0);
}


// ---------------------------------------------------------------------------
// Planet mode
// ---------------------------------------------------------------------------

namespace {

constexpr float PLANET_RADIUS = 40000.f;

FarLodSettings sphere_settings(unsigned int lod_count, unsigned int ring_radius, unsigned int first_lod) {
	FarLodSettings s;
	s.spherical = true;
	s.planet_centre = Vec3f(0.f, 0.f, 0.f);
	s.sphere_reference_radius = PLANET_RADIUS;
	s.sphere_root_lod = choose_sphere_root_lod(PLANET_RADIUS, 1.f);
	s.first_lod = static_cast<uint8_t>(first_lod);
	s.lod_count = lod_count;
	s.ring_radius_sectors = ring_radius;
	return s;
}

// Which sector of `required` contains a direction, if any. Returns how many do,
// which is the number that has to be exactly one.
unsigned int count_sectors_containing(
		const std::vector<SectorRequest> &required,
		uint8_t root_lod,
		const Vec3f &dir
) {
	unsigned int face = 0;
	float u = 0.f;
	float v = 0.f;
	dir_to_face_uv(dir, face, u, v);

	// Sector ranges are half-open so that adjacent sectors do not both claim a
	// shared boundary. A direction landing exactly on the outer edge of a face
	// would then belong to nothing, which happens constantly when the viewer
	// sits on a cube edge or corner and the samples are symmetric about it.
	// Nudge it inside; the sphere really is covered there.
	constexpr float EDGE = 1.f - 1e-6f;
	u = std::min(std::max(u, -1.f), EDGE);
	v = std::min(std::max(v, -1.f), EDGE);

	unsigned int hits = 0;
	for (const SectorRequest &r : required) {
		if (r.id.face != face) {
			continue;
		}
		const int32_t n = sectors_per_face_side(r.id.lod, root_lod);
		const float inv_n = 1.f / static_cast<float>(n);
		const float u0 = static_cast<float>(r.id.x) * 2.f * inv_n - 1.f;
		const float u1 = static_cast<float>(r.id.x + 1) * 2.f * inv_n - 1.f;
		const float v0 = static_cast<float>(r.id.z) * 2.f * inv_n - 1.f;
		const float v1 = static_cast<float>(r.id.z + 1) * 2.f * inv_n - 1.f;
		if (u >= u0 && u < u1 && v >= v0 && v < v1) {
			++hits;
		}
	}
	return hits;
}

Vec3f viewer_at(const Vec3f &dir, float radius) {
	return dir.normalized() * radius;
}

} // namespace

static void test_sphere_rings_have_no_duplicates() {
	TEST("planet rings name each sector at most once");

	const Vec3f dirs[] = {
		Vec3f(0.f, 1.f, 0.f), // straight at a face centre
		Vec3f(1.f, 1.f, 1.f), // a cube corner, the worst case
		Vec3f(1.f, 1.f, 0.f), // a cube edge
		Vec3f(0.3f, -0.9f, 0.31f),
		Vec3f(-1.f, 0.02f, 0.03f),
	};

	for (const Vec3f &d : dirs) {
		const FarLodSettings settings = sphere_settings(6, 4, 4);
		const std::vector<SectorRequest> required =
				required_set(settings, viewer_at(d, PLANET_RADIUS + 500.f));
		CHECK(!required.empty());

		std::unordered_set<SectorId, SectorIdHash> seen;
		for (const SectorRequest &r : required) {
			const bool inserted = seen.insert(r.id).second;
			CHECK(inserted);
		}
	}
}

static void test_sphere_sector_coords_are_in_range() {
	TEST("planet sectors stay inside their own face grid");

	const FarLodSettings settings = sphere_settings(6, 4, 4);
	const std::vector<SectorRequest> required =
			required_set(settings, viewer_at(Vec3f(1.f, 1.f, 1.f), PLANET_RADIUS + 500.f));

	for (const SectorRequest &r : required) {
		const int32_t n = sectors_per_face_side(r.id.lod, settings.sphere_root_lod);
		CHECK(r.id.face < CUBE_FACE_COUNT);
		CHECK(r.id.x >= 0 && r.id.x < n);
		CHECK(r.id.z >= 0 && r.id.z < n);
	}
}

static void test_sphere_coverage_near_the_viewer() {
	TEST("planet rings cover the ground under the viewer exactly once");

	// The sphere version of the property in HANDOFF 3.2. Sampled in a cone
	// around the viewer rather than over the whole sphere: past the outermost
	// ring nothing is drawn, which is correct and not a gap.
	const Vec3f dirs[] = {
		Vec3f(0.f, 1.f, 0.f),
		Vec3f(1.f, 1.f, 1.f),
		Vec3f(1.f, 1.f, 0.f),
		Vec3f(-0.2f, 0.4f, 0.9f),
	};

	for (const Vec3f &d : dirs) {
		const FarLodSettings settings = sphere_settings(6, 4, 4);
		const Vec3f viewer_dir = d.normalized();
		const std::vector<SectorRequest> required =
				required_set(settings, viewer_at(viewer_dir, PLANET_RADIUS + 500.f));

		// Cone half-angle: the innermost ring's reach, so every sample is well
		// inside what the rings claim to cover.
		const int32_t n0 = sectors_per_face_side(settings.first_lod, settings.sphere_root_lod);
		const float sector_angle = 1.5707963f / static_cast<float>(n0);
		const float cone = sector_angle * static_cast<float>(settings.ring_radius_sectors) * 0.5f;

		// An orthonormal pair to sweep the cone with.
		Vec3f tangent = cross(viewer_dir, Vec3f(0.f, 0.f, 1.f));
		if (tangent.length_squared() < 1e-6f) {
			tangent = cross(viewer_dir, Vec3f(1.f, 0.f, 0.f));
		}
		tangent = tangent.normalized();
		const Vec3f bitangent = cross(viewer_dir, tangent);

		unsigned int gaps = 0;
		unsigned int overlaps = 0;
		unsigned int sampled = 0;

		const int STEPS = 21;
		for (int iy = 0; iy < STEPS; ++iy) {
			for (int ix = 0; ix < STEPS; ++ix) {
				const float a = (static_cast<float>(ix) / (STEPS - 1) * 2.f - 1.f) * cone;
				const float b = (static_cast<float>(iy) / (STEPS - 1) * 2.f - 1.f) * cone;
				const Vec3f sample = (viewer_dir + tangent * a + bitangent * b).normalized();

				const unsigned int hits =
						count_sectors_containing(required, settings.sphere_root_lod, sample);
				++sampled;
				if (hits == 0) {
					++gaps;
				} else if (hits > 1) {
					++overlaps;
				}
			}
		}

		CHECK(sampled > 0);
		CHECK(gaps == 0);
		CHECK(overlaps == 0);
		if (gaps != 0 || overlaps != 0) {
			std::printf("    dir (%.2f, %.2f, %.2f): %u gaps, %u overlaps of %u\n", (double)d.x, (double)d.y,
					(double)d.z, gaps, overlaps, sampled);
		}
	}
}

static void test_sphere_coarsest_level_wraps_the_planet() {
	TEST("the coarsest planet level closes over the whole sphere");

	// Once a level's ring is as wide as a face it must enumerate all six faces
	// rather than wrap a square ring around the sphere and name sectors twice.
	FarLodSettings settings = sphere_settings(3, 4, 0);
	// Root lod chosen so the coarsest level has a 2x2 face grid, well inside the
	// ring radius.
	settings.sphere_root_lod = 3;
	settings.first_lod = 1;

	const std::vector<SectorRequest> required =
			required_set(settings, viewer_at(Vec3f(0.3f, 0.5f, 0.8f), PLANET_RADIUS + 500.f));

	std::unordered_set<SectorId, SectorIdHash> seen;
	for (const SectorRequest &r : required) {
		CHECK(seen.insert(r.id).second);
	}

	// The whole sphere is covered: sample directions everywhere, not just near
	// the viewer.
	const int N = 300;
	const float golden = 3.14159265f * (3.f - std::sqrt(5.f));
	unsigned int gaps = 0;
	unsigned int overlaps = 0;
	for (int i = 0; i < N; ++i) {
		const float y = 1.f - (static_cast<float>(i) / static_cast<float>(N - 1)) * 2.f;
		const float rr = std::sqrt(std::max(0.f, 1.f - y * y));
		const float theta = golden * static_cast<float>(i);
		const Vec3f dir(std::cos(theta) * rr, y, std::sin(theta) * rr);

		const unsigned int hits = count_sectors_containing(required, settings.sphere_root_lod, dir);
		if (hits == 0) {
			++gaps;
		} else if (hits > 1) {
			++overlaps;
		}
	}
	std::printf("    whole-sphere sample: %u gaps, %u overlaps of %d\n", gaps, overlaps, N);
	CHECK(gaps == 0);
	CHECK(overlaps == 0);
}

static void test_sphere_mixed_levels_cover_the_whole_planet() {
	TEST("planet levels tile the whole sphere when rings and full levels mix");

	// The configuration an actual planet uses: the fine levels are rings around
	// the viewer, and somewhere up the pyramid a level becomes wide enough to
	// wrap the sphere. The hole one level leaves for the next has to be exact
	// across that transition, and across face boundaries within it.
	FarLodSettings settings = sphere_settings(6, 4, 6);

	const Vec3f viewer_dirs[] = {
		Vec3f(0.f, 1.f, 0.f),
		Vec3f(1.f, 1.f, 1.f),
		Vec3f(0.2f, -0.3f, 0.93f),
	};

	for (const Vec3f &vd : viewer_dirs) {
		const std::vector<SectorRequest> required =
				required_set(settings, viewer_at(vd, PLANET_RADIUS + 2000.f));

		const int N = 4000;
		const float golden = 3.14159265f * (3.f - std::sqrt(5.f));
		unsigned int gaps = 0;
		unsigned int overlaps = 0;
		for (int i = 0; i < N; ++i) {
			const float y = 1.f - (static_cast<float>(i) / static_cast<float>(N - 1)) * 2.f;
			const float rr = std::sqrt(std::max(0.f, 1.f - y * y));
			const float theta = golden * static_cast<float>(i);
			const Vec3f dir(std::cos(theta) * rr, y, std::sin(theta) * rr);

			const unsigned int hits = count_sectors_containing(required, settings.sphere_root_lod, dir);
			if (hits == 0) {
				++gaps;
			} else if (hits > 1) {
				++overlaps;
			}
		}
		if (gaps != 0 || overlaps != 0) {
			std::printf("    viewer (%.2f, %.2f, %.2f): %u gaps, %u overlaps of %d\n", (double)vd.x, (double)vd.y,
					(double)vd.z, gaps, overlaps, N);
		}
		CHECK(gaps == 0);
		CHECK(overlaps == 0);
	}
}

static void test_sphere_resident_count_is_bounded() {
	TEST("planet residency stays bounded as the viewer orbits");

	FarLodTree tree;
	const FarLodSettings settings = sphere_settings(6, 4, 4);
	tree.configure(settings);

	std::vector<SectorRequest> to_load;
	std::vector<SectorId> to_unload;

	size_t peak = 0;
	for (int i = 0; i < 40; ++i) {
		const float t = static_cast<float>(i) * 0.17f;
		const Vec3f dir(std::cos(t), std::sin(t * 0.7f), std::sin(t));
		tree.update(viewer_at(dir, PLANET_RADIUS + 500.f), to_load, to_unload);
		peak = std::max(peak, tree.get_resident_count());
	}

	std::printf("    peak resident = %u sectors (bound %u)\n", static_cast<unsigned int>(peak),
			settings.get_max_resident_sectors());
	CHECK(peak > 0);
	CHECK(peak <= settings.get_max_resident_sectors());
}

int main() {
	std::printf("far-LOD ring selection tests\n\n");

	test_coverage_has_no_gaps_or_overlaps();
	test_detail_increases_toward_viewer();
	test_parent_child_relationship();
	test_hysteresis_prevents_thrashing();
	test_near_clip_withholds_sectors();
	test_withheld_sectors_are_built_when_they_come_into_view();
	test_resident_count_is_bounded();
	test_stationary_viewer_is_stable();
	test_load_order_is_nearest_first();
	test_first_lod_shifts_the_whole_pyramid();

	test_sphere_rings_have_no_duplicates();
	test_sphere_sector_coords_are_in_range();
	test_sphere_coverage_near_the_viewer();
	test_sphere_coarsest_level_wraps_the_planet();
	test_sphere_mixed_levels_cover_the_whole_planet();
	test_sphere_resident_count_is_bounded();

	std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
	return g_failures == 0 ? 0 : 1;
}
