// Host-compilable tests for the engine-independent far-LOD core.
//
// Build and run:
//   g++ -std=c++17 -O1 -g -fsanitize=address,undefined \
//       tests/test_far_core.cpp core/far_column_data.cpp core/far_extract.cpp core/far_mesher.cpp \
//       -o /tmp/test_far && /tmp/test_far
//
// The point of keeping the core free of engine headers is exactly this: the
// parts that are hard to get right (layer extraction, sheet labelling, seam
// behaviour, level registration) can be checked without launching Godot.

#include "../core/far_extract.h"
#include "../core/far_mesher.h"

#include <cstdio>
#include <functional>
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

#define CHECK_NEAR(a, b, eps)                                                                                          \
	do {                                                                                                               \
		++g_checks;                                                                                                    \
		const double va = (a);                                                                                         \
		const double vb = (b);                                                                                         \
		if (!(std::abs(va - vb) <= (eps))) {                                                                           \
			++g_failures;                                                                                              \
			std::printf(                                                                                               \
					"  FAIL [%s] %s:%d: %s (%.6f) !~ %s (%.6f), tol %.6f\n",                                           \
					g_current_test,                                                                                    \
					__FILE__,                                                                                          \
					__LINE__,                                                                                          \
					#a,                                                                                                \
					va,                                                                                                \
					#b,                                                                                                \
					vb,                                                                                                \
					double(eps)                                                                                        \
			);                                                                                                         \
		}                                                                                                              \
	} while (0)

#define TEST(name)                                                                                                     \
	g_current_test = name;                                                                                             \
	std::printf("- %s\n", name);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Builds an SDF slab from a height function. `sdf = y - h(x, z)`, so the field
// is negative below the surface (matter) and positive above (air), matching
// godot_voxel's convention.
struct SlabStorage {
	std::vector<float> sdf;
	std::vector<uint16_t> material;
	SdfSlab slab;
};

static void build_height_slab(
		SlabStorage &storage,
		const std::function<float(float, float)> &height_fn,
		unsigned int height_samples,
		float origin_y,
		float step,
		float world_origin_x = 0.f,
		float world_origin_z = 0.f
) {
	const unsigned int res = SECTOR_RES;
	storage.sdf.assign(static_cast<size_t>(res) * res * height_samples, 0.f);
	storage.material.assign(storage.sdf.size(), 3);

	storage.slab.sdf = storage.sdf.data();
	storage.slab.material = storage.material.data();
	storage.slab.res_xz = res;
	storage.slab.height = height_samples;
	storage.slab.origin_y = origin_y;
	storage.slab.step = step;

	for (unsigned int z = 0; z < res; ++z) {
		for (unsigned int x = 0; x < res; ++x) {
			const float wx = world_origin_x + static_cast<float>(x) * step;
			const float wz = world_origin_z + static_cast<float>(z) * step;
			const float h = height_fn(wx, wz);
			for (unsigned int y = 0; y < height_samples; ++y) {
				storage.sdf[storage.slab.index(x, y, z)] = storage.slab.world_y(y) - h;
			}
		}
	}
}

// Builds a slab with an explicit list of solid spans per column.
static void build_span_slab(
		SlabStorage &storage,
		const std::function<void(float, float, std::vector<std::pair<float, float>> &)> &span_fn,
		unsigned int height_samples,
		float origin_y,
		float step
) {
	const unsigned int res = SECTOR_RES;
	storage.sdf.assign(static_cast<size_t>(res) * res * height_samples, 1000.f);
	storage.material.assign(storage.sdf.size(), 1);

	storage.slab.sdf = storage.sdf.data();
	storage.slab.material = storage.material.data();
	storage.slab.res_xz = res;
	storage.slab.height = height_samples;
	storage.slab.origin_y = origin_y;
	storage.slab.step = step;

	std::vector<std::pair<float, float>> spans;
	for (unsigned int z = 0; z < res; ++z) {
		for (unsigned int x = 0; x < res; ++x) {
			spans.clear();
			span_fn(static_cast<float>(x) * step, static_cast<float>(z) * step, spans);
			for (unsigned int y = 0; y < height_samples; ++y) {
				const float wy = storage.slab.world_y(y);
				// Distance to the nearest span boundary, negative inside.
				float best = 1000.f;
				for (const auto &span : spans) {
					const float d = std::max(span.first - wy, wy - span.second);
					best = std::min(best, d);
				}
				storage.sdf[storage.slab.index(x, y, z)] = best;
			}
		}
	}
}

static ExtractSettings default_extract_settings(float step) {
	ExtractSettings settings;
	settings.min_gap = step * 0.25f;
	settings.min_thickness = step * 0.1f;
	settings.compute_ao = true;
	settings.ao_strength = 1.f;
	return settings;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

static void test_normal_packing() {
	TEST("octahedral normal packing round-trip");

	const Vec3f samples[] = {
		Vec3f(0.f, 1.f, 0.f),	 Vec3f(0.f, -1.f, 0.f),	  Vec3f(1.f, 0.f, 0.f),
		Vec3f(-1.f, 0.f, 0.f),	 Vec3f(0.f, 0.f, 1.f),	  Vec3f(0.707f, 0.707f, 0.f),
		Vec3f(0.3f, -0.8f, 0.5f), Vec3f(-0.5f, 0.1f, -0.85f),
	};

	for (const Vec3f &n_in : samples) {
		const Vec3f n = n_in.normalized();
		const Vec3f r = unpack_normal_oct16(pack_normal_oct16(n));
		// 16 bits per axis should land far inside a thousandth of a radian.
		CHECK_NEAR(dot(n, r), 1.0, 1e-4);
	}
}

static void test_flat_plane_extraction() {
	TEST("flat plane extracts one layer at the right height");

	const float step = 1.f;
	SlabStorage storage;
	build_height_slab(storage, [](float, float) { return 12.5f; }, 64, 0.f, step);

	std::vector<ColumnStack> stacks(SECTOR_COLUMN_COUNT);
	const unsigned int non_empty = extract_columns_from_sdf(storage.slab, default_extract_settings(step), stacks.data());

	CHECK(non_empty == SECTOR_COLUMN_COUNT);

	for (const ColumnStack &stack : stacks) {
		CHECK(stack.count == 1);
		if (stack.count != 1) {
			break;
		}
		const ColumnLayer &layer = stack.layers[0];
		CHECK_NEAR(layer.top, 12.5f, 1e-3);
		CHECK(layer.top_is_surface());
		// Solid to the bottom of the sampled range, so no bottom surface.
		CHECK(!layer.bottom_is_surface());
		const Vec3f n = unpack_normal_oct16(layer.normal);
		CHECK_NEAR(n.y, 1.0, 1e-3);
		CHECK(layer.material == 3);
	}
}

static void test_sloped_surface_normal() {
	TEST("sloped surface recovers the analytic normal");

	const float step = 1.f;
	const float slope = 0.5f; // dh/dx
	SlabStorage storage;
	build_height_slab(storage, [slope](float x, float) { return 20.f + slope * x; }, 96, 0.f, step);

	std::vector<ColumnStack> stacks(SECTOR_COLUMN_COUNT);
	extract_columns_from_sdf(storage.slab, default_extract_settings(step), stacks.data());

	// The analytic surface normal of y = 20 + 0.5x is normalize(-0.5, 1, 0).
	const Vec3f expected = Vec3f(-slope, 1.f, 0.f).normalized();

	// Check an interior column, away from the one-sided gradients at the border.
	const ColumnStack &stack = stacks[16 * SECTOR_RES + 16];
	CHECK(stack.count == 1);
	if (stack.count == 1) {
		CHECK_NEAR(stack.layers[0].top, 20.f + slope * 16.f, 1e-2);
		const Vec3f n = unpack_normal_oct16(stack.layers[0].normal);
		CHECK_NEAR(dot(n, expected), 1.0, 2e-3);
	}
}

static void test_overhang_produces_two_layers() {
	TEST("a cave produces two separate spans");

	const float step = 1.f;
	SlabStorage storage;
	build_span_slab(
			storage,
			[](float, float, std::vector<std::pair<float, float>> &spans) {
				spans.emplace_back(0.f, 20.f); // ground
				spans.emplace_back(30.f, 40.f); // ceiling slab above a 10-unit void
			},
			64,
			0.f,
			step
	);

	std::vector<ColumnStack> stacks(SECTOR_COLUMN_COUNT);
	extract_columns_from_sdf(storage.slab, default_extract_settings(step), stacks.data());

	const ColumnStack &stack = stacks[10 * SECTOR_RES + 10];
	CHECK(stack.count == 2);
	if (stack.count == 2) {
		// Stored top-down.
		CHECK_NEAR(stack.layers[0].top, 40.f, 0.6);
		CHECK_NEAR(stack.layers[0].bottom, 30.f, 0.6);
		CHECK_NEAR(stack.layers[1].top, 20.f, 0.6);
		CHECK(stack.layers[0].bottom_is_surface());
		CHECK(stack.layers[1].top_is_surface());
	}
}

static void test_thin_gap_is_merged() {
	TEST("a gap thinner than the tolerance collapses into one span");

	const float step = 4.f; // coarse LOD: a 1-unit gap is far below one cell
	SlabStorage storage;
	build_span_slab(
			storage,
			[](float, float, std::vector<std::pair<float, float>> &spans) {
				spans.emplace_back(0.f, 40.f);
				spans.emplace_back(40.5f, 60.f);
			},
			64,
			0.f,
			step
	);

	ExtractSettings settings = default_extract_settings(step);
	settings.min_gap = step * 0.5f; // 2 units, wider than the 0.5-unit gap

	std::vector<ColumnStack> stacks(SECTOR_COLUMN_COUNT);
	extract_columns_from_sdf(storage.slab, settings, stacks.data());

	const ColumnStack &stack = stacks[10 * SECTOR_RES + 10];
	CHECK(stack.count == 1);
	if (stack.count == 1) {
		CHECK(stack.layers[0].flags & LAYER_FLAG_MERGED);
		CHECK_NEAR(stack.layers[0].top, 60.f, 4.0);
	}
}

static void test_serialization_round_trip() {
	TEST("serialization round-trips within quantisation error");

	const float step = 1.f;
	SlabStorage storage;
	build_height_slab(
			storage,
			[](float x, float z) { return 30.f + 8.f * std::sin(x * 0.2f) + 5.f * std::cos(z * 0.13f); },
			96,
			0.f,
			step
	);

	std::vector<ColumnStack> stacks(SECTOR_COLUMN_COUNT);
	extract_columns_from_sdf(storage.slab, default_extract_settings(step), stacks.data());

	ColumnField field;
	field.build_from_stacks(stacks.data());
	CHECK(field.get_layer_count() > 0);

	std::vector<uint8_t> bytes;
	field.serialize(bytes);
	CHECK(bytes.size() > 0);

	ColumnField restored;
	CHECK(restored.deserialize(bytes.data(), bytes.size()));
	CHECK(restored.get_layer_count() == field.get_layer_count());

	// 16-bit quantisation across the sector's own Y range.
	const float range = field.get_max_y() - field.get_min_y();
	const double eps = static_cast<double>(range) / 65535.0 + 1e-5;

	for (unsigned int z = 0; z < SECTOR_RES; ++z) {
		for (unsigned int x = 0; x < SECTOR_RES; ++x) {
			unsigned int ca = 0;
			unsigned int cb = 0;
			const ColumnLayer *a = field.get_column(x, z, ca);
			const ColumnLayer *b = restored.get_column(x, z, cb);
			CHECK(ca == cb);
			for (unsigned int i = 0; i < ca && i < cb; ++i) {
				CHECK_NEAR(a[i].top, b[i].top, eps);
				CHECK_NEAR(a[i].bottom, b[i].bottom, eps);
				CHECK(a[i].normal == b[i].normal);
				CHECK(a[i].material == b[i].material);
				CHECK(a[i].ao == b[i].ao);
			}
		}
	}

	// Corrupt input must be rejected rather than misread.
	ColumnField bad;
	CHECK(!bad.deserialize(bytes.data(), 4));
	std::vector<uint8_t> garbage(bytes);
	garbage[0] ^= 0xffu;
	CHECK(!bad.deserialize(garbage.data(), garbage.size()));
}


static constexpr unsigned int MAX_DOWNSAMPLE_GROUP_FOR_TEST = 9;

static void test_downsample_picks_the_dominant_material() {
	TEST("downsampling keeps the material most of the area actually has");

	// Heights, normals and AO are all weight-blended when a coarse column is
	// reduced from several fine ones. Material used to be taken from whichever
	// span reached the cluster first, which with a 3x3 tent filter means the
	// corner of the neighbourhood rather than the bulk of it -- so a single
	// stray column could repaint a whole coarse column, and at distance the
	// materials drifted.

	constexpr uint16_t MINORITY = 3;
	constexpr uint16_t MAJORITY = 5;

	// One group of columns at the same height, so they all land in one cluster.
	// The first one carries the minority material, the rest the majority.
	ColumnStack stacks[MAX_DOWNSAMPLE_GROUP_FOR_TEST];
	float weights[MAX_DOWNSAMPLE_GROUP_FOR_TEST];

	const unsigned int count = 9;
	for (unsigned int i = 0; i < count; ++i) {
		ColumnLayer layer;
		layer.top = 100.f;
		layer.bottom = 60.f;
		layer.normal = pack_normal_oct16(Vec3f(0.f, 1.f, 0.f));
		layer.ao = 255;
		layer.material = (i == 0) ? MINORITY : MAJORITY;
		layer.flags = 0;

		stacks[i].clear();
		stacks[i].push(layer);
		// Equal say for every contributor, so "first" and "heaviest" differ.
		weights[i] = 1.f;
	}

	ColumnStack out;
	downsample_column_group(stacks, weights, count, 4.f, out);

	CHECK(out.count == 1);
	if (out.count > 0) {
		CHECK(out.layers[0].material == MAJORITY);
	}

	// And with the weights reversed, the answer follows the weight rather than
	// the order: one heavily weighted minority column wins.
	for (unsigned int i = 0; i < count; ++i) {
		weights[i] = (i == 0) ? 100.f : 1.f;
	}
	ColumnStack weighted;
	downsample_column_group(stacks, weights, count, 4.f, weighted);
	CHECK(weighted.count == 1);
	if (weighted.count > 0) {
		CHECK(weighted.layers[0].material == MINORITY);
	}
}

static void test_downsample_keeps_levels_registered() {
	TEST("downsampling keeps LOD levels registered with each other");

	// Four child sectors of a smooth hill, each covering its own quadrant of
	// world space, then reduced into one parent sector.
	const float step = 1.f;
	const auto height_fn = [](float x, float z) {
		return 40.f + 10.f * std::sin(x * 0.03f) + 7.f * std::cos(z * 0.025f);
	};

	SlabStorage storage[4];
	ColumnField fields[4];
	const ColumnField *field_ptrs[4];

	for (unsigned int q = 0; q < 4; ++q) {
		const float ox = (q & 1u) ? static_cast<float>(SECTOR_CELLS) * step : 0.f;
		const float oz = (q >> 1) ? static_cast<float>(SECTOR_CELLS) * step : 0.f;
		build_height_slab(storage[q], height_fn, 128, 0.f, step, ox, oz);

		std::vector<ColumnStack> stacks(SECTOR_COLUMN_COUNT);
		extract_columns_from_sdf(storage[q].slab, default_extract_settings(step), stacks.data());
		fields[q].build_from_stacks(stacks.data());
		field_ptrs[q] = &fields[q];
	}

	ColumnField parent;
	downsample_field(field_ptrs, step * 2.f, parent);

	CHECK(parent.get_layer_count() > 0);

	// Every parent column should sit on the analytic surface. The tent filter
	// smooths, so allow a tolerance proportional to the local curvature rather
	// than demanding an exact match.
	double max_error = 0.0;
	for (unsigned int dz = 0; dz < SECTOR_RES; ++dz) {
		for (unsigned int dx = 0; dx < SECTOR_RES; ++dx) {
			const float wx = static_cast<float>(dx) * 2.f * step;
			const float wz = static_cast<float>(dz) * 2.f * step;
			const float expected = height_fn(wx, wz);
			const float got = parent.get_top_surface(dx, dz, -1000.f);
			CHECK(got > -999.f);
			max_error = std::max(max_error, std::abs(static_cast<double>(got) - expected));
		}
	}
	CHECK_NEAR(max_error, 0.0, 0.5);

	// The shared border ring must be point-sampled, so a parent border column
	// has to match the child sample exactly (up to float noise). This is the
	// property that keeps adjacent parent sectors crack-free.
	for (unsigned int dx = 0; dx <= SECTOR_CELLS; ++dx) {
		const unsigned int gx = dx * 2;
		const unsigned int qx = (gx <= SECTOR_CELLS) ? 0u : 1u;
		const unsigned int lx = gx - qx * SECTOR_CELLS;
		const float child_top = fields[qx].get_top_surface(lx, 0, -1000.f);
		const float parent_top = parent.get_top_surface(dx, 0, -1000.f);
		CHECK_NEAR(parent_top, child_top, 1e-3);
	}
}

static void test_mesh_is_well_formed() {
	TEST("generated mesh is well formed");

	const float step = 1.f;
	SlabStorage storage;
	build_height_slab(
			storage,
			[](float x, float z) { return 40.f + 9.f * std::sin(x * 0.18f) + 6.f * std::cos(z * 0.11f); },
			128,
			0.f,
			step
	);

	std::vector<ColumnStack> stacks(SECTOR_COLUMN_COUNT);
	extract_columns_from_sdf(storage.slab, default_extract_settings(step), stacks.data());

	ColumnField field;
	field.build_from_stacks(stacks.data());

	MesherSettings settings;
	settings.cell_size = step;
	FarMeshArrays mesh;
	build_far_mesh(field, settings, mesh);

	CHECK(!mesh.is_empty());
	CHECK(mesh.indices.size() % 3 == 0);
	CHECK(mesh.has_bounds);

	// Every index in range.
	for (uint32_t index : mesh.indices) {
		CHECK(index < mesh.vertices.size());
		if (index >= mesh.vertices.size()) {
			break;
		}
	}

	// A smooth heightfield should be one continuous sheet: every cell covered,
	// so the cap count is exactly two triangles per cell.
	CHECK(mesh.cap_triangles == SECTOR_CELLS * SECTOR_CELLS * 2);
	// ... and no interior walls, because nothing exceeds the slope tolerance.
	CHECK(mesh.wall_triangles == 0);
	// Skirts all round.
	CHECK(mesh.skirt_triangles == SECTOR_CELLS * 4 * 2);

	// No degenerate triangles, and normals must be finite and unit length.
	unsigned int degenerate = 0;
	for (size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
		const Vec3f &a = mesh.vertices[mesh.indices[i]].position;
		const Vec3f &b = mesh.vertices[mesh.indices[i + 1]].position;
		const Vec3f &c = mesh.vertices[mesh.indices[i + 2]].position;
		if (cross(b - a, c - a).length() < 1e-9f) {
			++degenerate;
		}
	}
	CHECK(degenerate == 0);

	for (const FarVertex &v : mesh.vertices) {
		CHECK(std::isfinite(v.position.x) && std::isfinite(v.position.y) && std::isfinite(v.position.z));
		CHECK_NEAR(v.normal.length(), 1.0, 1e-3);
		CHECK(v.ao >= 0.f && v.ao <= 1.f);
	}

	// Winding: Godot front faces are clockwise from the front, so the
	// right-hand-rule normal of each cap triangle must OPPOSE the shading
	// normal. Check on the cap triangles, which are emitted first.
	unsigned int wrong_winding = 0;
	for (size_t i = 0; i + 2 < mesh.cap_triangles * 3; i += 3) {
		const FarVertex &a = mesh.vertices[mesh.indices[i]];
		const FarVertex &b = mesh.vertices[mesh.indices[i + 1]];
		const FarVertex &c = mesh.vertices[mesh.indices[i + 2]];
		const Vec3f rhr = cross(b.position - a.position, c.position - a.position);
		const Vec3f shading = (a.normal + b.normal + c.normal).normalized();
		if (dot(rhr.normalized(), shading) > 0.f) {
			++wrong_winding;
		}
	}
	CHECK(wrong_winding == 0);
}

static void test_cliff_stays_steep_and_solid() {
	TEST("a cliff stays steep and leaves no hole");

	const float step = 1.f;
	// A 20-unit vertical step at x = 16.
	SlabStorage storage;
	build_height_slab(storage, [](float x, float) { return (x < 16.f) ? 20.f : 40.f; }, 96, 0.f, step);

	std::vector<ColumnStack> stacks(SECTOR_COLUMN_COUNT);
	extract_columns_from_sdf(storage.slab, default_extract_settings(step), stacks.data());

	ColumnField field;
	field.build_from_stacks(stacks.data());

	MesherSettings settings;
	settings.cell_size = step;
	FarMeshArrays mesh;
	build_far_mesh(field, settings, mesh);

	// The surface is continuous, so every cell must be covered. A hole here
	// would be far worse than any shape inaccuracy.
	CHECK(mesh.cap_triangles == SECTOR_CELLS * SECTOR_CELLS * 2);

	// The face spanning the step must be near-vertical: a 20-unit drop across
	// one cell. What must NOT happen is that drop being spread over several
	// cells, which is the "terrain melting downhill" artifact.
	unsigned int steep_faces = 0;
	unsigned int stretched_ramps = 0;
	for (size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
		const Vec3f &pa = mesh.vertices[mesh.indices[i]].position;
		const Vec3f &pb = mesh.vertices[mesh.indices[i + 1]].position;
		const Vec3f &pc = mesh.vertices[mesh.indices[i + 2]].position;
		const float dy = std::max({ pa.y, pb.y, pc.y }) - std::min({ pa.y, pb.y, pc.y });
		const float dxz = std::max(
				std::max({ pa.x, pb.x, pc.x }) - std::min({ pa.x, pb.x, pc.x }),
				std::max({ pa.z, pb.z, pc.z }) - std::min({ pa.z, pb.z, pc.z })
		);
		if (dy > 10.f) {
			if (dxz <= step * 1.01f) {
				++steep_faces;
			} else {
				++stretched_ramps;
			}
		}
	}
	CHECK(steep_faces > 0);
	CHECK(stretched_ramps == 0);

	// And the stored normals along the step must be close to horizontal, which
	// is what makes it read as a cliff rather than a bright slope.
	const ColumnStack &at_edge = stacks[10 * SECTOR_RES + 16];
	CHECK(at_edge.count >= 1);
	if (at_edge.count >= 1) {
		const Vec3f n = unpack_normal_oct16(at_edge.layers[0].normal);
		CHECK(std::abs(n.y) < 0.75f);
	}
}

static void test_sustained_steep_slope_has_no_holes() {
	TEST("a sustained steep slope is covered, not dropped");

	// Regression test for a real failure mode. When sheet connection was a
	// slope threshold, a slope steeper than that threshold everywhere meant no
	// adjacent pair ever connected, every span became a singleton sheet, no
	// cell had four corners agreeing, and the mesher silently emitted nothing.
	// An entire mountainside rendered as a hole. Connection is permissive now,
	// so coverage must be complete at any slope.
	const float step = 1.f;
	const float SLOPES[] = { 0.5f, 1.8f, 2.6f, 8.0f, 40.0f };

	for (const float slope : SLOPES) {
		SlabStorage storage;
		build_height_slab(storage, [slope](float x, float) { return 100.f + slope * x; }, 2048, 0.f, step);

		std::vector<ColumnStack> stacks(SECTOR_COLUMN_COUNT);
		extract_columns_from_sdf(storage.slab, default_extract_settings(step), stacks.data());

		ColumnField field;
		field.build_from_stacks(stacks.data());

		MesherSettings settings;
		settings.cell_size = step;
		FarMeshArrays mesh;
		build_far_mesh(field, settings, mesh);

		// Full coverage at every slope. This is the assertion that would have
		// caught the bug.
		CHECK(mesh.cap_triangles == SECTOR_CELLS * SECTOR_CELLS * 2);
		CHECK(!mesh.is_empty());
	}
}

static void test_island_edge_produces_a_wall() {
	TEST("a floating island edge produces a wall");

	// Walls exist for discontinuities in layer *structure*, not for steepness.
	// A slab that simply stops partway across the sector is exactly that case:
	// the upper sheet ends, and there is nothing on the far side to connect to.
	const float step = 1.f;
	SlabStorage storage;
	build_span_slab(
			storage,
			[](float x, float, std::vector<std::pair<float, float>> &spans) {
				spans.emplace_back(0.f, 20.f); // ground everywhere
				if (x < 16.f) {
					spans.emplace_back(40.f, 50.f); // island over half the sector
				}
			},
			96,
			0.f,
			step
	);

	std::vector<ColumnStack> stacks(SECTOR_COLUMN_COUNT);
	extract_columns_from_sdf(storage.slab, default_extract_settings(step), stacks.data());

	ColumnField field;
	field.build_from_stacks(stacks.data());

	MesherSettings settings;
	settings.cell_size = step;
	FarMeshArrays mesh;
	build_far_mesh(field, settings, mesh);

	// The island's rim runs the full depth of the sector.
	CHECK(mesh.wall_triangles > 0);

	// The ground below is uninterrupted, so it alone accounts for one full
	// sheet of caps; the island adds more on top.
	CHECK(mesh.cap_triangles > SECTOR_CELLS * SECTOR_CELLS * 2);
}

static void test_empty_field_meshes_to_nothing() {
	TEST("an empty field produces an empty mesh");

	std::vector<ColumnStack> stacks(SECTOR_COLUMN_COUNT);
	ColumnField field;
	field.build_from_stacks(stacks.data());

	MesherSettings settings;
	FarMeshArrays mesh;
	build_far_mesh(field, settings, mesh);

	CHECK(mesh.is_empty());
	CHECK(!mesh.has_bounds);

	std::vector<uint8_t> bytes;
	field.serialize(bytes);
	ColumnField restored;
	CHECK(restored.deserialize(bytes.data(), bytes.size()));
	CHECK(restored.get_layer_count() == 0);
}

static void test_column_stack_overflow() {
	TEST("a column with too many spans degrades gracefully");

	ColumnStack stack;
	// Push far more spans than the cap allows.
	for (int i = 0; i < 40; ++i) {
		ColumnLayer layer;
		layer.top = 200.f - static_cast<float>(i) * 5.f;
		layer.bottom = layer.top - 2.f;
		layer.normal = pack_normal_oct16(Vec3f(0.f, 1.f, 0.f));
		layer.flags = LAYER_FLAG_TOP_IS_SURFACE | LAYER_FLAG_BOTTOM_IS_SURFACE;
		stack.push(layer);
	}

	CHECK(stack.count <= MAX_LAYERS_PER_COLUMN);
	CHECK(stack.count > 0);

	// The highest surface must survive: dropping it would punch a hole in the
	// horizon, which is the one artifact that is never acceptable.
	CHECK_NEAR(stack.layers[0].top, 200.f, 1e-3);

	// Spans must remain sorted and non-overlapping.
	for (unsigned int i = 0; i + 1 < stack.count; ++i) {
		CHECK(stack.layers[i].top >= stack.layers[i + 1].top);
		CHECK(stack.layers[i].bottom >= stack.layers[i + 1].top - 1e-3f);
	}
}

static void test_sectors_tile_exactly() {
	TEST("adjacent sectors agree on their shared border columns");

	// This is the property that lets sectors be generated and meshed with no
	// cross-sector communication. If it ever breaks, every sector boundary at
	// the same LOD opens a crack.
	const float step = 1.f;
	const auto height_fn = [](float x, float z) {
		return 35.f + 11.f * std::sin(x * 0.07f) + 4.f * std::cos(z * 0.09f);
	};

	SlabStorage left;
	SlabStorage right;
	build_height_slab(left, height_fn, 96, 0.f, step, 0.f, 0.f);
	build_height_slab(right, height_fn, 96, 0.f, step, static_cast<float>(SECTOR_CELLS) * step, 0.f);

	std::vector<ColumnStack> stacks_l(SECTOR_COLUMN_COUNT);
	std::vector<ColumnStack> stacks_r(SECTOR_COLUMN_COUNT);
	// AO is disabled here: it is a neighbourhood operation and legitimately
	// differs at a sector edge, which would mask the geometric property under
	// test.
	ExtractSettings settings = default_extract_settings(step);
	settings.compute_ao = false;
	extract_columns_from_sdf(left.slab, settings, stacks_l.data());
	extract_columns_from_sdf(right.slab, settings, stacks_r.data());

	ColumnField field_l;
	ColumnField field_r;
	field_l.build_from_stacks(stacks_l.data());
	field_r.build_from_stacks(stacks_r.data());

	for (unsigned int z = 0; z < SECTOR_RES; ++z) {
		// Last column of the left sector is the same world position as the
		// first column of the right one.
		const float tl = field_l.get_top_surface(SECTOR_CELLS, z, -1000.f);
		const float tr = field_r.get_top_surface(0, z, -1000.f);
		CHECK_NEAR(tl, tr, 1e-4);
	}
}

static void test_mesh_scales_with_lod() {
	TEST("mesh cost stays flat across LOD levels");

	// A sector always meshes to the same cell count regardless of LOD. That is
	// the whole economic argument for the format: each level costs the same and
	// covers four times the area, so total cost grows logarithmically with view
	// distance instead of quadratically.
	const auto height_fn = [](float x, float z) {
		return 50.f + 14.f * std::sin(x * 0.01f) + 9.f * std::cos(z * 0.013f);
	};

	uint32_t first_cap_count = 0;
	for (unsigned int lod = 0; lod < 5; ++lod) {
		const float step = static_cast<float>(1u << lod);
		SlabStorage storage;
		build_height_slab(storage, height_fn, 160, 0.f, step);

		std::vector<ColumnStack> stacks(SECTOR_COLUMN_COUNT);
		extract_columns_from_sdf(storage.slab, default_extract_settings(step), stacks.data());

		ColumnField field;
		field.build_from_stacks(stacks.data());

		MesherSettings settings;
		settings.cell_size = step;
		FarMeshArrays mesh;
		build_far_mesh(field, settings, mesh);

		if (lod == 0) {
			first_cap_count = mesh.cap_triangles;
		}
		CHECK(mesh.cap_triangles == first_cap_count);
		CHECK(mesh.vertices.size() < 8000);
	}
}

static void test_memory_footprint() {
	TEST("column data stays small enough to keep many sectors resident");

	const float step = 1.f;
	SlabStorage storage;
	build_height_slab(
			storage,
			[](float x, float z) { return 60.f + 20.f * std::sin(x * 0.05f) * std::cos(z * 0.04f); },
			192,
			0.f,
			step
	);

	std::vector<ColumnStack> stacks(SECTOR_COLUMN_COUNT);
	extract_columns_from_sdf(storage.slab, default_extract_settings(step), stacks.data());

	ColumnField field;
	field.build_from_stacks(stacks.data());

	const size_t resident = field.get_size_in_bytes();
	// A dense 33x33x192 voxel volume at one byte per voxel would be ~209 KB.
	const size_t equivalent_volume = SECTOR_RES * SECTOR_RES * 192;

	std::printf(
			"    resident=%zu B, equivalent dense volume=%zu B, ratio=%.1fx\n",
			resident,
			equivalent_volume,
			static_cast<double>(equivalent_volume) / static_cast<double>(resident)
	);

	CHECK(resident < equivalent_volume / 4);

	std::vector<uint8_t> bytes;
	field.serialize(bytes);
	std::printf("    serialized=%zu B\n", bytes.size());
	CHECK(bytes.size() < resident);
}

int main() {
	std::printf("far-LOD core tests\n\n");

	test_normal_packing();
	test_flat_plane_extraction();
	test_sloped_surface_normal();
	test_overhang_produces_two_layers();
	test_thin_gap_is_merged();
	test_serialization_round_trip();
	test_downsample_picks_the_dominant_material();
	test_downsample_keeps_levels_registered();
	test_mesh_is_well_formed();
	test_cliff_stays_steep_and_solid();
	test_sustained_steep_slope_has_no_holes();
	test_island_edge_produces_a_wall();
	test_empty_field_meshes_to_nothing();
	test_column_stack_overflow();
	test_sectors_tile_exactly();
	test_mesh_scales_with_lod();
	test_memory_footprint();

	std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
	return g_failures == 0 ? 0 : 1;
}
