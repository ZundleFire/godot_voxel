#include "test_terrain_noise.h"
#include "../../util/math/funcs.h"
#include "../../util/noise/voxel_terrain_noise.h"
#include "../../util/testing/test_macros.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace zylann::voxel::tests {

namespace {

// Deterministic pseudo-random sample positions (no engine RNG — tests must
// produce the same result every run)
void fill_positions(std::vector<float> &x, std::vector<float> &y, std::vector<float> &z, unsigned int count) {
	x.resize(count);
	y.resize(count);
	z.resize(count);
	uint32_t state = 0x12345678u;
	for (unsigned int i = 0; i < count; ++i) {
		state = state * 1664525u + 1013904223u;
		x[i] = float(state & 0xffffu) - 32768.f;
		state = state * 1664525u + 1013904223u;
		y[i] = float(state & 0xffffu) - 32768.f;
		state = state * 1664525u + 1013904223u;
		z[i] = float(state & 0xffffu) - 32768.f;
	}
}

TerrainHeightParams make_height_params() {
	TerrainHeightParams p = make_default_terrain_height_params();
	p.seed = 42;
	p.amplitude = 80.f;
	p.feature_scale = 700.f;
	p.num_octaves = 6;
	p.aesthetic_bias = 0.4f;
	// Enable every shaping stage so the whole pipeline is covered
	p.shaping.warp_strength = 50.f;
	p.shaping.mountain_blend = 0.5f;
	p.shaping.continent_blend = 0.5f;
	p.shaping.canyon_blend = 0.3f;
	p.shaping.terrace_strength = 0.2f;
	return p;
}

} // namespace

void test_terrain_noise_determinism() {
	const unsigned int count = 512;
	std::vector<float> x, y, z;
	fill_positions(x, y, z, count);

	const TerrainHeightParams p = make_height_params();

	std::vector<float> out_a(count), out_b(count);
	terrain_height_2d_series(x.data(), y.data(), out_a.data(), count, p);
	terrain_height_2d_series(x.data(), y.data(), out_b.data(), count, p);
	ZN_TEST_ASSERT(memcmp(out_a.data(), out_b.data(), count * sizeof(float)) == 0);

	terrain_height_3d_series(x.data(), y.data(), z.data(), out_a.data(), count, p);
	terrain_height_3d_series(x.data(), y.data(), z.data(), out_b.data(), count, p);
	ZN_TEST_ASSERT(memcmp(out_a.data(), out_b.data(), count * sizeof(float)) == 0);

	// A different seed must produce different terrain
	TerrainHeightParams p2 = p;
	p2.seed = 43;
	terrain_height_2d_series(x.data(), y.data(), out_a.data(), count, p);
	terrain_height_2d_series(x.data(), y.data(), out_b.data(), count, p2);
	ZN_TEST_ASSERT(memcmp(out_a.data(), out_b.data(), count * sizeof(float)) != 0);
}

void test_terrain_noise_ranges() {
	const unsigned int count = 512;
	std::vector<float> x, y, z;
	fill_positions(x, y, z, count);

	const TerrainHeightParams hp = make_height_params();

	// 2D height stays within [-amplitude, amplitude]
	std::vector<float> height(count);
	terrain_height_2d_series(x.data(), y.data(), height.data(), count, hp);
	for (unsigned int i = 0; i < count; ++i) {
		ZN_TEST_ASSERT(std::isfinite(height[i]));
		ZN_TEST_ASSERT(height[i] >= -hp.amplitude && height[i] <= hp.amplitude);
	}

	// 3D radial distance stays within planet_radius ± amplitude
	std::vector<float> radial(count);
	terrain_height_3d_series(x.data(), y.data(), z.data(), radial.data(), count, hp);
	for (unsigned int i = 0; i < count; ++i) {
		ZN_TEST_ASSERT(std::isfinite(radial[i]));
		ZN_TEST_ASSERT(
				radial[i] >= hp.planet_radius - hp.amplitude &&
				radial[i] <= hp.planet_radius + hp.amplitude);
	}

	// Climate outputs are normalized [0, 1]
	TerrainClimateParams cp = make_default_terrain_climate_params();
	cp.seed = 42;
	cp.amplitude = hp.amplitude;
	std::vector<float> temperature(count), moisture(count), normalized_height(count);
	terrain_climate_2d_series(
			x.data(), y.data(), height.data(),
			temperature.data(), moisture.data(), normalized_height.data(), count, cp);
	for (unsigned int i = 0; i < count; ++i) {
		ZN_TEST_ASSERT(temperature[i] >= 0.f && temperature[i] <= 1.f);
		ZN_TEST_ASSERT(moisture[i] >= 0.f && moisture[i] <= 1.f);
		ZN_TEST_ASSERT(normalized_height[i] >= 0.f && normalized_height[i] <= 1.f);
	}

	// Rivers: masks and depths [0, 1]
	TerrainRiversParams rp = make_default_terrain_rivers_params();
	rp.seed = 42;
	std::vector<float> river_mask(count), river_depth(count);
	terrain_rivers_2d_series(
			x.data(), y.data(), normalized_height.data(),
			river_mask.data(), river_depth.data(), count, rp);
	for (unsigned int i = 0; i < count; ++i) {
		ZN_TEST_ASSERT(river_mask[i] >= 0.f && river_mask[i] <= 1.f);
		ZN_TEST_ASSERT(river_depth[i] >= 0.f && river_depth[i] <= 1.f);
	}

	// Material blend: biome ids in [0, 10], masks in [0, 1]
	std::vector<float> biome_id(count), ocean(count), coast(count), river(count),
			vegetation(count), desert(count), tundra(count), mountain(count), snow(count);
	terrain_material_blend_series(
			normalized_height.data(), temperature.data(), moisture.data(), river_mask.data(),
			biome_id.data(), ocean.data(), coast.data(), river.data(), vegetation.data(),
			desert.data(), tundra.data(), mountain.data(), snow.data(),
			count, 0.5f, 0.5f);
	for (unsigned int i = 0; i < count; ++i) {
		ZN_TEST_ASSERT(biome_id[i] >= 0.f && biome_id[i] <= 10.f);
		ZN_TEST_ASSERT(biome_id[i] == std::floor(biome_id[i]));
		ZN_TEST_ASSERT(ocean[i] >= 0.f && ocean[i] <= 1.f);
		ZN_TEST_ASSERT(snow[i] >= 0.f && snow[i] <= 1.f);
	}

	// Sphere stamp: strength [0, 1], UVs centered at 0.5 at the stamp center
	std::vector<float> local_x(count), local_y(count), uv_x(count), uv_y(count), strength(count);
	sphere_stamp_series(
			x.data(), y.data(), z.data(),
			local_x.data(), local_y.data(), uv_x.data(), uv_y.data(), strength.data(), count,
			0.f, 1.f, 0.f, 500.f, 0.3f, 0.f, 1000.f);
	for (unsigned int i = 0; i < count; ++i) {
		ZN_TEST_ASSERT(std::isfinite(local_x[i]) && std::isfinite(local_y[i]));
		ZN_TEST_ASSERT(strength[i] >= 0.f && strength[i] <= 1.f);
	}
	// A position exactly at the stamp center maps to UV (0.5, 0.5), strength 1
	const float cx = 0.f, cy = 1000.f, cz = 0.f;
	float slx, sly, sux, suy, ss;
	sphere_stamp_series(&cx, &cy, &cz, &slx, &sly, &sux, &suy, &ss, 1,
			0.f, 1.f, 0.f, 500.f, 0.3f, 0.f, 1000.f);
	ZN_TEST_ASSERT(std::abs(sux - 0.5f) < 1e-4f && std::abs(suy - 0.5f) < 1e-4f);
	ZN_TEST_ASSERT(ss > 0.999f);
}

} // namespace zylann::voxel::tests
