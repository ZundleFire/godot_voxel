#include "test_terrain_noise.h"
#include "../../util/math/funcs.h"
#include "../../util/noise/voxel_terrain_noise.h"
#include "../../util/testing/test_macros.h"
#include "../../util/io/log.h"
#include "../../util/string/format.h"
#include "../../util/thread/thread.h"
#include "../../engine/voxel_engine.h"
#include "../../generators/graph/voxel_generator_graph.h"
#include "../../storage/voxel_buffer.h"
#ifdef VOXEL_ENABLE_GPU
#include "../../engine/gpu/compute_shader.h"
#endif

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

void test_planet_erosion() {
	const unsigned int count = 512;
	std::vector<float> x, y, z;
	fill_positions(x, y, z, count);

	const TerrainErosionParams p = make_default_terrain_erosion_params();
	const float max_h = get_terrain_erosion_max_height(p);

	std::vector<float> height(count), ridge(count), erosion(count), again(count);
	planet_erosion_series(x.data(), y.data(), z.data(), height.data(), ridge.data(), erosion.data(), count, p);
	planet_erosion_series(x.data(), y.data(), z.data(), again.data(), nullptr, nullptr, count, p);
	ZN_TEST_ASSERT(memcmp(height.data(), again.data(), count * sizeof(float)) == 0);

	float min_h = max_h;
	float max_seen_h = -max_h;
	float sum_abs_h = 0.f;
	for (unsigned int i = 0; i < count; ++i) {
		ZN_TEST_ASSERT(std::isfinite(height[i]));
		ZN_TEST_ASSERT(height[i] >= -max_h && height[i] <= max_h);
		ZN_TEST_ASSERT(ridge[i] >= -1.f && ridge[i] <= 1.f);
		ZN_TEST_ASSERT(erosion[i] >= 0.f && erosion[i] <= 1.f);
		min_h = math::min(min_h, height[i]);
		max_seen_h = math::max(max_seen_h, height[i]);
		sum_abs_h += std::abs(height[i]);
	}
	print_line(format("PlanetErosion relief (m): min {} max {} mean|h| {} bound {}",
			min_h, max_seen_h, sum_abs_h / count, max_h));

	// Direction-only: scaling positions radially must not change the relief
	std::vector<float> x2(count), y2(count), z2(count);
	for (unsigned int i = 0; i < count; ++i) {
		x2[i] = x[i] * 1.5f;
		y2[i] = y[i] * 1.5f;
		z2[i] = z[i] * 1.5f;
	}
	planet_erosion_series(x2.data(), y2.data(), z2.data(), again.data(), nullptr, nullptr, count, p);
	for (unsigned int i = 0; i < count; ++i) {
		ZN_TEST_ASSERT(std::abs(again[i] - height[i]) <= 1e-3f * max_h);
	}

	// Erosion must actually change the terrain
	TerrainErosionParams flat = p;
	flat.strength = 0.f;
	planet_erosion_series(x.data(), y.data(), z.data(), again.data(), nullptr, nullptr, count, flat);
	ZN_TEST_ASSERT(memcmp(height.data(), again.data(), count * sizeof(float)) != 0);

	// Graph node: SDF = altitude - erosion height, must match the direct kernel
	Ref<VoxelGeneratorGraph> generator;
	generator.instantiate();
	{
		pg::VoxelGraphFunction &g = **generator->get_main_function();
		const uint32_t n_x = g.create_node(pg::VoxelGraphFunction::NODE_INPUT_X, Vector2());
		const uint32_t n_y = g.create_node(pg::VoxelGraphFunction::NODE_INPUT_Y, Vector2());
		const uint32_t n_z = g.create_node(pg::VoxelGraphFunction::NODE_INPUT_Z, Vector2());
		const uint32_t n_alt = g.create_node(pg::VoxelGraphFunction::NODE_PLANET_ALTITUDE, Vector2());
		const uint32_t n_ero = g.create_node(pg::VoxelGraphFunction::NODE_PLANET_EROSION, Vector2());
		const uint32_t n_sub = g.create_node(pg::VoxelGraphFunction::NODE_SUBTRACT, Vector2());
		const uint32_t n_out = g.create_node(pg::VoxelGraphFunction::NODE_OUTPUT_SDF, Vector2());
		for (uint32_t n : { n_alt, n_ero }) {
			g.add_connection(n_x, 0, n, 0);
			g.add_connection(n_y, 0, n, 1);
			g.add_connection(n_z, 0, n, 2);
		}
		g.set_node_param(n_alt, 0, p.planet_radius);
		g.add_connection(n_alt, 0, n_sub, 0);
		g.add_connection(n_ero, 0, n_sub, 1);
		g.add_connection(n_sub, 0, n_out, 0);
		const pg::CompilationResult result = generator->compile(false);
		if (!result.success) {
			ZN_PRINT_ERROR(result.message.utf8().get_data());
		}
		ZN_TEST_ASSERT(result.success);
	}
	{
		// Land near the surface so the SDF isn't clipped
		const float dx = 0.36f, dy = 0.8f, dz = 0.48f;
		float h;
		planet_erosion_series(&dx, &dy, &dz, &h, nullptr, nullptr, 1, p);
		const float r = p.planet_radius + h;
		const Vector3i pos(Math::round(dx * r), Math::round(dy * r), Math::round(dz * r));
		const float px = pos.x, py = pos.y, pz = pos.z;
		float hp;
		planet_erosion_series(&px, &py, &pz, &hp, nullptr, nullptr, 1, p);
		const float expected = Math::sqrt(px * px + py * py + pz * pz) - p.planet_radius - hp;
		const float sdf = generator->generate_single(pos, VoxelBuffer::CHANNEL_SDF).f;
		ZN_TEST_ASSERT(std::abs(sdf - expected) < 0.05f);
	}

#ifdef VOXEL_ENABLE_GPU
	// GLSL twin must compile
	VoxelEngine::get_singleton().try_initialize_gpu_features();
	// The device is created asynchronously on the GPU task thread
	for (int i = 0; i < 3000 && !VoxelEngine::get_singleton().has_rendering_device(); ++i) {
		Thread::sleep_usec(1000);
	}
	if (VoxelEngine::get_singleton().has_rendering_device()) {
		generator->compile_shaders();
		const std::shared_ptr<ComputeShader> shader = generator->get_block_rendering_shader();
		ZN_TEST_ASSERT(shader != nullptr);
		// Compiled asynchronously on the GPU task thread
		for (int i = 0; i < 60000 && !shader->is_compilation_complete(); ++i) {
			Thread::sleep_usec(1000);
		}
		ZN_TEST_ASSERT(shader->is_compilation_complete());
		ZN_TEST_ASSERT(shader->is_valid());
	} else {
		ZN_PRINT_WARNING("No RenderingDevice, skipping PlanetErosion shader compilation check");
	}
#endif
}

} // namespace zylann::voxel::tests
