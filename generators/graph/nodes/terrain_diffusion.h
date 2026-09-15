#ifndef VOXEL_GRAPH_NODES_TERRAIN_DIFFUSION_H
#define VOXEL_GRAPH_NODES_TERRAIN_DIFFUSION_H

// Terrain diffusion graph nodes — infinite hierarchical terrain with
// geographic shaping, climate, rivers, sphere stamps and biome material
// blending. Ported from the VCET Unreal plugin's TerrainDiffusion node set
// (MIT). Batch math dispatches to ISPC when available, scalar otherwise
// (see util/noise/voxel_terrain_noise.h).

#include "../../../util/math/funcs.h"
#include "../../../util/noise/voxel_terrain_noise.h"
#include "../../../util/profiling.h"
#include "../node_type_db.h"

#include <cstddef>
#include <iterator>

#ifdef VOXEL_ENABLE_GPU
#include "../../../util/noise/voxel_erosion_filter_glsl.h"
#endif

namespace zylann::voxel::pg {

namespace {

// PlanetErosion params map 1:1 to TerrainErosionParams fields, in declaration order
struct ErosionParamField {
	const char *name;
	uint32_t offset;
	bool is_int;
};

#define EROSION_FIELD(name, is_int) { #name, offsetof(TerrainErosionParams, name), is_int }
static const ErosionParamField g_erosion_param_fields[] = {
	EROSION_FIELD(seed, true),
	EROSION_FIELD(planet_radius, false),
	EROSION_FIELD(tile_size, false),
	EROSION_FIELD(triplanar_sharpness, false),
	EROSION_FIELD(height_frequency, false),
	EROSION_FIELD(height_amp, false),
	EROSION_FIELD(height_octaves, true),
	EROSION_FIELD(height_lacunarity, false),
	EROSION_FIELD(height_gain, false),
	EROSION_FIELD(scale, false),
	EROSION_FIELD(strength, false),
	EROSION_FIELD(gully_weight, false),
	EROSION_FIELD(detail, false),
	EROSION_FIELD(rounding_ridge, false),
	EROSION_FIELD(rounding_crease, false),
	EROSION_FIELD(rounding_input_mult, false),
	EROSION_FIELD(rounding_octave_mult, false),
	EROSION_FIELD(onset_initial, false),
	EROSION_FIELD(onset_octave, false),
	EROSION_FIELD(ridge_onset_initial, false),
	EROSION_FIELD(ridge_onset_octave, false),
	EROSION_FIELD(assumed_slope, false),
	EROSION_FIELD(assumed_slope_blend, false),
	EROSION_FIELD(cell_scale, false),
	EROSION_FIELD(normalization, false),
	EROSION_FIELD(octaves, true),
	EROSION_FIELD(lacunarity, false),
	EROSION_FIELD(gain, false),
	EROSION_FIELD(height_offset, false),
	EROSION_FIELD(height_offset_fade_blend, false),
};
#undef EROSION_FIELD

// Shared by InfiniteTerrain2D/3D registrations
inline void add_terrain_height_params(NodeType &t) {
	t.params.push_back(NodeType::Param("seed", Variant::INT, 1337));
	t.params.push_back(NodeType::Param("amplitude", Variant::FLOAT, 50.f));
	t.params.push_back(NodeType::Param("feature_scale", Variant::FLOAT, 500.f));
	t.params.push_back(NodeType::Param("lacunarity", Variant::FLOAT, 2.f));
	t.params.push_back(NodeType::Param("gain", Variant::FLOAT, 0.5f));
	t.params.push_back(NodeType::Param("aesthetic_bias", Variant::FLOAT, 0.f));
	t.params.push_back(NodeType::Param("num_octaves", Variant::INT, 5));
	t.params.push_back(NodeType::Param("warp_strength", Variant::FLOAT, 0.f));
	t.params.push_back(NodeType::Param("warp_scale", Variant::FLOAT, 500.f));
	t.params.push_back(NodeType::Param("mountain_blend", Variant::FLOAT, 0.f));
	t.params.push_back(NodeType::Param("mountain_scale", Variant::FLOAT, 2000.f));
	t.params.push_back(NodeType::Param("continent_blend", Variant::FLOAT, 0.f));
	t.params.push_back(NodeType::Param("continent_scale", Variant::FLOAT, 8000.f));
	t.params.push_back(NodeType::Param("island_bias", Variant::FLOAT, 0.f));
	t.params.push_back(NodeType::Param("canyon_blend", Variant::FLOAT, 0.f));
	t.params.push_back(NodeType::Param("canyon_scale", Variant::FLOAT, 1500.f));
	t.params.push_back(NodeType::Param("terrace_strength", Variant::FLOAT, 0.f));
	t.params.push_back(NodeType::Param("terrace_count", Variant::FLOAT, 8.f));
}

inline TerrainHeightParams get_terrain_height_params(CompileContext &ctx) {
	TerrainHeightParams p = make_default_terrain_height_params();
	p.seed = ctx.get_param(0);
	p.amplitude = ctx.get_param(1);
	p.feature_scale = ctx.get_param(2);
	p.lacunarity = ctx.get_param(3);
	p.gain = ctx.get_param(4);
	p.aesthetic_bias = ctx.get_param(5);
	p.num_octaves = math::clamp(int(ctx.get_param(6)), 1, 255);
	p.shaping.warp_strength = ctx.get_param(7);
	p.shaping.warp_scale = ctx.get_param(8);
	p.shaping.mountain_blend = ctx.get_param(9);
	p.shaping.mountain_scale = ctx.get_param(10);
	p.shaping.continent_blend = ctx.get_param(11);
	p.shaping.continent_scale = ctx.get_param(12);
	p.shaping.island_bias = ctx.get_param(13);
	p.shaping.canyon_blend = ctx.get_param(14);
	p.shaping.canyon_scale = ctx.get_param(15);
	p.shaping.terrace_strength = ctx.get_param(16);
	p.shaping.terrace_count = ctx.get_param(17);
	return p;
}

} // namespace

void register_terrain_diffusion_nodes(Span<NodeType> types) {
	using namespace math;

	{
		NodeType &t = types[VoxelGraphFunction::NODE_INFINITE_TERRAIN_2D];
		t.name = "InfiniteTerrain2D";
		t.category = CATEGORY_GENERATE;
		t.inputs.push_back(NodeType::Port("x", 0.f, VoxelGraphFunction::AUTO_CONNECT_X));
		t.inputs.push_back(NodeType::Port("y", 0.f, VoxelGraphFunction::AUTO_CONNECT_Z));
		t.outputs.push_back(NodeType::Port("height"));
		add_terrain_height_params(t);

		t.compile_func = [](CompileContext &ctx) {
			ctx.set_params(get_terrain_height_params(ctx));
		};

		t.process_buffer_func = [](Runtime::ProcessBufferContext &ctx) {
			ZN_PROFILE_SCOPE_NAMED("NODE_INFINITE_TERRAIN_2D");
			const Runtime::Buffer &x = ctx.get_input(0);
			const Runtime::Buffer &y = ctx.get_input(1);
			Runtime::Buffer &out = ctx.get_output(0);
			const TerrainHeightParams p = ctx.get_params<TerrainHeightParams>();
			terrain_height_2d_series(x.data, y.data, out.data, out.size, p);
		};

		t.range_analysis_func = [](Runtime::RangeAnalysisContext &ctx) {
			const TerrainHeightParams p = ctx.get_params<TerrainHeightParams>();
			const float a = Math::abs(p.amplitude);
			ctx.set_output(0, Interval(-a, a));
		};
	}
	{
		NodeType &t = types[VoxelGraphFunction::NODE_INFINITE_TERRAIN_3D];
		t.name = "InfiniteTerrain3D";
		t.category = CATEGORY_GENERATE;
		t.inputs.push_back(NodeType::Port("x", 0.f, VoxelGraphFunction::AUTO_CONNECT_X));
		t.inputs.push_back(NodeType::Port("y", 0.f, VoxelGraphFunction::AUTO_CONNECT_Y));
		t.inputs.push_back(NodeType::Port("z", 0.f, VoxelGraphFunction::AUTO_CONNECT_Z));
		t.outputs.push_back(NodeType::Port("radial_distance"));
		add_terrain_height_params(t);
		t.params.push_back(NodeType::Param("planet_radius", Variant::FLOAT, 1000.f));

		t.compile_func = [](CompileContext &ctx) {
			TerrainHeightParams p = get_terrain_height_params(ctx);
			p.planet_radius = ctx.get_param(18);
			ctx.set_params(p);
		};

		t.process_buffer_func = [](Runtime::ProcessBufferContext &ctx) {
			ZN_PROFILE_SCOPE_NAMED("NODE_INFINITE_TERRAIN_3D");
			const Runtime::Buffer &x = ctx.get_input(0);
			const Runtime::Buffer &y = ctx.get_input(1);
			const Runtime::Buffer &z = ctx.get_input(2);
			Runtime::Buffer &out = ctx.get_output(0);
			const TerrainHeightParams p = ctx.get_params<TerrainHeightParams>();
			terrain_height_3d_series(x.data, y.data, z.data, out.data, out.size, p);
		};

		t.range_analysis_func = [](Runtime::RangeAnalysisContext &ctx) {
			const TerrainHeightParams p = ctx.get_params<TerrainHeightParams>();
			const float a = Math::abs(p.amplitude);
			ctx.set_output(0, Interval(p.planet_radius - a, p.planet_radius + a));
		};
	}
	{
		NodeType &t = types[VoxelGraphFunction::NODE_TERRAIN_CLIMATE_2D];
		t.name = "TerrainClimate2D";
		t.category = CATEGORY_GENERATE;
		t.inputs.push_back(NodeType::Port("x", 0.f, VoxelGraphFunction::AUTO_CONNECT_X));
		t.inputs.push_back(NodeType::Port("y", 0.f, VoxelGraphFunction::AUTO_CONNECT_Z));
		t.inputs.push_back(NodeType::Port("height", 0.f));
		t.outputs.push_back(NodeType::Port("temperature"));
		t.outputs.push_back(NodeType::Port("moisture"));
		t.outputs.push_back(NodeType::Port("normalized_height"));
		t.params.push_back(NodeType::Param("seed", Variant::INT, 1337));
		t.params.push_back(NodeType::Param("amplitude", Variant::FLOAT, 50.f));
		t.params.push_back(NodeType::Param("sea_level", Variant::FLOAT, 0.f));
		t.params.push_back(NodeType::Param("climate_scale", Variant::FLOAT, 4000.f));
		t.params.push_back(NodeType::Param("elevation_cooling", Variant::FLOAT, 0.6f));
		t.params.push_back(NodeType::Param("temperature_variation", Variant::FLOAT, 0.5f));

		t.compile_func = [](CompileContext &ctx) {
			TerrainClimateParams p = make_default_terrain_climate_params();
			p.seed = ctx.get_param(0);
			p.amplitude = ctx.get_param(1);
			p.sea_level = ctx.get_param(2);
			p.climate_scale = ctx.get_param(3);
			p.elevation_cooling = ctx.get_param(4);
			p.temperature_variation = ctx.get_param(5);
			ctx.set_params(p);
		};

		t.process_buffer_func = [](Runtime::ProcessBufferContext &ctx) {
			ZN_PROFILE_SCOPE_NAMED("NODE_TERRAIN_CLIMATE_2D");
			const Runtime::Buffer &x = ctx.get_input(0);
			const Runtime::Buffer &y = ctx.get_input(1);
			const Runtime::Buffer &height = ctx.get_input(2);
			Runtime::Buffer &temperature = ctx.get_output(0);
			Runtime::Buffer &moisture = ctx.get_output(1);
			Runtime::Buffer &normalized_height = ctx.get_output(2);
			const TerrainClimateParams p = ctx.get_params<TerrainClimateParams>();
			terrain_climate_2d_series(
					x.data, y.data, height.data,
					temperature.data, moisture.data, normalized_height.data,
					temperature.size, p);
		};

		t.range_analysis_func = [](Runtime::RangeAnalysisContext &ctx) {
			ctx.set_output(0, Interval(0.f, 1.f));
			ctx.set_output(1, Interval(0.f, 1.f));
			ctx.set_output(2, Interval(0.f, 1.f));
		};
	}
	{
		NodeType &t = types[VoxelGraphFunction::NODE_TERRAIN_CLIMATE_3D];
		t.name = "TerrainClimate3D";
		t.category = CATEGORY_GENERATE;
		t.inputs.push_back(NodeType::Port("x", 0.f, VoxelGraphFunction::AUTO_CONNECT_X));
		t.inputs.push_back(NodeType::Port("y", 0.f, VoxelGraphFunction::AUTO_CONNECT_Y));
		t.inputs.push_back(NodeType::Port("z", 0.f, VoxelGraphFunction::AUTO_CONNECT_Z));
		t.inputs.push_back(NodeType::Port("radial_distance", 0.f));
		t.outputs.push_back(NodeType::Port("temperature"));
		t.outputs.push_back(NodeType::Port("moisture"));
		t.outputs.push_back(NodeType::Port("normalized_height"));
		t.params.push_back(NodeType::Param("seed", Variant::INT, 1337));
		t.params.push_back(NodeType::Param("amplitude", Variant::FLOAT, 50.f));
		t.params.push_back(NodeType::Param("sea_level", Variant::FLOAT, 0.f));
		t.params.push_back(NodeType::Param("climate_scale", Variant::FLOAT, 4000.f));
		t.params.push_back(NodeType::Param("elevation_cooling", Variant::FLOAT, 0.6f));
		t.params.push_back(NodeType::Param("temperature_variation", Variant::FLOAT, 0.5f));
		t.params.push_back(NodeType::Param("planet_radius", Variant::FLOAT, 1000.f));

		t.compile_func = [](CompileContext &ctx) {
			TerrainClimateParams p = make_default_terrain_climate_params();
			p.seed = ctx.get_param(0);
			p.amplitude = ctx.get_param(1);
			p.sea_level = ctx.get_param(2);
			p.climate_scale = ctx.get_param(3);
			p.elevation_cooling = ctx.get_param(4);
			p.temperature_variation = ctx.get_param(5);
			p.planet_radius = ctx.get_param(6);
			ctx.set_params(p);
		};

		t.process_buffer_func = [](Runtime::ProcessBufferContext &ctx) {
			ZN_PROFILE_SCOPE_NAMED("NODE_TERRAIN_CLIMATE_3D");
			const Runtime::Buffer &x = ctx.get_input(0);
			const Runtime::Buffer &y = ctx.get_input(1);
			const Runtime::Buffer &z = ctx.get_input(2);
			const Runtime::Buffer &radial_distance = ctx.get_input(3);
			Runtime::Buffer &temperature = ctx.get_output(0);
			Runtime::Buffer &moisture = ctx.get_output(1);
			Runtime::Buffer &normalized_height = ctx.get_output(2);
			const TerrainClimateParams p = ctx.get_params<TerrainClimateParams>();
			terrain_climate_3d_series(
					x.data, y.data, z.data, radial_distance.data,
					temperature.data, moisture.data, normalized_height.data,
					temperature.size, p);
		};

		t.range_analysis_func = [](Runtime::RangeAnalysisContext &ctx) {
			ctx.set_output(0, Interval(0.f, 1.f));
			ctx.set_output(1, Interval(0.f, 1.f));
			ctx.set_output(2, Interval(0.f, 1.f));
		};
	}
	{
		struct Params {
			float center_x;
			float center_y;
			float center_z;
			float stamp_size;
			float falloff;
			float rotation_degrees;
			float planet_radius;
		};

		NodeType &t = types[VoxelGraphFunction::NODE_SPHERE_STAMP];
		t.name = "SphereStamp";
		t.category = CATEGORY_GENERATE;
		t.inputs.push_back(NodeType::Port("x", 0.f, VoxelGraphFunction::AUTO_CONNECT_X));
		t.inputs.push_back(NodeType::Port("y", 0.f, VoxelGraphFunction::AUTO_CONNECT_Y));
		t.inputs.push_back(NodeType::Port("z", 0.f, VoxelGraphFunction::AUTO_CONNECT_Z));
		t.outputs.push_back(NodeType::Port("local_x"));
		t.outputs.push_back(NodeType::Port("local_y"));
		t.outputs.push_back(NodeType::Port("uv_x"));
		t.outputs.push_back(NodeType::Port("uv_y"));
		t.outputs.push_back(NodeType::Port("strength"));
		t.params.push_back(NodeType::Param("center_x", Variant::FLOAT, 0.f));
		t.params.push_back(NodeType::Param("center_y", Variant::FLOAT, 1.f));
		t.params.push_back(NodeType::Param("center_z", Variant::FLOAT, 0.f));
		t.params.push_back(NodeType::Param("stamp_size", Variant::FLOAT, 500.f));
		t.params.push_back(NodeType::Param("falloff", Variant::FLOAT, 0.3f));
		t.params.push_back(NodeType::Param("rotation_degrees", Variant::FLOAT, 0.f));
		t.params.push_back(NodeType::Param("planet_radius", Variant::FLOAT, 1000.f));

		t.compile_func = [](CompileContext &ctx) {
			Params p;
			p.center_x = ctx.get_param(0);
			p.center_y = ctx.get_param(1);
			p.center_z = ctx.get_param(2);
			p.stamp_size = ctx.get_param(3);
			p.falloff = ctx.get_param(4);
			p.rotation_degrees = ctx.get_param(5);
			p.planet_radius = ctx.get_param(6);
			ctx.set_params(p);
		};

		t.process_buffer_func = [](Runtime::ProcessBufferContext &ctx) {
			ZN_PROFILE_SCOPE_NAMED("NODE_SPHERE_STAMP");
			const Runtime::Buffer &x = ctx.get_input(0);
			const Runtime::Buffer &y = ctx.get_input(1);
			const Runtime::Buffer &z = ctx.get_input(2);
			Runtime::Buffer &local_x = ctx.get_output(0);
			Runtime::Buffer &local_y = ctx.get_output(1);
			Runtime::Buffer &uv_x = ctx.get_output(2);
			Runtime::Buffer &uv_y = ctx.get_output(3);
			Runtime::Buffer &strength = ctx.get_output(4);
			const Params p = ctx.get_params<Params>();
			sphere_stamp_series(
					x.data, y.data, z.data,
					local_x.data, local_y.data, uv_x.data, uv_y.data, strength.data,
					local_x.size,
					p.center_x, p.center_y, p.center_z,
					p.stamp_size, p.falloff, p.rotation_degrees, p.planet_radius);
		};

		t.range_analysis_func = [](Runtime::RangeAnalysisContext &ctx) {
			const Params p = ctx.get_params<Params>();
			// Arc distances are bounded by half the planet circumference
			const float max_arc = Math::PI * Math::abs(p.planet_radius);
			const float uv_extent = max_arc / MAX(p.stamp_size, 1e-3f);
			ctx.set_output(0, Interval(-max_arc, max_arc));
			ctx.set_output(1, Interval(-max_arc, max_arc));
			ctx.set_output(2, Interval(0.5f - uv_extent, 0.5f + uv_extent));
			ctx.set_output(3, Interval(0.5f - uv_extent, 0.5f + uv_extent));
			ctx.set_output(4, Interval(0.f, 1.f));
		};
	}
	{
		NodeType &t = types[VoxelGraphFunction::NODE_TERRAIN_RIVERS_2D];
		t.name = "TerrainRivers2D";
		t.category = CATEGORY_GENERATE;
		t.inputs.push_back(NodeType::Port("x", 0.f, VoxelGraphFunction::AUTO_CONNECT_X));
		t.inputs.push_back(NodeType::Port("y", 0.f, VoxelGraphFunction::AUTO_CONNECT_Z));
		t.inputs.push_back(NodeType::Port("normalized_height", 0.f));
		t.outputs.push_back(NodeType::Port("river_mask"));
		t.outputs.push_back(NodeType::Port("river_depth"));
		t.params.push_back(NodeType::Param("seed", Variant::INT, 1337));
		t.params.push_back(NodeType::Param("sea_level", Variant::FLOAT, 0.5f));
		t.params.push_back(NodeType::Param("river_scale", Variant::FLOAT, 300.f));
		t.params.push_back(NodeType::Param("river_width", Variant::FLOAT, 0.35f));
		t.params.push_back(NodeType::Param("flow_strength", Variant::FLOAT, 0.55f));

		t.compile_func = [](CompileContext &ctx) {
			TerrainRiversParams p = make_default_terrain_rivers_params();
			p.seed = ctx.get_param(0);
			p.sea_level = ctx.get_param(1);
			p.river_scale = ctx.get_param(2);
			p.river_width = ctx.get_param(3);
			p.flow_strength = ctx.get_param(4);
			ctx.set_params(p);
		};

		t.process_buffer_func = [](Runtime::ProcessBufferContext &ctx) {
			ZN_PROFILE_SCOPE_NAMED("NODE_TERRAIN_RIVERS_2D");
			const Runtime::Buffer &x = ctx.get_input(0);
			const Runtime::Buffer &y = ctx.get_input(1);
			const Runtime::Buffer &normalized_height = ctx.get_input(2);
			Runtime::Buffer &river_mask = ctx.get_output(0);
			Runtime::Buffer &river_depth = ctx.get_output(1);
			const TerrainRiversParams p = ctx.get_params<TerrainRiversParams>();
			terrain_rivers_2d_series(
					x.data, y.data, normalized_height.data,
					river_mask.data, river_depth.data, river_mask.size, p);
		};

		t.range_analysis_func = [](Runtime::RangeAnalysisContext &ctx) {
			ctx.set_output(0, Interval(0.f, 1.f));
			ctx.set_output(1, Interval(0.f, 1.f));
		};
	}
	{
		struct Params {
			float sea_level;
			float biome_contrast;
		};

		NodeType &t = types[VoxelGraphFunction::NODE_TERRAIN_MATERIAL_BLEND];
		t.name = "TerrainMaterialBlend";
		t.category = CATEGORY_MATERIAL;
		t.inputs.push_back(NodeType::Port("normalized_height", 0.f));
		t.inputs.push_back(NodeType::Port("temperature", 0.5f));
		t.inputs.push_back(NodeType::Port("moisture", 0.5f));
		t.inputs.push_back(NodeType::Port("river_mask", 0.f));
		t.outputs.push_back(NodeType::Port("biome_id"));
		t.outputs.push_back(NodeType::Port("ocean_mask"));
		t.outputs.push_back(NodeType::Port("coast_mask"));
		t.outputs.push_back(NodeType::Port("river_feature_mask"));
		t.outputs.push_back(NodeType::Port("vegetation_mask"));
		t.outputs.push_back(NodeType::Port("desert_mask"));
		t.outputs.push_back(NodeType::Port("tundra_mask"));
		t.outputs.push_back(NodeType::Port("mountain_mask"));
		t.outputs.push_back(NodeType::Port("snow_mask"));
		t.params.push_back(NodeType::Param("sea_level", Variant::FLOAT, 0.5f));
		t.params.push_back(NodeType::Param("biome_contrast", Variant::FLOAT, 0.5f));

		t.compile_func = [](CompileContext &ctx) {
			Params p;
			p.sea_level = ctx.get_param(0);
			p.biome_contrast = ctx.get_param(1);
			ctx.set_params(p);
		};

		t.process_buffer_func = [](Runtime::ProcessBufferContext &ctx) {
			ZN_PROFILE_SCOPE_NAMED("NODE_TERRAIN_MATERIAL_BLEND");
			const Runtime::Buffer &normalized_height = ctx.get_input(0);
			const Runtime::Buffer &temperature = ctx.get_input(1);
			const Runtime::Buffer &moisture = ctx.get_input(2);
			const Runtime::Buffer &river_mask = ctx.get_input(3);
			Runtime::Buffer &biome_id = ctx.get_output(0);
			Runtime::Buffer &ocean = ctx.get_output(1);
			Runtime::Buffer &coast = ctx.get_output(2);
			Runtime::Buffer &river = ctx.get_output(3);
			Runtime::Buffer &vegetation = ctx.get_output(4);
			Runtime::Buffer &desert = ctx.get_output(5);
			Runtime::Buffer &tundra = ctx.get_output(6);
			Runtime::Buffer &mountain = ctx.get_output(7);
			Runtime::Buffer &snow = ctx.get_output(8);
			const Params p = ctx.get_params<Params>();
			terrain_material_blend_series(
					normalized_height.data, temperature.data, moisture.data, river_mask.data,
					biome_id.data, ocean.data, coast.data, river.data, vegetation.data,
					desert.data, tundra.data, mountain.data, snow.data,
					biome_id.size, p.sea_level, p.biome_contrast);
		};

		t.range_analysis_func = [](Runtime::RangeAnalysisContext &ctx) {
			ctx.set_output(0, Interval(0.f, 10.f));
			for (unsigned int i = 1; i <= 8; ++i) {
				ctx.set_output(i, Interval(0.f, 1.f));
			}
		};
	}
	{
		NodeType &t = types[VoxelGraphFunction::NODE_PLANET_EROSION];
		t.name = "PlanetErosion";
		t.category = CATEGORY_GENERATE;
		t.inputs.push_back(NodeType::Port("x", 0.f, VoxelGraphFunction::AUTO_CONNECT_X));
		t.inputs.push_back(NodeType::Port("y", 0.f, VoxelGraphFunction::AUTO_CONNECT_Y));
		t.inputs.push_back(NodeType::Port("z", 0.f, VoxelGraphFunction::AUTO_CONNECT_Z));
		t.outputs.push_back(NodeType::Port("height"));
		t.outputs.push_back(NodeType::Port("ridge"));
		t.outputs.push_back(NodeType::Port("erosion"));
		{
			const TerrainErosionParams defaults = make_default_terrain_erosion_params();
			for (const ErosionParamField &f : g_erosion_param_fields) {
				const char *field = reinterpret_cast<const char *>(&defaults) + f.offset;
				if (f.is_int) {
					t.params.push_back(NodeType::Param(f.name, Variant::INT, *reinterpret_cast<const int32_t *>(field)));
				} else {
					t.params.push_back(NodeType::Param(f.name, Variant::FLOAT, *reinterpret_cast<const float *>(field)));
				}
			}
		}

		t.compile_func = [](CompileContext &ctx) {
			TerrainErosionParams p = make_default_terrain_erosion_params();
			for (unsigned int i = 0; i < std::size(g_erosion_param_fields); ++i) {
				const ErosionParamField &f = g_erosion_param_fields[i];
				char *field = reinterpret_cast<char *>(&p) + f.offset;
				if (f.is_int) {
					*reinterpret_cast<int32_t *>(field) = int(ctx.get_param(i));
				} else {
					*reinterpret_cast<float *>(field) = float(ctx.get_param(i));
				}
			}
			p.height_octaves = math::clamp(p.height_octaves, 0, 16);
			p.octaves = math::clamp(p.octaves, 0, 16);
			if (p.tile_size <= 0.f || p.scale <= 0.f || p.cell_scale <= 0.f) {
				ctx.make_error("tile_size, scale and cell_scale must be > 0");
				return;
			}
			ctx.set_params(p);
		};

		t.process_buffer_func = [](Runtime::ProcessBufferContext &ctx) {
			ZN_PROFILE_SCOPE_NAMED("NODE_PLANET_EROSION");
			const Runtime::Buffer &x = ctx.get_input(0);
			const Runtime::Buffer &y = ctx.get_input(1);
			const Runtime::Buffer &z = ctx.get_input(2);
			Runtime::Buffer &height = ctx.get_output(0);
			Runtime::Buffer &ridge = ctx.get_output(1);
			Runtime::Buffer &erosion = ctx.get_output(2);
			const TerrainErosionParams p = ctx.get_params<TerrainErosionParams>();
			planet_erosion_series(x.data, y.data, z.data, height.data, ridge.data, erosion.data, height.size, p);
		};

		t.range_analysis_func = [](Runtime::RangeAnalysisContext &ctx) {
			const TerrainErosionParams p = ctx.get_params<TerrainErosionParams>();
			const float h = get_terrain_erosion_max_height(p);
			ctx.set_output(0, Interval(-h, h));
			ctx.set_output(1, Interval(-1.f, 1.f));
			ctx.set_output(2, Interval(0.f, 1.f));
		};

#ifdef VOXEL_ENABLE_GPU
		t.shader_gen_func = [](ShaderGenContext &ctx) {
			ctx.require_lib_code("vg_erosion_filter", g_erosion_filter_shader);
			const char *out = ctx.get_output_name(0);
			ctx.add_format("VgErosionParams {}_prm = VgErosionParams(", out);
			for (unsigned int i = 0; i < std::size(g_erosion_param_fields); ++i) {
				if (i > 0) {
					ctx.add_format("{}", ", ");
				}
				if (g_erosion_param_fields[i].is_int) {
					ctx.add_format("{}", int(ctx.get_param(i)));
				} else {
					ctx.add_format("{}", float(ctx.get_param(i)));
				}
			}
			ctx.add_format(
					");\n"
					"vec3 {}_r = vg_planet_erosion(vec3({}, {}, {}), {}_prm);\n"
					"{} = {}_r.x;\n"
					"{} = {}_r.y;\n"
					"{} = {}_r.z;\n",
					out,
					ctx.get_input_name(0),
					ctx.get_input_name(1),
					ctx.get_input_name(2),
					out,
					ctx.get_output_name(0),
					out,
					ctx.get_output_name(1),
					out,
					ctx.get_output_name(2),
					out);
		};
#endif
	}
}

} // namespace zylann::voxel::pg

#endif // VOXEL_GRAPH_NODES_TERRAIN_DIFFUSION_H
