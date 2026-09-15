#include "voxel_terrain_noise.h"

#ifdef VOXEL_ISPC_ENABLED

// ISPC-generated header (created by the ISPC compiler from voxel_procedural_noise_ispc.ispc).
#include "voxel_procedural_noise_ispc_ispc.h"

namespace zylann::voxel {

#define TERRAIN_SHAPING_ARGS(s) \
	(s).warp_strength, (s).warp_scale, (s).mountain_blend, (s).mountain_scale, \
			(s).continent_blend, (s).continent_scale, (s).island_bias, (s).canyon_blend, \
			(s).canyon_scale, (s).terrace_strength, (s).terrace_count

void terrain_height_2d_series(
		const float *x, const float *y, float *out, unsigned int count,
		const TerrainHeightParams &p) {
	ispc::VoxelTerrainHeight2D_Batch(
			x, y,
			p.amplitude, p.feature_scale, p.lacunarity, p.gain, p.aesthetic_bias,
			p.num_octaves, p.seed,
			TERRAIN_SHAPING_ARGS(p.shaping),
			out, count);
}

void terrain_height_3d_series(
		const float *x, const float *y, const float *z, float *out, unsigned int count,
		const TerrainHeightParams &p) {
	ispc::VoxelTerrainHeight3D_Batch(
			x, y, z,
			p.amplitude, p.feature_scale, p.lacunarity, p.gain, p.aesthetic_bias,
			p.num_octaves, p.seed, p.planet_radius,
			TERRAIN_SHAPING_ARGS(p.shaping),
			out, count);
}

void terrain_climate_2d_series(
		const float *x, const float *y, const float *height,
		float *out_temperature, float *out_moisture, float *out_normalized_height,
		unsigned int count, const TerrainClimateParams &p) {
	ispc::VoxelTerrainClimate2D_Batch(
			x, y, height,
			p.amplitude, p.sea_level, p.climate_scale, p.elevation_cooling,
			p.temperature_variation, p.seed,
			out_temperature, out_moisture, out_normalized_height, count);
}

void terrain_climate_3d_series(
		const float *x, const float *y, const float *z, const float *radial_distance,
		float *out_temperature, float *out_moisture, float *out_normalized_height,
		unsigned int count, const TerrainClimateParams &p) {
	ispc::VoxelTerrainClimate3D_Batch(
			x, y, z, radial_distance,
			p.amplitude, p.sea_level, p.climate_scale, p.elevation_cooling,
			p.temperature_variation, p.seed, p.planet_radius,
			out_temperature, out_moisture, out_normalized_height, count);
}

void sphere_stamp_series(
		const float *x, const float *y, const float *z,
		float *out_local_x, float *out_local_y, float *out_uv_x, float *out_uv_y,
		float *out_strength, unsigned int count,
		float center_x, float center_y, float center_z,
		float stamp_size, float falloff, float rotation_degrees, float planet_radius) {
	ispc::VoxelSphereStamp_Batch(
			x, y, z,
			center_x, center_y, center_z,
			stamp_size, falloff, rotation_degrees, planet_radius,
			out_local_x, out_local_y, out_uv_x, out_uv_y, out_strength, count);
}

void terrain_rivers_2d_series(
		const float *x, const float *y, const float *normalized_height,
		float *out_river_mask, float *out_river_depth,
		unsigned int count, const TerrainRiversParams &p) {
	ispc::VoxelTerrainRivers2D_Batch(
			x, y, normalized_height,
			p.sea_level, p.river_scale, p.river_width, p.flow_strength, p.seed,
			out_river_mask, out_river_depth, count);
}

void terrain_material_blend_series(
		const float *normalized_height, const float *temperature, const float *moisture,
		const float *river_mask,
		float *out_biome_id, float *out_ocean_mask, float *out_coast_mask,
		float *out_river_feature_mask, float *out_vegetation_mask, float *out_desert_mask,
		float *out_tundra_mask, float *out_mountain_mask, float *out_snow_mask,
		unsigned int count, float sea_level, float biome_contrast) {
	ispc::VoxelTerrainMaterialBlend_Batch(
			normalized_height, temperature, moisture, river_mask,
			sea_level, biome_contrast,
			out_biome_id, out_ocean_mask, out_coast_mask, out_river_feature_mask,
			out_vegetation_mask, out_desert_mask, out_tundra_mask, out_mountain_mask,
			out_snow_mask, count);
}

void planet_erosion_series(
		const float *x, const float *y, const float *z,
		float *out_height, float *out_ridge, float *out_erosion,
		unsigned int count, const TerrainErosionParams &p) {
	static_assert(sizeof(ispc::FCErosionParams) == sizeof(TerrainErosionParams));
	ispc::VoxelPlanetErosion_Batch(
			x, y, z, reinterpret_cast<const ispc::FCErosionParams *>(&p),
			out_height, out_ridge, out_erosion, count);
}

} // namespace zylann::voxel

#else // VOXEL_ISPC_ENABLED

// Scalar fallback — same single-source math as the ISPC kernels.
#include "voxel_procedural_noise_impl.h"

namespace zylann::voxel {

namespace {
using namespace zylann::procedural_noise;

FCTerrainShaping to_fc_shaping(const TerrainShapingParams &s) {
	FCTerrainShaping f;
	f.WarpStrength = s.warp_strength;
	f.WarpScale = s.warp_scale;
	f.MountainBlend = s.mountain_blend;
	f.MountainScale = s.mountain_scale;
	f.ContinentBlend = s.continent_blend;
	f.ContinentScale = s.continent_scale;
	f.IslandBias = s.island_bias;
	f.CanyonBlend = s.canyon_blend;
	f.CanyonScale = s.canyon_scale;
	f.TerraceStrength = s.terrace_strength;
	f.TerraceCount = s.terrace_count;
	return f;
}
} // namespace

void terrain_height_2d_series(
		const float *x, const float *y, float *out, unsigned int count,
		const TerrainHeightParams &p) {
	const FCTerrainShaping shaping = to_fc_shaping(p.shaping);
	for (unsigned int i = 0; i < count; ++i) {
		out[i] = FCTerrainHeight2D(
				p.seed, MakeFloat2(x[i], y[i]),
				p.amplitude, p.feature_scale, p.lacunarity, p.gain, p.aesthetic_bias,
				p.num_octaves, shaping);
	}
}

void terrain_height_3d_series(
		const float *x, const float *y, const float *z, float *out, unsigned int count,
		const TerrainHeightParams &p) {
	const FCTerrainShaping shaping = to_fc_shaping(p.shaping);
	for (unsigned int i = 0; i < count; ++i) {
		out[i] = FCTerrainHeight3D(
				p.seed, MakeFloat3(x[i], y[i], z[i]),
				p.amplitude, p.feature_scale, p.lacunarity, p.gain, p.aesthetic_bias,
				p.num_octaves, p.planet_radius, shaping);
	}
}

void terrain_climate_2d_series(
		const float *x, const float *y, const float *height,
		float *out_temperature, float *out_moisture, float *out_normalized_height,
		unsigned int count, const TerrainClimateParams &p) {
	for (unsigned int i = 0; i < count; ++i) {
		const FCClimateResult r = FCTerrainClimate2D(
				p.seed, MakeFloat2(x[i], y[i]), height[i],
				p.amplitude, p.sea_level, p.climate_scale, p.elevation_cooling,
				p.temperature_variation);
		out_temperature[i] = r.Temperature;
		out_moisture[i] = r.Moisture;
		out_normalized_height[i] = r.NormalizedHeight;
	}
}

void terrain_climate_3d_series(
		const float *x, const float *y, const float *z, const float *radial_distance,
		float *out_temperature, float *out_moisture, float *out_normalized_height,
		unsigned int count, const TerrainClimateParams &p) {
	for (unsigned int i = 0; i < count; ++i) {
		const FCClimateResult r = FCTerrainClimate3D(
				p.seed, MakeFloat3(x[i], y[i], z[i]), radial_distance[i],
				p.amplitude, p.sea_level, p.climate_scale, p.elevation_cooling,
				p.temperature_variation, p.planet_radius);
		out_temperature[i] = r.Temperature;
		out_moisture[i] = r.Moisture;
		out_normalized_height[i] = r.NormalizedHeight;
	}
}

void sphere_stamp_series(
		const float *x, const float *y, const float *z,
		float *out_local_x, float *out_local_y, float *out_uv_x, float *out_uv_y,
		float *out_strength, unsigned int count,
		float center_x, float center_y, float center_z,
		float stamp_size, float falloff, float rotation_degrees, float planet_radius) {
	for (unsigned int i = 0; i < count; ++i) {
		const FCSphereStampResult r = FCSphereStamp(
				MakeFloat3(x[i], y[i], z[i]),
				MakeFloat3(center_x, center_y, center_z),
				stamp_size, falloff, rotation_degrees, planet_radius);
		out_local_x[i] = r.LocalX;
		out_local_y[i] = r.LocalY;
		out_uv_x[i] = r.UVX;
		out_uv_y[i] = r.UVY;
		out_strength[i] = r.Strength;
	}
}

void terrain_rivers_2d_series(
		const float *x, const float *y, const float *normalized_height,
		float *out_river_mask, float *out_river_depth,
		unsigned int count, const TerrainRiversParams &p) {
	for (unsigned int i = 0; i < count; ++i) {
		const FCRiversResult r = FCTerrainRivers2D(
				p.seed, MakeFloat2(x[i], y[i]), normalized_height[i],
				p.sea_level, p.river_scale, p.river_width, p.flow_strength);
		out_river_mask[i] = r.Mask;
		out_river_depth[i] = r.Depth;
	}
}

void terrain_material_blend_series(
		const float *normalized_height, const float *temperature, const float *moisture,
		const float *river_mask,
		float *out_biome_id, float *out_ocean_mask, float *out_coast_mask,
		float *out_river_feature_mask, float *out_vegetation_mask, float *out_desert_mask,
		float *out_tundra_mask, float *out_mountain_mask, float *out_snow_mask,
		unsigned int count, float sea_level, float biome_contrast) {
	for (unsigned int i = 0; i < count; ++i) {
		const FCMaterialBlendResult r = FCTerrainMaterialBlend(
				normalized_height[i], temperature[i], moisture[i], sea_level, river_mask[i],
				biome_contrast);
		out_biome_id[i] = r.BiomeId;
		out_ocean_mask[i] = r.OceanMask;
		out_coast_mask[i] = r.CoastMask;
		out_river_feature_mask[i] = r.RiverFeatureMask;
		out_vegetation_mask[i] = r.VegetationMask;
		out_desert_mask[i] = r.DesertMask;
		out_tundra_mask[i] = r.TundraMask;
		out_mountain_mask[i] = r.MountainMask;
		out_snow_mask[i] = r.SnowMask;
	}
}

void planet_erosion_series(
		const float *x, const float *y, const float *z,
		float *out_height, float *out_ridge, float *out_erosion,
		unsigned int count, const TerrainErosionParams &p) {
	static_assert(sizeof(FCErosionParams) == sizeof(TerrainErosionParams));
	const FCErosionParams &fp = reinterpret_cast<const FCErosionParams &>(p);
	for (unsigned int i = 0; i < count; ++i) {
		const FCErodedTerrain r = FCPlanetErosion(MakeFloat3(x[i], y[i], z[i]), fp);
		out_height[i] = r.Height;
		if (out_ridge != nullptr) {
			out_ridge[i] = r.Ridge;
		}
		if (out_erosion != nullptr) {
			out_erosion[i] = r.Erosion;
		}
	}
}

} // namespace zylann::voxel

#endif // VOXEL_ISPC_ENABLED
