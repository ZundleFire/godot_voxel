// ISPC kernels for VoxelProceduralNoise — multi-octave stylized procedural noise.
// The noise math lives in voxel_procedural_noise_impl.h (shared with the scalar C++ path).
// Ported from the VCET Unreal plugin (MIT), see the impl header for full license.

#include "voxel_procedural_noise_impl.h"

// Must stay in sync with ProceduralOctaveConfig in voxel_procedural_noise_simd.h
struct ProceduralOctaveConfig {
	int32 type;
	float strength;
};

export void VoxelProceduralNoise2D_Batch(
		const uniform float pos_x[],
		const uniform float pos_y[],
		const uniform float amplitude,
		const uniform float feature_scale,
		const uniform float lacunarity,
		const uniform float gain,
		const uniform float voronoi_smoothness,
		const uniform float wavelet_phase,
		const uniform float scratch_smoothness,
		const uniform int32 in_seed,
		const uniform ProceduralOctaveConfig octaves[],
		const uniform int32 num_octaves,
		uniform float output[],
		const uniform int32 count) {
	foreach (index = 0 ... count) {
		varying float sum = 0.f;
		varying float amplitude_sum = 0.f;

		varying float octave_amplitude = 1.f;
		varying float2 position = MakeFloat2(pos_x[index], pos_y[index]) / feature_scale;
		uniform int32 seed = in_seed;

		for (uniform int32 octave_index = 0; octave_index < num_octaves; octave_index++) {
			const uniform ProceduralOctaveConfig octave = octaves[octave_index];

			const varying float noise = FCSampleNoise2D(
					octave.type,
					(uniform uint32)seed,
					position,
					voronoi_smoothness,
					wavelet_phase,
					scratch_smoothness);

			const varying float new_sum = sum + noise * octave.strength * octave_amplitude;
			sum = select(FCIsFinite(new_sum), new_sum, sum);

			amplitude_sum = amplitude_sum + abs(octave.strength * octave_amplitude);
			octave_amplitude = octave_amplitude * gain;
			position = position * lacunarity;
			seed = FCNextOctaveSeed(seed);
		}

		output[index] = sum / (amplitude_sum == 0.f ? 1.f : amplitude_sum) * amplitude;
	}
}

export void VoxelProceduralNoise3D_Batch(
		const uniform float pos_x[],
		const uniform float pos_y[],
		const uniform float pos_z[],
		const uniform float amplitude,
		const uniform float feature_scale,
		const uniform float lacunarity,
		const uniform float gain,
		const uniform float voronoi_smoothness,
		const uniform float wavelet_phase,
		const uniform float scratch_smoothness,
		const uniform int32 in_seed,
		const uniform ProceduralOctaveConfig octaves[],
		const uniform int32 num_octaves,
		uniform float output[],
		const uniform int32 count) {
	foreach (index = 0 ... count) {
		varying float sum = 0.f;
		varying float amplitude_sum = 0.f;

		varying float octave_amplitude = 1.f;
		varying float3 position = MakeFloat3(pos_x[index], pos_y[index], pos_z[index]) / feature_scale;
		uniform int32 seed = in_seed;

		for (uniform int32 octave_index = 0; octave_index < num_octaves; octave_index++) {
			const uniform ProceduralOctaveConfig octave = octaves[octave_index];

			const varying float noise = FCSampleNoise3D(
					octave.type,
					(uniform uint32)seed,
					position,
					voronoi_smoothness,
					wavelet_phase,
					scratch_smoothness);

			const varying float new_sum = sum + noise * octave.strength * octave_amplitude;
			sum = select(FCIsFinite(new_sum), new_sum, sum);

			amplitude_sum = amplitude_sum + abs(octave.strength * octave_amplitude);
			octave_amplitude = octave_amplitude * gain;
			position = position * lacunarity;
			seed = FCNextOctaveSeed(seed);
		}

		output[index] = sum / (amplitude_sum == 0.f ? 1.f : amplitude_sum) * amplitude;
	}
}

///////////////////////////////////////////////////////////////////////////////
// Terrain diffusion kernels — ported from the VCET Unreal plugin (MIT).
// Per-point math lives in voxel_procedural_noise_impl.h (shared with the
// scalar C++ path). Positions are varying arrays; all parameters are uniform
// (graph node parameters, constant per block).
///////////////////////////////////////////////////////////////////////////////

#define TERRAIN_SHAPING_PARAMS \
		const uniform float warp_strength, \
		const uniform float warp_scale, \
		const uniform float mountain_blend, \
		const uniform float mountain_scale, \
		const uniform float continent_blend, \
		const uniform float continent_scale, \
		const uniform float island_bias, \
		const uniform float canyon_blend, \
		const uniform float canyon_scale, \
		const uniform float terrace_strength, \
		const uniform float terrace_count

#define MAKE_TERRAIN_SHAPING(shaping) \
		FCTerrainShaping shaping; \
		shaping.WarpStrength = warp_strength; \
		shaping.WarpScale = warp_scale; \
		shaping.MountainBlend = mountain_blend; \
		shaping.MountainScale = mountain_scale; \
		shaping.ContinentBlend = continent_blend; \
		shaping.ContinentScale = continent_scale; \
		shaping.IslandBias = island_bias; \
		shaping.CanyonBlend = canyon_blend; \
		shaping.CanyonScale = canyon_scale; \
		shaping.TerraceStrength = terrace_strength; \
		shaping.TerraceCount = terrace_count;

export void VoxelTerrainHeight2D_Batch(
		const uniform float pos_x[],
		const uniform float pos_y[],
		const uniform float amplitude,
		const uniform float feature_scale,
		const uniform float lacunarity,
		const uniform float gain,
		const uniform float aesthetic_bias,
		const uniform int32 num_octaves,
		const uniform int32 seed,
		TERRAIN_SHAPING_PARAMS,
		uniform float output[],
		const uniform int32 count) {
	MAKE_TERRAIN_SHAPING(shaping)
	foreach (index = 0 ... count) {
		output[index] = FCTerrainHeight2D(
				seed,
				MakeFloat2(pos_x[index], pos_y[index]),
				amplitude, feature_scale, lacunarity, gain, aesthetic_bias,
				num_octaves, shaping);
	}
}

export void VoxelTerrainHeight3D_Batch(
		const uniform float pos_x[],
		const uniform float pos_y[],
		const uniform float pos_z[],
		const uniform float amplitude,
		const uniform float feature_scale,
		const uniform float lacunarity,
		const uniform float gain,
		const uniform float aesthetic_bias,
		const uniform int32 num_octaves,
		const uniform int32 seed,
		const uniform float planet_radius,
		TERRAIN_SHAPING_PARAMS,
		uniform float output[],
		const uniform int32 count) {
	MAKE_TERRAIN_SHAPING(shaping)
	foreach (index = 0 ... count) {
		output[index] = FCTerrainHeight3D(
				seed,
				MakeFloat3(pos_x[index], pos_y[index], pos_z[index]),
				amplitude, feature_scale, lacunarity, gain, aesthetic_bias,
				num_octaves, planet_radius, shaping);
	}
}

export void VoxelTerrainClimate2D_Batch(
		const uniform float pos_x[],
		const uniform float pos_y[],
		const uniform float height[],
		const uniform float amplitude,
		const uniform float sea_level,
		const uniform float climate_scale,
		const uniform float elevation_cooling,
		const uniform float temperature_variation,
		const uniform int32 seed,
		uniform float out_temperature[],
		uniform float out_moisture[],
		uniform float out_normalized_height[],
		const uniform int32 count) {
	foreach (index = 0 ... count) {
		const FCClimateResult r = FCTerrainClimate2D(
				seed,
				MakeFloat2(pos_x[index], pos_y[index]),
				height[index],
				amplitude, sea_level, climate_scale, elevation_cooling, temperature_variation);
		out_temperature[index] = r.Temperature;
		out_moisture[index] = r.Moisture;
		out_normalized_height[index] = r.NormalizedHeight;
	}
}

export void VoxelTerrainClimate3D_Batch(
		const uniform float pos_x[],
		const uniform float pos_y[],
		const uniform float pos_z[],
		const uniform float radial_distance[],
		const uniform float amplitude,
		const uniform float sea_level,
		const uniform float climate_scale,
		const uniform float elevation_cooling,
		const uniform float temperature_variation,
		const uniform int32 seed,
		const uniform float planet_radius,
		uniform float out_temperature[],
		uniform float out_moisture[],
		uniform float out_normalized_height[],
		const uniform int32 count) {
	foreach (index = 0 ... count) {
		const FCClimateResult r = FCTerrainClimate3D(
				seed,
				MakeFloat3(pos_x[index], pos_y[index], pos_z[index]),
				radial_distance[index],
				amplitude, sea_level, climate_scale, elevation_cooling, temperature_variation,
				planet_radius);
		out_temperature[index] = r.Temperature;
		out_moisture[index] = r.Moisture;
		out_normalized_height[index] = r.NormalizedHeight;
	}
}

export void VoxelSphereStamp_Batch(
		const uniform float pos_x[],
		const uniform float pos_y[],
		const uniform float pos_z[],
		const uniform float center_x,
		const uniform float center_y,
		const uniform float center_z,
		const uniform float stamp_size,
		const uniform float falloff,
		const uniform float rotation_degrees,
		const uniform float planet_radius,
		uniform float out_local_x[],
		uniform float out_local_y[],
		uniform float out_uv_x[],
		uniform float out_uv_y[],
		uniform float out_strength[],
		const uniform int32 count) {
	foreach (index = 0 ... count) {
		const FCSphereStampResult r = FCSphereStamp(
				MakeFloat3(pos_x[index], pos_y[index], pos_z[index]),
				MakeFloat3(center_x, center_y, center_z),
				stamp_size, falloff, rotation_degrees, planet_radius);
		out_local_x[index] = r.LocalX;
		out_local_y[index] = r.LocalY;
		out_uv_x[index] = r.UVX;
		out_uv_y[index] = r.UVY;
		out_strength[index] = r.Strength;
	}
}

export void VoxelTerrainRivers2D_Batch(
		const uniform float pos_x[],
		const uniform float pos_y[],
		const uniform float normalized_height[],
		const uniform float sea_level,
		const uniform float river_scale,
		const uniform float river_width,
		const uniform float flow_strength,
		const uniform int32 seed,
		uniform float out_river_mask[],
		uniform float out_river_depth[],
		const uniform int32 count) {
	foreach (index = 0 ... count) {
		const FCRiversResult r = FCTerrainRivers2D(
				seed,
				MakeFloat2(pos_x[index], pos_y[index]),
				normalized_height[index],
				sea_level, river_scale, river_width, flow_strength);
		out_river_mask[index] = r.Mask;
		out_river_depth[index] = r.Depth;
	}
}

export void VoxelTerrainMaterialBlend_Batch(
		const uniform float normalized_height[],
		const uniform float temperature[],
		const uniform float moisture[],
		const uniform float river_mask[],
		const uniform float sea_level,
		const uniform float biome_contrast,
		uniform float out_biome_id[],
		uniform float out_ocean_mask[],
		uniform float out_coast_mask[],
		uniform float out_river_feature_mask[],
		uniform float out_vegetation_mask[],
		uniform float out_desert_mask[],
		uniform float out_tundra_mask[],
		uniform float out_mountain_mask[],
		uniform float out_snow_mask[],
		const uniform int32 count) {
	foreach (index = 0 ... count) {
		const FCMaterialBlendResult r = FCTerrainMaterialBlend(
				normalized_height[index],
				temperature[index],
				moisture[index],
				sea_level,
				river_mask[index],
				biome_contrast);
		out_biome_id[index] = r.BiomeId;
		out_ocean_mask[index] = r.OceanMask;
		out_coast_mask[index] = r.CoastMask;
		out_river_feature_mask[index] = r.RiverFeatureMask;
		out_vegetation_mask[index] = r.VegetationMask;
		out_desert_mask[index] = r.DesertMask;
		out_tundra_mask[index] = r.TundraMask;
		out_mountain_mask[index] = r.MountainMask;
		out_snow_mask[index] = r.SnowMask;
	}
}
