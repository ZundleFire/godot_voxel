#ifndef VOXEL_TERRAIN_NOISE_H
#define VOXEL_TERRAIN_NOISE_H

// Terrain diffusion noise — batch series API for the voxel graph nodes.
// Ported from the VCET Unreal plugin's TerrainDiffusion kernels (MIT).
// Dispatches to ISPC kernels when available (VOXEL_ISPC_ENABLED), otherwise
// to the scalar path compiled from the same single-source implementation
// (voxel_procedural_noise_impl.h), so both paths are bit-identical in math.

#include <cstdint>

namespace zylann::voxel {

// These structs are stored directly as compiled voxel-graph node params,
// which requires them to be trivial — no default member initializers.
// Defaults live in the node parameter declarations (nodes/terrain_diffusion.h).
// Initializer helpers below give safe defaults for direct C++ use.

// Optional geographic shaping stages. Every stage is an exact no-op when its
// blend/strength is 0.
struct TerrainShapingParams {
	float warp_strength; // world units of positional warp
	float warp_scale; // feature scale of the warp field
	float mountain_blend; // 0-1
	float mountain_scale; // spacing of mountain range bands
	float continent_blend; // 0-1
	float continent_scale; // size of landmasses
	float island_bias; // -1 (archipelago) .. +1 (mostly land)
	float canyon_blend; // 0-1
	float canyon_scale; // spacing of canyon networks
	float terrace_strength; // 0-1
	float terrace_count; // number of terrace levels
};

struct TerrainHeightParams {
	float amplitude;
	float feature_scale;
	float lacunarity;
	float gain;
	float aesthetic_bias; // 0 = realistic (eroded fBm), 1 = fantastical (ridged)
	int32_t num_octaves;
	int32_t seed;
	float planet_radius; // 3D only
	TerrainShapingParams shaping;
};

struct TerrainClimateParams {
	float amplitude; // must match the height node's amplitude
	float sea_level; // normalized height [-1, 1] of the water surface
	float climate_scale;
	float elevation_cooling; // 0-1
	float temperature_variation; // 0-1
	int32_t seed;
	float planet_radius; // 3D only
};

struct TerrainRiversParams {
	float sea_level; // normalized height [0, 1], matches climate output space
	float river_scale; // spacing of the river network
	float river_width; // 0-1
	float flow_strength; // 0-1
	int32_t seed;
};

inline TerrainShapingParams make_default_terrain_shaping_params() {
	return TerrainShapingParams{ 0.f, 500.f, 0.f, 2000.f, 0.f, 8000.f, 0.f, 0.f, 1500.f, 0.f, 8.f };
}

inline TerrainHeightParams make_default_terrain_height_params() {
	return TerrainHeightParams{ 50.f, 500.f, 2.f, 0.5f, 0.f, 5, 1337, 1000.f,
		make_default_terrain_shaping_params() };
}

inline TerrainClimateParams make_default_terrain_climate_params() {
	return TerrainClimateParams{ 50.f, 0.f, 4000.f, 0.6f, 0.5f, 1337, 1000.f };
}

inline TerrainRiversParams make_default_terrain_rivers_params() {
	return TerrainRiversParams{ 0.5f, 300.f, 0.35f, 0.55f, 1337 };
}

// Height in world units: [-amplitude, amplitude].
void terrain_height_2d_series(
		const float *x, const float *y, float *out, unsigned int count,
		const TerrainHeightParams &p);

// Radial surface distance from planet center: planet_radius + height.
void terrain_height_3d_series(
		const float *x, const float *y, const float *z, float *out, unsigned int count,
		const TerrainHeightParams &p);

// All outputs normalized [0, 1].
void terrain_climate_2d_series(
		const float *x, const float *y, const float *height,
		float *out_temperature, float *out_moisture, float *out_normalized_height,
		unsigned int count, const TerrainClimateParams &p);

void terrain_climate_3d_series(
		const float *x, const float *y, const float *z, const float *radial_distance,
		float *out_temperature, float *out_moisture, float *out_normalized_height,
		unsigned int count, const TerrainClimateParams &p);

// Azimuthal-equidistant tangent frame around a stamp direction on a planet.
void sphere_stamp_series(
		const float *x, const float *y, const float *z,
		float *out_local_x, float *out_local_y, float *out_uv_x, float *out_uv_y,
		float *out_strength, unsigned int count,
		float center_x, float center_y, float center_z,
		float stamp_size, float falloff, float rotation_degrees, float planet_radius);

void terrain_rivers_2d_series(
		const float *x, const float *y, const float *normalized_height,
		float *out_river_mask, float *out_river_depth,
		unsigned int count, const TerrainRiversParams &p);

// Biome ids: 0 DeepOcean, 1 ShallowOcean, 2 Coast, 3 River, 4 Tropical,
// 5 Forest, 6 Plains, 7 Desert, 8 Tundra, 9 Mountain, 10 Snow.
void terrain_material_blend_series(
		const float *normalized_height, const float *temperature, const float *moisture,
		const float *river_mask,
		float *out_biome_id, float *out_ocean_mask, float *out_coast_mask,
		float *out_river_feature_mask, float *out_vegetation_mask, float *out_desert_mask,
		float *out_tundra_mask, float *out_mountain_mask, float *out_snow_mask,
		unsigned int count, float sea_level, float biome_contrast);

} // namespace zylann::voxel

#endif // VOXEL_TERRAIN_NOISE_H
