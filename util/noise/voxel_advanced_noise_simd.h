#ifndef VOXEL_ADVANCED_NOISE_SIMD_H
#define VOXEL_ADVANCED_NOISE_SIMD_H

// SIMD dispatch wrapper for VoxelAdvancedNoise.
// When ISPC is available (VOXEL_ISPC_ENABLED), uses vectorized ISPC kernels.
// Otherwise provides inline scalar fallback (no-op — caller uses VoxelAdvancedNoise directly).

#include <cstdint>

namespace zylann {

// Matches the OctaveConfig layout expected by the ISPC kernel.
// Must stay in sync with the `OctaveConfig` struct in voxel_advanced_noise_ispc.ispc.
struct ISPCOctaveConfig {
	int32_t type;
	float strength;
};

#ifdef VOXEL_ISPC_ENABLED

// Returns true if ISPC SIMD noise is available at runtime.
bool voxel_ispc_available();

// Fill a 3D grid with multi-octave noise using ISPC SIMD.
// Parameters mirror VoxelAdvancedNoise::fill_buffer_3d().
void voxel_ispc_fill_noise_3d(
		float origin_x, float origin_y, float origin_z,
		int32_t size_x, int32_t size_y, int32_t size_z,
		float spacing,
		float amplitude, float feature_scale,
		float lacunarity, float gain, float cellular_jitter,
		int32_t seed,
		const ISPCOctaveConfig *octaves, int32_t num_octaves,
		float *output, int32_t total_count);

// Fill a 2D grid with multi-octave noise using ISPC SIMD.
void voxel_ispc_fill_noise_2d(
		float origin_x, float origin_y,
		int32_t size_x, int32_t size_y,
		float spacing,
		float amplitude, float feature_scale,
		float lacunarity, float gain, float cellular_jitter,
		int32_t seed,
		const ISPCOctaveConfig *octaves, int32_t num_octaves,
		float *output, int32_t total_count);

// Process an arbitrary batch of 3D positions (structure-of-arrays).
void voxel_ispc_noise_3d_batch(
		const float *pos_x, const float *pos_y, const float *pos_z,
		float amplitude, float feature_scale,
		float lacunarity, float gain, float cellular_jitter,
		int32_t seed,
		const ISPCOctaveConfig *octaves, int32_t num_octaves,
		float *output, int32_t count);

// -----------------------------------------------------------------------
// Standalone noise type batch functions
// -----------------------------------------------------------------------

void voxel_ispc_perlin_3d_batch(const float *pos_x, const float *pos_y, const float *pos_z,
		int32_t seed, float *output, int32_t count);
void voxel_ispc_perlin_2d_batch(const float *pos_x, const float *pos_y,
		int32_t seed, float *output, int32_t count);
void voxel_ispc_simplex_3d_batch(const float *pos_x, const float *pos_y, const float *pos_z,
		int32_t seed, float *output, int32_t count);
void voxel_ispc_simplex_2d_batch(const float *pos_x, const float *pos_y,
		int32_t seed, float *output, int32_t count);
void voxel_ispc_cellular_3d_batch(const float *pos_x, const float *pos_y, const float *pos_z,
		int32_t seed, float jitter, float *output, int32_t count);
void voxel_ispc_cellular_2d_batch(const float *pos_x, const float *pos_y,
		int32_t seed, float jitter, float *output, int32_t count);
void voxel_ispc_value_3d_batch(const float *pos_x, const float *pos_y, const float *pos_z,
		int32_t seed, float *output, int32_t count);
void voxel_ispc_value_2d_batch(const float *pos_x, const float *pos_y,
		int32_t seed, float *output, int32_t count);
void voxel_ispc_true_distance_cellular_3d_batch(const float *pos_x, const float *pos_y, const float *pos_z,
		int32_t seed, float jitter, float *output, int32_t count);
void voxel_ispc_true_distance_cellular_2d_batch(const float *pos_x, const float *pos_y,
		int32_t seed, float jitter, float *output, int32_t count);

// -----------------------------------------------------------------------
// Domain warp functions
// -----------------------------------------------------------------------

void voxel_ispc_domain_warp_3d_grid(
		float origin_x, float origin_y, float origin_z,
		int32_t size_x, int32_t size_y, int32_t size_z, float spacing,
		float amplitude, float feature_scale,
		float lacunarity, float gain, float weighted_strength,
		int32_t num_octaves, int32_t seed,
		float *out_x, float *out_y, float *out_z, int32_t total_count);

void voxel_ispc_domain_warp_2d_grid(
		float origin_x, float origin_y,
		int32_t size_x, int32_t size_y, float spacing,
		float amplitude, float feature_scale,
		float lacunarity, float gain, float weighted_strength,
		int32_t num_octaves, int32_t seed,
		float *out_x, float *out_y, int32_t total_count);

void voxel_ispc_domain_warp_3d_batch(
		const float *pos_x, const float *pos_y, const float *pos_z,
		float amplitude, float feature_scale,
		float lacunarity, float gain, float weighted_strength,
		int32_t num_octaves, int32_t seed,
		float *out_x, float *out_y, float *out_z, int32_t count);

#else

// Stub when ISPC is not compiled in.
inline bool voxel_ispc_available() {
	return false;
}

#endif // VOXEL_ISPC_ENABLED

} // namespace zylann

#endif // VOXEL_ADVANCED_NOISE_SIMD_H
