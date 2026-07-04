#ifndef VOXEL_PROCEDURAL_NOISE_SIMD_H
#define VOXEL_PROCEDURAL_NOISE_SIMD_H

// SIMD dispatch wrapper for VoxelProceduralNoise.
// When ISPC is available (VOXEL_ISPC_ENABLED), uses vectorized ISPC kernels.
// Otherwise the caller falls back to the scalar path in VoxelProceduralNoise.

#include <cstdint>

namespace zylann {

// Must stay in sync with ProceduralOctaveConfig in voxel_procedural_noise_ispc.ispc
struct ProceduralOctaveConfig {
	int32_t type;
	float strength;
};

#ifdef VOXEL_ISPC_ENABLED

// Process an arbitrary batch of 2D positions (structure-of-arrays).
void voxel_ispc_procedural_noise_2d_batch(
		const float *pos_x, const float *pos_y,
		float amplitude, float feature_scale,
		float lacunarity, float gain,
		float voronoi_smoothness, float wavelet_phase, float scratch_smoothness,
		int32_t seed,
		const ProceduralOctaveConfig *octaves, int32_t num_octaves,
		float *output, int32_t count);

// Process an arbitrary batch of 3D positions (structure-of-arrays).
void voxel_ispc_procedural_noise_3d_batch(
		const float *pos_x, const float *pos_y, const float *pos_z,
		float amplitude, float feature_scale,
		float lacunarity, float gain,
		float voronoi_smoothness, float wavelet_phase, float scratch_smoothness,
		int32_t seed,
		const ProceduralOctaveConfig *octaves, int32_t num_octaves,
		float *output, int32_t count);

#endif // VOXEL_ISPC_ENABLED

} // namespace zylann

#endif // VOXEL_PROCEDURAL_NOISE_SIMD_H
