#include "voxel_procedural_noise_simd.h"

#ifdef VOXEL_ISPC_ENABLED

// ISPC-generated header (created by the ISPC compiler from voxel_procedural_noise_ispc.ispc).
#include "voxel_procedural_noise_ispc_ispc.h"

namespace zylann {

void voxel_ispc_procedural_noise_2d_batch(
		const float *pos_x, const float *pos_y,
		float amplitude, float feature_scale,
		float lacunarity, float gain,
		float voronoi_smoothness, float wavelet_phase, float scratch_smoothness,
		int32_t seed,
		const ProceduralOctaveConfig *octaves, int32_t num_octaves,
		float *output, int32_t count) {
	// Both structs are { int32 type; float strength; } — same layout, safe to reinterpret.
	ispc::VoxelProceduralNoise2D_Batch(
			pos_x, pos_y,
			amplitude, feature_scale,
			lacunarity, gain,
			voronoi_smoothness, wavelet_phase, scratch_smoothness,
			seed,
			reinterpret_cast<const ispc::ProceduralOctaveConfig *>(octaves), num_octaves,
			output, count);
}

void voxel_ispc_procedural_noise_3d_batch(
		const float *pos_x, const float *pos_y, const float *pos_z,
		float amplitude, float feature_scale,
		float lacunarity, float gain,
		float voronoi_smoothness, float wavelet_phase, float scratch_smoothness,
		int32_t seed,
		const ProceduralOctaveConfig *octaves, int32_t num_octaves,
		float *output, int32_t count) {
	ispc::VoxelProceduralNoise3D_Batch(
			pos_x, pos_y, pos_z,
			amplitude, feature_scale,
			lacunarity, gain,
			voronoi_smoothness, wavelet_phase, scratch_smoothness,
			seed,
			reinterpret_cast<const ispc::ProceduralOctaveConfig *>(octaves), num_octaves,
			output, count);
}

} // namespace zylann

#endif // VOXEL_ISPC_ENABLED
