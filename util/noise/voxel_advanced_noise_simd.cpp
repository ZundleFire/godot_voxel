#include "voxel_advanced_noise_simd.h"

#ifdef VOXEL_ISPC_ENABLED

// Include the ISPC-generated headers (created by ISPC compiler from the .ispc files).
// These provide extern "C" declarations for the exported ISPC functions.
#include "voxel_advanced_noise_ispc_ispc.h"
#include "voxel_domain_warp_ispc_ispc.h"

namespace zylann {

bool voxel_ispc_available() {
	return true;
}

void voxel_ispc_fill_noise_3d(
		float origin_x, float origin_y, float origin_z,
		int32_t size_x, int32_t size_y, int32_t size_z,
		float spacing,
		float amplitude, float feature_scale,
		float lacunarity, float gain, float cellular_jitter,
		int32_t seed,
		const ISPCOctaveConfig *octaves, int32_t num_octaves,
		float *output, int32_t total_count) {
	// The ISPC OctaveConfig struct layout must match ISPCOctaveConfig.
	// Both have { int32 type; float strength; } — same layout, safe to reinterpret.
	ispc::VoxelAdvancedNoise3D_FillBuffer(
			origin_x, origin_y, origin_z,
			size_x, size_y, size_z,
			spacing,
			amplitude, feature_scale,
			lacunarity, gain, cellular_jitter,
			seed,
			reinterpret_cast<const ispc::OctaveConfig *>(octaves), num_octaves,
			output, total_count);
}

void voxel_ispc_fill_noise_2d(
		float origin_x, float origin_y,
		int32_t size_x, int32_t size_y,
		float spacing,
		float amplitude, float feature_scale,
		float lacunarity, float gain, float cellular_jitter,
		int32_t seed,
		const ISPCOctaveConfig *octaves, int32_t num_octaves,
		float *output, int32_t total_count) {
	ispc::VoxelAdvancedNoise2D_FillBuffer(
			origin_x, origin_y,
			size_x, size_y,
			spacing,
			amplitude, feature_scale,
			lacunarity, gain, cellular_jitter,
			seed,
			reinterpret_cast<const ispc::OctaveConfig *>(octaves), num_octaves,
			output, total_count);
}

void voxel_ispc_noise_3d_batch(
		const float *pos_x, const float *pos_y, const float *pos_z,
		float amplitude, float feature_scale,
		float lacunarity, float gain, float cellular_jitter,
		int32_t seed,
		const ISPCOctaveConfig *octaves, int32_t num_octaves,
		float *output, int32_t count) {
	ispc::VoxelAdvancedNoise3D_Batch(
			pos_x, pos_y, pos_z,
			amplitude, feature_scale,
			lacunarity, gain, cellular_jitter,
			seed,
			reinterpret_cast<const ispc::OctaveConfig *>(octaves), num_octaves,
			output, count);
}

// ---------------------------------------------------------------------------
// Standalone noise batch functions
// ---------------------------------------------------------------------------

void voxel_ispc_perlin_3d_batch(const float *pos_x, const float *pos_y, const float *pos_z,
		int32_t seed, float *output, int32_t count) {
	ispc::VoxelNoise_Perlin3D_Batch(pos_x, pos_y, pos_z, seed, output, count);
}

void voxel_ispc_perlin_2d_batch(const float *pos_x, const float *pos_y,
		int32_t seed, float *output, int32_t count) {
	ispc::VoxelNoise_Perlin2D_Batch(pos_x, pos_y, seed, output, count);
}

void voxel_ispc_simplex_3d_batch(const float *pos_x, const float *pos_y, const float *pos_z,
		int32_t seed, float *output, int32_t count) {
	ispc::VoxelNoise_Simplex3D_Batch(pos_x, pos_y, pos_z, seed, output, count);
}

void voxel_ispc_simplex_2d_batch(const float *pos_x, const float *pos_y,
		int32_t seed, float *output, int32_t count) {
	ispc::VoxelNoise_Simplex2D_Batch(pos_x, pos_y, seed, output, count);
}

void voxel_ispc_cellular_3d_batch(const float *pos_x, const float *pos_y, const float *pos_z,
		int32_t seed, float jitter, float *output, int32_t count) {
	ispc::VoxelNoise_Cellular3D_Batch(pos_x, pos_y, pos_z, seed, jitter, output, count);
}

void voxel_ispc_cellular_2d_batch(const float *pos_x, const float *pos_y,
		int32_t seed, float jitter, float *output, int32_t count) {
	ispc::VoxelNoise_Cellular2D_Batch(pos_x, pos_y, seed, jitter, output, count);
}

void voxel_ispc_value_3d_batch(const float *pos_x, const float *pos_y, const float *pos_z,
		int32_t seed, float *output, int32_t count) {
	ispc::VoxelNoise_Value3D_Batch(pos_x, pos_y, pos_z, seed, output, count);
}

void voxel_ispc_value_2d_batch(const float *pos_x, const float *pos_y,
		int32_t seed, float *output, int32_t count) {
	ispc::VoxelNoise_Value2D_Batch(pos_x, pos_y, seed, output, count);
}

void voxel_ispc_true_distance_cellular_3d_batch(const float *pos_x, const float *pos_y, const float *pos_z,
		int32_t seed, float jitter, float *output, int32_t count) {
	ispc::VoxelNoise_TrueDistanceCellular3D_Batch(pos_x, pos_y, pos_z, seed, jitter, output, count);
}

void voxel_ispc_true_distance_cellular_2d_batch(const float *pos_x, const float *pos_y,
		int32_t seed, float jitter, float *output, int32_t count) {
	ispc::VoxelNoise_TrueDistanceCellular2D_Batch(pos_x, pos_y, seed, jitter, output, count);
}

// ---------------------------------------------------------------------------
// Domain warp functions
// ---------------------------------------------------------------------------

void voxel_ispc_domain_warp_3d_grid(
		float origin_x, float origin_y, float origin_z,
		int32_t size_x, int32_t size_y, int32_t size_z, float spacing,
		float amplitude, float feature_scale,
		float lacunarity, float gain, float weighted_strength,
		int32_t num_octaves, int32_t seed,
		float *out_x, float *out_y, float *out_z, int32_t total_count) {
	ispc::VoxelDomainWarp3D_FillBuffer(
			origin_x, origin_y, origin_z,
			size_x, size_y, size_z, spacing,
			amplitude, feature_scale,
			lacunarity, gain, weighted_strength,
			num_octaves, seed,
			out_x, out_y, out_z, total_count);
}

void voxel_ispc_domain_warp_2d_grid(
		float origin_x, float origin_y,
		int32_t size_x, int32_t size_y, float spacing,
		float amplitude, float feature_scale,
		float lacunarity, float gain, float weighted_strength,
		int32_t num_octaves, int32_t seed,
		float *out_x, float *out_y, int32_t total_count) {
	ispc::VoxelDomainWarp2D_FillBuffer(
			origin_x, origin_y,
			size_x, size_y, spacing,
			amplitude, feature_scale,
			lacunarity, gain, weighted_strength,
			num_octaves, seed,
			out_x, out_y, total_count);
}

void voxel_ispc_domain_warp_3d_batch(
		const float *pos_x, const float *pos_y, const float *pos_z,
		float amplitude, float feature_scale,
		float lacunarity, float gain, float weighted_strength,
		int32_t num_octaves, int32_t seed,
		float *out_x, float *out_y, float *out_z, int32_t count) {
	ispc::VoxelDomainWarp3D_Batch(
			pos_x, pos_y, pos_z,
			amplitude, feature_scale,
			lacunarity, gain, weighted_strength,
			num_octaves, seed,
			out_x, out_y, out_z, count);
}

} // namespace zylann

#endif // VOXEL_ISPC_ENABLED
