#ifndef VOXEL_NOISE_BATCH_H
#define VOXEL_NOISE_BATCH_H

#include <core/object/class_db.h>
#include <core/variant/variant.h>

namespace zylann {

// VoxelNoiseBatch — static utility class exposed to GDScript for SIMD-accelerated
// batch noise operations. Designed to be called from VoxelGeneratorScript._generate_block()
// to process entire blocks of voxels at once via ISPC SIMD instead of per-voxel scalar calls.
//
// All methods accept position arrays in Structure-of-Arrays (SoA) layout for SIMD efficiency.
// Use prepare_positions_3d() to convert a block origin + size + lod into SoA arrays.
//
// Example GDScript usage in _generate_block():
//
//   var positions := VoxelNoiseBatch.prepare_positions_3d(origin_in_voxels, size, lod)
//   var noise_values := VoxelNoiseBatch.perlin_3d_batch(positions[0], positions[1], positions[2], seed)
//   for i in range(noise_values.size()):
//       out_buffer.set_voxel_f(noise_values[i], ...)
//
class VoxelNoiseBatch : public Object {
	GDCLASS(VoxelNoiseBatch, Object)
public:
	// -----------------------------------------------------------------------
	// Position preparation helpers
	// -----------------------------------------------------------------------

	// Prepare SoA position arrays for a 3D voxel block.
	// Returns [PackedFloat32Array x, PackedFloat32Array y, PackedFloat32Array z]
	// in world coordinates, accounting for LOD scale.
	static Array prepare_positions_3d(Vector3i origin, Vector3i size, int lod);

	// Prepare SoA position arrays for a 2D grid (XZ plane at y=0).
	// Returns [PackedFloat32Array x, PackedFloat32Array z]
	static Array prepare_positions_2d(Vector2 origin, Vector2i size, float spacing);

	// -----------------------------------------------------------------------
	// Individual noise types — batch operations
	// All return PackedFloat32Array with one value per input position.
	// Coordinates should already be scaled (divided by feature_scale).
	// When ISPC is compiled in, these use SIMD. Otherwise scalar fallback.
	// -----------------------------------------------------------------------

	static PackedFloat32Array perlin_3d_batch(
			const PackedFloat32Array &pos_x,
			const PackedFloat32Array &pos_y,
			const PackedFloat32Array &pos_z,
			int seed);

	static PackedFloat32Array perlin_2d_batch(
			const PackedFloat32Array &pos_x,
			const PackedFloat32Array &pos_y,
			int seed);

	static PackedFloat32Array simplex_3d_batch(
			const PackedFloat32Array &pos_x,
			const PackedFloat32Array &pos_y,
			const PackedFloat32Array &pos_z,
			int seed);

	static PackedFloat32Array simplex_2d_batch(
			const PackedFloat32Array &pos_x,
			const PackedFloat32Array &pos_y,
			int seed);

	static PackedFloat32Array cellular_3d_batch(
			const PackedFloat32Array &pos_x,
			const PackedFloat32Array &pos_y,
			const PackedFloat32Array &pos_z,
			int seed, float jitter);

	static PackedFloat32Array cellular_2d_batch(
			const PackedFloat32Array &pos_x,
			const PackedFloat32Array &pos_y,
			int seed, float jitter);

	static PackedFloat32Array value_3d_batch(
			const PackedFloat32Array &pos_x,
			const PackedFloat32Array &pos_y,
			const PackedFloat32Array &pos_z,
			int seed);

	static PackedFloat32Array value_2d_batch(
			const PackedFloat32Array &pos_x,
			const PackedFloat32Array &pos_y,
			int seed);

	static PackedFloat32Array true_distance_cellular_3d_batch(
			const PackedFloat32Array &pos_x,
			const PackedFloat32Array &pos_y,
			const PackedFloat32Array &pos_z,
			int seed, float jitter);

	static PackedFloat32Array true_distance_cellular_2d_batch(
			const PackedFloat32Array &pos_x,
			const PackedFloat32Array &pos_y,
			int seed, float jitter);

	// -----------------------------------------------------------------------
	// Domain warp — multi-octave coordinate warping
	// Returns [PackedFloat32Array warped_x, warped_y, warped_z] (3D)
	// or [PackedFloat32Array warped_x, warped_y] (2D)
	// -----------------------------------------------------------------------

	// 3D domain warp on a regular grid
	static Array domain_warp_3d_grid(
			Vector3 origin, Vector3i size, float spacing,
			float amplitude, float feature_scale,
			float lacunarity, float gain, float weighted_strength,
			int num_octaves, int seed);

	// 2D domain warp on a regular grid
	static Array domain_warp_2d_grid(
			Vector2 origin, Vector2i size, float spacing,
			float amplitude, float feature_scale,
			float lacunarity, float gain, float weighted_strength,
			int num_octaves, int seed);

	// 3D domain warp on arbitrary positions (SoA)
	static Array domain_warp_3d_batch(
			const PackedFloat32Array &pos_x,
			const PackedFloat32Array &pos_y,
			const PackedFloat32Array &pos_z,
			float amplitude, float feature_scale,
			float lacunarity, float gain, float weighted_strength,
			int num_octaves, int seed);

	// -----------------------------------------------------------------------
	// Info
	// -----------------------------------------------------------------------

	// Returns true if ISPC SIMD acceleration is available.
	static bool is_simd_available();

protected:
	static void _bind_methods();
};

} // namespace zylann

#endif // VOXEL_NOISE_BATCH_H
