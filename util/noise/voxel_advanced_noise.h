#ifndef VOXEL_ADVANCED_NOISE_H
#define VOXEL_ADVANCED_NOISE_H

#include "../containers/fixed_array.h"
#include "../containers/std_vector.h"
#include "../math/interval.h"
#include "voxel_advanced_noise_simd.h"
#include <core/io/resource.h>
#include <core/object/gdvirtual.gen.inc>

namespace zylann {

// Port of UE5 VoxelPlugin's AdvancedNoise3D node.
// Multi-octave noise with per-octave noise type and strength control.
// Supports mixing different noise types (Perlin, Simplex, Cellular, Value) per octave,
// with Smooth/Billowy/Ridged variants.
class VoxelAdvancedNoise : public Resource {
	GDCLASS(VoxelAdvancedNoise, Resource)
public:
	static const int MAX_OCTAVES = 32;

	enum NoiseType {
		NOISE_SMOOTH_PERLIN = 0,
		NOISE_BILLOWY_PERLIN = 1,
		NOISE_RIDGED_PERLIN = 2,
		NOISE_SMOOTH_CELLULAR = 3,
		NOISE_BILLOWY_CELLULAR = 4,
		NOISE_RIDGED_CELLULAR = 5,
		NOISE_SMOOTH_SIMPLEX = 6,
		NOISE_BILLOWY_SIMPLEX = 7,
		NOISE_RIDGED_SIMPLEX = 8,
		NOISE_SMOOTH_VALUE = 9,
		NOISE_BILLOWY_VALUE = 10,
		NOISE_RIDGED_VALUE = 11,
		NOISE_SMOOTH_TRUE_DISTANCE_CELLULAR = 12,
		NOISE_BILLOWY_TRUE_DISTANCE_CELLULAR = 13,
		NOISE_RIDGED_TRUE_DISTANCE_CELLULAR = 14,
		NOISE_TYPE_COUNT
	};

	struct OctaveConfig {
		NoiseType type = NOISE_SMOOTH_PERLIN;
		float strength = 1.0f;
	};

	VoxelAdvancedNoise();

	void set_seed(int seed);
	int get_seed() const;

	void set_amplitude(float amplitude);
	float get_amplitude() const;

	void set_feature_scale(float scale);
	float get_feature_scale() const;

	void set_lacunarity(float lacunarity);
	float get_lacunarity() const;

	void set_gain(float gain);
	float get_gain() const;

	void set_cellular_jitter(float jitter);
	float get_cellular_jitter() const;

	void set_num_octaves(int count);
	int get_num_octaves() const;

	void set_default_octave_type(NoiseType type);
	NoiseType get_default_octave_type() const;

	void set_octave_type(int octave_index, NoiseType type);
	NoiseType get_octave_type(int octave_index) const;

	void set_octave_strength(int octave_index, float strength);
	float get_octave_strength(int octave_index) const;

	// Sample the noise at a 3D position. Thread-safe (read-only state after configuration).
	float get_noise_3d(float x, float y, float z) const;
	float get_noise_3dv(Vector3 pos) const;

	// Sample the noise at a 2D position (XZ plane). Thread-safe.
	float get_noise_2d(float x, float y) const;
	float get_noise_2dv(Vector2 pos) const;

	// Fill a buffer with 3D noise values. More efficient than per-voxel calls.
	// Uses ISPC SIMD acceleration when available, otherwise scalar fallback.
	PackedFloat32Array fill_buffer_3d(Vector3 origin, Vector3i size, float spacing) const;

	// Returns true if ISPC SIMD acceleration is available for batch noise operations.
	static bool is_simd_available();

	// Get an estimate of the noise range (min, max) for interval analysis.
	math::Interval get_estimated_output_range() const;

protected:
	static void _bind_methods();

private:
	// Internal noise functions (stateless, use seed parameter)
	static float _perlin_3d(int seed, float x, float y, float z);
	static float _simplex_3d(int seed, float x, float y, float z);
	static float _cellular_3d(int seed, float x, float y, float z, float jitter);
	static float _value_3d(int seed, float x, float y, float z);

	static float _perlin_2d(int seed, float x, float y);
	static float _simplex_2d(int seed, float x, float y);
	static float _cellular_2d(int seed, float x, float y, float jitter);
	static float _value_2d(int seed, float x, float y);

	float _sample_octave_3d(NoiseType type, int seed, float x, float y, float z, float jitter) const;
	float _sample_octave_2d(NoiseType type, int seed, float x, float y, float jitter) const;

	int _seed = 0;
	float _amplitude = 10000.0f;
	float _feature_scale = 100000.0f;
	float _lacunarity = 2.0f;
	float _gain = 0.5f;
	float _cellular_jitter = 0.9f;
	int _num_octaves = 10;
	NoiseType _default_octave_type = NOISE_SMOOTH_PERLIN;
	FixedArray<OctaveConfig, MAX_OCTAVES> _octave_configs;
};

} // namespace zylann

VARIANT_ENUM_CAST(zylann::VoxelAdvancedNoise::NoiseType);

#endif // VOXEL_ADVANCED_NOISE_H
