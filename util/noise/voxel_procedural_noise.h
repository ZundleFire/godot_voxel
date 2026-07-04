#ifndef VOXEL_PROCEDURAL_NOISE_H
#define VOXEL_PROCEDURAL_NOISE_H

#include "../containers/fixed_array.h"
#include "../math/interval.h"
#include <core/io/resource.h>

namespace zylann {

// Port of the VCET Unreal plugin's ProceduralNoise2D/3D nodes
// (https://github.com/ZundleFire/VCET, MIT), themselves ported from the
// Procedural Noise Collection by @lumiey (MIT).
// Multi-octave stylized noise with per-octave noise type and strength control.
// 17 noise types: Perlin, Simplex, Value, Worley, Voronoi, Blue, HilbertBlue,
// Crater, Gabor, Curl, Scratch, Wavelet, Erosion, Paper, Stone, Wool,
// InterleavedGradient.
class VoxelProceduralNoise : public Resource {
	GDCLASS(VoxelProceduralNoise, Resource)
public:
	static const int MAX_OCTAVES = 32;

	// Values 0..16 must match FCProceduralNoiseType in voxel_procedural_noise_impl.h
	enum NoiseType {
		NOISE_PERLIN = 0,
		NOISE_SIMPLEX = 1,
		NOISE_VALUE = 2,
		NOISE_WORLEY = 3,
		NOISE_VORONOI = 4,
		NOISE_BLUE = 5,
		NOISE_HILBERT_BLUE = 6,
		NOISE_CRATER = 7,
		NOISE_GABOR = 8,
		NOISE_CURL = 9,
		NOISE_SCRATCH = 10,
		NOISE_WAVELET = 11,
		NOISE_EROSION = 12,
		NOISE_PAPER = 13,
		NOISE_STONE = 14,
		NOISE_WOOL = 15,
		NOISE_INTERLEAVED_GRADIENT = 16,
		NOISE_TYPE_COUNT = 17,
		// Octave-only sentinel: octave uses the resource's default noise type.
		// Mirrors the VCET plugin's `Default` enum value.
		NOISE_DEFAULT = 200
	};

	struct OctaveConfig {
		NoiseType type = NOISE_DEFAULT;
		float strength = 1.0f;
	};

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

	void set_voronoi_smoothness(float smoothness);
	float get_voronoi_smoothness() const;

	void set_wavelet_phase(float phase);
	float get_wavelet_phase() const;

	void set_scratch_smoothness(float smoothness);
	float get_scratch_smoothness() const;

	void set_num_octaves(int count);
	int get_num_octaves() const;

	void set_default_noise_type(NoiseType type);
	NoiseType get_default_noise_type() const;

	void set_octave_type(int octave_index, NoiseType type);
	NoiseType get_octave_type(int octave_index) const;

	void set_octave_strength(int octave_index, float strength);
	float get_octave_strength(int octave_index) const;

	// Sample the noise at a single position. Thread-safe (read-only state after configuration).
	float get_noise_2d(float x, float y) const;
	float get_noise_2dv(Vector2 pos) const;
	float get_noise_3d(float x, float y, float z) const;
	float get_noise_3dv(Vector3 pos) const;

	// Batch sampling on structure-of-arrays positions.
	// Uses ISPC SIMD acceleration when available, otherwise scalar fallback.
	void get_noise_2d_series(const float *x, const float *y, float *out, unsigned int count) const;
	void get_noise_3d_series(const float *x, const float *y, const float *z, float *out, unsigned int count) const;

	// Fill a buffer with 3D noise values on a regular grid. More efficient than per-voxel calls.
	PackedFloat32Array fill_buffer_3d(Vector3 origin, Vector3i size, float spacing) const;

	// Returns true if ISPC SIMD acceleration is available for batch noise operations.
	static bool is_simd_available();

	// Get an estimate of the noise range (min, max) for interval analysis.
	math::Interval get_estimated_output_range() const;

protected:
	static void _bind_methods();

private:
	NoiseType _resolve_octave_type(int octave_index) const;

	int _seed = 0;
	float _amplitude = 10000.0f;
	float _feature_scale = 100000.0f;
	float _lacunarity = 2.0f;
	float _gain = 0.5f;
	float _voronoi_smoothness = 1.0f;
	float _wavelet_phase = 0.0f;
	float _scratch_smoothness = 0.05f;
	int _num_octaves = 10;
	NoiseType _default_noise_type = NOISE_PERLIN;
	FixedArray<OctaveConfig, MAX_OCTAVES> _octave_configs;
};

} // namespace zylann

VARIANT_ENUM_CAST(zylann::VoxelProceduralNoise::NoiseType);

#endif // VOXEL_PROCEDURAL_NOISE_H
