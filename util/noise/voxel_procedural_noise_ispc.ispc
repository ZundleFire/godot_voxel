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
