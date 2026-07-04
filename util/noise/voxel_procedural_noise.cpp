#include "voxel_procedural_noise.h"
#include "../containers/std_vector.h"
#include "voxel_procedural_noise_impl.h"
#include "voxel_procedural_noise_simd.h"

#include <cmath>

namespace zylann {

namespace pn = procedural_noise;

void VoxelProceduralNoise::set_seed(int seed) {
	if (_seed == seed) {
		return;
	}
	_seed = seed;
	emit_changed();
}

int VoxelProceduralNoise::get_seed() const {
	return _seed;
}

void VoxelProceduralNoise::set_amplitude(float amplitude) {
	if (_amplitude == amplitude) {
		return;
	}
	_amplitude = amplitude;
	emit_changed();
}

float VoxelProceduralNoise::get_amplitude() const {
	return _amplitude;
}

void VoxelProceduralNoise::set_feature_scale(float scale) {
	scale = MAX(scale, 0.0001f);
	if (_feature_scale == scale) {
		return;
	}
	_feature_scale = scale;
	emit_changed();
}

float VoxelProceduralNoise::get_feature_scale() const {
	return _feature_scale;
}

void VoxelProceduralNoise::set_lacunarity(float lacunarity) {
	if (_lacunarity == lacunarity) {
		return;
	}
	_lacunarity = lacunarity;
	emit_changed();
}

float VoxelProceduralNoise::get_lacunarity() const {
	return _lacunarity;
}

void VoxelProceduralNoise::set_gain(float gain) {
	if (_gain == gain) {
		return;
	}
	_gain = gain;
	emit_changed();
}

float VoxelProceduralNoise::get_gain() const {
	return _gain;
}

void VoxelProceduralNoise::set_voronoi_smoothness(float smoothness) {
	if (_voronoi_smoothness == smoothness) {
		return;
	}
	_voronoi_smoothness = smoothness;
	emit_changed();
}

float VoxelProceduralNoise::get_voronoi_smoothness() const {
	return _voronoi_smoothness;
}

void VoxelProceduralNoise::set_wavelet_phase(float phase) {
	if (_wavelet_phase == phase) {
		return;
	}
	_wavelet_phase = phase;
	emit_changed();
}

float VoxelProceduralNoise::get_wavelet_phase() const {
	return _wavelet_phase;
}

void VoxelProceduralNoise::set_scratch_smoothness(float smoothness) {
	if (_scratch_smoothness == smoothness) {
		return;
	}
	_scratch_smoothness = smoothness;
	emit_changed();
}

float VoxelProceduralNoise::get_scratch_smoothness() const {
	return _scratch_smoothness;
}

void VoxelProceduralNoise::set_num_octaves(int count) {
	count = CLAMP(count, 1, MAX_OCTAVES);
	if (_num_octaves == count) {
		return;
	}
	_num_octaves = count;
	emit_changed();
}

int VoxelProceduralNoise::get_num_octaves() const {
	return _num_octaves;
}

void VoxelProceduralNoise::set_default_noise_type(NoiseType type) {
	ERR_FAIL_COND_MSG(type < 0 || type >= NOISE_TYPE_COUNT, "Invalid noise type");
	if (_default_noise_type == type) {
		return;
	}
	_default_noise_type = type;
	emit_changed();
}

VoxelProceduralNoise::NoiseType VoxelProceduralNoise::get_default_noise_type() const {
	return _default_noise_type;
}

void VoxelProceduralNoise::set_octave_type(int octave_index, NoiseType type) {
	ERR_FAIL_INDEX(octave_index, MAX_OCTAVES);
	ERR_FAIL_COND_MSG(
			(type < 0 || type >= NOISE_TYPE_COUNT) && type != NOISE_DEFAULT, "Invalid noise type");
	_octave_configs[octave_index].type = type;
	emit_changed();
}

VoxelProceduralNoise::NoiseType VoxelProceduralNoise::get_octave_type(int octave_index) const {
	ERR_FAIL_INDEX_V(octave_index, MAX_OCTAVES, NOISE_DEFAULT);
	return _octave_configs[octave_index].type;
}

void VoxelProceduralNoise::set_octave_strength(int octave_index, float strength) {
	ERR_FAIL_INDEX(octave_index, MAX_OCTAVES);
	_octave_configs[octave_index].strength = strength;
	emit_changed();
}

float VoxelProceduralNoise::get_octave_strength(int octave_index) const {
	ERR_FAIL_INDEX_V(octave_index, MAX_OCTAVES, 1.0f);
	return _octave_configs[octave_index].strength;
}

VoxelProceduralNoise::NoiseType VoxelProceduralNoise::_resolve_octave_type(int octave_index) const {
	const NoiseType type = _octave_configs[octave_index].type;
	return type == NOISE_DEFAULT ? _default_noise_type : type;
}

float VoxelProceduralNoise::get_noise_2d(float x, float y) const {
	float sum = 0.0f;
	float amplitude_sum = 0.0f;
	float amplitude = 1.0f;

	pn::float2 pos = pn::MakeFloat2(x / _feature_scale, y / _feature_scale);
	int32_t seed = _seed;

	for (int octave = 0; octave < _num_octaves; octave++) {
		const float strength = _octave_configs[octave].strength;
		const float noise = pn::FCSampleNoise2D(
				_resolve_octave_type(octave),
				(uint32_t)seed,
				pos,
				_voronoi_smoothness,
				_wavelet_phase,
				_scratch_smoothness);

		const float new_sum = sum + noise * strength * amplitude;
		if (std::isfinite(new_sum)) {
			sum = new_sum;
		}

		amplitude_sum += std::abs(strength * amplitude);
		amplitude *= _gain;
		pos = pos * _lacunarity;
		seed = pn::FCNextOctaveSeed(seed);
	}

	return sum / (amplitude_sum == 0.0f ? 1.0f : amplitude_sum) * _amplitude;
}

float VoxelProceduralNoise::get_noise_2dv(Vector2 pos) const {
	return get_noise_2d(pos.x, pos.y);
}

float VoxelProceduralNoise::get_noise_3d(float x, float y, float z) const {
	float sum = 0.0f;
	float amplitude_sum = 0.0f;
	float amplitude = 1.0f;

	pn::float3 pos = pn::MakeFloat3(x / _feature_scale, y / _feature_scale, z / _feature_scale);
	int32_t seed = _seed;

	for (int octave = 0; octave < _num_octaves; octave++) {
		const float strength = _octave_configs[octave].strength;
		const float noise = pn::FCSampleNoise3D(
				_resolve_octave_type(octave),
				(uint32_t)seed,
				pos,
				_voronoi_smoothness,
				_wavelet_phase,
				_scratch_smoothness);

		const float new_sum = sum + noise * strength * amplitude;
		if (std::isfinite(new_sum)) {
			sum = new_sum;
		}

		amplitude_sum += std::abs(strength * amplitude);
		amplitude *= _gain;
		pos = pos * _lacunarity;
		seed = pn::FCNextOctaveSeed(seed);
	}

	return sum / (amplitude_sum == 0.0f ? 1.0f : amplitude_sum) * _amplitude;
}

float VoxelProceduralNoise::get_noise_3dv(Vector3 pos) const {
	return get_noise_3d(pos.x, pos.y, pos.z);
}

void VoxelProceduralNoise::get_noise_2d_series(
		const float *x, const float *y, float *out, unsigned int count) const {
#ifdef VOXEL_ISPC_ENABLED
	ProceduralOctaveConfig octaves[MAX_OCTAVES];
	for (int i = 0; i < _num_octaves; i++) {
		octaves[i].type = static_cast<int32_t>(_resolve_octave_type(i));
		octaves[i].strength = _octave_configs[i].strength;
	}
	voxel_ispc_procedural_noise_2d_batch(
			x, y,
			_amplitude, _feature_scale,
			_lacunarity, _gain,
			_voronoi_smoothness, _wavelet_phase, _scratch_smoothness,
			_seed,
			octaves, _num_octaves,
			out, count);
#else
	for (unsigned int i = 0; i < count; ++i) {
		out[i] = get_noise_2d(x[i], y[i]);
	}
#endif
}

void VoxelProceduralNoise::get_noise_3d_series(
		const float *x, const float *y, const float *z, float *out, unsigned int count) const {
#ifdef VOXEL_ISPC_ENABLED
	ProceduralOctaveConfig octaves[MAX_OCTAVES];
	for (int i = 0; i < _num_octaves; i++) {
		octaves[i].type = static_cast<int32_t>(_resolve_octave_type(i));
		octaves[i].strength = _octave_configs[i].strength;
	}
	voxel_ispc_procedural_noise_3d_batch(
			x, y, z,
			_amplitude, _feature_scale,
			_lacunarity, _gain,
			_voronoi_smoothness, _wavelet_phase, _scratch_smoothness,
			_seed,
			octaves, _num_octaves,
			out, count);
#else
	for (unsigned int i = 0; i < count; ++i) {
		out[i] = get_noise_3d(x[i], y[i], z[i]);
	}
#endif
}

PackedFloat32Array VoxelProceduralNoise::fill_buffer_3d(Vector3 origin, Vector3i size, float spacing) const {
	PackedFloat32Array output;
	ERR_FAIL_COND_V(size.x <= 0 || size.y <= 0 || size.z <= 0, output);
	const int total = size.x * size.y * size.z;
	output.resize(total);

	// Build SoA positions then run the batch path (ISPC when available).
	StdVector<float> xs;
	StdVector<float> ys;
	StdVector<float> zs;
	xs.resize(total);
	ys.resize(total);
	zs.resize(total);

	int index = 0;
	for (int z = 0; z < size.z; z++) {
		for (int y = 0; y < size.y; y++) {
			for (int x = 0; x < size.x; x++) {
				xs[index] = origin.x + x * spacing;
				ys[index] = origin.y + y * spacing;
				zs[index] = origin.z + z * spacing;
				++index;
			}
		}
	}

	get_noise_3d_series(xs.data(), ys.data(), zs.data(), output.ptrw(), total);
	return output;
}

bool VoxelProceduralNoise::is_simd_available() {
#ifdef VOXEL_ISPC_ENABLED
	return true;
#else
	return false;
#endif
}

math::Interval VoxelProceduralNoise::get_estimated_output_range() const {
	// The octave sum is normalized to roughly [-1, 1] before amplitude scaling.
	return math::Interval(-_amplitude, _amplitude);
}

void VoxelProceduralNoise::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_seed", "seed"), &VoxelProceduralNoise::set_seed);
	ClassDB::bind_method(D_METHOD("get_seed"), &VoxelProceduralNoise::get_seed);

	ClassDB::bind_method(D_METHOD("set_amplitude", "amplitude"), &VoxelProceduralNoise::set_amplitude);
	ClassDB::bind_method(D_METHOD("get_amplitude"), &VoxelProceduralNoise::get_amplitude);

	ClassDB::bind_method(D_METHOD("set_feature_scale", "scale"), &VoxelProceduralNoise::set_feature_scale);
	ClassDB::bind_method(D_METHOD("get_feature_scale"), &VoxelProceduralNoise::get_feature_scale);

	ClassDB::bind_method(D_METHOD("set_lacunarity", "lacunarity"), &VoxelProceduralNoise::set_lacunarity);
	ClassDB::bind_method(D_METHOD("get_lacunarity"), &VoxelProceduralNoise::get_lacunarity);

	ClassDB::bind_method(D_METHOD("set_gain", "gain"), &VoxelProceduralNoise::set_gain);
	ClassDB::bind_method(D_METHOD("get_gain"), &VoxelProceduralNoise::get_gain);

	ClassDB::bind_method(
			D_METHOD("set_voronoi_smoothness", "smoothness"), &VoxelProceduralNoise::set_voronoi_smoothness);
	ClassDB::bind_method(D_METHOD("get_voronoi_smoothness"), &VoxelProceduralNoise::get_voronoi_smoothness);

	ClassDB::bind_method(D_METHOD("set_wavelet_phase", "phase"), &VoxelProceduralNoise::set_wavelet_phase);
	ClassDB::bind_method(D_METHOD("get_wavelet_phase"), &VoxelProceduralNoise::get_wavelet_phase);

	ClassDB::bind_method(
			D_METHOD("set_scratch_smoothness", "smoothness"), &VoxelProceduralNoise::set_scratch_smoothness);
	ClassDB::bind_method(D_METHOD("get_scratch_smoothness"), &VoxelProceduralNoise::get_scratch_smoothness);

	ClassDB::bind_method(D_METHOD("set_num_octaves", "count"), &VoxelProceduralNoise::set_num_octaves);
	ClassDB::bind_method(D_METHOD("get_num_octaves"), &VoxelProceduralNoise::get_num_octaves);

	ClassDB::bind_method(
			D_METHOD("set_default_noise_type", "type"), &VoxelProceduralNoise::set_default_noise_type);
	ClassDB::bind_method(D_METHOD("get_default_noise_type"), &VoxelProceduralNoise::get_default_noise_type);

	ClassDB::bind_method(D_METHOD("set_octave_type", "octave_index", "type"), &VoxelProceduralNoise::set_octave_type);
	ClassDB::bind_method(D_METHOD("get_octave_type", "octave_index"), &VoxelProceduralNoise::get_octave_type);

	ClassDB::bind_method(
			D_METHOD("set_octave_strength", "octave_index", "strength"), &VoxelProceduralNoise::set_octave_strength);
	ClassDB::bind_method(D_METHOD("get_octave_strength", "octave_index"), &VoxelProceduralNoise::get_octave_strength);

	ClassDB::bind_method(D_METHOD("get_noise_2d", "x", "y"), &VoxelProceduralNoise::get_noise_2d);
	ClassDB::bind_method(D_METHOD("get_noise_2dv", "pos"), &VoxelProceduralNoise::get_noise_2dv);
	ClassDB::bind_method(D_METHOD("get_noise_3d", "x", "y", "z"), &VoxelProceduralNoise::get_noise_3d);
	ClassDB::bind_method(D_METHOD("get_noise_3dv", "pos"), &VoxelProceduralNoise::get_noise_3dv);

	ClassDB::bind_method(
			D_METHOD("fill_buffer_3d", "origin", "size", "spacing"), &VoxelProceduralNoise::fill_buffer_3d);

	ClassDB::bind_static_method(
			"VoxelProceduralNoise", D_METHOD("is_simd_available"), &VoxelProceduralNoise::is_simd_available);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "seed"), "set_seed", "get_seed");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "amplitude"), "set_amplitude", "get_amplitude");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "feature_scale"), "set_feature_scale", "get_feature_scale");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "lacunarity"), "set_lacunarity", "get_lacunarity");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "gain"), "set_gain", "get_gain");
	ADD_PROPERTY(
			PropertyInfo(Variant::FLOAT, "voronoi_smoothness"), "set_voronoi_smoothness", "get_voronoi_smoothness");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "wavelet_phase"), "set_wavelet_phase", "get_wavelet_phase");
	ADD_PROPERTY(
			PropertyInfo(Variant::FLOAT, "scratch_smoothness"), "set_scratch_smoothness", "get_scratch_smoothness");
	ADD_PROPERTY(
			PropertyInfo(Variant::INT, "num_octaves", PROPERTY_HINT_RANGE, "1,32,1"),
			"set_num_octaves",
			"get_num_octaves");
	ADD_PROPERTY(
			PropertyInfo(
					Variant::INT,
					"default_noise_type",
					PROPERTY_HINT_ENUM,
					"Perlin,Simplex,Value,Worley,Voronoi,Blue,HilbertBlue,Crater,Gabor,Curl,Scratch,Wavelet,"
					"Erosion,Paper,Stone,Wool,InterleavedGradient"),
			"set_default_noise_type",
			"get_default_noise_type");

	BIND_ENUM_CONSTANT(NOISE_PERLIN);
	BIND_ENUM_CONSTANT(NOISE_SIMPLEX);
	BIND_ENUM_CONSTANT(NOISE_VALUE);
	BIND_ENUM_CONSTANT(NOISE_WORLEY);
	BIND_ENUM_CONSTANT(NOISE_VORONOI);
	BIND_ENUM_CONSTANT(NOISE_BLUE);
	BIND_ENUM_CONSTANT(NOISE_HILBERT_BLUE);
	BIND_ENUM_CONSTANT(NOISE_CRATER);
	BIND_ENUM_CONSTANT(NOISE_GABOR);
	BIND_ENUM_CONSTANT(NOISE_CURL);
	BIND_ENUM_CONSTANT(NOISE_SCRATCH);
	BIND_ENUM_CONSTANT(NOISE_WAVELET);
	BIND_ENUM_CONSTANT(NOISE_EROSION);
	BIND_ENUM_CONSTANT(NOISE_PAPER);
	BIND_ENUM_CONSTANT(NOISE_STONE);
	BIND_ENUM_CONSTANT(NOISE_WOOL);
	BIND_ENUM_CONSTANT(NOISE_INTERLEAVED_GRADIENT);
	BIND_ENUM_CONSTANT(NOISE_TYPE_COUNT);
	BIND_ENUM_CONSTANT(NOISE_DEFAULT);
}

} // namespace zylann
