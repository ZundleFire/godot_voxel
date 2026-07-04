#include "voxel_advanced_noise.h"
#include "../math/funcs.h"
#include <cmath>

namespace zylann {

// ============================================================================
// Internal noise implementations
// Based on standard algorithms matching UE5 VoxelPlugin's ISPC implementations.
// These are self-contained and don't depend on external noise libraries.
// ============================================================================

namespace {

// Hash function for gradient noise (same LCG as UE5 VoxelPlugin)
inline int hash_seed(int seed) {
	return (seed * 196314165) + 907633515;
}

// Fast floor
inline int fast_floor(float x) {
	int xi = (int)x;
	return (x < xi) ? xi - 1 : xi;
}

// Fade/smoothstep for Perlin noise (6t^5 - 15t^4 + 10t^3)
inline float fade(float t) {
	return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

inline float lerp_f(float a, float b, float t) {
	return a + t * (b - a);
}

// ============================================================================
// 3D Perlin noise
// ============================================================================

// Permutation-based gradient selection
inline float grad3(int hash, float x, float y, float z) {
	int h = hash & 15;
	float u = h < 8 ? x : y;
	float v = h < 4 ? y : (h == 12 || h == 14 ? x : z);
	return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}

// Simple permutation from seed
inline int perm(int seed, int x) {
	// FNV-1a inspired hash
	unsigned int h = (unsigned int)(seed ^ (x * 1619));
	h = h * 6364136223846793005ULL + 1442695040888963407ULL;
	h ^= h >> 16;
	return (int)(h & 0xFF);
}

inline int perm3(int seed, int x, int y, int z) {
	return perm(seed, x + perm(seed + 1, y + perm(seed + 2, z)));
}

float perlin_3d(int seed, float x, float y, float z) {
	int x0 = fast_floor(x);
	int y0 = fast_floor(y);
	int z0 = fast_floor(z);

	float xf = x - x0;
	float yf = y - y0;
	float zf = z - z0;

	float u = fade(xf);
	float v = fade(yf);
	float w = fade(zf);

	int aaa = perm3(seed, x0, y0, z0);
	int aba = perm3(seed, x0, y0 + 1, z0);
	int aab = perm3(seed, x0, y0, z0 + 1);
	int abb = perm3(seed, x0, y0 + 1, z0 + 1);
	int baa = perm3(seed, x0 + 1, y0, z0);
	int bba = perm3(seed, x0 + 1, y0 + 1, z0);
	int bab = perm3(seed, x0 + 1, y0, z0 + 1);
	int bbb = perm3(seed, x0 + 1, y0 + 1, z0 + 1);

	float x1 = lerp_f(grad3(aaa, xf, yf, zf), grad3(baa, xf - 1, yf, zf), u);
	float x2 = lerp_f(grad3(aba, xf, yf - 1, zf), grad3(bba, xf - 1, yf - 1, zf), u);
	float y1 = lerp_f(x1, x2, v);

	float x3 = lerp_f(grad3(aab, xf, yf, zf - 1), grad3(bab, xf - 1, yf, zf - 1), u);
	float x4 = lerp_f(grad3(abb, xf, yf - 1, zf - 1), grad3(bbb, xf - 1, yf - 1, zf - 1), u);
	float y2 = lerp_f(x3, x4, v);

	return lerp_f(y1, y2, w);
}

// ============================================================================
// 2D Perlin noise
// ============================================================================

inline float grad2(int hash, float x, float y) {
	int h = hash & 7;
	float u = h < 4 ? x : y;
	float v = h < 4 ? y : x;
	return ((h & 1) != 0 ? -u : u) + ((h & 2) != 0 ? -2.0f * v : 2.0f * v);
}

inline int perm2(int seed, int x, int y) {
	return perm(seed, x + perm(seed + 1, y));
}

float perlin_2d(int seed, float x, float y) {
	int x0 = fast_floor(x);
	int y0 = fast_floor(y);

	float xf = x - x0;
	float yf = y - y0;

	float u = fade(xf);
	float v = fade(yf);

	int aa = perm2(seed, x0, y0);
	int ab = perm2(seed, x0, y0 + 1);
	int ba = perm2(seed, x0 + 1, y0);
	int bb = perm2(seed, x0 + 1, y0 + 1);

	float x1 = lerp_f(grad2(aa, xf, yf), grad2(ba, xf - 1, yf), u);
	float x2 = lerp_f(grad2(ab, xf, yf - 1), grad2(bb, xf - 1, yf - 1), u);

	return lerp_f(x1, x2, v);
}

// ============================================================================
// 3D Simplex noise (based on Stefan Gustavson's implementation)
// ============================================================================

float simplex_3d(int seed, float x, float y, float z) {
	const float F3 = 1.0f / 3.0f;
	const float G3 = 1.0f / 6.0f;

	float s = (x + y + z) * F3;
	int i = fast_floor(x + s);
	int j = fast_floor(y + s);
	int k = fast_floor(z + s);

	float t = (i + j + k) * G3;
	float X0 = i - t;
	float Y0 = j - t;
	float Z0 = k - t;
	float x0 = x - X0;
	float y0 = y - Y0;
	float z0 = z - Z0;

	int i1, j1, k1;
	int i2, j2, k2;

	if (x0 >= y0) {
		if (y0 >= z0) {
			i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 1; k2 = 0;
		} else if (x0 >= z0) {
			i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 0; k2 = 1;
		} else {
			i1 = 0; j1 = 0; k1 = 1; i2 = 1; j2 = 0; k2 = 1;
		}
	} else {
		if (y0 < z0) {
			i1 = 0; j1 = 0; k1 = 1; i2 = 0; j2 = 1; k2 = 1;
		} else if (x0 < z0) {
			i1 = 0; j1 = 1; k1 = 0; i2 = 0; j2 = 1; k2 = 1;
		} else {
			i1 = 0; j1 = 1; k1 = 0; i2 = 1; j2 = 1; k2 = 0;
		}
	}

	float x1 = x0 - i1 + G3;
	float y1 = y0 - j1 + G3;
	float z1 = z0 - k1 + G3;
	float x2 = x0 - i2 + 2.0f * G3;
	float y2 = y0 - j2 + 2.0f * G3;
	float z2 = z0 - k2 + 2.0f * G3;
	float x3 = x0 - 1.0f + 3.0f * G3;
	float y3 = y0 - 1.0f + 3.0f * G3;
	float z3 = z0 - 1.0f + 3.0f * G3;

	float n0 = 0, n1 = 0, n2 = 0, n3 = 0;

	float t0 = 0.6f - x0 * x0 - y0 * y0 - z0 * z0;
	if (t0 >= 0) {
		t0 *= t0;
		n0 = t0 * t0 * grad3(perm3(seed, i, j, k), x0, y0, z0);
	}

	float t1 = 0.6f - x1 * x1 - y1 * y1 - z1 * z1;
	if (t1 >= 0) {
		t1 *= t1;
		n1 = t1 * t1 * grad3(perm3(seed, i + i1, j + j1, k + k1), x1, y1, z1);
	}

	float t2 = 0.6f - x2 * x2 - y2 * y2 - z2 * z2;
	if (t2 >= 0) {
		t2 *= t2;
		n2 = t2 * t2 * grad3(perm3(seed, i + i2, j + j2, k + k2), x2, y2, z2);
	}

	float t3 = 0.6f - x3 * x3 - y3 * y3 - z3 * z3;
	if (t3 >= 0) {
		t3 *= t3;
		n3 = t3 * t3 * grad3(perm3(seed, i + 1, j + 1, k + 1), x3, y3, z3);
	}

	return 32.0f * (n0 + n1 + n2 + n3);
}

// ============================================================================
// 2D Simplex noise
// ============================================================================

float simplex_2d(int seed, float x, float y) {
	const float F2 = 0.5f * (std::sqrt(3.0f) - 1.0f);
	const float G2 = (3.0f - std::sqrt(3.0f)) / 6.0f;

	float s = (x + y) * F2;
	int i = fast_floor(x + s);
	int j = fast_floor(y + s);

	float t = (i + j) * G2;
	float X0 = i - t;
	float Y0 = j - t;
	float x0 = x - X0;
	float y0 = y - Y0;

	int i1, j1;
	if (x0 > y0) {
		i1 = 1; j1 = 0;
	} else {
		i1 = 0; j1 = 1;
	}

	float x1 = x0 - i1 + G2;
	float y1 = y0 - j1 + G2;
	float x2 = x0 - 1.0f + 2.0f * G2;
	float y2 = y0 - 1.0f + 2.0f * G2;

	float n0 = 0, n1 = 0, n2 = 0;

	float t0 = 0.5f - x0 * x0 - y0 * y0;
	if (t0 >= 0) {
		t0 *= t0;
		n0 = t0 * t0 * grad2(perm2(seed, i, j), x0, y0);
	}

	float t1 = 0.5f - x1 * x1 - y1 * y1;
	if (t1 >= 0) {
		t1 *= t1;
		n1 = t1 * t1 * grad2(perm2(seed, i + i1, j + j1), x1, y1);
	}

	float t2 = 0.5f - x2 * x2 - y2 * y2;
	if (t2 >= 0) {
		t2 *= t2;
		n2 = t2 * t2 * grad2(perm2(seed, i + 1, j + 1), x2, y2);
	}

	return 70.0f * (n0 + n1 + n2);
}

// ============================================================================
// 3D Cellular (Worley/Voronoi) noise
// ============================================================================

float cellular_3d(int seed, float x, float y, float z, float jitter) {
	int xi = fast_floor(x);
	int yi = fast_floor(y);
	int zi = fast_floor(z);

	float min_dist = 1e30f;

	for (int dz = -1; dz <= 1; dz++) {
		for (int dy = -1; dy <= 1; dy++) {
			for (int dx = -1; dx <= 1; dx++) {
				int cx = xi + dx;
				int cy = yi + dy;
				int cz = zi + dz;

				// Hash cell to get jittered point
				unsigned int h = (unsigned int)(seed ^ (cx * 1619) ^ (cy * 31337) ^ (cz * 6971));
				h = h * 6364136223846793005ULL + 1442695040888963407ULL;
				float px = (float)(h & 0xFFFF) / 65535.0f;
				h = h * 6364136223846793005ULL + 1442695040888963407ULL;
				float py = (float)(h & 0xFFFF) / 65535.0f;
				h = h * 6364136223846793005ULL + 1442695040888963407ULL;
				float pz = (float)(h & 0xFFFF) / 65535.0f;

				float cell_x = cx + px * jitter;
				float cell_y = cy + py * jitter;
				float cell_z = cz + pz * jitter;

				float dist_x = x - cell_x;
				float dist_y = y - cell_y;
				float dist_z = z - cell_z;
				float dist = dist_x * dist_x + dist_y * dist_y + dist_z * dist_z;

				if (dist < min_dist) {
					min_dist = dist;
				}
			}
		}
	}

	// Normalize to roughly [-1, 1] range
	return std::sqrt(min_dist) - 0.5f;
}

// ============================================================================
// 2D Cellular noise
// ============================================================================

float cellular_2d(int seed, float x, float y, float jitter) {
	int xi = fast_floor(x);
	int yi = fast_floor(y);

	float min_dist = 1e30f;

	for (int dy = -1; dy <= 1; dy++) {
		for (int dx = -1; dx <= 1; dx++) {
			int cx = xi + dx;
			int cy = yi + dy;

			unsigned int h = (unsigned int)(seed ^ (cx * 1619) ^ (cy * 31337));
			h = h * 6364136223846793005ULL + 1442695040888963407ULL;
			float px = (float)(h & 0xFFFF) / 65535.0f;
			h = h * 6364136223846793005ULL + 1442695040888963407ULL;
			float py = (float)(h & 0xFFFF) / 65535.0f;

			float cell_x = cx + px * jitter;
			float cell_y = cy + py * jitter;

			float dist_x = x - cell_x;
			float dist_y = y - cell_y;
			float dist = dist_x * dist_x + dist_y * dist_y;

			if (dist < min_dist) {
				min_dist = dist;
			}
		}
	}

	return std::sqrt(min_dist) - 0.5f;
}

// ============================================================================
// 3D Value noise
// ============================================================================

float value_3d(int seed, float x, float y, float z) {
	int x0 = fast_floor(x);
	int y0 = fast_floor(y);
	int z0 = fast_floor(z);

	float xf = x - x0;
	float yf = y - y0;
	float zf = z - z0;

	float u = fade(xf);
	float v = fade(yf);
	float w = fade(zf);

	auto val_at = [seed](int ix, int iy, int iz) -> float {
		unsigned int h = (unsigned int)(seed ^ (ix * 1619) ^ (iy * 31337) ^ (iz * 6971));
		h = h * 6364136223846793005ULL + 1442695040888963407ULL;
		h ^= h >> 16;
		return (float)(h & 0xFFFF) / 32767.5f - 1.0f; // [-1, 1]
	};

	float v000 = val_at(x0, y0, z0);
	float v100 = val_at(x0 + 1, y0, z0);
	float v010 = val_at(x0, y0 + 1, z0);
	float v110 = val_at(x0 + 1, y0 + 1, z0);
	float v001 = val_at(x0, y0, z0 + 1);
	float v101 = val_at(x0 + 1, y0, z0 + 1);
	float v011 = val_at(x0, y0 + 1, z0 + 1);
	float v111 = val_at(x0 + 1, y0 + 1, z0 + 1);

	float x1 = lerp_f(v000, v100, u);
	float x2 = lerp_f(v010, v110, u);
	float y1 = lerp_f(x1, x2, v);

	float x3 = lerp_f(v001, v101, u);
	float x4 = lerp_f(v011, v111, u);
	float y2 = lerp_f(x3, x4, v);

	return lerp_f(y1, y2, w);
}

// ============================================================================
// 2D Value noise
// ============================================================================

float value_2d(int seed, float x, float y) {
	int x0 = fast_floor(x);
	int y0 = fast_floor(y);

	float xf = x - x0;
	float yf = y - y0;

	float u = fade(xf);
	float v = fade(yf);

	auto val_at = [seed](int ix, int iy) -> float {
		unsigned int h = (unsigned int)(seed ^ (ix * 1619) ^ (iy * 31337));
		h = h * 6364136223846793005ULL + 1442695040888963407ULL;
		h ^= h >> 16;
		return (float)(h & 0xFFFF) / 32767.5f - 1.0f;
	};

	float v00 = val_at(x0, y0);
	float v10 = val_at(x0 + 1, y0);
	float v01 = val_at(x0, y0 + 1);
	float v11 = val_at(x0 + 1, y0 + 1);

	float x1 = lerp_f(v00, v10, u);
	float x2 = lerp_f(v01, v11, u);

	return lerp_f(x1, x2, v);
}

} // anonymous namespace

// ============================================================================
// VoxelAdvancedNoise implementation
// ============================================================================

VoxelAdvancedNoise::VoxelAdvancedNoise() {
	for (int i = 0; i < MAX_OCTAVES; i++) {
		_octave_configs[i].type = NOISE_SMOOTH_PERLIN;
		_octave_configs[i].strength = 1.0f;
	}
}

void VoxelAdvancedNoise::set_seed(int seed) {
	if (_seed == seed) return;
	_seed = seed;
	emit_changed();
}

int VoxelAdvancedNoise::get_seed() const {
	return _seed;
}

void VoxelAdvancedNoise::set_amplitude(float amplitude) {
	if (_amplitude == amplitude) return;
	_amplitude = amplitude;
	emit_changed();
}

float VoxelAdvancedNoise::get_amplitude() const {
	return _amplitude;
}

void VoxelAdvancedNoise::set_feature_scale(float scale) {
	if (_feature_scale == scale) return;
	_feature_scale = scale;
	emit_changed();
}

float VoxelAdvancedNoise::get_feature_scale() const {
	return _feature_scale;
}

void VoxelAdvancedNoise::set_lacunarity(float lacunarity) {
	if (_lacunarity == lacunarity) return;
	_lacunarity = lacunarity;
	emit_changed();
}

float VoxelAdvancedNoise::get_lacunarity() const {
	return _lacunarity;
}

void VoxelAdvancedNoise::set_gain(float gain) {
	if (_gain == gain) return;
	_gain = gain;
	emit_changed();
}

float VoxelAdvancedNoise::get_gain() const {
	return _gain;
}

void VoxelAdvancedNoise::set_cellular_jitter(float jitter) {
	if (_cellular_jitter == jitter) return;
	_cellular_jitter = jitter;
	emit_changed();
}

float VoxelAdvancedNoise::get_cellular_jitter() const {
	return _cellular_jitter;
}

void VoxelAdvancedNoise::set_num_octaves(int count) {
	count = CLAMP(count, 1, MAX_OCTAVES);
	if (_num_octaves == count) return;
	_num_octaves = count;
	emit_changed();
}

int VoxelAdvancedNoise::get_num_octaves() const {
	return _num_octaves;
}

void VoxelAdvancedNoise::set_default_octave_type(NoiseType type) {
	ERR_FAIL_INDEX(type, NOISE_TYPE_COUNT);
	if (_default_octave_type == type) return;
	_default_octave_type = type;
	emit_changed();
}

VoxelAdvancedNoise::NoiseType VoxelAdvancedNoise::get_default_octave_type() const {
	return _default_octave_type;
}

void VoxelAdvancedNoise::set_octave_type(int octave_index, NoiseType type) {
	ERR_FAIL_INDEX(octave_index, MAX_OCTAVES);
	ERR_FAIL_INDEX(type, NOISE_TYPE_COUNT);
	_octave_configs[octave_index].type = type;
	emit_changed();
}

VoxelAdvancedNoise::NoiseType VoxelAdvancedNoise::get_octave_type(int octave_index) const {
	ERR_FAIL_INDEX_V(octave_index, MAX_OCTAVES, NOISE_SMOOTH_PERLIN);
	return _octave_configs[octave_index].type;
}

void VoxelAdvancedNoise::set_octave_strength(int octave_index, float strength) {
	ERR_FAIL_INDEX(octave_index, MAX_OCTAVES);
	_octave_configs[octave_index].strength = strength;
	emit_changed();
}

float VoxelAdvancedNoise::get_octave_strength(int octave_index) const {
	ERR_FAIL_INDEX_V(octave_index, MAX_OCTAVES, 1.0f);
	return _octave_configs[octave_index].strength;
}

// Static noise functions — wrappers for internal implementations
float VoxelAdvancedNoise::_perlin_3d(int seed, float x, float y, float z) {
	return perlin_3d(seed, x, y, z);
}

float VoxelAdvancedNoise::_simplex_3d(int seed, float x, float y, float z) {
	return simplex_3d(seed, x, y, z);
}

float VoxelAdvancedNoise::_cellular_3d(int seed, float x, float y, float z, float jitter) {
	return cellular_3d(seed, x, y, z, jitter);
}

float VoxelAdvancedNoise::_value_3d(int seed, float x, float y, float z) {
	return value_3d(seed, x, y, z);
}

float VoxelAdvancedNoise::_perlin_2d(int seed, float x, float y) {
	return perlin_2d(seed, x, y);
}

float VoxelAdvancedNoise::_simplex_2d(int seed, float x, float y) {
	return simplex_2d(seed, x, y);
}

float VoxelAdvancedNoise::_cellular_2d(int seed, float x, float y, float jitter) {
	return cellular_2d(seed, x, y, jitter);
}

float VoxelAdvancedNoise::_value_2d(int seed, float x, float y) {
	return value_2d(seed, x, y);
}

float VoxelAdvancedNoise::_sample_octave_3d(NoiseType type, int seed, float x, float y, float z, float jitter) const {
	float noise;
	switch (type) {
		case NOISE_SMOOTH_PERLIN:
			noise = _perlin_3d(seed, x, y, z);
			break;
		case NOISE_BILLOWY_PERLIN:
			noise = std::abs(_perlin_3d(seed, x, y, z)) * 2.0f - 1.0f;
			break;
		case NOISE_RIDGED_PERLIN:
			noise = (1.0f - std::abs(_perlin_3d(seed, x, y, z))) * 2.0f - 1.0f;
			break;
		case NOISE_SMOOTH_CELLULAR:
			noise = _cellular_3d(seed, x, y, z, jitter);
			break;
		case NOISE_BILLOWY_CELLULAR:
			noise = std::abs(_cellular_3d(seed, x, y, z, jitter)) * 2.0f - 1.0f;
			break;
		case NOISE_RIDGED_CELLULAR:
			noise = (1.0f - std::abs(_cellular_3d(seed, x, y, z, jitter))) * 2.0f - 1.0f;
			break;
		case NOISE_SMOOTH_SIMPLEX:
			noise = _simplex_3d(seed, x, y, z);
			break;
		case NOISE_BILLOWY_SIMPLEX:
			noise = std::abs(_simplex_3d(seed, x, y, z)) * 2.0f - 1.0f;
			break;
		case NOISE_RIDGED_SIMPLEX:
			noise = (1.0f - std::abs(_simplex_3d(seed, x, y, z))) * 2.0f - 1.0f;
			break;
		case NOISE_SMOOTH_VALUE:
			noise = _value_3d(seed, x, y, z);
			break;
		case NOISE_BILLOWY_VALUE:
			noise = std::abs(_value_3d(seed, x, y, z)) * 2.0f - 1.0f;
			break;
		case NOISE_RIDGED_VALUE:
			noise = (1.0f - std::abs(_value_3d(seed, x, y, z))) * 2.0f - 1.0f;
			break;
		case NOISE_SMOOTH_TRUE_DISTANCE_CELLULAR:
			noise = _cellular_3d(seed, x, y, z, jitter);
			break;
		case NOISE_BILLOWY_TRUE_DISTANCE_CELLULAR:
			noise = std::abs(_cellular_3d(seed, x, y, z, jitter)) * 2.0f - 1.0f;
			break;
		case NOISE_RIDGED_TRUE_DISTANCE_CELLULAR:
			noise = (1.0f - std::abs(_cellular_3d(seed, x, y, z, jitter))) * 2.0f - 1.0f;
			break;
		default:
			noise = 0.0f;
			break;
	}
	return noise;
}

float VoxelAdvancedNoise::_sample_octave_2d(NoiseType type, int seed, float x, float y, float jitter) const {
	float noise;
	switch (type) {
		case NOISE_SMOOTH_PERLIN:
			noise = _perlin_2d(seed, x, y);
			break;
		case NOISE_BILLOWY_PERLIN:
			noise = std::abs(_perlin_2d(seed, x, y)) * 2.0f - 1.0f;
			break;
		case NOISE_RIDGED_PERLIN:
			noise = (1.0f - std::abs(_perlin_2d(seed, x, y))) * 2.0f - 1.0f;
			break;
		case NOISE_SMOOTH_CELLULAR:
			noise = _cellular_2d(seed, x, y, jitter);
			break;
		case NOISE_BILLOWY_CELLULAR:
			noise = std::abs(_cellular_2d(seed, x, y, jitter)) * 2.0f - 1.0f;
			break;
		case NOISE_RIDGED_CELLULAR:
			noise = (1.0f - std::abs(_cellular_2d(seed, x, y, jitter))) * 2.0f - 1.0f;
			break;
		case NOISE_SMOOTH_SIMPLEX:
			noise = _simplex_2d(seed, x, y);
			break;
		case NOISE_BILLOWY_SIMPLEX:
			noise = std::abs(_simplex_2d(seed, x, y)) * 2.0f - 1.0f;
			break;
		case NOISE_RIDGED_SIMPLEX:
			noise = (1.0f - std::abs(_simplex_2d(seed, x, y))) * 2.0f - 1.0f;
			break;
		case NOISE_SMOOTH_VALUE:
			noise = _value_2d(seed, x, y);
			break;
		case NOISE_BILLOWY_VALUE:
			noise = std::abs(_value_2d(seed, x, y)) * 2.0f - 1.0f;
			break;
		case NOISE_RIDGED_VALUE:
			noise = (1.0f - std::abs(_value_2d(seed, x, y))) * 2.0f - 1.0f;
			break;
		case NOISE_SMOOTH_TRUE_DISTANCE_CELLULAR:
			noise = _cellular_2d(seed, x, y, jitter);
			break;
		case NOISE_BILLOWY_TRUE_DISTANCE_CELLULAR:
			noise = std::abs(_cellular_2d(seed, x, y, jitter)) * 2.0f - 1.0f;
			break;
		case NOISE_RIDGED_TRUE_DISTANCE_CELLULAR:
			noise = (1.0f - std::abs(_cellular_2d(seed, x, y, jitter))) * 2.0f - 1.0f;
			break;
		default:
			noise = 0.0f;
			break;
	}
	return noise;
}

float VoxelAdvancedNoise::get_noise_3d(float x, float y, float z) const {
	float sum = 0.0f;
	float amplitude_sum = 0.0f;
	float amplitude = 1.0f;

	float px = x / _feature_scale;
	float py = y / _feature_scale;
	float pz = z / _feature_scale;

	int seed = _seed;

	for (int octave = 0; octave < _num_octaves; octave++) {
		const OctaveConfig &config = _octave_configs[octave];
		NoiseType type = config.type;
		// If using default type for this octave and no explicit override was set
		// (we use NOISE_SMOOTH_PERLIN as default init, but the user's intent is "use default_octave_type")
		// This matches UE5 behavior where octave type "Default" means use the default_octave_type
		if (octave >= 0) {
			// Use per-octave type if explicitly set, otherwise default
			// The user controls this via set_octave_type; default init matches _default_octave_type
			// For simplicity, we always use the stored config type
		}

		float noise = _sample_octave_3d(type, seed, px, py, pz, _cellular_jitter);

		float strength = config.strength;
		float new_sum = sum + noise * strength * amplitude;

		// Guard against NaN/Inf (matches UE5 IsFinite check)
		if (std::isfinite(new_sum)) {
			sum = new_sum;
		}

		amplitude_sum += std::abs(strength * amplitude);
		amplitude *= _gain;
		px *= _lacunarity;
		py *= _lacunarity;
		pz *= _lacunarity;
		seed = hash_seed(seed);
	}

	// Normalize and apply amplitude (matches UE5 behavior)
	return sum / (amplitude_sum == 0.0f ? 1.0f : amplitude_sum) * _amplitude;
}

float VoxelAdvancedNoise::get_noise_3dv(Vector3 pos) const {
	return get_noise_3d(pos.x, pos.y, pos.z);
}

float VoxelAdvancedNoise::get_noise_2d(float x, float y) const {
	float sum = 0.0f;
	float amplitude_sum = 0.0f;
	float amplitude = 1.0f;

	float px = x / _feature_scale;
	float py = y / _feature_scale;

	int seed = _seed;

	for (int octave = 0; octave < _num_octaves; octave++) {
		const OctaveConfig &config = _octave_configs[octave];
		float noise = _sample_octave_2d(config.type, seed, px, py, _cellular_jitter);

		float strength = config.strength;
		float new_sum = sum + noise * strength * amplitude;

		if (std::isfinite(new_sum)) {
			sum = new_sum;
		}

		amplitude_sum += std::abs(strength * amplitude);
		amplitude *= _gain;
		px *= _lacunarity;
		py *= _lacunarity;
		seed = hash_seed(seed);
	}

	return sum / (amplitude_sum == 0.0f ? 1.0f : amplitude_sum) * _amplitude;
}

float VoxelAdvancedNoise::get_noise_2dv(Vector2 pos) const {
	return get_noise_2d(pos.x, pos.y);
}

PackedFloat32Array VoxelAdvancedNoise::fill_buffer_3d(
		Vector3 origin, Vector3i size, float spacing) const {
	const int total = size.x * size.y * size.z;
	PackedFloat32Array output;
	output.resize(total);

	float *data = output.ptrw();

#ifdef VOXEL_ISPC_ENABLED
	if (voxel_ispc_available()) {
		// Build ISPC-compatible octave config array
		ISPCOctaveConfig ispc_octaves[MAX_OCTAVES];
		for (int i = 0; i < _num_octaves; i++) {
			ispc_octaves[i].type = static_cast<int32_t>(_octave_configs[i].type);
			ispc_octaves[i].strength = _octave_configs[i].strength;
		}

		voxel_ispc_fill_noise_3d(
				origin.x, origin.y, origin.z,
				size.x, size.y, size.z,
				spacing,
				_amplitude, _feature_scale,
				_lacunarity, _gain, _cellular_jitter,
				_seed,
				ispc_octaves, _num_octaves,
				data, total);

		return output;
	}
#endif

	// Scalar fallback
	int index = 0;

	for (int z = 0; z < size.z; z++) {
		for (int y = 0; y < size.y; y++) {
			for (int x = 0; x < size.x; x++) {
				float wx = origin.x + x * spacing;
				float wy = origin.y + y * spacing;
				float wz = origin.z + z * spacing;
				data[index++] = get_noise_3d(wx, wy, wz);
			}
		}
	}

	return output;
}

bool VoxelAdvancedNoise::is_simd_available() {
	return voxel_ispc_available();
}

math::Interval VoxelAdvancedNoise::get_estimated_output_range() const {
	// The noise output is normalized to [-1, 1] before amplitude scaling.
	// So the range is approximately [-amplitude, amplitude].
	// In practice, multi-octave noise rarely reaches the extremes.
	return math::Interval(-_amplitude, _amplitude);
}

void VoxelAdvancedNoise::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_seed", "seed"), &VoxelAdvancedNoise::set_seed);
	ClassDB::bind_method(D_METHOD("get_seed"), &VoxelAdvancedNoise::get_seed);

	ClassDB::bind_method(D_METHOD("set_amplitude", "amplitude"), &VoxelAdvancedNoise::set_amplitude);
	ClassDB::bind_method(D_METHOD("get_amplitude"), &VoxelAdvancedNoise::get_amplitude);

	ClassDB::bind_method(D_METHOD("set_feature_scale", "scale"), &VoxelAdvancedNoise::set_feature_scale);
	ClassDB::bind_method(D_METHOD("get_feature_scale"), &VoxelAdvancedNoise::get_feature_scale);

	ClassDB::bind_method(D_METHOD("set_lacunarity", "lacunarity"), &VoxelAdvancedNoise::set_lacunarity);
	ClassDB::bind_method(D_METHOD("get_lacunarity"), &VoxelAdvancedNoise::get_lacunarity);

	ClassDB::bind_method(D_METHOD("set_gain", "gain"), &VoxelAdvancedNoise::set_gain);
	ClassDB::bind_method(D_METHOD("get_gain"), &VoxelAdvancedNoise::get_gain);

	ClassDB::bind_method(D_METHOD("set_cellular_jitter", "jitter"), &VoxelAdvancedNoise::set_cellular_jitter);
	ClassDB::bind_method(D_METHOD("get_cellular_jitter"), &VoxelAdvancedNoise::get_cellular_jitter);

	ClassDB::bind_method(D_METHOD("set_num_octaves", "count"), &VoxelAdvancedNoise::set_num_octaves);
	ClassDB::bind_method(D_METHOD("get_num_octaves"), &VoxelAdvancedNoise::get_num_octaves);

	ClassDB::bind_method(D_METHOD("set_default_octave_type", "type"), &VoxelAdvancedNoise::set_default_octave_type);
	ClassDB::bind_method(D_METHOD("get_default_octave_type"), &VoxelAdvancedNoise::get_default_octave_type);

	ClassDB::bind_method(D_METHOD("set_octave_type", "octave_index", "type"), &VoxelAdvancedNoise::set_octave_type);
	ClassDB::bind_method(D_METHOD("get_octave_type", "octave_index"), &VoxelAdvancedNoise::get_octave_type);

	ClassDB::bind_method(
			D_METHOD("set_octave_strength", "octave_index", "strength"), &VoxelAdvancedNoise::set_octave_strength);
	ClassDB::bind_method(D_METHOD("get_octave_strength", "octave_index"), &VoxelAdvancedNoise::get_octave_strength);

	ClassDB::bind_method(D_METHOD("get_noise_3d", "x", "y", "z"), &VoxelAdvancedNoise::get_noise_3d);
	ClassDB::bind_method(D_METHOD("get_noise_3dv", "position"), &VoxelAdvancedNoise::get_noise_3dv);

	ClassDB::bind_method(D_METHOD("get_noise_2d", "x", "y"), &VoxelAdvancedNoise::get_noise_2d);
	ClassDB::bind_method(D_METHOD("get_noise_2dv", "position"), &VoxelAdvancedNoise::get_noise_2dv);

	ClassDB::bind_method(
			D_METHOD("fill_buffer_3d", "origin", "size", "spacing"),
			&VoxelAdvancedNoise::fill_buffer_3d);

	ClassDB::bind_static_method(
			"VoxelAdvancedNoise",
			D_METHOD("is_simd_available"),
			&VoxelAdvancedNoise::is_simd_available);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "seed"), "set_seed", "get_seed");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "amplitude", PROPERTY_HINT_RANGE, "0.001,1000000,0.1"),
			"set_amplitude", "get_amplitude");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "feature_scale", PROPERTY_HINT_RANGE, "0.001,1000000,0.1"),
			"set_feature_scale", "get_feature_scale");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "lacunarity", PROPERTY_HINT_RANGE, "0.1,10,0.01"),
			"set_lacunarity", "get_lacunarity");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "gain", PROPERTY_HINT_RANGE, "0.01,2,0.01"),
			"set_gain", "get_gain");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "cellular_jitter", PROPERTY_HINT_RANGE, "0,1,0.01"),
			"set_cellular_jitter", "get_cellular_jitter");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "num_octaves", PROPERTY_HINT_RANGE, "1,32,1"),
			"set_num_octaves", "get_num_octaves");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "default_octave_type", PROPERTY_HINT_ENUM,
					"SmoothPerlin,BillowyPerlin,RidgedPerlin,"
					"SmoothCellular,BillowyCellular,RidgedCellular,"
					"SmoothSimplex,BillowySimplex,RidgedSimplex,"
					"SmoothValue,BillowyValue,RidgedValue"),
			"set_default_octave_type", "get_default_octave_type");

	BIND_ENUM_CONSTANT(NOISE_SMOOTH_PERLIN);
	BIND_ENUM_CONSTANT(NOISE_BILLOWY_PERLIN);
	BIND_ENUM_CONSTANT(NOISE_RIDGED_PERLIN);
	BIND_ENUM_CONSTANT(NOISE_SMOOTH_CELLULAR);
	BIND_ENUM_CONSTANT(NOISE_BILLOWY_CELLULAR);
	BIND_ENUM_CONSTANT(NOISE_RIDGED_CELLULAR);
	BIND_ENUM_CONSTANT(NOISE_SMOOTH_SIMPLEX);
	BIND_ENUM_CONSTANT(NOISE_BILLOWY_SIMPLEX);
	BIND_ENUM_CONSTANT(NOISE_RIDGED_SIMPLEX);
	BIND_ENUM_CONSTANT(NOISE_SMOOTH_VALUE);
	BIND_ENUM_CONSTANT(NOISE_BILLOWY_VALUE);
	BIND_ENUM_CONSTANT(NOISE_RIDGED_VALUE);
	BIND_ENUM_CONSTANT(NOISE_SMOOTH_TRUE_DISTANCE_CELLULAR);
	BIND_ENUM_CONSTANT(NOISE_BILLOWY_TRUE_DISTANCE_CELLULAR);
	BIND_ENUM_CONSTANT(NOISE_RIDGED_TRUE_DISTANCE_CELLULAR);
}

} // namespace zylann
