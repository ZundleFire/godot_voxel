#include "voxel_noise_batch.h"
#include "voxel_advanced_noise.h"
#include "voxel_advanced_noise_simd.h"

#include <cmath>

namespace zylann {

// ============================================================================
// Scalar fallback noise functions (same algorithms as VoxelAdvancedNoise)
// Used when ISPC is not available. These are defined in voxel_advanced_noise.cpp
// as static functions in the anonymous namespace — we replicate lightweight
// versions here to avoid cross-TU dependencies.
// ============================================================================

namespace {

inline int batch_fast_floor(float x) {
	int xi = (int)x;
	return (x < xi) ? xi - 1 : xi;
}

inline float batch_fade(float t) {
	return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

inline float batch_lerp(float a, float b, float t) {
	return a + t * (b - a);
}

inline float batch_grad3(int hash, float x, float y, float z) {
	int h = hash & 15;
	float u = h < 8 ? x : y;
	float v = h < 4 ? y : (h == 12 || h == 14 ? x : z);
	return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}

inline float batch_grad2(int hash, float x, float y) {
	int h = hash & 7;
	float u = h < 4 ? x : y;
	float v = h < 4 ? y : x;
	return ((h & 1) != 0 ? -u : u) + ((h & 2) != 0 ? -2.0f * v : 2.0f * v);
}

inline int batch_perm(int seed, int x) {
	unsigned int h = (unsigned int)(seed ^ (x * 1619));
	h = h * 6364136223846793005ULL + 1442695040888963407ULL;
	h ^= h >> 16;
	return (int)(h & 0xFF);
}

inline int batch_perm2(int seed, int x, int y) {
	return batch_perm(seed, x + batch_perm(seed + 1, y));
}

inline int batch_perm3(int seed, int x, int y, int z) {
	return batch_perm(seed, x + batch_perm(seed + 1, y + batch_perm(seed + 2, z)));
}

float scalar_perlin_3d(int seed, float x, float y, float z) {
	int x0 = batch_fast_floor(x), y0 = batch_fast_floor(y), z0 = batch_fast_floor(z);
	float xf = x - x0, yf = y - y0, zf = z - z0;
	float u = batch_fade(xf), v = batch_fade(yf), w = batch_fade(zf);

	float x1 = batch_lerp(batch_grad3(batch_perm3(seed, x0, y0, z0), xf, yf, zf),
			batch_grad3(batch_perm3(seed, x0 + 1, y0, z0), xf - 1, yf, zf), u);
	float x2 = batch_lerp(batch_grad3(batch_perm3(seed, x0, y0 + 1, z0), xf, yf - 1, zf),
			batch_grad3(batch_perm3(seed, x0 + 1, y0 + 1, z0), xf - 1, yf - 1, zf), u);
	float y1 = batch_lerp(x1, x2, v);

	float x3 = batch_lerp(batch_grad3(batch_perm3(seed, x0, y0, z0 + 1), xf, yf, zf - 1),
			batch_grad3(batch_perm3(seed, x0 + 1, y0, z0 + 1), xf - 1, yf, zf - 1), u);
	float x4 = batch_lerp(batch_grad3(batch_perm3(seed, x0, y0 + 1, z0 + 1), xf, yf - 1, zf - 1),
			batch_grad3(batch_perm3(seed, x0 + 1, y0 + 1, z0 + 1), xf - 1, yf - 1, zf - 1), u);
	float y2 = batch_lerp(x3, x4, v);

	return batch_lerp(y1, y2, w);
}

float scalar_perlin_2d(int seed, float x, float y) {
	int x0 = batch_fast_floor(x), y0 = batch_fast_floor(y);
	float xf = x - x0, yf = y - y0;
	float u = batch_fade(xf), v = batch_fade(yf);

	float x1 = batch_lerp(batch_grad2(batch_perm2(seed, x0, y0), xf, yf),
			batch_grad2(batch_perm2(seed, x0 + 1, y0), xf - 1, yf), u);
	float x2 = batch_lerp(batch_grad2(batch_perm2(seed, x0, y0 + 1), xf, yf - 1),
			batch_grad2(batch_perm2(seed, x0 + 1, y0 + 1), xf - 1, yf - 1), u);
	return batch_lerp(x1, x2, v);
}

float scalar_simplex_3d(int seed, float x, float y, float z) {
	const float F3 = 1.0f / 3.0f;
	const float G3 = 1.0f / 6.0f;
	float s = (x + y + z) * F3;
	int i = batch_fast_floor(x + s), j = batch_fast_floor(y + s), k = batch_fast_floor(z + s);
	float t = (i + j + k) * G3;
	float x0 = x - (i - t), y0 = y - (j - t), z0 = z - (k - t);

	int i1, j1, k1, i2, j2, k2;
	if (x0 >= y0) {
		if (y0 >= z0) { i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 1; k2 = 0; }
		else if (x0 >= z0) { i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 0; k2 = 1; }
		else { i1 = 0; j1 = 0; k1 = 1; i2 = 1; j2 = 0; k2 = 1; }
	} else {
		if (y0 < z0) { i1 = 0; j1 = 0; k1 = 1; i2 = 0; j2 = 1; k2 = 1; }
		else if (x0 < z0) { i1 = 0; j1 = 1; k1 = 0; i2 = 0; j2 = 1; k2 = 1; }
		else { i1 = 0; j1 = 1; k1 = 0; i2 = 1; j2 = 1; k2 = 0; }
	}

	float x1 = x0 - i1 + G3, y1 = y0 - j1 + G3, z1 = z0 - k1 + G3;
	float x2 = x0 - i2 + 2.0f * G3, y2 = y0 - j2 + 2.0f * G3, z2 = z0 - k2 + 2.0f * G3;
	float x3 = x0 - 1.0f + 3.0f * G3, y3 = y0 - 1.0f + 3.0f * G3, z3 = z0 - 1.0f + 3.0f * G3;

	float n0 = 0, n1 = 0, n2 = 0, n3 = 0;
	float t0 = 0.6f - x0 * x0 - y0 * y0 - z0 * z0;
	if (t0 >= 0) { t0 *= t0; n0 = t0 * t0 * batch_grad3(batch_perm3(seed, i, j, k), x0, y0, z0); }
	float t1 = 0.6f - x1 * x1 - y1 * y1 - z1 * z1;
	if (t1 >= 0) { t1 *= t1; n1 = t1 * t1 * batch_grad3(batch_perm3(seed, i + i1, j + j1, k + k1), x1, y1, z1); }
	float t2 = 0.6f - x2 * x2 - y2 * y2 - z2 * z2;
	if (t2 >= 0) { t2 *= t2; n2 = t2 * t2 * batch_grad3(batch_perm3(seed, i + i2, j + j2, k + k2), x2, y2, z2); }
	float t3 = 0.6f - x3 * x3 - y3 * y3 - z3 * z3;
	if (t3 >= 0) { t3 *= t3; n3 = t3 * t3 * batch_grad3(batch_perm3(seed, i + 1, j + 1, k + 1), x3, y3, z3); }
	return 32.0f * (n0 + n1 + n2 + n3);
}

float scalar_simplex_2d(int seed, float x, float y) {
	const float F2 = 0.5f * (std::sqrt(3.0f) - 1.0f);
	const float G2 = (3.0f - std::sqrt(3.0f)) / 6.0f;
	float s = (x + y) * F2;
	int i = batch_fast_floor(x + s), j = batch_fast_floor(y + s);
	float t = (i + j) * G2;
	float x0 = x - (i - t), y0 = y - (j - t);

	int i1, j1;
	if (x0 > y0) { i1 = 1; j1 = 0; } else { i1 = 0; j1 = 1; }

	float x1 = x0 - i1 + G2, y1 = y0 - j1 + G2;
	float x2 = x0 - 1.0f + 2.0f * G2, y2 = y0 - 1.0f + 2.0f * G2;

	float n0 = 0, n1 = 0, n2 = 0;
	float t0 = 0.5f - x0 * x0 - y0 * y0;
	if (t0 >= 0) { t0 *= t0; n0 = t0 * t0 * batch_grad2(batch_perm2(seed, i, j), x0, y0); }
	float t1 = 0.5f - x1 * x1 - y1 * y1;
	if (t1 >= 0) { t1 *= t1; n1 = t1 * t1 * batch_grad2(batch_perm2(seed, i + i1, j + j1), x1, y1); }
	float t2 = 0.5f - x2 * x2 - y2 * y2;
	if (t2 >= 0) { t2 *= t2; n2 = t2 * t2 * batch_grad2(batch_perm2(seed, i + 1, j + 1), x2, y2); }
	return 70.0f * (n0 + n1 + n2);
}

float scalar_cellular_3d(int seed, float x, float y, float z, float jitter) {
	int xi = batch_fast_floor(x), yi = batch_fast_floor(y), zi = batch_fast_floor(z);
	float min_dist = 1e30f;
	for (int dz = -1; dz <= 1; dz++) {
		for (int dy = -1; dy <= 1; dy++) {
			for (int dx = -1; dx <= 1; dx++) {
				int cx = xi + dx, cy = yi + dy, cz = zi + dz;
				unsigned int h = (unsigned int)(seed ^ (cx * 1619) ^ (cy * 31337) ^ (cz * 6971));
				h = h * 6364136223846793005ULL + 1442695040888963407ULL;
				float px = (float)(h & 0xFFFF) / 65535.0f;
				h = h * 6364136223846793005ULL + 1442695040888963407ULL;
				float py = (float)(h & 0xFFFF) / 65535.0f;
				h = h * 6364136223846793005ULL + 1442695040888963407ULL;
				float pz = (float)(h & 0xFFFF) / 65535.0f;
				float dist_x = x - (cx + px * jitter), dist_y = y - (cy + py * jitter), dist_z = z - (cz + pz * jitter);
				float dist = dist_x * dist_x + dist_y * dist_y + dist_z * dist_z;
				if (dist < min_dist) min_dist = dist;
			}
		}
	}
	return std::sqrt(min_dist) - 0.5f;
}

float scalar_cellular_2d(int seed, float x, float y, float jitter) {
	int xi = batch_fast_floor(x), yi = batch_fast_floor(y);
	float min_dist = 1e30f;
	for (int dy = -1; dy <= 1; dy++) {
		for (int dx = -1; dx <= 1; dx++) {
			int cx = xi + dx, cy = yi + dy;
			unsigned int h = (unsigned int)(seed ^ (cx * 1619) ^ (cy * 31337));
			h = h * 6364136223846793005ULL + 1442695040888963407ULL;
			float px = (float)(h & 0xFFFF) / 65535.0f;
			h = h * 6364136223846793005ULL + 1442695040888963407ULL;
			float py = (float)(h & 0xFFFF) / 65535.0f;
			float dist_x = x - (cx + px * jitter), dist_y = y - (cy + py * jitter);
			float dist = dist_x * dist_x + dist_y * dist_y;
			if (dist < min_dist) min_dist = dist;
		}
	}
	return std::sqrt(min_dist) - 0.5f;
}

float scalar_value_3d(int seed, float x, float y, float z) {
	int x0 = batch_fast_floor(x), y0 = batch_fast_floor(y), z0 = batch_fast_floor(z);
	float xf = x - x0, yf = y - y0, zf = z - z0;
	float u = batch_fade(xf), v = batch_fade(yf), w = batch_fade(zf);

	auto val_at = [seed](int ix, int iy, int iz) -> float {
		unsigned int h = (unsigned int)(seed ^ (ix * 1619) ^ (iy * 31337) ^ (iz * 6971));
		h = h * 6364136223846793005ULL + 1442695040888963407ULL;
		h ^= h >> 16;
		return (float)(h & 0xFFFF) / 32767.5f - 1.0f;
	};

	float lx1 = batch_lerp(val_at(x0, y0, z0), val_at(x0 + 1, y0, z0), u);
	float lx2 = batch_lerp(val_at(x0, y0 + 1, z0), val_at(x0 + 1, y0 + 1, z0), u);
	float ly1 = batch_lerp(lx1, lx2, v);
	float lx3 = batch_lerp(val_at(x0, y0, z0 + 1), val_at(x0 + 1, y0, z0 + 1), u);
	float lx4 = batch_lerp(val_at(x0, y0 + 1, z0 + 1), val_at(x0 + 1, y0 + 1, z0 + 1), u);
	float ly2 = batch_lerp(lx3, lx4, v);
	return batch_lerp(ly1, ly2, w);
}

float scalar_value_2d(int seed, float x, float y) {
	int x0 = batch_fast_floor(x), y0 = batch_fast_floor(y);
	float xf = x - x0, yf = y - y0;
	float u = batch_fade(xf), v = batch_fade(yf);

	auto val_at = [seed](int ix, int iy) -> float {
		unsigned int h = (unsigned int)(seed ^ (ix * 1619) ^ (iy * 31337));
		h = h * 6364136223846793005ULL + 1442695040888963407ULL;
		h ^= h >> 16;
		return (float)(h & 0xFFFF) / 32767.5f - 1.0f;
	};

	float lx1 = batch_lerp(val_at(x0, y0), val_at(x0 + 1, y0), u);
	float lx2 = batch_lerp(val_at(x0, y0 + 1), val_at(x0 + 1, y0 + 1), u);
	return batch_lerp(lx1, lx2, v);
}

} // anonymous namespace

// ============================================================================
// Position preparation
// ============================================================================

Array VoxelNoiseBatch::prepare_positions_3d(Vector3i origin, Vector3i size, int lod) {
	const int total = size.x * size.y * size.z;
	const float lod_scale = static_cast<float>(1 << lod);

	PackedFloat32Array px, py, pz;
	px.resize(total);
	py.resize(total);
	pz.resize(total);

	float *dx = px.ptrw();
	float *dy = py.ptrw();
	float *dz = pz.ptrw();

	int idx = 0;
	for (int z = 0; z < size.z; z++) {
		for (int y = 0; y < size.y; y++) {
			for (int x = 0; x < size.x; x++) {
				dx[idx] = origin.x + x * lod_scale;
				dy[idx] = origin.y + y * lod_scale;
				dz[idx] = origin.z + z * lod_scale;
				idx++;
			}
		}
	}

	Array result;
	result.resize(3);
	result[0] = px;
	result[1] = py;
	result[2] = pz;
	return result;
}

Array VoxelNoiseBatch::prepare_positions_2d(Vector2 origin, Vector2i size, float spacing) {
	const int total = size.x * size.y;

	PackedFloat32Array px, py;
	px.resize(total);
	py.resize(total);

	float *dx = px.ptrw();
	float *dy = py.ptrw();

	int idx = 0;
	for (int y = 0; y < size.y; y++) {
		for (int x = 0; x < size.x; x++) {
			dx[idx] = origin.x + x * spacing;
			dy[idx] = origin.y + y * spacing;
			idx++;
		}
	}

	Array result;
	result.resize(2);
	result[0] = px;
	result[1] = py;
	return result;
}

// ============================================================================
// Batch noise — ISPC dispatch with scalar fallback
// ============================================================================

PackedFloat32Array VoxelNoiseBatch::perlin_3d_batch(
		const PackedFloat32Array &pos_x,
		const PackedFloat32Array &pos_y,
		const PackedFloat32Array &pos_z,
		int seed) {
	const int count = pos_x.size();
	ERR_FAIL_COND_V(pos_y.size() != count || pos_z.size() != count, PackedFloat32Array());

	PackedFloat32Array output;
	output.resize(count);

#ifdef VOXEL_ISPC_ENABLED
	if (voxel_ispc_available()) {
		voxel_ispc_perlin_3d_batch(pos_x.ptr(), pos_y.ptr(), pos_z.ptr(),
				seed, output.ptrw(), count);
		return output;
	}
#endif

	const float *rx = pos_x.ptr();
	const float *ry = pos_y.ptr();
	const float *rz = pos_z.ptr();
	float *out = output.ptrw();
	for (int i = 0; i < count; i++) {
		out[i] = scalar_perlin_3d(seed, rx[i], ry[i], rz[i]);
	}
	return output;
}

PackedFloat32Array VoxelNoiseBatch::perlin_2d_batch(
		const PackedFloat32Array &pos_x,
		const PackedFloat32Array &pos_y,
		int seed) {
	const int count = pos_x.size();
	ERR_FAIL_COND_V(pos_y.size() != count, PackedFloat32Array());

	PackedFloat32Array output;
	output.resize(count);

#ifdef VOXEL_ISPC_ENABLED
	if (voxel_ispc_available()) {
		voxel_ispc_perlin_2d_batch(pos_x.ptr(), pos_y.ptr(), seed, output.ptrw(), count);
		return output;
	}
#endif

	const float *rx = pos_x.ptr();
	const float *ry = pos_y.ptr();
	float *out = output.ptrw();
	for (int i = 0; i < count; i++) {
		out[i] = scalar_perlin_2d(seed, rx[i], ry[i]);
	}
	return output;
}

PackedFloat32Array VoxelNoiseBatch::simplex_3d_batch(
		const PackedFloat32Array &pos_x,
		const PackedFloat32Array &pos_y,
		const PackedFloat32Array &pos_z,
		int seed) {
	const int count = pos_x.size();
	ERR_FAIL_COND_V(pos_y.size() != count || pos_z.size() != count, PackedFloat32Array());

	PackedFloat32Array output;
	output.resize(count);

#ifdef VOXEL_ISPC_ENABLED
	if (voxel_ispc_available()) {
		voxel_ispc_simplex_3d_batch(pos_x.ptr(), pos_y.ptr(), pos_z.ptr(),
				seed, output.ptrw(), count);
		return output;
	}
#endif

	const float *rx = pos_x.ptr();
	const float *ry = pos_y.ptr();
	const float *rz = pos_z.ptr();
	float *out = output.ptrw();
	for (int i = 0; i < count; i++) {
		out[i] = scalar_simplex_3d(seed, rx[i], ry[i], rz[i]);
	}
	return output;
}

PackedFloat32Array VoxelNoiseBatch::simplex_2d_batch(
		const PackedFloat32Array &pos_x,
		const PackedFloat32Array &pos_y,
		int seed) {
	const int count = pos_x.size();
	ERR_FAIL_COND_V(pos_y.size() != count, PackedFloat32Array());

	PackedFloat32Array output;
	output.resize(count);

#ifdef VOXEL_ISPC_ENABLED
	if (voxel_ispc_available()) {
		voxel_ispc_simplex_2d_batch(pos_x.ptr(), pos_y.ptr(), seed, output.ptrw(), count);
		return output;
	}
#endif

	const float *rx = pos_x.ptr();
	const float *ry = pos_y.ptr();
	float *out = output.ptrw();
	for (int i = 0; i < count; i++) {
		out[i] = scalar_simplex_2d(seed, rx[i], ry[i]);
	}
	return output;
}

PackedFloat32Array VoxelNoiseBatch::cellular_3d_batch(
		const PackedFloat32Array &pos_x,
		const PackedFloat32Array &pos_y,
		const PackedFloat32Array &pos_z,
		int seed, float jitter) {
	const int count = pos_x.size();
	ERR_FAIL_COND_V(pos_y.size() != count || pos_z.size() != count, PackedFloat32Array());

	PackedFloat32Array output;
	output.resize(count);

#ifdef VOXEL_ISPC_ENABLED
	if (voxel_ispc_available()) {
		voxel_ispc_cellular_3d_batch(pos_x.ptr(), pos_y.ptr(), pos_z.ptr(),
				seed, jitter, output.ptrw(), count);
		return output;
	}
#endif

	const float *rx = pos_x.ptr();
	const float *ry = pos_y.ptr();
	const float *rz = pos_z.ptr();
	float *out = output.ptrw();
	for (int i = 0; i < count; i++) {
		out[i] = scalar_cellular_3d(seed, rx[i], ry[i], rz[i], jitter);
	}
	return output;
}

PackedFloat32Array VoxelNoiseBatch::cellular_2d_batch(
		const PackedFloat32Array &pos_x,
		const PackedFloat32Array &pos_y,
		int seed, float jitter) {
	const int count = pos_x.size();
	ERR_FAIL_COND_V(pos_y.size() != count, PackedFloat32Array());

	PackedFloat32Array output;
	output.resize(count);

#ifdef VOXEL_ISPC_ENABLED
	if (voxel_ispc_available()) {
		voxel_ispc_cellular_2d_batch(pos_x.ptr(), pos_y.ptr(), seed, jitter, output.ptrw(), count);
		return output;
	}
#endif

	const float *rx = pos_x.ptr();
	const float *ry = pos_y.ptr();
	float *out = output.ptrw();
	for (int i = 0; i < count; i++) {
		out[i] = scalar_cellular_2d(seed, rx[i], ry[i], jitter);
	}
	return output;
}

PackedFloat32Array VoxelNoiseBatch::value_3d_batch(
		const PackedFloat32Array &pos_x,
		const PackedFloat32Array &pos_y,
		const PackedFloat32Array &pos_z,
		int seed) {
	const int count = pos_x.size();
	ERR_FAIL_COND_V(pos_y.size() != count || pos_z.size() != count, PackedFloat32Array());

	PackedFloat32Array output;
	output.resize(count);

#ifdef VOXEL_ISPC_ENABLED
	if (voxel_ispc_available()) {
		voxel_ispc_value_3d_batch(pos_x.ptr(), pos_y.ptr(), pos_z.ptr(),
				seed, output.ptrw(), count);
		return output;
	}
#endif

	const float *rx = pos_x.ptr();
	const float *ry = pos_y.ptr();
	const float *rz = pos_z.ptr();
	float *out = output.ptrw();
	for (int i = 0; i < count; i++) {
		out[i] = scalar_value_3d(seed, rx[i], ry[i], rz[i]);
	}
	return output;
}

PackedFloat32Array VoxelNoiseBatch::value_2d_batch(
		const PackedFloat32Array &pos_x,
		const PackedFloat32Array &pos_y,
		int seed) {
	const int count = pos_x.size();
	ERR_FAIL_COND_V(pos_y.size() != count, PackedFloat32Array());

	PackedFloat32Array output;
	output.resize(count);

#ifdef VOXEL_ISPC_ENABLED
	if (voxel_ispc_available()) {
		voxel_ispc_value_2d_batch(pos_x.ptr(), pos_y.ptr(), seed, output.ptrw(), count);
		return output;
	}
#endif

	const float *rx = pos_x.ptr();
	const float *ry = pos_y.ptr();
	float *out = output.ptrw();
	for (int i = 0; i < count; i++) {
		out[i] = scalar_value_2d(seed, rx[i], ry[i]);
	}
	return output;
}

PackedFloat32Array VoxelNoiseBatch::true_distance_cellular_3d_batch(
		const PackedFloat32Array &pos_x,
		const PackedFloat32Array &pos_y,
		const PackedFloat32Array &pos_z,
		int seed, float jitter) {
	const int count = pos_x.size();
	ERR_FAIL_COND_V(pos_y.size() != count || pos_z.size() != count, PackedFloat32Array());

	PackedFloat32Array output;
	output.resize(count);

#ifdef VOXEL_ISPC_ENABLED
	if (voxel_ispc_available()) {
		voxel_ispc_true_distance_cellular_3d_batch(pos_x.ptr(), pos_y.ptr(), pos_z.ptr(),
				seed, jitter, output.ptrw(), count);
		return output;
	}
#endif

	// Scalar fallback uses standard cellular noise
	const float *rx = pos_x.ptr();
	const float *ry = pos_y.ptr();
	const float *rz = pos_z.ptr();
	float *out = output.ptrw();
	for (int i = 0; i < count; i++) {
		out[i] = scalar_cellular_3d(seed, rx[i], ry[i], rz[i], jitter);
	}
	return output;
}

PackedFloat32Array VoxelNoiseBatch::true_distance_cellular_2d_batch(
		const PackedFloat32Array &pos_x,
		const PackedFloat32Array &pos_y,
		int seed, float jitter) {
	const int count = pos_x.size();
	ERR_FAIL_COND_V(pos_y.size() != count, PackedFloat32Array());

	PackedFloat32Array output;
	output.resize(count);

#ifdef VOXEL_ISPC_ENABLED
	if (voxel_ispc_available()) {
		voxel_ispc_true_distance_cellular_2d_batch(pos_x.ptr(), pos_y.ptr(),
				seed, jitter, output.ptrw(), count);
		return output;
	}
#endif

	const float *rx = pos_x.ptr();
	const float *ry = pos_y.ptr();
	float *out = output.ptrw();
	for (int i = 0; i < count; i++) {
		out[i] = scalar_cellular_2d(seed, rx[i], ry[i], jitter);
	}
	return output;
}

// ============================================================================
// Domain warp
// ============================================================================

Array VoxelNoiseBatch::domain_warp_3d_grid(
		Vector3 origin, Vector3i size, float spacing,
		float amplitude, float feature_scale,
		float lacunarity, float gain, float weighted_strength,
		int num_octaves, int seed) {
	const int total = size.x * size.y * size.z;

	PackedFloat32Array out_x, out_y, out_z;
	out_x.resize(total);
	out_y.resize(total);
	out_z.resize(total);

#ifdef VOXEL_ISPC_ENABLED
	if (voxel_ispc_available()) {
		voxel_ispc_domain_warp_3d_grid(
				origin.x, origin.y, origin.z,
				size.x, size.y, size.z, spacing,
				amplitude, feature_scale,
				lacunarity, gain, weighted_strength,
				num_octaves, seed,
				out_x.ptrw(), out_y.ptrw(), out_z.ptrw(), total);

		Array result;
		result.resize(3);
		result[0] = out_x;
		result[1] = out_y;
		result[2] = out_z;
		return result;
	}
#endif

	// Scalar fallback — simple single-octave hash-based warp
	float *dx = out_x.ptrw();
	float *dy = out_y.ptrw();
	float *dz = out_z.ptrw();

	int idx = 0;
	for (int z = 0; z < size.z; z++) {
		for (int y = 0; y < size.y; y++) {
			for (int x = 0; x < size.x; x++) {
				float wx = origin.x + x * spacing;
				float wy = origin.y + y * spacing;
				float wz = origin.z + z * spacing;

				// Simple scalar domain warp: use noise to offset coordinates
				float sx = wx / feature_scale;
				float sy = wy / feature_scale;
				float sz = wz / feature_scale;

				float warp_x = scalar_perlin_3d(seed, sx, sy, sz) * amplitude;
				float warp_y = scalar_perlin_3d(seed + 31337, sx, sy, sz) * amplitude;
				float warp_z = scalar_perlin_3d(seed + 67890, sx, sy, sz) * amplitude;

				dx[idx] = wx + warp_x;
				dy[idx] = wy + warp_y;
				dz[idx] = wz + warp_z;
				idx++;
			}
		}
	}

	Array result;
	result.resize(3);
	result[0] = out_x;
	result[1] = out_y;
	result[2] = out_z;
	return result;
}

Array VoxelNoiseBatch::domain_warp_2d_grid(
		Vector2 origin, Vector2i size, float spacing,
		float amplitude, float feature_scale,
		float lacunarity, float gain, float weighted_strength,
		int num_octaves, int seed) {
	const int total = size.x * size.y;

	PackedFloat32Array out_x, out_y;
	out_x.resize(total);
	out_y.resize(total);

#ifdef VOXEL_ISPC_ENABLED
	if (voxel_ispc_available()) {
		voxel_ispc_domain_warp_2d_grid(
				origin.x, origin.y,
				size.x, size.y, spacing,
				amplitude, feature_scale,
				lacunarity, gain, weighted_strength,
				num_octaves, seed,
				out_x.ptrw(), out_y.ptrw(), total);

		Array result;
		result.resize(2);
		result[0] = out_x;
		result[1] = out_y;
		return result;
	}
#endif

	// Scalar fallback
	float *dx = out_x.ptrw();
	float *dy = out_y.ptrw();

	int idx = 0;
	for (int y = 0; y < size.y; y++) {
		for (int x = 0; x < size.x; x++) {
			float wx = origin.x + x * spacing;
			float wy = origin.y + y * spacing;

			float sx = wx / feature_scale;
			float sy = wy / feature_scale;

			float warp_x = scalar_perlin_2d(seed, sx, sy) * amplitude;
			float warp_y = scalar_perlin_2d(seed + 31337, sx, sy) * amplitude;

			dx[idx] = wx + warp_x;
			dy[idx] = wy + warp_y;
			idx++;
		}
	}

	Array result;
	result.resize(2);
	result[0] = out_x;
	result[1] = out_y;
	return result;
}

Array VoxelNoiseBatch::domain_warp_3d_batch(
		const PackedFloat32Array &pos_x,
		const PackedFloat32Array &pos_y,
		const PackedFloat32Array &pos_z,
		float amplitude, float feature_scale,
		float lacunarity, float gain, float weighted_strength,
		int num_octaves, int seed) {
	const int count = pos_x.size();
	ERR_FAIL_COND_V(pos_y.size() != count || pos_z.size() != count, Array());

	PackedFloat32Array out_x, out_y, out_z;
	out_x.resize(count);
	out_y.resize(count);
	out_z.resize(count);

#ifdef VOXEL_ISPC_ENABLED
	if (voxel_ispc_available()) {
		voxel_ispc_domain_warp_3d_batch(
				pos_x.ptr(), pos_y.ptr(), pos_z.ptr(),
				amplitude, feature_scale,
				lacunarity, gain, weighted_strength,
				num_octaves, seed,
				out_x.ptrw(), out_y.ptrw(), out_z.ptrw(), count);

		Array result;
		result.resize(3);
		result[0] = out_x;
		result[1] = out_y;
		result[2] = out_z;
		return result;
	}
#endif

	// Scalar fallback
	const float *rx = pos_x.ptr();
	const float *ry = pos_y.ptr();
	const float *rz = pos_z.ptr();
	float *dx = out_x.ptrw();
	float *dy = out_y.ptrw();
	float *dz = out_z.ptrw();

	for (int i = 0; i < count; i++) {
		float sx = rx[i] / feature_scale;
		float sy = ry[i] / feature_scale;
		float sz = rz[i] / feature_scale;

		float warp_x = scalar_perlin_3d(seed, sx, sy, sz) * amplitude;
		float warp_y = scalar_perlin_3d(seed + 31337, sx, sy, sz) * amplitude;
		float warp_z = scalar_perlin_3d(seed + 67890, sx, sy, sz) * amplitude;

		dx[i] = rx[i] + warp_x;
		dy[i] = ry[i] + warp_y;
		dz[i] = rz[i] + warp_z;
	}

	Array result;
	result.resize(3);
	result[0] = out_x;
	result[1] = out_y;
	result[2] = out_z;
	return result;
}

// ============================================================================
// Info
// ============================================================================

bool VoxelNoiseBatch::is_simd_available() {
	return voxel_ispc_available();
}

// ============================================================================
// GDScript bindings
// ============================================================================

void VoxelNoiseBatch::_bind_methods() {
	// Position helpers
	ClassDB::bind_static_method("VoxelNoiseBatch",
			D_METHOD("prepare_positions_3d", "origin", "size", "lod"),
			&VoxelNoiseBatch::prepare_positions_3d);
	ClassDB::bind_static_method("VoxelNoiseBatch",
			D_METHOD("prepare_positions_2d", "origin", "size", "spacing"),
			&VoxelNoiseBatch::prepare_positions_2d);

	// Individual noise batch functions
	ClassDB::bind_static_method("VoxelNoiseBatch",
			D_METHOD("perlin_3d_batch", "pos_x", "pos_y", "pos_z", "seed"),
			&VoxelNoiseBatch::perlin_3d_batch);
	ClassDB::bind_static_method("VoxelNoiseBatch",
			D_METHOD("perlin_2d_batch", "pos_x", "pos_y", "seed"),
			&VoxelNoiseBatch::perlin_2d_batch);

	ClassDB::bind_static_method("VoxelNoiseBatch",
			D_METHOD("simplex_3d_batch", "pos_x", "pos_y", "pos_z", "seed"),
			&VoxelNoiseBatch::simplex_3d_batch);
	ClassDB::bind_static_method("VoxelNoiseBatch",
			D_METHOD("simplex_2d_batch", "pos_x", "pos_y", "seed"),
			&VoxelNoiseBatch::simplex_2d_batch);

	ClassDB::bind_static_method("VoxelNoiseBatch",
			D_METHOD("cellular_3d_batch", "pos_x", "pos_y", "pos_z", "seed", "jitter"),
			&VoxelNoiseBatch::cellular_3d_batch);
	ClassDB::bind_static_method("VoxelNoiseBatch",
			D_METHOD("cellular_2d_batch", "pos_x", "pos_y", "seed", "jitter"),
			&VoxelNoiseBatch::cellular_2d_batch);

	ClassDB::bind_static_method("VoxelNoiseBatch",
			D_METHOD("value_3d_batch", "pos_x", "pos_y", "pos_z", "seed"),
			&VoxelNoiseBatch::value_3d_batch);
	ClassDB::bind_static_method("VoxelNoiseBatch",
			D_METHOD("value_2d_batch", "pos_x", "pos_y", "seed"),
			&VoxelNoiseBatch::value_2d_batch);

	ClassDB::bind_static_method("VoxelNoiseBatch",
			D_METHOD("true_distance_cellular_3d_batch", "pos_x", "pos_y", "pos_z", "seed", "jitter"),
			&VoxelNoiseBatch::true_distance_cellular_3d_batch);
	ClassDB::bind_static_method("VoxelNoiseBatch",
			D_METHOD("true_distance_cellular_2d_batch", "pos_x", "pos_y", "seed", "jitter"),
			&VoxelNoiseBatch::true_distance_cellular_2d_batch);

	// Domain warp
	ClassDB::bind_static_method("VoxelNoiseBatch",
			D_METHOD("domain_warp_3d_grid", "origin", "size", "spacing",
					"amplitude", "feature_scale", "lacunarity", "gain", "weighted_strength",
					"num_octaves", "seed"),
			&VoxelNoiseBatch::domain_warp_3d_grid);
	ClassDB::bind_static_method("VoxelNoiseBatch",
			D_METHOD("domain_warp_2d_grid", "origin", "size", "spacing",
					"amplitude", "feature_scale", "lacunarity", "gain", "weighted_strength",
					"num_octaves", "seed"),
			&VoxelNoiseBatch::domain_warp_2d_grid);
	ClassDB::bind_static_method("VoxelNoiseBatch",
			D_METHOD("domain_warp_3d_batch", "pos_x", "pos_y", "pos_z",
					"amplitude", "feature_scale", "lacunarity", "gain", "weighted_strength",
					"num_octaves", "seed"),
			&VoxelNoiseBatch::domain_warp_3d_batch);

	// Info
	ClassDB::bind_static_method("VoxelNoiseBatch",
			D_METHOD("is_simd_available"),
			&VoxelNoiseBatch::is_simd_available);
}

} // namespace zylann
