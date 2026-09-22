#ifndef VOXEL_FAR_MATH_H
#define VOXEL_FAR_MATH_H

// Engine-independent math used by the far-LOD core.
//
// The core of this module deliberately does NOT include any Godot or godot_voxel
// header, so that the load-bearing algorithms (column extraction, downsampling,
// quadtree, meshing) can be unit-tested on the host with a plain C++17 compiler.
// The Godot glue in `node/` converts to and from engine types at the boundary.

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace zylann::voxel::far {

struct Vec3f {
	float x = 0.f;
	float y = 0.f;
	float z = 0.f;

	Vec3f() = default;
	Vec3f(float p_x, float p_y, float p_z) : x(p_x), y(p_y), z(p_z) {}

	inline Vec3f operator+(const Vec3f &o) const {
		return Vec3f(x + o.x, y + o.y, z + o.z);
	}
	inline Vec3f operator-(const Vec3f &o) const {
		return Vec3f(x - o.x, y - o.y, z - o.z);
	}
	inline Vec3f operator*(float s) const {
		return Vec3f(x * s, y * s, z * s);
	}
	inline Vec3f &operator+=(const Vec3f &o) {
		x += o.x;
		y += o.y;
		z += o.z;
		return *this;
	}

	inline float length_squared() const {
		return x * x + y * y + z * z;
	}
	inline float length() const {
		return std::sqrt(length_squared());
	}

	inline Vec3f normalized() const {
		const float l = length();
		if (l < 1e-20f) {
			return Vec3f(0.f, 1.f, 0.f);
		}
		const float inv = 1.f / l;
		return Vec3f(x * inv, y * inv, z * inv);
	}
};

inline Vec3f cross(const Vec3f &a, const Vec3f &b) {
	return Vec3f(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

inline float dot(const Vec3f &a, const Vec3f &b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

struct Vec2i {
	int32_t x = 0;
	int32_t y = 0;

	Vec2i() = default;
	Vec2i(int32_t p_x, int32_t p_y) : x(p_x), y(p_y) {}

	inline bool operator==(const Vec2i &o) const {
		return x == o.x && y == o.y;
	}
	inline bool operator!=(const Vec2i &o) const {
		return !(*this == o);
	}
	// Ordering so Vec2i can key an std::map / be sorted.
	inline bool operator<(const Vec2i &o) const {
		return (y != o.y) ? (y < o.y) : (x < o.x);
	}
};

struct Vec3i {
	int32_t x = 0;
	int32_t y = 0;
	int32_t z = 0;

	Vec3i() = default;
	Vec3i(int32_t p_x, int32_t p_y, int32_t p_z) : x(p_x), y(p_y), z(p_z) {}
};

// Integer floor-division and floor-modulo. Plain `/` truncates toward zero,
// which produces a discontinuity at the origin when mapping world coordinates
// to sector coordinates. Every world->sector mapping in this module goes
// through these two.
inline int32_t floordiv(int32_t a, int32_t b) {
	return (a >= 0) ? (a / b) : -((-a + b - 1) / b);
}

inline int32_t floormod(int32_t a, int32_t b) {
	const int32_t r = a % b;
	return (r < 0) ? (r + b) : r;
}

inline float lerpf(float a, float b, float t) {
	return a + (b - a) * t;
}

inline float clampf(float v, float lo, float hi) {
	return (v < lo) ? lo : ((v > hi) ? hi : v);
}

// ---------------------------------------------------------------------------
// Octahedral normal packing.
//
// Far terrain stores one normal per surface layer. Storing three floats would
// triple the size of the column data, and the whole point of the format is that
// it stays small enough to keep hundreds of thousands of sectors resident.
// Octahedral encoding at 16 bits per axis gives a worst-case angular error
// around 0.1 degrees, which is far below anything visible at LOD distances,
// for 4 bytes instead of 12.
// ---------------------------------------------------------------------------

inline float oct_wrap(float v, float w) {
	return (1.f - std::abs(w)) * ((v >= 0.f) ? 1.f : -1.f);
}

inline uint32_t pack_normal_oct16(const Vec3f &n_in) {
	const Vec3f n = n_in.normalized();
	const float inv_l1 = 1.f / (std::abs(n.x) + std::abs(n.y) + std::abs(n.z) + 1e-20f);
	float px = n.x * inv_l1;
	float pz = n.z * inv_l1;
	if (n.y < 0.f) {
		const float ox = oct_wrap(px, pz);
		const float oz = oct_wrap(pz, px);
		px = ox;
		pz = oz;
	}
	const uint32_t ux = static_cast<uint32_t>(std::lround(clampf(px * 0.5f + 0.5f, 0.f, 1.f) * 65535.f));
	const uint32_t uz = static_cast<uint32_t>(std::lround(clampf(pz * 0.5f + 0.5f, 0.f, 1.f) * 65535.f));
	return (uz << 16) | ux;
}

inline Vec3f unpack_normal_oct16(uint32_t packed) {
	const float px = (static_cast<float>(packed & 0xffffu) / 65535.f) * 2.f - 1.f;
	const float pz = (static_cast<float>(packed >> 16) / 65535.f) * 2.f - 1.f;
	Vec3f n(px, 1.f - std::abs(px) - std::abs(pz), pz);
	if (n.y < 0.f) {
		const float ox = oct_wrap(n.x, n.z);
		const float oz = oct_wrap(n.z, n.x);
		n.x = ox;
		n.z = oz;
	}
	return n.normalized();
}

// Average packed normals without fully unpacking to a mean-of-unit-vectors and
// renormalizing per pair. Used by the downsampler, which merges 4 layers into 1.
inline uint32_t blend_normals_oct16(const uint32_t *packed, const float *weights, unsigned int count) {
	Vec3f acc;
	float total = 0.f;
	for (unsigned int i = 0; i < count; ++i) {
		const float w = weights[i];
		if (w <= 0.f) {
			continue;
		}
		acc += unpack_normal_oct16(packed[i]) * w;
		total += w;
	}
	if (total <= 0.f || acc.length_squared() < 1e-12f) {
		return pack_normal_oct16(Vec3f(0.f, 1.f, 0.f));
	}
	return pack_normal_oct16(acc.normalized());
}

} // namespace zylann::voxel::far

#endif // VOXEL_FAR_MATH_H
