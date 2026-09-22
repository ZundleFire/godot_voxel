#include "gpu_vertex_pack.h"
#include <cmath>
#include <cstring>
#include <limits>

namespace zylann::voxel::gpu_driven {

namespace {

inline float sign_not_zero(float v) {
	return v >= 0.f ? 1.f : -1.f;
}

inline uint32_t snorm7(float v) {
	const float c = v < -1.f ? -1.f : (v > 1.f ? 1.f : v);
	return static_cast<uint32_t>(std::lround(c * 63.f)) & 0x7fu;
}

inline float decode_snorm7(uint32_t bits) {
	// Sign-extend 7 bits, like GLSL bitfieldExtract on an int
	const int32_t v = static_cast<int32_t>(bits << 25) >> 25;
	return static_cast<float>(v) / 63.f;
}

inline uint32_t float_bits(float f) {
	uint32_t u;
	memcpy(&u, &f, sizeof(u));
	return u;
}

inline void expand_bounds(PackedMesh &m, const float p[3]) {
	for (unsigned int i = 0; i < 3; ++i) {
		if (p[i] < m.aabb_min[i]) {
			m.aabb_min[i] = p[i];
		}
		if (p[i] > m.aabb_max[i]) {
			m.aabb_max[i] = p[i];
		}
	}
}

} // namespace

uint32_t encode_oct14(float x, float y, float z) {
	const float l1 = std::fabs(x) + std::fabs(y) + std::fabs(z);
	if (l1 < 1e-20f) {
		return snorm7(0.f) | (snorm7(1.f) << 7);
	}
	float ox = x / l1;
	float oy = y / l1;
	if (z < 0.f) {
		const float tx = (1.f - std::fabs(oy)) * sign_not_zero(ox);
		const float ty = (1.f - std::fabs(ox)) * sign_not_zero(oy);
		ox = tx;
		oy = ty;
	}
	return snorm7(ox) | (snorm7(oy) << 7);
}

void decode_oct14(uint32_t e, float out[3]) {
	// Mirrors the vertex shader
	const float ox = decode_snorm7(e & 0x7fu);
	const float oy = decode_snorm7((e >> 7) & 0x7fu);
	float x = ox;
	float y = oy;
	const float z = 1.f - std::fabs(ox) - std::fabs(oy);
	if (z < 0.f) {
		x = (1.f - std::fabs(oy)) * sign_not_zero(ox);
		y = (1.f - std::fabs(ox)) * sign_not_zero(oy);
	}
	const float len = std::sqrt(x * x + y * y + z * z);
	out[0] = x / len;
	out[1] = y / len;
	out[2] = z / len;
}

uint16_t float_to_half(float f) {
	const uint32_t x = float_bits(f);
	const uint32_t sign = (x >> 16) & 0x8000u;
	const uint32_t abs_bits = x & 0x7fffffffu;
	if (abs_bits > 0x7f800000u) {
		return static_cast<uint16_t>(sign | 0x7e00u); // NaN
	}
	uint32_t mant = x & 0x7fffffu;
	const int exp = static_cast<int>((x >> 23) & 0xffu) - 127 + 15;
	if (exp >= 31) {
		return static_cast<uint16_t>(sign | 0x7bffu);
	}
	if (exp <= 0) {
		// Subnormal half
		if (exp < -10) {
			return static_cast<uint16_t>(sign);
		}
		mant |= 0x800000u;
		const uint32_t shift = static_cast<uint32_t>(14 - exp);
		uint32_t h = mant >> shift;
		const uint32_t rem = mant & ((1u << shift) - 1);
		const uint32_t halfway = 1u << (shift - 1);
		if (rem > halfway || (rem == halfway && (h & 1u))) {
			++h;
		}
		return static_cast<uint16_t>(sign | h);
	}
	uint32_t h = sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
	const uint32_t rem = mant & 0x1fffu;
	// A carry into the exponent is still the correctly rounded value
	if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) {
		++h;
	}
	if ((h & 0x7fffu) >= 0x7c00u) {
		h = sign | 0x7bffu;
	}
	return static_cast<uint16_t>(h);
}

float half_to_float(uint16_t h) {
	const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
	const uint32_t exp = (h >> 10) & 0x1fu;
	uint32_t mant = h & 0x3ffu;
	uint32_t bits;
	if (exp == 0) {
		if (mant == 0) {
			bits = sign;
		} else {
			// Normalize the subnormal
			int e = -1;
			do {
				++e;
				mant <<= 1;
			} while ((mant & 0x400u) == 0);
			bits = sign | (static_cast<uint32_t>(127 - 15 - e) << 23) | ((mant & 0x3ffu) << 13);
		}
	} else if (exp == 31) {
		bits = sign | 0x7f800000u | (mant << 13);
	} else {
		bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
	}
	float f;
	memcpy(&f, &bits, sizeof(f));
	return f;
}

void decode_secondary(const PackedVertex &v, float out[3]) {
	out[0] = v.position[0] + half_to_float(static_cast<uint16_t>(v.secondary_xy & 0xffffu));
	out[1] = v.position[1] + half_to_float(static_cast<uint16_t>(v.secondary_xy >> 16));
	out[2] = v.position[2] + half_to_float(static_cast<uint16_t>(v.secondary_z & 0xffffu));
}

uint32_t compact_flags(uint32_t idata) {
	return (idata & 0x3fu) | (((idata >> 8) & 0x3fu) << 6) | (((idata >> 16) & 0x3fu) << 12);
}

uint32_t pack_mixel4(uint32_t packed_indices, uint32_t packed_weights, bool &clamped) {
	uint32_t w8[4];
	uint32_t w4[4];
	uint32_t wsum4 = 0;
	uint32_t best = 0;
	for (unsigned int k = 0; k < 4; ++k) {
		w8[k] = (packed_weights >> (k * 8)) & 0xffu;
		w4[k] = (w8[k] * 15 + 127) / 255;
		wsum4 += w4[k];
		if (w8[k] > w8[best]) {
			best = k;
		}
	}
	// Weights too small to survive 4 bits: keep the dominant material rather than rendering nothing
	if (wsum4 == 0 && w8[best] > 0) {
		w4[best] = 1;
	}
	uint32_t out = 0;
	for (unsigned int k = 0; k < 4; ++k) {
		uint32_t idx = (packed_indices >> (k * 8)) & 0xffu;
		if (idx > MAX_MATERIAL_INDEX) {
			if (w8[k] > 0) {
				clamped = true;
			}
			idx = MAX_MATERIAL_INDEX;
		}
		out |= (idx | (w4[k] << 4)) << (k * 8);
	}
	return out;
}

void append_surface(PackedMesh &out, const SurfaceView &s, Custom0Kind custom0_kind, uint32_t extra_flags) {
	if (s.vertex_count == 0 || s.index_count == 0 || s.positions == nullptr || s.indices == nullptr) {
		return;
	}

	if (out.vertices.empty()) {
		for (unsigned int i = 0; i < 3; ++i) {
			out.aabb_min[i] = std::numeric_limits<float>::max();
			out.aabb_max[i] = std::numeric_limits<float>::lowest();
		}
	}

	const uint32_t base = static_cast<uint32_t>(out.vertices.size());
	out.vertices.reserve(out.vertices.size() + s.vertex_count);

	for (uint32_t vi = 0; vi < s.vertex_count; ++vi) {
		PackedVertex v;
		memcpy(v.position, s.positions + vi * 3, sizeof(v.position));
		// Offset 0 unless Transvoxel provides a secondary position
		v.secondary_xy = 0;
		v.secondary_z = 0;
		uint32_t idata = 0;

		if (s.custom0 != nullptr) {
			const float *c0 = s.custom0 + vi * 4;
			if (custom0_kind == CUSTOM0_TRANSVOXEL) {
				v.secondary_xy = float_to_half(c0[0] - v.position[0]) |
						(static_cast<uint32_t>(float_to_half(c0[1] - v.position[1])) << 16);
				v.secondary_z = float_to_half(c0[2] - v.position[2]);
				idata = float_bits(c0[3]);
			} else if (custom0_kind == CUSTOM0_TRANSITION_TAG) {
				idata = (float_bits(c0[3]) & 0xffu) << 16;
			}
		}
		idata |= extra_flags;

		const uint32_t normal = s.normals != nullptr
				? encode_oct14(s.normals[vi * 3 + 0], s.normals[vi * 3 + 1], s.normals[vi * 3 + 2])
				: encode_oct14(0.f, 1.f, 0.f);
		v.normal_flags = compact_flags(idata) | (normal << 18);

		const uint32_t mat_indices = s.custom1_components > 0 ? s.custom1[vi * s.custom1_components + 0] : 0;
		const uint32_t mat_weights = s.custom1_components > 1 ? s.custom1[vi * s.custom1_components + 1] : 0;
		bool clamped = false;
		v.materials = pack_mixel4(mat_indices, mat_weights, clamped);
		out.clamped_materials += clamped ? 1 : 0;

		v.surface = s.custom2 != nullptr ? s.custom2[vi] : 0;
		out.vertices.push_back(v);

		expand_bounds(out, v.position);
		if (custom0_kind == CUSTOM0_TRANSVOXEL) {
			float secondary[3];
			decode_secondary(v, secondary);
			expand_bounds(out, secondary);
		}
	}

	out.indices.reserve(out.indices.size() + s.index_count);
	for (uint32_t ii = 0; ii < s.index_count; ++ii) {
		out.indices.push_back(base + static_cast<uint32_t>(s.indices[ii]));
	}
}

void finalize(PackedMesh &mesh) {
	mesh.index_count = static_cast<uint32_t>(mesh.indices.size());
	mesh.wide_indices = mesh.vertices.size() > 65536;
	if (mesh.wide_indices) {
		mesh.index_words = std::move(mesh.indices);
	} else {
		mesh.index_words.resize((mesh.index_count + 1) / 2, 0);
		for (uint32_t i = 0; i < mesh.index_count; ++i) {
			mesh.index_words[i >> 1] |= mesh.indices[i] << ((i & 1) * 16);
		}
	}
	mesh.indices = std::vector<uint32_t>();
}

} // namespace zylann::voxel::gpu_driven
