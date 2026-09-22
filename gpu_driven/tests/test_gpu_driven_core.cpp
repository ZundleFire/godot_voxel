// Host tests for gpu_driven/core (no Godot). Run gpu_driven/tests/run_tests.bat.
#include "../core/gpu_range_allocator.h"
#include "../core/gpu_vertex_pack.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace zylann::voxel::gpu_driven;

#define CHECK(cond)                                                                                                    \
	do {                                                                                                               \
		if (!(cond)) {                                                                                                 \
			std::printf("FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond);                                             \
			std::exit(1);                                                                                              \
		}                                                                                                              \
	} while (0)

static bool near(float a, float b, float eps) {
	return std::fabs(a - b) <= eps;
}

static float bitsf(uint32_t u) {
	float f;
	std::memcpy(&f, &u, 4);
	return f;
}

static bool same_bits(float a, float b) {
	return std::memcmp(&a, &b, sizeof(float)) == 0;
}

static uint32_t flags_of(const PackedVertex &v) {
	return v.normal_flags & 0x3ffffu;
}

static void normal_of(const PackedVertex &v, float out[3]) {
	decode_oct14(v.normal_flags >> 18, out);
}

static void test_oct_normal() {
	std::mt19937 rng(1234);
	std::uniform_real_distribution<float> d(-1.f, 1.f);
	float worst_dot = 1.f;
	for (int i = 0; i < 20000; ++i) {
		float n[3] = { d(rng), d(rng), d(rng) };
		const float len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
		if (len < 1e-3f) {
			continue;
		}
		n[0] /= len;
		n[1] /= len;
		n[2] /= len;
		float r[3];
		decode_oct14(encode_oct14(n[0], n[1], n[2]), r);
		const float dp = n[0] * r[0] + n[1] * r[1] + n[2] * r[2];
		if (dp < worst_dot) {
			worst_dot = dp;
		}
	}
	// 7+7 bit octahedral: worst case error around 2-3 degrees
	CHECK(worst_dot > std::cos(4.f * 3.14159265f / 180.f));
	CHECK(encode_oct14(0.3f, -0.9f, 0.1f) < (1u << 14));

	float r[3];
	decode_oct14(encode_oct14(0, 0, -1), r);
	CHECK(r[2] < -0.999f);
	decode_oct14(encode_oct14(0, 1, 0), r);
	CHECK(r[1] > 0.999f);
	decode_oct14(encode_oct14(0, 0, 0), r);
	CHECK(!std::isnan(r[0]) && !std::isnan(r[1]) && !std::isnan(r[2]));
}

static void test_half() {
	// Every non-NaN half survives half -> float -> half
	for (uint32_t h = 0; h < 0x10000; ++h) {
		const float f = half_to_float(static_cast<uint16_t>(h));
		if (std::isnan(f)) {
			continue;
		}
		if ((h & 0x7fffu) == 0x7c00u) {
			// Infinity saturates to the largest finite half by design
			CHECK(float_to_half(f) == static_cast<uint16_t>((h & 0x8000u) | 0x7bffu));
			continue;
		}
		CHECK(float_to_half(f) == h);
	}
	// Rounding to nearest: error within half an ulp
	std::mt19937 rng(7);
	std::uniform_real_distribution<float> d(-3000.f, 3000.f);
	for (int i = 0; i < 100000; ++i) {
		const float f = d(rng);
		const float r = half_to_float(float_to_half(f));
		CHECK(std::fabs(r - f) <= std::fabs(f) / 2048.f + 1e-7f);
	}
	CHECK(half_to_float(float_to_half(0.f)) == 0.f);
	CHECK(half_to_float(float_to_half(1e9f)) == 65504.f);
	CHECK(half_to_float(float_to_half(-1e9f)) == -65504.f);
}

static void test_flags() {
	// idata: cell border 0x21, vertex border 0x14, transition 0x3f
	CHECK(compact_flags(0x003f1421) == (0x21u | (0x14u << 6) | (0x3fu << 12)));
	CHECK(compact_flags(0) == 0);
	// Flags never reach the normal bits
	CHECK(compact_flags(0xffffffff) < (1u << 18));
}

static void test_pack_transvoxel() {
	// Two triangles with Transvoxel CUSTOM0. Positions use values with long mantissas: they must survive packing
	// bit-for-bit, since LOD seams rely on neighbor blocks producing identical vertices.
	const float positions[] = { 0.1f, 0.2f, 0.3f, 8.f, 0.f, 0.f, 0.f, 4.f, 64.f, 60.f, 60.f, 16.000002f };
	const float normals[] = { 0, 1, 0, 0, 1, 0, 0, 0, -1, 1, 0, 0 };
	const uint32_t idata0 = 0x00020301; // transition side bits 0x02, vertex border 0x03, cell border 0x01
	float custom0[16] = {
		1.3f, 2.f, 3.f, bitsf(idata0), //
		0, 0, 0, bitsf(0), //
		0, 0, 0, bitsf(0), //
		-4, 68, 8, bitsf(0x3f), //
	};
	// 2 components per vertex: packed indices, packed weights
	const uint32_t custom1[] = { 0x04030201, 0xff000000, 1, 2, 3, 0x00000080, 5, 6 };
	const uint32_t custom2[] = { 10, 11, 12, 13 };
	const int32_t indices[] = { 0, 1, 2, 1, 3, 2 };

	SurfaceView s;
	s.positions = positions;
	s.normals = normals;
	s.custom0 = custom0;
	s.custom1 = custom1;
	s.custom1_components = 2;
	s.custom2 = custom2;
	s.vertex_count = 4;
	s.indices = indices;
	s.index_count = 6;

	PackedMesh m;
	append_surface(m, s, CUSTOM0_TRANSVOXEL, 0);
	CHECK(m.vertices.size() == 4);
	CHECK(m.indices.size() == 6);

	for (int vi = 0; vi < 4; ++vi) {
		float secondary[3];
		decode_secondary(m.vertices[vi], secondary);
		for (int c = 0; c < 3; ++c) {
			CHECK(same_bits(m.vertices[vi].position[c], positions[vi * 3 + c]));
			// Half-float offset: relative error of the offset around 1/2048
			const float offset = custom0[vi * 4 + c] - positions[vi * 3 + c];
			CHECK(near(secondary[c], custom0[vi * 4 + c], std::fabs(offset) / 1024.f + 1e-6f));
		}
	}

	const PackedVertex &v0 = m.vertices[0];
	CHECK(flags_of(v0) == compact_flags(idata0));
	// Slot 3: index 4 weight 15, others weight 0
	CHECK(v0.materials == ((0x01u) | (0x02u << 8) | (0x03u << 16) | ((0x04u | (15u << 4)) << 24)));
	// Vertex 2: index 3, weight 0x80 in slot 0 -> 8
	CHECK((m.vertices[2].materials & 0xffu) == (3u | (8u << 4)));
	// Vertex 3: index 5, weight 6 rounds to 0 in 4 bits, kept at 1 as the dominant material
	CHECK((m.vertices[3].materials & 0xffu) == (5u | (1u << 4)));
	CHECK(v0.surface == 10 && m.vertices[3].surface == 13);
	CHECK(m.clamped_materials == 0);

	float n[3];
	normal_of(m.vertices[2], n);
	CHECK(n[2] < -0.99f);

	// Bounds include the secondary positions (vertex 3 secondary: -4, 68, 8)
	CHECK(near(m.aabb_min[0], -4.f, 1e-5f));
	CHECK(near(m.aabb_max[1], 68.f, 1e-5f));
	CHECK(near(m.aabb_max[2], 64.f, 1e-5f));
	CHECK(near(m.aabb_min[1], 0.f, 1e-5f));

	// Second surface: indices are rebased, extra flags are OR'ed
	PackedMesh m2 = m;
	append_surface(m2, s, CUSTOM0_TRANSVOXEL, 0x00100000);
	CHECK(m2.vertices.size() == 8);
	CHECK(m2.indices[6] == 4 && m2.indices[11] == 6);
	CHECK(flags_of(m2.vertices[5]) == (0x10u << 12));
	CHECK(flags_of(m2.vertices[4]) == compact_flags(idata0 | 0x00100000));

	// 16-bit index packing: 12 indices -> 6 words, low half first
	finalize(m2);
	CHECK(!m2.wide_indices);
	CHECK(m2.index_count == 12);
	CHECK(m2.index_words.size() == 6);
	CHECK(m2.index_words[0] == (0u | (1u << 16)));
	CHECK(m2.index_words[3] == (4u | (5u << 16)));
	CHECK(m2.indices.empty() && !m2.is_empty());
}

static void test_materials_edge_cases() {
	bool clamped = false;
	// Tiny weight that rounds to 0 in 4 bits: the dominant material is kept
	uint32_t m = pack_mixel4(0x00000005, 0x00000003, clamped);
	CHECK((m & 0xffu) == (5u | (1u << 4)));
	CHECK(!clamped);
	// Index above 15 with weight: clamped and reported
	m = pack_mixel4(0x00000014, 0x000000ff, clamped);
	CHECK(clamped);
	CHECK((m & 0x0fu) == 15u);
	// Index above 15 but unused (weight 0) is not reported
	clamped = false;
	pack_mixel4(0x14000000, 0x000000ff, clamped);
	CHECK(!clamped);
	// No material data at all stays zero (shader falls back to neutral)
	CHECK(pack_mixel4(0, 0, clamped) == 0);
}

static void test_wide_indices() {
	// More than 65536 vertices: indices stay 32-bit
	std::vector<float> positions(70000 * 3, 0.f);
	const int32_t indices[] = { 0, 69999, 65536 };
	SurfaceView s;
	s.positions = positions.data();
	s.vertex_count = 70000;
	s.indices = indices;
	s.index_count = 3;
	PackedMesh m;
	append_surface(m, s, CUSTOM0_NONE, 0);
	finalize(m);
	CHECK(m.wide_indices);
	CHECK(m.index_words.size() == 3 && m.index_words[1] == 69999 && m.index_words[2] == 65536);

	// Odd index count with 16-bit: last word's high half is padding
	PackedMesh small;
	const float p3[] = { 0, 0, 0, 1, 0, 0, 0, 1, 0 };
	const int32_t i3[] = { 2, 1, 0 };
	SurfaceView s3;
	s3.positions = p3;
	s3.vertex_count = 3;
	s3.indices = i3;
	s3.index_count = 3;
	append_surface(small, s3, CUSTOM0_NONE, 0);
	finalize(small);
	CHECK(small.index_words.size() == 2 && small.index_words[1] == 0u);
	CHECK(small.index_words[0] == (2u | (1u << 16)));
}

static void test_pack_surface_nets_and_minimal() {
	const float positions[] = { 0, 0, 0, 1, 0, 0, 0, 1, 0 };
	float custom0[12] = { 0, 0, 0, bitsf(0), 0, 0, 0, bitsf(4), 0, 0, 0, bitsf(0) };
	const uint32_t custom1[] = { 7, 8, 9 }; // 1 component: indices only, weights 0
	const int32_t indices[] = { 0, 1, 2 };

	SurfaceView s;
	s.positions = positions;
	s.custom0 = custom0;
	s.custom1 = custom1;
	s.custom1_components = 1;
	s.vertex_count = 3;
	s.indices = indices;
	s.index_count = 3;

	PackedMesh m;
	append_surface(m, s, CUSTOM0_TRANSITION_TAG, 0);
	// Transition tag moves to the transition bits, no cell/vertex border bits
	CHECK(flags_of(m.vertices[1]) == (4u << 12));
	CHECK(flags_of(m.vertices[0]) == 0);
	// Secondary position defaults to primary (zero offset)
	CHECK(m.vertices[1].secondary_xy == 0 && m.vertices[1].secondary_z == 0);
	CHECK((m.vertices[2].materials & 0xffu) == 9u);
	CHECK(m.vertices[0].surface == 0);
	// No normals: defaults to +Y
	float n[3];
	normal_of(m.vertices[0], n);
	CHECK(n[1] > 0.99f);

	// Empty surfaces are ignored and leave the mesh empty
	PackedMesh e;
	SurfaceView empty;
	append_surface(e, empty, CUSTOM0_NONE, 0);
	finalize(e);
	CHECK(e.is_empty() && e.vertices.empty());
}

static void test_allocator_basic() {
	RangeAllocator a;
	CHECK(a.allocate(1) == RangeAllocator::INVALID);
	a.grow(100);
	CHECK(a.allocate(0) == RangeAllocator::INVALID);
	const uint32_t x = a.allocate(30);
	const uint32_t y = a.allocate(30);
	const uint32_t z = a.allocate(30);
	CHECK(x == 0 && y == 30 && z == 60);
	CHECK(a.allocate(20) == RangeAllocator::INVALID);
	CHECK(a.get_used() == 90);

	a.free(y, 30);
	CHECK(a.allocate(31) == RangeAllocator::INVALID);
	a.free(x, 30);
	// x and y coalesce into one 60-unit range
	CHECK(a.allocate(55) == 0);
	a.free(0, 55);
	a.free(z, 30);
	// Everything free again, one range
	CHECK(a.get_free_range_count() == 1);
	CHECK(a.get_used() == 0);

	// Grow merges with a trailing free range
	const uint32_t w = a.allocate(90);
	CHECK(w == 0);
	a.grow(200);
	CHECK(a.get_free_range_count() == 1);
	CHECK(a.allocate(110) == 90);
}

static void test_allocator_stress() {
	// Compare against a brute-force occupancy map
	const uint32_t cap = 4096;
	RangeAllocator a;
	a.grow(cap);
	std::vector<uint8_t> occupied(cap, 0);
	struct Block {
		uint32_t offset, size;
	};
	std::vector<Block> live;
	std::mt19937 rng(42);

	for (int it = 0; it < 20000; ++it) {
		const bool do_alloc = live.empty() || (rng() % 100) < 55;
		if (do_alloc) {
			const uint32_t size = 1 + rng() % 64;
			const uint32_t off = a.allocate(size);
			if (off == RangeAllocator::INVALID) {
				continue;
			}
			CHECK(off + size <= cap);
			for (uint32_t i = off; i < off + size; ++i) {
				CHECK(occupied[i] == 0);
				occupied[i] = 1;
			}
			live.push_back({ off, size });
		} else {
			const size_t k = rng() % live.size();
			const Block b = live[k];
			live[k] = live.back();
			live.pop_back();
			a.free(b.offset, b.size);
			for (uint32_t i = b.offset; i < b.offset + b.size; ++i) {
				occupied[i] = 0;
			}
		}
		uint32_t used = 0;
		for (const Block &b : live) {
			used += b.size;
		}
		CHECK(a.get_used() == used);
	}
	for (const Block &b : live) {
		a.free(b.offset, b.size);
	}
	CHECK(a.get_free_range_count() == 1);
	CHECK(a.allocate(cap) == 0);
}

int main() {
	test_oct_normal();
	test_half();
	test_flags();
	test_pack_transvoxel();
	test_materials_edge_cases();
	test_wide_indices();
	test_pack_surface_nets_and_minimal();
	test_allocator_basic();
	test_allocator_stress();
	std::printf("gpu_driven core: all tests passed\n");
	return 0;
}
