#pragma once

// No Godot headers: compiles standalone against gpu_driven/tests/.

#include <cstdint>
#include <vector>

namespace zylann::voxel::gpu_driven {

// 32 bytes (8 x uint32), decoded by the vertex shader in voxel_gpu_driven_renderer.cpp.
// Positions stay float32, relative to the block origin, exactly as the mesher output them: Transvoxel LOD seams
// need a fine block's transition vertices to land exactly on its coarser neighbor's vertices, and quantizing each
// block on its own LOD grid breaks that (hairline cracks). Everything else is packed.
struct PackedVertex {
	float position[3];
	// Bits 0-5 cell border mask, 6-11 vertex border mask, 12-17 transition sides (Transvoxel idata, compacted),
	// 18-24 / 25-31 octahedral normal x / y as 7-bit snorm
	uint32_t normal_flags;
	// Transvoxel secondary position as a half-float offset from `position`: x | y << 16, then z (high half unused)
	uint32_t secondary_xy;
	uint32_t secondary_z;
	// MIXEL4, one byte per slot: material index (low nibble) | weight 0-15 (high nibble)
	uint32_t materials;
	uint32_t surface; // CUSTOM2 bits (erosion, ridge, moisture, temperature)
};
static_assert(sizeof(PackedVertex) == 32, "Must match shader stride");

// MIXEL4 indices must fit 4 bits
constexpr uint32_t MAX_MATERIAL_INDEX = 15;

enum Custom0Kind {
	CUSTOM0_NONE,
	// xyz = secondary position, w = idata bits (VoxelMesherTransvoxel)
	CUSTOM0_TRANSVOXEL,
	// w = transition tag (1 << side) or 0, xyz unused (VoxelMesherSurfaceNets)
	CUSTOM0_TRANSITION_TAG
};

// Non-owning view of one mesher output surface. Optional arrays may be null.
struct SurfaceView {
	const float *positions = nullptr; // 3 per vertex
	const float *normals = nullptr; // 3 per vertex
	const float *custom0 = nullptr; // 4 per vertex
	const uint32_t *custom1 = nullptr; // custom1_components per vertex
	unsigned int custom1_components = 0; // 0, 1 or 2
	const uint32_t *custom2 = nullptr; // 1 per vertex
	uint32_t vertex_count = 0;
	const int32_t *indices = nullptr;
	uint32_t index_count = 0;
};

struct PackedMesh {
	std::vector<PackedVertex> vertices;
	// Built by append_surface, relative to the start of `vertices`. finalize() moves them into `index_words`.
	std::vector<uint32_t> indices;
	// Two 16-bit indices per word (low half first), or one 32-bit index per word if `wide_indices`
	std::vector<uint32_t> index_words;
	uint32_t index_count = 0;
	// More than 65536 vertices: indices can't be 16-bit
	bool wide_indices = false;
	// Bounds of all primary and secondary positions
	float aabb_min[3] = { 0.f, 0.f, 0.f };
	float aabb_max[3] = { 0.f, 0.f, 0.f };
	// Vertices referencing a material index above MAX_MATERIAL_INDEX (clamped, so rendered with the wrong material)
	uint32_t clamped_materials = 0;

	bool is_empty() const {
		return index_count == 0 && indices.empty();
	}
};

// Appends a surface. `extra_flags` uses the Transvoxel idata layout (bits 16-21 transition sides) and is OR'ed into
// every vertex's flags (used to tag separate transition surfaces with their side bit).
void append_surface(PackedMesh &out, const SurfaceView &surface, Custom0Kind custom0_kind, uint32_t extra_flags);
// Call once after the last append_surface: packs indices into `index_words`.
void finalize(PackedMesh &mesh);

// Octahedral normal, 7-bit snorm per component, as stored in PackedVertex::normal_flags bits 18-31
uint32_t encode_oct14(float x, float y, float z);
void decode_oct14(uint32_t e, float out[3]);

// Transvoxel idata (cell 0-5, vertex border 8-13, transition 16-21) <-> PackedVertex flag bits 0-17
uint32_t compact_flags(uint32_t idata);
// IEEE half, round to nearest even, like GLSL unpackHalf2x16 expects. Overflow saturates to the largest finite half.
uint16_t float_to_half(float f);
float half_to_float(uint16_t h);
// Secondary position decoded the way the shader does
void decode_secondary(const PackedVertex &v, float out[3]);
uint32_t pack_mixel4(uint32_t packed_indices, uint32_t packed_weights, bool &clamped);

} // namespace zylann::voxel::gpu_driven
