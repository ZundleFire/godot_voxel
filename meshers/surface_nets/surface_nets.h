#ifndef VOXEL_SURFACE_NETS_H
#define VOXEL_SURFACE_NETS_H

#include "../../storage/voxel_buffer.h"
#include "../../util/containers/fixed_array.h"
#include "../../util/containers/std_vector.h"
#include "../../util/math/vector2f.h"
#include "../../util/math/vector3f.h"
#include "../../util/math/vector3i.h"

namespace zylann::voxel::surface_nets {

// How many extra voxels are needed towards the negative axes.
// A quad is owned by the block containing its edge's corner, and the 4 cells sharing that edge reach
// one cell back along the two tangent axes, so cells at -1 must be solvable.
static const int MIN_PADDING = 1;
// How many extra voxels are needed towards the positive axes.
// Cells at block_size-1 read their far corner one voxel past the block edge.
static const int MAX_PADDING = 1;

struct MeshArrays {
	StdVector<Vector3f> vertices;
	StdVector<Vector3f> normals;
	// 0 for regular-mesh vertices (always visible). For transition-strip vertices, `1 << direction`
	// (see Cube::Side) -- consumed by the default shader (surface_nets_minimal.gdshader) to hide
	// the strip unless that side's neighbor is actually coarser at render time.
	StdVector<uint8_t> transition_tag;
	// Mixel4 texturing: x = 4 texture indices packed as bytes, y = their 4 blend weights, both
	// reinterpreted as floats. Goes to ARRAY_CUSTOM1, same encoding VoxelMesherTransvoxel uses, so
	// the shaders that read it (including the graph's auto material) work unchanged.
	// Empty when the mesher runs with texturing disabled.
	StdVector<Vector2f> texturing_data;
	StdVector<int32_t> indices;

	void clear() {
		vertices.clear();
		normals.clear();
		transition_tag.clear();
		texturing_data.clear();
		indices.clear();
	}
};

// Scratch buffers reused across builds. `VoxelMesher::build` runs on several threads at once, so
// hold one of these per thread (see VoxelMesherSurfaceNets::build) rather than sharing one.
struct Cache {
	// The whole padded SDF channel decoded once, in the same ZXY order VoxelBuffer stores it in.
	// Sampling through this instead of calling `VoxelBuffer::get_voxel_f` per cell corner is the
	// difference between ~1 and ~14 decodes per voxel.
	StdVector<float> sdf;
	// Index of the vertex emitted by each cell of the regular grid, or -1. Kept here so meshing a
	// block doesn't allocate.
	StdVector<int32_t> cell_vertex;
	// Mixel4 material channels, materialized in the same layout as `sdf` (so a uniform or compressed
	// channel is expanded here once rather than branched on per corner). Only filled, and only
	// consulted, when `materials_enabled`.
	StdVector<uint16_t> material_indices;
	StdVector<uint16_t> material_weights;
	// The 4 texture layers this whole block blends between, ascending, packed one per byte.
	//
	// Chosen once per block rather than per cell on purpose: the shader declares `v_indices` as
	// `flat`, so a triangle uses a single vertex's index set while interpolating all three vertices'
	// weights. Picking per cell lets neighbouring vertices disagree about what slot 2 means, and the
	// mismatch renders as bands of unrelated colour along the boundary between two materials. One set
	// per block makes that impossible. A block needing a 5th layer loses its least-present one, which
	// degrades quietly instead.
	uint32_t material_packed_indices = 0;
	// Layer (0..15) -> slot in the 4 above, or -1 if this block dropped that layer.
	FixedArray<int8_t, 16> material_layer_to_slot;
	bool materials_enabled = false;
};

// Decodes `sdf_channel` of `voxels` into `cache.sdf`. Call once per block, before the build
// functions below that take a `Cache`.
// `with_materials` additionally decodes CHANNEL_INDICES and CHANNEL_WEIGHTS for mixel4 texturing;
// it requires both to be 16-bit and is ignored otherwise.
void fill_cache(const VoxelBuffer &voxels, unsigned int sdf_channel, bool with_materials, Cache &cache);

// Naive Surface Nets: one vertex per SDF-sign-crossing cell, positioned at the average of the
// crossing points of that cell's 12 edges.
// Output positions are relative to the block origin and scaled by `1 << lod_index`, i.e. expressed
// in LOD0 voxel units -- the same convention as VoxelMesherTransvoxel. This matters because
// VoxelLodTerrain gives mesh blocks a translation-only transform (identity basis), so the mesher is
// what has to apply the LOD scale.
void build_regular_mesh(
		const VoxelBuffer &voxels,
		unsigned int lod_index,
		Cache &cache,
		MeshArrays &output
);

// Convenience overload using a temporary cache. Allocates; prefer the one above in hot paths.
void build_regular_mesh(
		const VoxelBuffer &voxels,
		unsigned int sdf_channel,
		unsigned int lod_index,
		MeshArrays &output
);

// Closes LOD seams by extruding the mesh's open boundary edges along -normal, hanging a curtain
// beneath the surface.
//
// Why a curtain and not a stitch: two neighbouring blocks at different LODs produce surfaces that
// disagree by the coarse block's decimation error. On flat ground that error is ~half a cell and a
// stitch ribbon can close it exactly, by resampling the fine block's own padded buffer on the coarse
// grid to reproduce the vertices the coarse neighbour will emit -- which is what this used to do. But
// that reconstruction is only valid when the surface is roughly perpendicular to the shared face; on
// slopes the coarse surface can sit several cells away from where the ribbon looks for it, and the
// ribbon leaves holes exactly where terrain is interesting. The decimation error has no useful bound
// in general, so instead of trying to meet the neighbour the block hides the disagreement.
//
// `depth_in_cells` is in this LOD's cells; deeper hides bigger mismatches but risks poking through
// thin geometry. Boundary edges are the ones used by exactly one triangle, which is the block's outer
// rim plus the rim of any hole naive Surface Nets left behind -- both worth skirting. Because the
// curtain points into matter it's safe to render unconditionally, so unlike the old ribbon it needs
// no shader support and works with any material.
void build_skirts(MeshArrays &output, unsigned int lod_index, float depth_in_cells);

} // namespace zylann::voxel::surface_nets

#endif // VOXEL_SURFACE_NETS_H
