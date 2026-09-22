#ifndef VOXEL_FAR_EXTRACT_H
#define VOXEL_FAR_EXTRACT_H

#include "far_column_data.h"
#include "far_math.h"

namespace zylann::voxel::far {

// ---------------------------------------------------------------------------
// Turning a signed distance field into column data.
//
// This is the step that makes the whole approach work for smooth terrain. The
// cube-voxel version of this idea only has to run-length encode a boolean
// (solid / not solid) down each column. With an SDF the same walk becomes:
// track the sign, and every time it flips, interpolate the exact crossing and
// sample the gradient there.
//
// The result is strictly more information than the cube version carries: a
// continuous surface height plus a true normal, rather than a snapped integer
// height whose normal has to be guessed from neighbours later.
// ---------------------------------------------------------------------------

// A vertical slab of SDF samples covering one sector.
//
// Layout is column-major on purpose: `index = (z * res_xz + x) * height + y`.
// Extraction's inner loop walks a single column top to bottom, so putting Y
// contiguous makes that walk a linear scan instead of a strided one. The copy
// out of a VoxelBuffer pays for the transpose once, and it is a streaming read.
struct SdfSlab {
	const float *sdf = nullptr;
	// Optional. Same layout and extent as `sdf`. When null every layer gets
	// material 0.
	const uint16_t *material = nullptr;

	unsigned int res_xz = SECTOR_RES;
	// Number of samples along Y.
	unsigned int height = 0;

	// World-space Y of sample index 0.
	float origin_y = 0.f;
	// World units between adjacent samples on every axis. Equals `1 << lod`.
	float step = 1.f;

	inline size_t index(unsigned int x, unsigned int y, unsigned int z) const {
		return (static_cast<size_t>(z) * res_xz + x) * height + y;
	}

	inline float at(unsigned int x, unsigned int y, unsigned int z) const {
		return sdf[index(x, y, z)];
	}

	inline float world_y(unsigned int y) const {
		return origin_y + static_cast<float>(y) * step;
	}

	inline bool is_valid() const {
		return sdf != nullptr && height >= 2 && res_xz == SECTOR_RES;
	}
};

struct ExtractSettings {
	// Spans separated by less than this are merged. Scaled by cell size by the
	// caller; the default corresponds to "less than a quarter of a cell".
	float min_gap = 0.f;
	// Spans thinner than this are dropped (except the topmost).
	float min_thickness = 0.f;
	// Whether to compute the cheap screen-independent AO term.
	bool compute_ao = true;
	// Strength of the AO term, 0 disables its effect without skipping the pass.
	float ao_strength = 1.f;
};

// Extracts one sector's worth of column stacks from an SDF slab.
// `out_stacks` must point to at least SECTOR_COLUMN_COUNT stacks.
// Returns the number of non-empty columns.
unsigned int extract_columns_from_sdf(const SdfSlab &slab, const ExtractSettings &settings, ColumnStack *out_stacks);

// Computes an ambient occlusion term for the top surface of each stack, from
// the height differences against the 8 neighbouring columns.
//
// Doing this at extraction time rather than at mesh time is deliberate: here the
// neighbourhood is still at the resolution the data was captured at. Once the
// field has been downsampled a few times, the height differences that produce
// the occlusion have been averaged away, and the same computation would return
// a nearly flat result. Baking it in and carrying it through the downsampler
// keeps valleys and cliff bases dark at every LOD.
void compute_column_ao(ColumnStack *stacks, unsigned int res_xz, float cell_size, float strength);

// ---------------------------------------------------------------------------
// Downsampling
// ---------------------------------------------------------------------------

// Merges a weighted neighbourhood of column stacks into one.
//
// Regenerating each LOD from the source generator instead would be simpler, but
// it costs a full generator evaluation per level and -- worse -- lets the levels
// disagree. A generator with any high-frequency component sampled at stride 2
// is not the average of itself sampled at stride 1, so independently generated
// levels produce surfaces that do not line up, and every LOD transition pops.
// Reducing downward guarantees the coarse level is a consistent reduction of
// the fine one.
//
// Layers whose tops agree within `merge_tolerance` collapse into one; layers
// further apart survive separately, which is what keeps a cliff running through
// the neighbourhood from being smeared into a ramp.
void downsample_column_group(
		const ColumnStack *stacks,
		const float *weights,
		unsigned int count,
		float merge_tolerance,
		ColumnStack &out_stack
);

// Downsamples a full field. `src_fields` are the four child sectors in
// [qz * 2 + qx] order (that is: -X-Z, +X-Z, -X+Z, +X+Z). Any of them may be
// null, in which case that quadrant contributes nothing.
//
// `cell_size` is the cell size of the *destination* level; it scales the
// tolerances, because what counts as "the same surface" has to widen as the
// columns get further apart.
//
// Because sectors are point grids, the destination column at (dx, dz) sits
// exactly on the source column at (dx * 2, dz * 2) of the combined 2N+1 grid --
// no half-cell shift. Interior columns get a 3x3 tent filter centred on that
// sample to suppress aliasing; the shared border ring is point-sampled instead,
// so that two adjacent destination sectors computing the same border column
// from their own (different) source groups still arrive at identical values.
// Filtering the border would make them disagree and open a crack.
void downsample_field(const ColumnField *const *src_fields, float cell_size, ColumnField &out_field);

} // namespace zylann::voxel::far

#endif // VOXEL_FAR_EXTRACT_H
