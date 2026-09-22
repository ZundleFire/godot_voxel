#ifndef VOXEL_FAR_MESHER_H
#define VOXEL_FAR_MESHER_H

#include "far_column_data.h"
#include "far_math.h"

#include <vector>

namespace zylann::voxel::far {

// ---------------------------------------------------------------------------
// Turning column data into triangles.
//
// The naive reading of a column field is "a heightmap", and meshing a heightmap
// is trivial. That reading is wrong in two ways that matter, and handling both
// is most of what this file does.
//
// 1. A column can hold several solid spans. Connecting column tops blindly
//    would stitch the roof of a cave to the floor of the one next to it.
//
// 2. Even with one span per column, a cliff is a near-vertical surface. Joining
//    the two columns on either side of it with a quad produces a stretched ramp,
//    which at distance reads as terrain melting downhill -- the single most
//    recognisable artifact of naive far-LOD terrain.
//
// The fix for both is the same: decide which spans across the grid belong to
// the *same continuous surface*, and only connect those. Surfaces are labelled
// with a union-find pass over adjacent columns, where two spans join only if
// they are mutually each other's closest match. That mutual test is what keeps
// a cave floor attached to the floor next door rather than to the ceiling.
//
// Problem 2 then resolves without a slope threshold. A cliff in an SDF is a
// continuous steep surface, so it stays one sheet and is drawn as a steep quad
// spanning a single cell -- and because each layer carries the normal it was
// captured with, that quad still shades like a vertical face. One cell is the
// horizontal resolution at that LOD, so no sharper representation existed.
// The "melting downhill" artifact comes from ramps spanning *many* cells, which
// cannot happen here.
//
// Walls are reserved for discontinuities in layer *structure*: the lip of an
// overhang, the edge of a floating island, a cave mouth -- places where a sheet
// simply stops and there is nothing on the other side to connect to. See
// `max_connect_gap_cells` for why making walls slope-driven instead is a trap.
//
// Winding: Godot treats clockwise-from-the-front as front-facing, which means
// the right-hand-rule normal of an emitted triangle points *opposite* the
// surface normal. This was checked against the cube tables in godot_voxel
// rather than assumed. Caps use hardcoded orders derived from that; walls and
// skirts go through `emit_quad_oriented`, which picks the order from the
// desired normal so it cannot be got wrong.
// ---------------------------------------------------------------------------

struct FarVertex {
	// Sector-local position. Keeping vertices local and putting the sector
	// origin in the instance transform is what keeps precision usable at the
	// distances this module exists to render -- absolute coordinates in the
	// hundreds of thousands quantise visibly in 32-bit floats.
	Vec3f position;
	Vec3f normal;
	// Baked ambient occlusion, 0..1.
	float ao = 1.f;
	// Material index as a float, for the shader to look up in a palette.
	float material = 0.f;
};

struct FarMeshArrays {
	std::vector<FarVertex> vertices;
	std::vector<uint32_t> indices;

	Vec3f aabb_min;
	Vec3f aabb_max;
	bool has_bounds = false;

	// Counts kept separately for profiling / tuning the tolerances.
	uint32_t cap_triangles = 0;
	uint32_t wall_triangles = 0;
	uint32_t skirt_triangles = 0;

	void clear();

	inline bool is_empty() const {
		return indices.empty();
	}
};

struct SectorGeometry;

struct MesherSettings {
	// World units between adjacent columns. Equals `1 << lod`.
	float cell_size = 1.f;

	// Where this sector's columns are in the world. Null means the original
	// flat interpretation: the grid is the XZ plane and the column axis is +Y.
	// Non-null lets the same mesher run on a planet, where the grid is a cube
	// face and the column axis is radial -- see far_geometry.h.
	const SectorGeometry *geometry = nullptr;

	// How far apart two adjacent spans may be, in cells, and still be treated as
	// the same continuous surface.
	//
	// This is a guard against nonsense pairings, NOT a slope limit, and the
	// difference matters enough to be worth spelling out.
	//
	// The obvious design is to make this a slope threshold: connect gentle
	// neighbours into a surface, and draw a vertical wall wherever the slope is
	// too steep. That is right for cube worlds, where every surface really is
	// either flat or a cliff. Applied to an SDF it fails badly, and not
	// gracefully: on a *sustained* steep slope no adjacent pair connects, so
	// every span ends up in a sheet of its own, no cell ever has four corners
	// agreeing on a sheet, and the mesher emits nothing at all. A whole
	// mountainside turns into a hole with a skirt around it.
	//
	// So connection is deliberately permissive. A steep-but-continuous surface
	// stays one sheet and is drawn as a steep quad, which is the correct
	// reading of the data: the SDF there really is a continuous steep surface.
	// Crucially, the stored per-layer normals survive, so a near-vertical face
	// still shades like a cliff even though its geometry spans one cell
	// horizontally -- and one cell is the resolution limit at that LOD anyway,
	// so nothing sharper was available.
	//
	// What this value actually prevents is connecting across a genuine void:
	// ground at y=20 in one column and a floating island at y=900 in the next
	// should not be joined into a single sheet stretched between them. The
	// mutual-nearest-match rule already separates a cave floor from its ceiling
	// on its own, since each is the other's nearest counterpart; this is the
	// backstop for the case where a column has no counterpart at all.
	//
	// Discontinuities in *layer structure* -- the lip of an overhang, the edge
	// of an island, a cave mouth -- are still found and still get walls. Those
	// are the cases where a wall is genuinely the right primitive.
	float max_connect_gap_cells = 64.f;

	// Emit vertical faces at surface discontinuities.
	bool generate_walls = true;

	// Emit downward-facing caps under overhangs and floating geometry. Costs
	// triangles for something the camera rarely sees at distance; worth turning
	// off for worlds without much overhang.
	bool generate_bottoms = true;

	// A wall with nothing below it would otherwise run to the bottom of the
	// sampled range. Clamped to this many cells.
	float max_wall_depth_cells = 64.f;

	// Emit a downward apron around the sector border. This is what hides the
	// cracks at boundaries between different LOD levels, where no amount of
	// padding can make the two sides agree. Cheap (a few hundred triangles) and
	// invisible when the neighbour happens to be at the same level.
	bool generate_skirts = true;
	float skirt_depth_cells = 2.0f;
};

// Builds the mesh for one sector. `out` is cleared first.
void build_far_mesh(const ColumnField &field, const MesherSettings &settings, FarMeshArrays &out);

} // namespace zylann::voxel::far

#endif // VOXEL_FAR_MESHER_H
