#ifndef VOXEL_FAR_SPHERE_H
#define VOXEL_FAR_SPHERE_H

#include "far_column_data.h"
#include "far_math.h"

#include <cstdint>

namespace zylann::voxel::far {

// ---------------------------------------------------------------------------
// Cubesphere parameterisation of a planet surface.
//
// The far field's whole economic argument (README 2.1) is that the surface is a
// 2D field and only its *position along one axis* varies. On a flat world that
// axis is Y and the 2D field is the XZ plane. On a planet the axis is radial and
// the 2D field is the sphere -- nothing else about the representation changes.
//
// What is needed is therefore a way to cover a sphere with square grids, so that
// the existing 33x33 sector, its extractor and its mesher all keep working. Six
// cube faces projected outward do exactly that.
//
// Why not the icosphere in `modules/eden_icosphere`, which has less distortion:
// its tiles are triangles with barycentric grids, and the mesher walks a square
// grid of columns, emitting quads from four corners. Triangular tiles would mean
// rewriting the mesher -- including the sheet-labelling rules in HANDOFF 3.1,
// the part most expensive to get wrong. A cube face keeps the grid square and
// costs only the corner distortion the tangent warp below mostly removes.
//
// Faces meet exactly. Both faces either side of a cube edge subdivide that edge
// into the same 2^k angular steps, because the warp is symmetric and depends
// only on the parameter along the edge. So a column on the shared edge is at the
// same world position from either side, which is the padding invariant of
// README 2.2 carried onto the sphere -- sectors still tile with no seam, and
// meshing still never has to read a neighbour.
// ---------------------------------------------------------------------------

static constexpr unsigned int CUBE_FACE_COUNT = 6;

// Axes of one cube face. `dir = normalize(normal + u * right + v * up)` for
// u, v in [-1, 1] -- and the formula stays meaningful outside that range, which
// is what makes `remap_sector_across_faces()` able to just re-project.
struct CubeFaceAxes {
	Vec3f right;
	Vec3f up;
	Vec3f normal;
};

const CubeFaceAxes &get_cube_face_axes(unsigned int face);

// Tangent warp.
//
// Without it, a face's grid lines are evenly spaced on the *plane*, so once
// projected the cells near a face corner subtend about 1.7x the angle of the
// ones at its centre. Cell size is what the whole LOD scheme is expressed in, so
// that ratio propagates into everything. Warping the parameter through tan()
// spaces the grid lines evenly in *angle* instead, which brings the worst-case
// ratio down to about 1.3 for one tanf per axis.
float warp_face_param(float s);
float unwarp_face_param(float t);

// Direction on the unit sphere for a face parameter pair in [-1, 1].
Vec3f face_uv_to_dir(unsigned int face, float u, float v);

// Inverse. `dir` need not be normalised.
void dir_to_face_uv(const Vec3f &dir, unsigned int &out_face, float &out_u, float &out_v);

// Orthonormal frame at a surface point: `out_radial` is the outward normal,
// `out_tangent_u` and `out_tangent_v` follow the face's u and v.
//
// The extractor needs this because it computes its gradient in slab index space.
// On a flat world those axes are world X/Y/Z and the gradient is already a world
// normal; on a sphere they are this frame, and the gradient has to be rotated
// through it or every normal comes out wrong.
void face_uv_basis(unsigned int face, float u, float v, Vec3f &out_tangent_u, Vec3f &out_tangent_v,
		Vec3f &out_radial);

// Moves sector coordinates that have run off the edge of a face onto whichever
// face they actually belong to. `n` is the number of sectors along a face side.
//
// Done by re-projecting rather than from an adjacency table: the out-of-range
// coordinate still names a real direction through the face's own axes, so
// converting it to a direction and back through `dir_to_face_uv` lands on the
// right face with no table to get wrong. It is also self-consistent by
// construction, which a hand-written 24-entry table of edge rotations is not.
//
// Accurate to any depth, not just one cell. The two ends of the shared edge are
// re-projected one step out -- which is exact, because past the edge the tangent
// warp matches the neighbour's grid to first order -- and the rest is integer
// arithmetic on the neighbour's own grid. Extending a face's parameterisation
// several cells past its edge and re-projecting that does NOT work: the match is
// only first order, so by a few cells out it drifts enough for two cells to land
// on one.
//
// A coordinate off the edge in both axes is past a cube corner, where three
// faces meet rather than four. That is handled as two hops, and returns false
// only when it really names nothing.
bool remap_sector_across_faces(unsigned int face, int32_t x, int32_t z, int32_t n, unsigned int &out_face,
		int32_t &out_x, int32_t &out_z);

// Number of sectors along one cube face edge at `lod`. `root_lod` is the level
// at which a single sector covers a whole face.
//
// Lives here rather than in far_geometry.h so that the LOD tree can use it
// without pulling in SectorGeometry, which needs SectorId and would make the
// include graph circular.
inline int32_t sectors_per_face_side(uint8_t lod, uint8_t root_lod) {
	return lod >= root_lod ? 1 : (int32_t(1) << (root_lod - lod));
}

// Smallest `root_lod` whose LOD-0 cells are at most `target_cell_size` across,
// for a planet of the given radius.
//
// This is what ties the sphere's LOD numbering to the flat one: `first_lod`
// keeps meaning "cells are about 2^first_lod units across", so the tuning advice
// in HANDOFF step 3 carries over unchanged.
uint8_t choose_sphere_root_lod(float planet_radius, float target_cell_size);

} // namespace zylann::voxel::far

#endif // VOXEL_FAR_SPHERE_H
