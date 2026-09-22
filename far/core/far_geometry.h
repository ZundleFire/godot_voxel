#ifndef VOXEL_FAR_GEOMETRY_H
#define VOXEL_FAR_GEOMETRY_H

#include "far_column_data.h"
#include "far_lod_tree.h"
#include "far_math.h"
#include "far_sphere.h"

namespace zylann::voxel::far {

// ---------------------------------------------------------------------------
// Where one sector's columns actually are in the world.
//
// This is the whole of what changes between a flat world and a planet. The
// column format, the extractor, the mesher, the LOD rings and the cache all
// work in sector-grid coordinates and a scalar distance along the column axis;
// none of them care what that axis means. Put the meaning in one place and the
// rest keeps working unchanged:
//
//   flat    column axis is +Y      grid is the XZ plane
//   sphere  column axis is radial  grid is a cube face projected outward
//
// Precomputed per column rather than recomputed at each use. A sector has 1089
// columns and both the extractor and the mesher walk them several times, so the
// ~39 KB this costs during a task is cheaper than repeating six trig calls per
// column per pass.
// ---------------------------------------------------------------------------

struct SectorGeometry {
	bool spherical = false;

	// Flat only: world units between adjacent columns. Equals `1 << lod`.
	float cell_size = 1.f;

	// Spherical only: planet centre in world space.
	Vec3f planet_centre;

	// Vertices are emitted relative to this point, and the renderer puts it in
	// the mesh instance's transform. At 200 km out, absolute float coordinates
	// resolve to about 16 units, which would make the far terrain visibly crawl
	// as the camera moves; sector-local coordinates stay in the thousands
	// however far away the sector is.
	Vec3f local_origin;

	// Outward direction and surface frame for each column, indexed
	// `z * SECTOR_RES + x`. Only filled when spherical.
	Vec3f dir[SECTOR_COLUMN_COUNT];
	Vec3f tangent_u[SECTOR_COLUMN_COUNT];
	Vec3f tangent_v[SECTOR_COLUMN_COUNT];

	void build_flat(const SectorId &id, float p_cell_size);

	// `sectors_per_side` is how many sectors cover one cube face edge at this
	// sector's LOD. `reference_radius` positions `local_origin` and should be
	// somewhere in the middle of the terrain's radial range.
	void build_spherical(
			const SectorId &id,
			int32_t sectors_per_side,
			const Vec3f &p_planet_centre,
			float reference_radius
	);

	inline unsigned int column_index(unsigned int x, unsigned int z) const {
		return z * SECTOR_RES + x;
	}

	// `h` is the value stored in a column layer: a world Y when flat, a radius
	// from the planet centre when spherical. Result is relative to
	// `local_origin`.
	inline Vec3f position(unsigned int x, unsigned int z, float h) const {
		if (!spherical) {
			return Vec3f(static_cast<float>(x) * cell_size, h, static_cast<float>(z) * cell_size);
		}
		const unsigned int i = column_index(x, z);
		return planet_centre + dir[i] * h - local_origin;
	}

	// Rotates a gradient expressed in slab index space (x, y, z) into world
	// space. On a flat world the slab axes already are world axes and this is
	// the identity; on a sphere they are the column's surface frame, and
	// skipping this leaves every normal wrong.
	inline Vec3f slab_gradient_to_world(unsigned int x, unsigned int z, const Vec3f &g) const {
		if (!spherical) {
			return g;
		}
		const unsigned int i = column_index(x, z);
		return tangent_u[i] * g.x + dir[i] * g.y + tangent_v[i] * g.z;
	}

	// Outward axis at a column: +Y when flat, radial when spherical. Used by the
	// mesher for skirt and wall directions.
	inline Vec3f up_at(unsigned int x, unsigned int z) const {
		return spherical ? dir[column_index(x, z)] : Vec3f(0.f, 1.f, 0.f);
	}
};

// Rotates every extracted normal from slab index space into world space.
//
// The extractor takes its gradient from index-space neighbours. On a flat world
// those axes are world X/Y/Z and the result is already a world normal; on a
// sphere they are each column's surface frame, and without this every normal
// points somewhere arbitrary. No-op when `geom` is flat.
void rotate_stack_normals_to_world(ColumnStack *stacks, const SectorGeometry &geom);

// Arc length one sector spans at `radius`. This is the sphere's answer to
// `SectorId::get_world_size()`, and dividing it by SECTOR_CELLS gives the cell
// size every tolerance in the extractor and mesher is expressed in.
float arc_length_of_sector(const SectorId &id, uint8_t root_lod, float radius);

} // namespace zylann::voxel::far

#endif // VOXEL_FAR_GEOMETRY_H
