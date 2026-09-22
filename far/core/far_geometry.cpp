#include "far_geometry.h"
#include "far_sphere.h"

namespace zylann::voxel::far {

void SectorGeometry::build_flat(const SectorId &id, float p_cell_size) {
	spherical = false;
	cell_size = p_cell_size;
	// Flat sectors keep the original convention: vertices are relative to the
	// sector's minimum corner, and the renderer adds that corner back.
	local_origin = Vec3f(id.get_world_x(), 0.f, id.get_world_z());
}

void SectorGeometry::build_spherical(
		const SectorId &id,
		int32_t sectors_per_side,
		const Vec3f &p_planet_centre,
		float reference_radius
) {
	spherical = true;
	planet_centre = p_planet_centre;

	if (sectors_per_side < 1) {
		sectors_per_side = 1;
	}

	// Parameter span of this sector within its face. Columns run to SECTOR_RES,
	// one past SECTOR_CELLS, so the last column of one sector lands exactly on
	// the first column of its neighbour -- the padding invariant of README 2.2,
	// and it holds across a face boundary too because both faces subdivide the
	// shared edge identically.
	const float inv_side = 1.f / static_cast<float>(sectors_per_side);
	const float u0 = static_cast<float>(id.x) * 2.f * inv_side - 1.f;
	const float v0 = static_cast<float>(id.z) * 2.f * inv_side - 1.f;
	const float du = 2.f * inv_side / static_cast<float>(SECTOR_CELLS);

	for (unsigned int z = 0; z < SECTOR_RES; ++z) {
		const float v = v0 + static_cast<float>(z) * du;
		for (unsigned int x = 0; x < SECTOR_RES; ++x) {
			const float u = u0 + static_cast<float>(x) * du;
			const unsigned int i = column_index(x, z);
			face_uv_basis(id.face, u, v, tangent_u[i], tangent_v[i], dir[i]);
		}
	}

	// Centre of the sector's surface patch. Using the middle column rather than
	// a corner keeps the local coordinates emitted by `position()` as small as
	// possible, which is the entire point of having a local origin.
	const unsigned int centre_index = column_index(SECTOR_CELLS / 2, SECTOR_CELLS / 2);
	local_origin = planet_centre + dir[centre_index] * reference_radius;
}

void rotate_stack_normals_to_world(ColumnStack *stacks, const SectorGeometry &geom) {
	if (!geom.spherical) {
		return;
	}
	for (unsigned int z = 0; z < SECTOR_RES; ++z) {
		for (unsigned int x = 0; x < SECTOR_RES; ++x) {
			ColumnStack &stack = stacks[geom.column_index(x, z)];
			for (unsigned int i = 0; i < stack.count; ++i) {
				const Vec3f n = unpack_normal_oct16(stack.layers[i].normal);
				const Vec3f w = geom.slab_gradient_to_world(x, z, n);
				stack.layers[i].normal = pack_normal_oct16(w.normalized());
			}
		}
	}
}

float arc_length_of_sector(const SectorId &id, uint8_t root_lod, float radius) {
	const int32_t n = sectors_per_face_side(id.lod, root_lod);
	const float inv_n = 1.f / static_cast<float>(n);
	const float u0 = static_cast<float>(id.x) * 2.f * inv_n - 1.f;
	const float u1 = static_cast<float>(id.x + 1) * 2.f * inv_n - 1.f;
	const float v = (static_cast<float>(id.z) + 0.5f) * 2.f * inv_n - 1.f;

	const Vec3f a = face_uv_to_dir(id.face, u0, v);
	const Vec3f b = face_uv_to_dir(id.face, u1, v);
	// Via the chord: acos(dot) loses most of its precision as the angle shrinks,
	// and at fine LODs this angle is very small indeed.
	const float chord = (a - b).length();
	const float angle = 2.f * std::asin(std::min(1.f, chord * 0.5f));
	return angle * radius;
}

} // namespace zylann::voxel::far
