#include "far_sphere.h"

namespace zylann::voxel::far {

namespace {

constexpr float PI_F = 3.14159265358979323846f;
constexpr float QUARTER_PI = PI_F * 0.25f;

// Face order is +X, -X, +Y, -Y, +Z, -Z. The exact handedness of each face does
// not matter to anything outside this file -- `remap_sector_across_faces()`
// re-projects rather than assuming a layout, and the tests check the properties
// that do matter (faces tile the sphere once, neighbours are mutual, shared
// edges agree). It only has to be consistent.
// Every face satisfies right x up = -normal, which is the handedness a flat
// world has (grid u = +X, grid v = +Z, column axis = +Y, and X x Z = -Y). Get
// this wrong on a face and its caps wind backwards, are back-face culled, and
// the face renders as a hole -- which is how four of these were found to be
// flipped. Pinned by test_all_faces_share_the_flat_handedness.
const CubeFaceAxes CUBE_FACES[CUBE_FACE_COUNT] = {
	{ Vec3f(0.f, 0.f, 1.f), Vec3f(0.f, 1.f, 0.f), Vec3f(1.f, 0.f, 0.f) }, // +X
	{ Vec3f(0.f, 0.f, -1.f), Vec3f(0.f, 1.f, 0.f), Vec3f(-1.f, 0.f, 0.f) }, // -X
	{ Vec3f(1.f, 0.f, 0.f), Vec3f(0.f, 0.f, 1.f), Vec3f(0.f, 1.f, 0.f) }, // +Y
	{ Vec3f(1.f, 0.f, 0.f), Vec3f(0.f, 0.f, -1.f), Vec3f(0.f, -1.f, 0.f) }, // -Y
	{ Vec3f(-1.f, 0.f, 0.f), Vec3f(0.f, 1.f, 0.f), Vec3f(0.f, 0.f, 1.f) }, // +Z
	{ Vec3f(1.f, 0.f, 0.f), Vec3f(0.f, 1.f, 0.f), Vec3f(0.f, 0.f, -1.f) }, // -Z
};

inline float component(const Vec3f &v, unsigned int axis) {
	return axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
}

} // namespace

const CubeFaceAxes &get_cube_face_axes(unsigned int face) {
	return CUBE_FACES[face < CUBE_FACE_COUNT ? face : 0];
}

float warp_face_param(float s) {
	return std::tan(s * QUARTER_PI);
}

float unwarp_face_param(float t) {
	return std::atan(t) / QUARTER_PI;
}

Vec3f face_uv_to_dir(unsigned int face, float u, float v) {
	const CubeFaceAxes &axes = get_cube_face_axes(face);
	const float wu = warp_face_param(u);
	const float wv = warp_face_param(v);
	const Vec3f d = axes.normal + axes.right * wu + axes.up * wv;
	return d.normalized();
}

void dir_to_face_uv(const Vec3f &dir, unsigned int &out_face, float &out_u, float &out_v) {
	// Dominant axis picks the face; its sign picks which of the pair.
	const float ax = std::abs(dir.x);
	const float ay = std::abs(dir.y);
	const float az = std::abs(dir.z);

	unsigned int axis = 0;
	float m = ax;
	if (ay > m) {
		axis = 1;
		m = ay;
	}
	if (az > m) {
		axis = 2;
		m = az;
	}

	const bool positive = component(dir, axis) >= 0.f;
	out_face = axis * 2 + (positive ? 0 : 1);

	if (m < 1e-20f) {
		// Degenerate input. Any face will do; report its centre.
		out_u = 0.f;
		out_v = 0.f;
		return;
	}

	const CubeFaceAxes &axes = get_cube_face_axes(out_face);
	// Project onto the face plane. `dot(dir, normal)` is the dominant component,
	// so dividing by it puts the point on the plane at distance 1.
	const float inv_w = 1.f / dot(dir, axes.normal);
	out_u = unwarp_face_param(dot(dir, axes.right) * inv_w);
	out_v = unwarp_face_param(dot(dir, axes.up) * inv_w);
}

void face_uv_basis(unsigned int face, float u, float v, Vec3f &out_tangent_u, Vec3f &out_tangent_v,
		Vec3f &out_radial) {
	const CubeFaceAxes &axes = get_cube_face_axes(face);
	const float wu = warp_face_param(u);
	const float wv = warp_face_param(v);

	const Vec3f p = axes.normal + axes.right * wu + axes.up * wv;
	out_radial = p.normalized();

	// d(p/|p|)/du, with the warp's derivative folded in. The warp scales both
	// tangents by a positive factor and is then normalised away, so it is only
	// the projection onto the tangent plane that matters here.
	Vec3f tu = axes.right - out_radial * dot(out_radial, axes.right);
	if (tu.length_squared() < 1e-20f) {
		tu = axes.up;
	}
	out_tangent_u = tu.normalized();

	// Complete a right-handed frame rather than projecting `up` separately, so
	// the basis is guaranteed orthonormal even at a face corner.
	out_tangent_v = cross(out_radial, out_tangent_u);
	if (dot(out_tangent_v, axes.up) < 0.f) {
		out_tangent_v = out_tangent_v * -1.f;
	}
}

namespace {

// Where a one-step excursion past an edge lands, by re-projecting through the
// face's own axes. Exact for a single step: past the edge the tangent warp
// matches the neighbour's grid spacing to first order, which is enough when the
// step is one cell. It is NOT exact further out -- see the note in
// `remap_sector_across_faces`.
bool reproject_one_step(unsigned int face, int32_t x, int32_t z, int32_t n, unsigned int &out_face, int32_t &out_x,
		int32_t &out_z) {
	const float inv_n = 1.f / static_cast<float>(n);
	const float u = (static_cast<float>(x) + 0.5f) * 2.f * inv_n - 1.f;
	const float v = (static_cast<float>(z) + 0.5f) * 2.f * inv_n - 1.f;

	unsigned int nf = 0;
	float nu = 0.f;
	float nv = 0.f;
	dir_to_face_uv(face_uv_to_dir(face, u, v), nf, nu, nv);
	if (nf == face) {
		return false;
	}

	const float fx = (nu + 1.f) * 0.5f * static_cast<float>(n);
	const float fz = (nv + 1.f) * 0.5f * static_cast<float>(n);
	out_face = nf;
	out_x = std::min(std::max(static_cast<int32_t>(std::floor(fx)), 0), n - 1);
	out_z = std::min(std::max(static_cast<int32_t>(std::floor(fz)), 0), n - 1);
	return true;
}

// Crossing a single edge, to any depth.
//
// `axis_is_x` says which coordinate ran out of range, `high` which side, and
// `overflow` how many cells past the boundary (1 for the first cell outside).
// `along` is the in-range coordinate.
//
// The two ends of the edge are re-projected one step out, which is accurate,
// and everything else is integer arithmetic on the neighbour's own grid. That
// avoids both a hand-written table of edge rotations and the drift that comes
// from extending a face's parameterisation several cells past its edge.
bool cross_edge(unsigned int face, bool axis_is_x, bool high, int32_t along, int32_t overflow, int32_t n,
		unsigned int &out_face, int32_t &out_x, int32_t &out_z, int32_t *out_along_step_x,
		int32_t *out_along_step_z) {
	const int32_t outside = high ? n : -1;

	unsigned int f0 = 0;
	int32_t x0 = 0;
	int32_t z0 = 0;
	unsigned int f1 = 0;
	int32_t x1 = 0;
	int32_t z1 = 0;

	const int32_t ax0 = axis_is_x ? outside : 0;
	const int32_t az0 = axis_is_x ? 0 : outside;
	const int32_t ax1 = axis_is_x ? outside : (n - 1);
	const int32_t az1 = axis_is_x ? (n - 1) : outside;

	if (!reproject_one_step(face, ax0, az0, n, f0, x0, z0)) {
		return false;
	}
	if (!reproject_one_step(face, ax1, az1, n, f1, x1, z1)) {
		return false;
	}
	if (f0 != f1 || n < 2) {
		return false;
	}

	// One step along the edge, expressed on the neighbour. Exactly one of these
	// is non-zero, because the two grids share the edge.
	const int32_t along_dx = (x1 - x0) / (n - 1);
	const int32_t along_dz = (z1 - z0) / (n - 1);

	// The cell at depth 1, opposite `along`.
	const int32_t bx = x0 + along_dx * along;
	const int32_t bz = z0 + along_dz * along;

	// Depth runs along whichever axis the edge is not. The first cell inside
	// the neighbour sits at one extreme of that axis, so inward is away from it.
	int32_t depth_dx = 0;
	int32_t depth_dz = 0;
	if (along_dx == 0) {
		depth_dx = (bx == 0) ? 1 : -1;
	} else {
		depth_dz = (bz == 0) ? 1 : -1;
	}

	out_face = f0;
	out_x = bx + depth_dx * (overflow - 1);
	out_z = bz + depth_dz * (overflow - 1);
	if (out_along_step_x != nullptr) {
		*out_along_step_x = along_dx;
		*out_along_step_z = along_dz;
	}
	return true;
}

} // namespace

bool remap_sector_across_faces(unsigned int face, int32_t x, int32_t z, int32_t n, unsigned int &out_face,
		int32_t &out_x, int32_t &out_z) {
	if (n <= 0) {
		return false;
	}

	const bool out_x_range = (x < 0 || x >= n);
	const bool out_z_range = (z < 0 || z >= n);

	if (!out_x_range && !out_z_range) {
		out_face = face;
		out_x = x;
		out_z = z;
		return true;
	}

	if (out_x_range && out_z_range) {
		// Past a cube corner. Three faces meet there rather than four, so this
		// cannot be a single hop -- cross the x edge first, then continue in
		// the direction that this face's z maps to on the face we land on.
		const bool high_x = (x >= n);
		const int32_t overflow_x = high_x ? (x - n + 1) : (-x);
		const int32_t clamped_z = std::min(std::max(z, 0), n - 1);

		unsigned int f1 = 0;
		int32_t x1 = 0;
		int32_t z1 = 0;
		int32_t along_dx = 0;
		int32_t along_dz = 0;
		if (!cross_edge(face, true, high_x, clamped_z, overflow_x, n, f1, x1, z1, &along_dx, &along_dz)) {
			return false;
		}

		// How far past the z edge we still have to go, and which way that is on
		// the new face. `along_*` is exactly one step of this face's z there.
		const bool high_z = (z >= n);
		const int32_t overflow_z = high_z ? (z - (n - 1)) : (-z);
		const int32_t sign = high_z ? 1 : -1;

		const int32_t fx = x1 + along_dx * sign * overflow_z;
		const int32_t fz = z1 + along_dz * sign * overflow_z;

		if (fx >= 0 && fx < n && fz >= 0 && fz < n) {
			out_face = f1;
			out_x = fx;
			out_z = fz;
			return true;
		}

		// Still outside: one more single-edge hop onto the third face.
		const bool second_is_x = (fx < 0 || fx >= n);
		if (second_is_x && (fz < 0 || fz >= n)) {
			// Genuinely past the corner of the corner. Nothing there.
			return false;
		}
		const bool second_high = second_is_x ? (fx >= n) : (fz >= n);
		const int32_t second_overflow =
				second_is_x ? (second_high ? (fx - n + 1) : (-fx)) : (second_high ? (fz - n + 1) : (-fz));
		const int32_t second_along = second_is_x ? fz : fx;
		return cross_edge(f1, second_is_x, second_high, second_along, second_overflow, n, out_face, out_x, out_z,
				nullptr, nullptr);
	}

	const bool axis_is_x = out_x_range;
	const bool high = axis_is_x ? (x >= n) : (z >= n);
	const int32_t overflow = axis_is_x ? (high ? (x - n + 1) : (-x)) : (high ? (z - n + 1) : (-z));
	const int32_t along = axis_is_x ? z : x;

	return cross_edge(face, axis_is_x, high, along, overflow, n, out_face, out_x, out_z, nullptr, nullptr);
}

uint8_t choose_sphere_root_lod(float planet_radius, float target_cell_size) {
	if (!(planet_radius > 0.f) || !(target_cell_size > 0.f)) {
		return 1;
	}
	// A cube face spans a quarter turn, so its edge is about (pi/2) * R long.
	constexpr float HALF_PI = 1.57079632679f;
	const float face_arc = HALF_PI * planet_radius;
	const float cells_per_side = face_arc / target_cell_size;
	const float sectors_needed = cells_per_side / static_cast<float>(SECTOR_CELLS);

	uint8_t root = 0;
	while (root < 31 && (float(int64_t(1) << root)) < sectors_needed) {
		++root;
	}
	// At least one split, so there is always a coarser level to fall back to.
	return root < 1 ? 1 : root;
}

} // namespace zylann::voxel::far
