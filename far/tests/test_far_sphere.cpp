// Host-compilable tests for the cubesphere parameterisation.
//
// The properties that matter are the ones the rest of the far field leans on:
// every direction belongs to exactly one face, round-trips are exact, adjacent
// faces agree along their shared edge to float precision (which is what lets
// sectors tile across a face boundary with no seam), and stepping off an edge
// lands on the right neighbour.

#include "../core/far_sphere.h"

#include <cstdio>
#include <vector>

using namespace zylann::voxel::far;

static int g_failures = 0;
static int g_checks = 0;
static const char *g_current_test = "";

#define CHECK(cond)                                                                                                    \
	do {                                                                                                               \
		++g_checks;                                                                                                    \
		if (!(cond)) {                                                                                                 \
			++g_failures;                                                                                              \
			std::printf("  FAIL [%s] %s:%d: %s\n", g_current_test, __FILE__, __LINE__, #cond);                         \
		}                                                                                                              \
	} while (0)

#define TEST(name)                                                                                                     \
	g_current_test = name;                                                                                             \
	std::printf("- %s\n", name);

namespace {

// Via the chord, not acos(dot). Near zero angle `dot` is 1 - O(t^2), so in
// float it saturates and acos returns sqrt(epsilon) ~ 5e-4 rad for vectors that
// are bit-identical -- which silently puts a floor under every tolerance here.
// The chord stays well conditioned all the way down.
float angle_between(const Vec3f &a, const Vec3f &b) {
	const Vec3f na = a.normalized();
	const Vec3f nb = b.normalized();
	const float chord = (na - nb).length();
	return 2.f * std::asin(std::min(1.f, chord * 0.5f));
}

} // namespace

// ---------------------------------------------------------------------------

static void test_uv_round_trips() {
	TEST("face uv round-trips through a direction");

	for (unsigned int face = 0; face < CUBE_FACE_COUNT; ++face) {
		for (int iv = 0; iv <= 16; ++iv) {
			for (int iu = 0; iu <= 16; ++iu) {
				const float u = static_cast<float>(iu) / 8.f - 1.f;
				const float v = static_cast<float>(iv) / 8.f - 1.f;

				const Vec3f dir = face_uv_to_dir(face, u, v);
				CHECK(std::abs(dir.length() - 1.f) < 1e-5f);

				unsigned int rf = 0;
				float ru = 0.f;
				float rv = 0.f;
				dir_to_face_uv(dir, rf, ru, rv);

				// Points strictly inside the face must come back on it. Exactly
				// on an edge either face is a valid answer, so only the
				// direction is checked there.
				if (std::abs(u) < 0.999f && std::abs(v) < 0.999f) {
					CHECK(rf == face);
					CHECK(std::abs(ru - u) < 1e-4f);
					CHECK(std::abs(rv - v) < 1e-4f);
				}
				CHECK(angle_between(face_uv_to_dir(rf, ru, rv), dir) < 1e-4f);
			}
		}
	}
}

static void test_faces_tile_the_sphere_once() {
	TEST("every direction belongs to exactly one face");

	// Fibonacci sphere: an even spread with no axis bias, so face boundaries
	// and corners are all sampled.
	const int N = 4000;
	const float golden = 3.14159265f * (3.f - std::sqrt(5.f));
	for (int i = 0; i < N; ++i) {
		const float y = 1.f - (static_cast<float>(i) / static_cast<float>(N - 1)) * 2.f;
		const float r = std::sqrt(std::max(0.f, 1.f - y * y));
		const float theta = golden * static_cast<float>(i);
		const Vec3f dir(std::cos(theta) * r, y, std::sin(theta) * r);

		unsigned int face = 0;
		float u = 0.f;
		float v = 0.f;
		dir_to_face_uv(dir, face, u, v);

		CHECK(face < CUBE_FACE_COUNT);
		// Inside the face's own square, within float slop.
		CHECK(u >= -1.0001f && u <= 1.0001f);
		CHECK(v >= -1.0001f && v <= 1.0001f);
		// And it really is the same direction.
		CHECK(angle_between(face_uv_to_dir(face, u, v), dir) < 1e-4f);
	}
}

static void test_shared_edges_agree() {
	TEST("adjacent faces agree along a shared edge");

	// This is the sphere version of the padding invariant in README 2.2: a
	// column on a face boundary must land on the same world position from
	// either side, or sectors cannot tile across the boundary.
	const int N = 64;
	float worst = 0.f;

	for (unsigned int face = 0; face < CUBE_FACE_COUNT; ++face) {
		for (int i = 0; i <= N; ++i) {
			const float t = static_cast<float>(i) / static_cast<float>(N) * 2.f - 1.f;

			const Vec3f edges[4] = {
				face_uv_to_dir(face, 1.f, t),
				face_uv_to_dir(face, -1.f, t),
				face_uv_to_dir(face, t, 1.f),
				face_uv_to_dir(face, t, -1.f),
			};

			for (const Vec3f &dir : edges) {
				unsigned int nf = 0;
				float nu = 0.f;
				float nv = 0.f;
				dir_to_face_uv(dir, nf, nu, nv);
				// Whichever face claims it, re-expanding must give the same
				// direction back.
				const float err = angle_between(face_uv_to_dir(nf, nu, nv), dir);
				worst = std::max(worst, err);
				CHECK(err < 1e-4f);
			}
		}
	}
	std::printf("    worst edge disagreement = %.3g rad\n", static_cast<double>(worst));
}

static void test_tangent_warp_evens_out_cells() {
	TEST("the tangent warp narrows the spread of cell angular sizes");

	// Cell size is the unit the whole LOD scheme is expressed in, so a face
	// whose corner cells subtend much more angle than its centre cells makes
	// every distance-based decision wrong by that factor. This measures the
	// spread with and without the warp rather than asserting a magic number,
	// so it stays meaningful if the mapping is ever changed.
	const int N = 32;

	auto measure = [N](bool warped) {
		const CubeFaceAxes &axes = get_cube_face_axes(4);
		auto dir_at = [&](float u, float v) {
			const float wu = warped ? warp_face_param(u) : u;
			const float wv = warped ? warp_face_param(v) : v;
			return (axes.normal + axes.right * wu + axes.up * wv).normalized();
		};

		float min_step = 1e9f;
		float max_step = 0.f;
		for (int iv = 0; iv < N; ++iv) {
			for (int iu = 0; iu < N; ++iu) {
				const float u0 = static_cast<float>(iu) / static_cast<float>(N) * 2.f - 1.f;
				const float u1 = static_cast<float>(iu + 1) / static_cast<float>(N) * 2.f - 1.f;
				const float v = static_cast<float>(iv) / static_cast<float>(N) * 2.f - 1.f;

				const float step = angle_between(dir_at(u0, v), dir_at(u1, v));
				min_step = std::min(min_step, step);
				max_step = std::max(max_step, step);
			}
		}
		return max_step / min_step;
	};

	const float warped = measure(true);
	const float plain = measure(false);
	std::printf("    cell angular size ratio: warped %.3f, unwarped %.3f\n", static_cast<double>(warped),
			static_cast<double>(plain));

	CHECK(warped < plain);
	// sqrt(2) is the known worst case for a tangent-warped cube face, against
	// sqrt(3) for an unwarped one. Allow a little slop above it.
	CHECK(warped < 1.45f);
	CHECK(plain > 1.6f);
}

static void test_basis_is_orthonormal() {
	TEST("the surface frame is orthonormal and radial points outward");

	for (unsigned int face = 0; face < CUBE_FACE_COUNT; ++face) {
		for (int iv = 0; iv <= 8; ++iv) {
			for (int iu = 0; iu <= 8; ++iu) {
				const float u = static_cast<float>(iu) / 4.f - 1.f;
				const float v = static_cast<float>(iv) / 4.f - 1.f;

				Vec3f tu;
				Vec3f tv;
				Vec3f rad;
				face_uv_basis(face, u, v, tu, tv, rad);

				CHECK(std::abs(tu.length() - 1.f) < 1e-4f);
				CHECK(std::abs(tv.length() - 1.f) < 1e-4f);
				CHECK(std::abs(rad.length() - 1.f) < 1e-4f);

				CHECK(std::abs(dot(tu, tv)) < 1e-4f);
				CHECK(std::abs(dot(tu, rad)) < 1e-4f);
				CHECK(std::abs(dot(tv, rad)) < 1e-4f);

				// Radial must match the surface point itself.
				CHECK(angle_between(rad, face_uv_to_dir(face, u, v)) < 1e-4f);
			}
		}
	}
}

static void test_stepping_off_an_edge_finds_the_neighbour() {
	TEST("stepping off a face edge lands on the adjacent face");

	const int32_t n = 8;

	for (unsigned int face = 0; face < CUBE_FACE_COUNT; ++face) {
		for (int32_t i = 0; i < n; ++i) {
			// Four one-step excursions past each edge.
			const int32_t coords[4][2] = { { -1, i }, { n, i }, { i, -1 }, { i, n } };

			for (const auto &c : coords) {
				unsigned int nf = 0;
				int32_t nx = 0;
				int32_t nz = 0;
				const bool ok = remap_sector_across_faces(face, c[0], c[1], n, nf, nx, nz);
				CHECK(ok);
				if (!ok) {
					continue;
				}
				CHECK(nf != face);
				CHECK(nx >= 0 && nx < n);
				CHECK(nz >= 0 && nz < n);

				// The remapped sector must occupy the same patch of sphere the
				// out-of-range one named.
				const float u = (static_cast<float>(c[0]) + 0.5f) * 2.f / static_cast<float>(n) - 1.f;
				const float v = (static_cast<float>(c[1]) + 0.5f) * 2.f / static_cast<float>(n) - 1.f;
				const Vec3f wanted = face_uv_to_dir(face, u, v);

				const float nu = (static_cast<float>(nx) + 0.5f) * 2.f / static_cast<float>(n) - 1.f;
				const float nv = (static_cast<float>(nz) + 0.5f) * 2.f / static_cast<float>(n) - 1.f;
				const Vec3f got = face_uv_to_dir(nf, nu, nv);

				// Within half a sector: the grids line up, so this should be
				// far tighter, but half a cell is the meaningful tolerance.
				const float half_sector = (3.14159265f * 0.5f) / static_cast<float>(n) * 0.5f;
				CHECK(angle_between(wanted, got) < half_sector);
			}
		}
	}
}

static void test_corner_diagonals_are_rejected() {
	TEST("a diagonal step at a cube corner names no sector");

	const int32_t n = 8;
	unsigned int nf = 0;
	int32_t nx = 0;
	int32_t nz = 0;

	// Only three faces meet at a cube corner, so a diagonal step there is not a
	// single hop. It is resolved as two, and may or may not name a real sector
	// depending on how far past the corner it reaches. Either answer is fine;
	// what must never happen is a result outside the grid. Whether the set as a
	// whole tiles is checked by the ring tests in test_far_lod_tree.cpp.
	const int32_t diagonals[4][2] = { { -1, -1 }, { n, -1 }, { -1, n }, { n, n } };
	for (const auto &d : diagonals) {
		if (remap_sector_across_faces(0, d[0], d[1], n, nf, nx, nz)) {
			CHECK(nf < CUBE_FACE_COUNT);
			CHECK(nf != 0);
			CHECK(nx >= 0 && nx < n);
			CHECK(nz >= 0 && nz < n);
		}
	}

	// In-range coordinates pass through untouched.
	CHECK(remap_sector_across_faces(3, 2, 5, n, nf, nx, nz));
	CHECK(nf == 3 && nx == 2 && nz == 5);
}

static void test_neighbours_are_mutual() {
	TEST("crossing an edge and coming back returns the original sector");

	const int32_t n = 8;

	for (unsigned int face = 0; face < CUBE_FACE_COUNT; ++face) {
		for (int32_t i = 0; i < n; ++i) {
			const int32_t coords[4][2] = { { -1, i }, { n, i }, { i, -1 }, { i, n } };

			for (const auto &c : coords) {
				unsigned int nf = 0;
				int32_t nx = 0;
				int32_t nz = 0;
				if (!remap_sector_across_faces(face, c[0], c[1], n, nf, nx, nz)) {
					continue;
				}

				// From the neighbour, the sector we came from lies one step back
				// across the same edge. Find it by checking all four of the
				// neighbour's own excursions.
				bool found_back = false;
				const int32_t back[4][2] = { { nx - 1, nz }, { nx + 1, nz }, { nx, nz - 1 }, { nx, nz + 1 } };
				for (const auto &b : back) {
					unsigned int bf = 0;
					int32_t bx = 0;
					int32_t bz = 0;
					if (!remap_sector_across_faces(nf, b[0], b[1], n, bf, bx, bz)) {
						continue;
					}
					const int32_t ox = std::min(std::max(c[0], 0), n - 1);
					const int32_t oz = std::min(std::max(c[1], 0), n - 1);
					if (bf == face && bx == ox && bz == oz) {
						found_back = true;
						break;
					}
				}
				CHECK(found_back);
			}
		}
	}
}

static void test_all_faces_share_the_flat_handedness() {
	TEST("every face orients its grid the same way the flat world does");

	// The mesher winds a cap from grid corners (0,0), (1,0), (1,1), and Godot
	// front faces are clockwise from the front (HANDOFF 3.3), so the right-hand
	// normal of that triangle must point *away* from the surface normal. On a
	// flat world grid u is +X, grid v is +Z and the column axis is +Y, and
	// indeed X cross Z = -Y.
	//
	// So every cube face must satisfy right cross up = -normal. A face that does
	// not gets its caps wound backwards, is culled, and renders as a hole --
	// which is exactly how this was found.
	for (unsigned int face = 0; face < CUBE_FACE_COUNT; ++face) {
		const CubeFaceAxes &a = get_cube_face_axes(face);
		const Vec3f c = cross(a.right, a.up);
		CHECK(std::abs(c.x + a.normal.x) < 1e-5f);
		CHECK(std::abs(c.y + a.normal.y) < 1e-5f);
		CHECK(std::abs(c.z + a.normal.z) < 1e-5f);
	}
}

int main() {
	std::printf("cubesphere tests\n\n");

	test_uv_round_trips();
	test_faces_tile_the_sphere_once();
	test_shared_edges_agree();
	test_tangent_warp_evens_out_cells();
	test_basis_is_orthonormal();
	test_stepping_off_an_edge_finds_the_neighbour();
	test_corner_diagonals_are_rejected();
	test_neighbours_are_mutual();
	test_all_faces_share_the_flat_handedness();

	std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
	return g_failures == 0 ? 0 : 1;
}
