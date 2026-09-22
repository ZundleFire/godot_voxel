#include "far_mesher.h"
#include "far_geometry.h"

#include <limits>

namespace zylann::voxel::far {

void FarMeshArrays::clear() {
	vertices.clear();
	indices.clear();
	aabb_min = Vec3f();
	aabb_max = Vec3f();
	has_bounds = false;
	cap_triangles = 0;
	wall_triangles = 0;
	skirt_triangles = 0;
}

namespace {

// ---------------------------------------------------------------------------
// Union-find over layers, used to label surface sheets.
// ---------------------------------------------------------------------------
class UnionFind {
public:
	void reset(size_t n) {
		_parent.resize(n);
		for (size_t i = 0; i < n; ++i) {
			_parent[i] = static_cast<uint32_t>(i);
		}
		_rank.assign(n, 0);
	}

	uint32_t find(uint32_t a) {
		// Path halving: iterative, no recursion, and leaves the tree nearly
		// flat after the first traversal.
		while (_parent[a] != a) {
			_parent[a] = _parent[_parent[a]];
			a = _parent[a];
		}
		return a;
	}

	void unite(uint32_t a, uint32_t b) {
		uint32_t ra = find(a);
		uint32_t rb = find(b);
		if (ra == rb) {
			return;
		}
		if (_rank[ra] < _rank[rb]) {
			const uint32_t tmp = ra;
			ra = rb;
			rb = tmp;
		}
		_parent[rb] = ra;
		if (_rank[ra] == _rank[rb]) {
			++_rank[ra];
		}
	}

private:
	std::vector<uint32_t> _parent;
	std::vector<uint8_t> _rank;
};

// One continuous surface passing through a cell, with the layer it uses at each
// of the cell's four corners.
struct CellSurface {
	uint32_t root = 0;
	// Flat layer indices at c00, c10, c11, c01 (counter-clockwise from -X-Z
	// when seen from above).
	uint32_t layers[4] = { 0, 0, 0, 0 };
};

struct MeshBuilder {
	FarMeshArrays *out = nullptr;

	inline void grow_bounds(const Vec3f &p) {
		if (!out->has_bounds) {
			out->aabb_min = p;
			out->aabb_max = p;
			out->has_bounds = true;
			return;
		}
		out->aabb_min.x = std::min(out->aabb_min.x, p.x);
		out->aabb_min.y = std::min(out->aabb_min.y, p.y);
		out->aabb_min.z = std::min(out->aabb_min.z, p.z);
		out->aabb_max.x = std::max(out->aabb_max.x, p.x);
		out->aabb_max.y = std::max(out->aabb_max.y, p.y);
		out->aabb_max.z = std::max(out->aabb_max.z, p.z);
	}

	inline uint32_t add_vertex(const Vec3f &position, const Vec3f &normal, float ao, float material) {
		FarVertex v;
		v.position = position;
		v.normal = normal;
		v.ao = ao;
		v.material = material;
		out->vertices.push_back(v);
		grow_bounds(position);
		return static_cast<uint32_t>(out->vertices.size() - 1);
	}

	inline void add_triangle(uint32_t a, uint32_t b, uint32_t c) {
		out->indices.push_back(a);
		out->indices.push_back(b);
		out->indices.push_back(c);
	}

	// Emits a quad given in order a, b, c, d around its perimeter, choosing the
	// winding so the visible face points along `desired_normal`.
	//
	// Godot front faces are clockwise as seen from the front, so the
	// right-hand-rule normal of the emitted triangles must oppose
	// `desired_normal`. Deriving the order here rather than hardcoding it is
	// what keeps walls correct regardless of which way an edge happened to be
	// traversed.
	void emit_quad_oriented(
			const Vec3f &a,
			const Vec3f &b,
			const Vec3f &c,
			const Vec3f &d,
			const Vec3f &desired_normal,
			float ao,
			float material
	) {
		const Vec3f rhr = cross(b - a, d - a);
		const bool flip = dot(rhr, desired_normal) > 0.f;

		const uint32_t ia = add_vertex(a, desired_normal, ao, material);
		const uint32_t ib = add_vertex(b, desired_normal, ao, material);
		const uint32_t ic = add_vertex(c, desired_normal, ao, material);
		const uint32_t id = add_vertex(d, desired_normal, ao, material);

		if (flip) {
			add_triangle(ia, id, ic);
			add_triangle(ia, ic, ib);
		} else {
			add_triangle(ia, ib, ic);
			add_triangle(ia, ic, id);
		}
	}
};

inline Vec3f column_position(
		const SectorGeometry *geom,
		unsigned int x,
		unsigned int z,
		float y,
		float cell_size
) {
	if (geom != nullptr) {
		return geom->position(x, z, y);
	}
	return Vec3f(static_cast<float>(x) * cell_size, y, static_cast<float>(z) * cell_size);
}

// Outward axis at a column: +Y on a flat world, radial on a planet.
inline Vec3f column_up(const SectorGeometry *geom, unsigned int x, unsigned int z) {
	return geom != nullptr ? geom->up_at(x, z) : Vec3f(0.f, 1.f, 0.f);
}

// In-surface direction for a grid step, used to orient walls and skirts. The
// flat constants are the tangents of a flat world, so this reduces to them.
inline Vec3f column_tangent(const SectorGeometry *geom, unsigned int x, unsigned int z, float dx, float dz) {
	if (geom == nullptr || !geom->spherical) {
		return Vec3f(dx, 0.f, dz);
	}
	const unsigned int i = geom->column_index(x, z);
	return (geom->tangent_u[i] * dx + geom->tangent_v[i] * dz).normalized();
}

// Mirrors a top-surface normal to face the other way along the column axis.
// The flat version negated x and z and forced y negative; this is the same
// operation about an arbitrary up.
inline Vec3f flip_normal_downward(const Vec3f &n, const Vec3f &up) {
	const float along = dot(n, up);
	const Vec3f tangential = n - up * along;
	return (tangential * -1.f - up * std::abs(along)).normalized();
}

// Finds the layer in `layers[begin..begin+count)` whose top is closest to
// `target`. Returns the flat index, or UINT32_MAX if there are none.
inline uint32_t closest_layer(const ColumnLayer *layers, uint32_t begin, uint32_t count, float target, float &out_dist) {
	uint32_t best = std::numeric_limits<uint32_t>::max();
	float best_dist = std::numeric_limits<float>::max();
	for (uint32_t i = 0; i < count; ++i) {
		const float d = std::abs(layers[begin + i].top - target);
		if (d < best_dist) {
			best_dist = d;
			best = begin + i;
		}
	}
	out_dist = best_dist;
	return best;
}

// Top of the highest layer strictly below `y` in a column, or `fallback`.
inline float next_surface_below(const ColumnLayer *layers, uint32_t begin, uint32_t count, float y, float fallback) {
	float best = fallback;
	bool found = false;
	for (uint32_t i = 0; i < count; ++i) {
		const float top = layers[begin + i].top;
		if (top < y - 1e-4f && (!found || top > best)) {
			best = top;
			found = true;
		}
	}
	return best;
}

} // namespace

// ---------------------------------------------------------------------------

void build_far_mesh(const ColumnField &field, const MesherSettings &settings, FarMeshArrays &out) {
	out.clear();

	const uint32_t layer_count = field.get_layer_count();
	if (layer_count == 0) {
		return;
	}

	const ColumnLayer *layers = field.get_layers_data();
	const SectorGeometry *geom = settings.geometry;
	const float cs = settings.cell_size;
	const float tol = cs * settings.max_connect_gap_cells;

	MeshBuilder builder;
	builder.out = &out;

	// -- Pass 1: label surface sheets -------------------------------------
	//
	// Two spans in adjacent columns join only if each is the other's closest
	// match. Without the mutual test, a column holding two spans near the same
	// height as one span next door would drag both of its spans into the same
	// sheet, and the cell would then have two candidate layers for one surface.

	UnionFind uf;
	uf.reset(layer_count);

	for (unsigned int z = 0; z < SECTOR_RES; ++z) {
		for (unsigned int x = 0; x < SECTOR_RES; ++x) {
			const uint32_t begin_a = field.get_column_offset(x, z);
			const uint32_t count_a = field.get_column_layer_count(x, z);
			if (count_a == 0) {
				continue;
			}

			for (unsigned int dir = 0; dir < 2; ++dir) {
				const unsigned int nx = x + (dir == 0 ? 1 : 0);
				const unsigned int nz = z + (dir == 1 ? 1 : 0);
				if (nx >= SECTOR_RES || nz >= SECTOR_RES) {
					continue;
				}

				const uint32_t begin_b = field.get_column_offset(nx, nz);
				const uint32_t count_b = field.get_column_layer_count(nx, nz);
				if (count_b == 0) {
					continue;
				}

				for (uint32_t i = 0; i < count_a; ++i) {
					const ColumnLayer &la = layers[begin_a + i];

					float dist_ab = 0.f;
					const uint32_t best_b = closest_layer(layers, begin_b, count_b, la.top, dist_ab);
					if (best_b == std::numeric_limits<uint32_t>::max()) {
						continue;
					}

					// A layer produced by merging several spans has an averaged
					// top rather than a measured one, so allow it more slack
					// before declaring a discontinuity. Without this, the first
					// downsample introduces a ring of spurious walls wherever
					// the averaging moved a height by more than the tolerance.
					float local_tol = tol;
					if ((la.flags & LAYER_FLAG_MERGED) || (layers[best_b].flags & LAYER_FLAG_MERGED)) {
						local_tol *= 2.f;
					}
					if (dist_ab > local_tol) {
						continue;
					}

					float dist_ba = 0.f;
					const uint32_t best_a = closest_layer(layers, begin_a, count_a, layers[best_b].top, dist_ba);
					if (best_a != begin_a + i) {
						continue;
					}

					uf.unite(begin_a + i, best_b);
				}
			}
		}
	}

	// -- Pass 2: find which sheets fully cover each cell -------------------

	std::vector<CellSurface> cell_surfaces;
	cell_surfaces.reserve(SECTOR_CELLS * SECTOR_CELLS);
	std::vector<uint32_t> cell_begin(SECTOR_CELLS * SECTOR_CELLS + 1, 0);

	// Corner offsets in the order c00, c10, c11, c01.
	static const unsigned int CORNER_DX[4] = { 0, 1, 1, 0 };
	static const unsigned int CORNER_DZ[4] = { 0, 0, 1, 1 };

	for (unsigned int cz = 0; cz < SECTOR_CELLS; ++cz) {
		for (unsigned int cx = 0; cx < SECTOR_CELLS; ++cx) {
			const unsigned int cell_index = cz * SECTOR_CELLS + cx;
			cell_begin[cell_index] = static_cast<uint32_t>(cell_surfaces.size());

			uint32_t corner_begin[4];
			uint32_t corner_count[4];
			for (unsigned int k = 0; k < 4; ++k) {
				const unsigned int x = cx + CORNER_DX[k];
				const unsigned int z = cz + CORNER_DZ[k];
				corner_begin[k] = field.get_column_offset(x, z);
				corner_count[k] = field.get_column_layer_count(x, z);
			}

			if (corner_count[0] == 0) {
				continue;
			}

			// Drive from corner 0's layers. A sheet that covers the cell must
			// pass through every corner, so it necessarily appears at corner 0.
			for (uint32_t i = 0; i < corner_count[0]; ++i) {
				const uint32_t root = uf.find(corner_begin[0] + i);

				CellSurface surface;
				surface.root = root;
				surface.layers[0] = corner_begin[0] + i;

				bool complete = true;
				for (unsigned int k = 1; k < 4 && complete; ++k) {
					complete = false;
					for (uint32_t j = 0; j < corner_count[k]; ++j) {
						if (uf.find(corner_begin[k] + j) == root) {
							surface.layers[k] = corner_begin[k] + j;
							complete = true;
							break;
						}
					}
				}

				if (complete) {
					cell_surfaces.push_back(surface);
				}
			}
		}
	}
	cell_begin[SECTOR_CELLS * SECTOR_CELLS] = static_cast<uint32_t>(cell_surfaces.size());

	// -- Pass 3: top caps --------------------------------------------------
	//
	// One vertex per (column, layer) that participates in any cap, shared
	// between the up-to-four cells around it. That is what makes the surface
	// shade smoothly; emitting per-quad vertices would give the stored normals
	// nothing to interpolate between.

	std::vector<uint32_t> vertex_of_layer(layer_count, std::numeric_limits<uint32_t>::max());

	for (unsigned int cz = 0; cz < SECTOR_CELLS; ++cz) {
		for (unsigned int cx = 0; cx < SECTOR_CELLS; ++cx) {
			const unsigned int cell_index = cz * SECTOR_CELLS + cx;
			for (uint32_t s = cell_begin[cell_index]; s < cell_begin[cell_index + 1]; ++s) {
				const CellSurface &surface = cell_surfaces[s];

				uint32_t vi[4];
				for (unsigned int k = 0; k < 4; ++k) {
					const uint32_t li = surface.layers[k];
					uint32_t &cached = vertex_of_layer[li];
					if (cached == std::numeric_limits<uint32_t>::max()) {
						const ColumnLayer &layer = layers[li];
						const unsigned int x = cx + CORNER_DX[k];
						const unsigned int z = cz + CORNER_DZ[k];
						cached = builder.add_vertex(
								column_position(geom, x, z, layer.top, cs),
								unpack_normal_oct16(layer.normal),
								static_cast<float>(layer.ao) / 255.f,
								static_cast<float>(layer.material)
						);
					}
					vi[k] = cached;
				}

				// Up-facing: (c00, c10, c11) and (c00, c11, c01).
				builder.add_triangle(vi[0], vi[1], vi[2]);
				builder.add_triangle(vi[0], vi[2], vi[3]);
				out.cap_triangles += 2;
			}
		}
	}

	// -- Pass 4: bottom caps ----------------------------------------------

	if (settings.generate_bottoms) {
		for (unsigned int cz = 0; cz < SECTOR_CELLS; ++cz) {
			for (unsigned int cx = 0; cx < SECTOR_CELLS; ++cx) {
				const unsigned int cell_index = cz * SECTOR_CELLS + cx;
				for (uint32_t s = cell_begin[cell_index]; s < cell_begin[cell_index + 1]; ++s) {
					const CellSurface &surface = cell_surfaces[s];

					// Only where every corner's span actually ends at a
					// surface. A span clipped by the bottom of the sampled
					// range has no underside, and capping it would paint a
					// floor across the world.
					bool all_surface = true;
					for (unsigned int k = 0; k < 4 && all_surface; ++k) {
						all_surface = layers[surface.layers[k]].bottom_is_surface();
					}
					if (!all_surface) {
						continue;
					}

					uint32_t vi[4];
					for (unsigned int k = 0; k < 4; ++k) {
						const ColumnLayer &layer = layers[surface.layers[k]];
						const unsigned int x = cx + CORNER_DX[k];
						const unsigned int z = cz + CORNER_DZ[k];
						// The underside gets the mirrored normal. The stored
						// normal belongs to the top surface, and reusing it
						// would light the underside as if it faced the sky.
						const Vec3f n = unpack_normal_oct16(layer.normal);
						vi[k] = builder.add_vertex(
								column_position(geom, x, z, layer.bottom, cs),
								flip_normal_downward(n, column_up(geom, x, z)),
								static_cast<float>(layer.ao) / 255.f * 0.6f,
								static_cast<float>(layer.material)
						);
					}

					// Down-facing: reverse of the top order.
					builder.add_triangle(vi[0], vi[2], vi[1]);
					builder.add_triangle(vi[0], vi[3], vi[2]);
					out.cap_triangles += 2;
				}
			}
		}
	}

	// -- Pass 5: walls at surface discontinuities --------------------------
	//
	// A sheet covering this cell but not the neighbouring one means the surface
	// stops at the shared edge. That edge gets a vertical face running down to
	// whatever surface is next below, which is the cliff itself.

	if (settings.generate_walls) {
		const float max_wall_depth = cs * settings.max_wall_depth_cells;

		// Neighbour cell offsets, paired with which two corners of *this* cell
		// lie on the shared edge, and the outward direction of the wall.
		struct EdgeSpec {
			int dx;
			int dz;
			unsigned int corner_a;
			unsigned int corner_b;
			Vec3f normal;
		};
		const EdgeSpec EDGES[4] = {
			{ 1, 0, 1, 2, Vec3f(1.f, 0.f, 0.f) }, // +X edge: corners c10, c11
			{ -1, 0, 3, 0, Vec3f(-1.f, 0.f, 0.f) }, // -X edge: corners c01, c00
			{ 0, 1, 2, 3, Vec3f(0.f, 0.f, 1.f) }, // +Z edge: corners c11, c01
			{ 0, -1, 0, 1, Vec3f(0.f, 0.f, -1.f) }, // -Z edge: corners c00, c10
		};

		for (unsigned int cz = 0; cz < SECTOR_CELLS; ++cz) {
			for (unsigned int cx = 0; cx < SECTOR_CELLS; ++cx) {
				const unsigned int cell_index = cz * SECTOR_CELLS + cx;

				for (uint32_t s = cell_begin[cell_index]; s < cell_begin[cell_index + 1]; ++s) {
					const CellSurface &surface = cell_surfaces[s];

					for (const EdgeSpec &edge : EDGES) {
						const int ncx = static_cast<int>(cx) + edge.dx;
						const int ncz = static_cast<int>(cz) + edge.dz;

						// Sector-border edges are left to the skirt pass. The
						// neighbour there is a different sector, possibly at a
						// different LOD, and this pass has no view of it.
						if (ncx < 0 || ncz < 0 || ncx >= static_cast<int>(SECTOR_CELLS) ||
								ncz >= static_cast<int>(SECTOR_CELLS)) {
							continue;
						}

						const unsigned int n_index =
								static_cast<unsigned int>(ncz) * SECTOR_CELLS + static_cast<unsigned int>(ncx);

						bool neighbour_has_sheet = false;
						for (uint32_t t = cell_begin[n_index]; t < cell_begin[n_index + 1]; ++t) {
							if (cell_surfaces[t].root == surface.root) {
								neighbour_has_sheet = true;
								break;
							}
						}
						if (neighbour_has_sheet) {
							continue;
						}

						// The two columns on the shared edge.
						const unsigned int ka = edge.corner_a;
						const unsigned int kb = edge.corner_b;
						const ColumnLayer &la = layers[surface.layers[ka]];
						const ColumnLayer &lb = layers[surface.layers[kb]];

						const unsigned int xa = cx + CORNER_DX[ka];
						const unsigned int za = cz + CORNER_DZ[ka];
						const unsigned int xb = cx + CORNER_DX[kb];
						const unsigned int zb = cz + CORNER_DZ[kb];

						const float bottom_a = std::max(
								next_surface_below(
										layers,
										field.get_column_offset(xa, za),
										field.get_column_layer_count(xa, za),
										la.top,
										la.bottom
								),
								la.top - max_wall_depth
						);
						const float bottom_b = std::max(
								next_surface_below(
										layers,
										field.get_column_offset(xb, zb),
										field.get_column_layer_count(xb, zb),
										lb.top,
										lb.bottom
								),
								lb.top - max_wall_depth
						);

						// Nothing to draw if the drop is negligible.
						if ((la.top - bottom_a) < 1e-3f && (lb.top - bottom_b) < 1e-3f) {
							continue;
						}

						const Vec3f ta = column_position(geom, xa, za, la.top, cs);
						const Vec3f tb = column_position(geom, xb, zb, lb.top, cs);
						const Vec3f bb = column_position(geom, xb, zb, bottom_b, cs);
						const Vec3f ba = column_position(geom, xa, za, bottom_a, cs);

						// Walls sit in shadow at the base of whatever is above
						// them; darkening them slightly is cheap and reads well
						// at distance.
						const float ao = static_cast<float>(la.ao) / 255.f * 0.75f;
						builder.emit_quad_oriented(
								ta,
								tb,
								bb,
								ba,
								column_tangent(geom, xa, za, static_cast<float>(edge.dx),
										static_cast<float>(edge.dz)),
								ao,
								static_cast<float>(la.material)
						);
						out.wall_triangles += 2;
					}
				}
			}
		}
	}

	// -- Pass 6: border skirts ---------------------------------------------
	//
	// An apron hanging straight down from the sector's outer ring. When the
	// neighbouring sector is at the same LOD the two surfaces already meet and
	// the apron is buried; when it is one level coarser the heights differ by
	// up to a cell and the apron is what the camera sees through the crack
	// instead of the sky.
	//
	// This costs a few hundred triangles per sector and needs no knowledge of
	// the neighbour at all, which is what keeps sector meshing independent and
	// parallel. Making seams exact instead would require every sector to wait
	// for its four neighbours to settle on a LOD first.

	if (settings.generate_skirts) {
		const float skirt_depth = cs * settings.skirt_depth_cells;

		struct BorderSpec {
			bool along_x;
			unsigned int fixed;
			Vec3f normal;
		};
		const BorderSpec BORDERS[4] = {
			{ true, 0, Vec3f(0.f, 0.f, -1.f) }, // z = 0
			{ true, SECTOR_CELLS, Vec3f(0.f, 0.f, 1.f) }, // z = max
			{ false, 0, Vec3f(-1.f, 0.f, 0.f) }, // x = 0
			{ false, SECTOR_CELLS, Vec3f(1.f, 0.f, 0.f) }, // x = max
		};

		for (const BorderSpec &border : BORDERS) {
			for (unsigned int i = 0; i < SECTOR_CELLS; ++i) {
				const unsigned int xa = border.along_x ? i : border.fixed;
				const unsigned int za = border.along_x ? border.fixed : i;
				const unsigned int xb = border.along_x ? (i + 1) : border.fixed;
				const unsigned int zb = border.along_x ? border.fixed : (i + 1);

				const uint32_t begin_a = field.get_column_offset(xa, za);
				const uint32_t count_a = field.get_column_layer_count(xa, za);
				const uint32_t begin_b = field.get_column_offset(xb, zb);
				const uint32_t count_b = field.get_column_layer_count(xb, zb);

				if (count_a == 0 || count_b == 0) {
					continue;
				}

				// Only the topmost surface needs an apron. Lower layers are
				// interior geometry; a crack there is already occluded.
				const ColumnLayer &la = layers[begin_a];
				const ColumnLayer &lb = layers[begin_b];

				const Vec3f ta = column_position(geom, xa, za, la.top, cs);
				const Vec3f tb = column_position(geom, xb, zb, lb.top, cs);
				const Vec3f bb = column_position(geom, xb, zb, lb.top - skirt_depth, cs);
				const Vec3f ba = column_position(geom, xa, za, la.top - skirt_depth, cs);

				builder.emit_quad_oriented(
						ta,
						tb,
						bb,
						ba,
						column_tangent(geom, xa, za, border.normal.x, border.normal.z),
						static_cast<float>(la.ao) / 255.f,
						static_cast<float>(la.material)
				);
				out.skirt_triangles += 2;
			}
		}
	}
}

} // namespace zylann::voxel::far
