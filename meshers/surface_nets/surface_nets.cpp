#include "surface_nets.h"
#include "../../constants/cube_tables.h"
#include "../../storage/mixel4.h"
#include "../../util/containers/std_unordered_map.h"
#include "../../util/math/conv.h"
#include "../../util/profiling.h"
#include <cstring>

namespace zylann::voxel::surface_nets {

namespace {

// SDF convention used throughout this engine (see VoxelBuffer::CHANNEL_SDF docs):
// negative = inside matter, positive = outside/air.

const Vector3i g_corner_offsets[8] = {
	Vector3i(0, 0, 0), // 0
	Vector3i(1, 0, 0), // 1
	Vector3i(0, 1, 0), // 2
	Vector3i(1, 1, 0), // 3
	Vector3i(0, 0, 1), // 4
	Vector3i(1, 0, 1), // 5
	Vector3i(0, 1, 1), // 6
	Vector3i(1, 1, 1), // 7
};

// Pairs of corner indices forming the 12 edges of a cube, grouped by axis (X, then Y, then Z).
const int g_edges[12][2] = {
	{ 0, 1 }, { 2, 3 }, { 4, 5 }, { 6, 7 }, // X
	{ 0, 2 }, { 1, 3 }, { 4, 6 }, { 5, 7 }, // Y
	{ 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 }, // Z
};

struct CellSolveResult {
	Vector3f local_position;
	Vector3f normal;
	// Where this cell's 8 corners live in the voxel buffer, and which of them contain matter.
	// Only needed for material sampling, but solving the cell already computes both.
	FixedArray<uint32_t, 8> corner_buffer_indices;
	uint8_t solid_mask;
};

// Read-only view over a decoded SDF channel, laid out like VoxelBuffer stores it.
struct SdfView {
	const float *data;
	Vector3i size;

	inline float at(const Vector3i p) const {
		return data[Vector3iUtil::get_zxy_index(p, size)];
	}
};

// Solves a Surface Nets vertex for the cube whose min corner is at buffer-space `corner_pos`,
// with edges of length `step` voxels (1 for regular cells, larger for the coarse transition
// layer). `local_origin` is where `corner_pos` sits in the caller's output coordinate space, and
// `pos_scale` converts that space into output units (2^lod_index -- see build_regular_mesh).
// Only samples the cell's own 8 corners (no data outside the cell needed), so this works right up
// to the edge of whatever padding is available.
bool solve_cell_vertex(
		const SdfView &sdf,
		const Vector3i corner_pos,
		const int step,
		const Vector3f local_origin,
		const float pos_scale,
		CellSolveResult &out
) {
	float s[8];
	uint8_t mask = 0;
	for (unsigned int i = 0; i < 8; ++i) {
		const uint32_t bi = Vector3iUtil::get_zxy_index(corner_pos + g_corner_offsets[i] * step, sdf.size);
		out.corner_buffer_indices[i] = bi;
		s[i] = sdf.data[bi];
		if (s[i] < 0.f) {
			mask |= (1 << i);
		}
	}
	if (mask == 0 || mask == 0xff) {
		// Cell is fully inside or fully outside, no surface crosses it.
		return false;
	}
	out.solid_mask = mask;

	Vector3f sum;
	int count = 0;
	for (unsigned int e = 0; e < 12; ++e) {
		const int ia = g_edges[e][0];
		const int ib = g_edges[e][1];
		const bool solid_a = s[ia] < 0.f;
		const bool solid_b = s[ib] < 0.f;
		if (solid_a == solid_b) {
			continue;
		}
		const float t = s[ia] / (s[ia] - s[ib]);
		const Vector3f pa = to_vec3f(g_corner_offsets[ia]) * float(step);
		const Vector3f pb = to_vec3f(g_corner_offsets[ib]) * float(step);
		sum += pa + (pb - pa) * t;
		++count;
	}
	// `count` is always >= 2 here since `mask` is neither 0 nor 0xff.
	const Vector3f local = sum / float(count);

	// Approximate normal from this cell's own corner samples only (naive Surface Nets;
	// no QEF, no data beyond this cell). Points toward increasing SDF, i.e. away from matter.
	const float gx = ((s[1] - s[0]) + (s[3] - s[2]) + (s[5] - s[4]) + (s[7] - s[6])) * 0.25f;
	const float gy = ((s[2] - s[0]) + (s[3] - s[1]) + (s[6] - s[4]) + (s[7] - s[5])) * 0.25f;
	const float gz = ((s[4] - s[0]) + (s[5] - s[1]) + (s[6] - s[2]) + (s[7] - s[3])) * 0.25f;
	const Vector3f gradient(gx, gy, gz);
	const float len2 = math::length_squared(gradient);
	out.normal = len2 > 1e-12f ? gradient / Math::sqrt(len2) : Vector3f(0.f, 1.f, 0.f);
	out.local_position = (local_origin + local) * pos_scale;
	return true;
}

// Little-endian, matching the decode4u8 every voxel shader uses:
// uvec4(i & 0xff, i >> 8 & 0xff, i >> 16 & 0xff, i >> 24 & 0xff).
inline uint32_t pack_bytes(const uint8_t a, const uint8_t b, const uint8_t c, const uint8_t d) {
	return static_cast<uint32_t>(a) | (static_cast<uint32_t>(b) << 8) | (static_cast<uint32_t>(c) << 16) |
			(static_cast<uint32_t>(d) << 24);
}

// Averages this cell's blend weights over the corners that contain matter, remapped into the block's
// 4 shared slots. One vertex per cell, so there is no edge to interpolate along the way Transvoxel
// does; the corner average is what puts the vertex's material where the surface actually is.
void push_cell_material(MeshArrays &output, const Cache &cache, const CellSolveResult &r) {
	uint32_t sums[4] = { 0, 0, 0, 0 };
	unsigned int solid_count = 0;

	for (unsigned int ci = 0; ci < 8; ++ci) {
		if ((r.solid_mask & (1 << ci)) == 0) {
			// Air corners carry no material worth blending in.
			continue;
		}
		++solid_count;
		const uint32_t bi = r.corner_buffer_indices[ci];
		const FixedArray<uint8_t, 4> layers = mixel4::decode_indices_from_packed_u16(cache.material_indices[bi]);
		const FixedArray<uint8_t, 4> weights = mixel4::decode_weights_from_packed_u16(cache.material_weights[bi]);
		for (unsigned int j = 0; j < 4; ++j) {
			const int8_t slot = cache.material_layer_to_slot[layers[j]];
			if (slot >= 0) {
				sums[slot] += weights[j];
			}
		}
	}

	uint8_t out_weights[4];
	for (unsigned int j = 0; j < 4; ++j) {
		out_weights[j] = solid_count > 0 ? static_cast<uint8_t>(sums[j] / solid_count) : uint8_t(0);
	}
	// A cell straddling the isosurface always has a solid corner, but all its weight can land on
	// layers this block dropped. The shader reads all-zero weights as "no data" and paints layer 0,
	// which shows up as speckle; pin the vertex to the block's dominant layer instead.
	if (out_weights[0] == 0 && out_weights[1] == 0 && out_weights[2] == 0 && out_weights[3] == 0) {
		out_weights[0] = 255;
	}

	// x = packed indices, y = packed weights, both bit-cast into the floats of ARRAY_CUSTOM1.
	const uint32_t packed_weights = pack_bytes(out_weights[0], out_weights[1], out_weights[2], out_weights[3]);
	Vector2f data;
	memcpy(&data.x, &cache.material_packed_indices, sizeof(float));
	memcpy(&data.y, &packed_weights, sizeof(float));
	output.texturing_data.push_back(data);
}

inline int32_t push_vertex(
		MeshArrays &output,
		const Cache &cache,
		const CellSolveResult &r,
		const uint8_t transition_tag = 0
) {
	const int32_t vi = output.vertices.size();
	output.vertices.push_back(r.local_position);
	output.normals.push_back(r.normal);
	output.transition_tag.push_back(transition_tag);
	if (cache.materials_enabled) {
		push_cell_material(output, cache, r);
	}
	return vi;
}

// Emits a quad (as two triangles) for the 4 cells sharing a bipolar edge, winding so the
// triangle faces outward (toward the "outside" corner of the edge).
void emit_quad(MeshArrays &output, const bool solid_at_edge_start, int32_t v0, int32_t v1, int32_t v2, int32_t v3) {
	// Empirically the opposite of the derived convention renders correctly (front faces were
	// culled, undersides were visible) -- flipped here rather than re-deriving from scratch.
	if (!solid_at_edge_start) {
		output.indices.push_back(v0);
		output.indices.push_back(v1);
		output.indices.push_back(v2);
		output.indices.push_back(v0);
		output.indices.push_back(v2);
		output.indices.push_back(v3);
	} else {
		output.indices.push_back(v0);
		output.indices.push_back(v2);
		output.indices.push_back(v1);
		output.indices.push_back(v0);
		output.indices.push_back(v3);
		output.indices.push_back(v2);
	}
}


} // namespace

namespace {

// Materializes a 16-bit channel into a flat array laid out like the buffer, so callers can index it
// without caring whether the channel was uniform, compressed or raw.
bool fill_u16_channel(const VoxelBuffer &voxels, const unsigned int channel, StdVector<uint16_t> &dst) {
	if (voxels.get_channel_depth(channel) != VoxelBuffer::DEPTH_16_BIT) {
		return false;
	}
	const size_t volume = Vector3iUtil::get_volume_u64(voxels.get_size());
	dst.clear();
	if (voxels.is_uniform(channel)) {
		dst.resize(volume, static_cast<uint16_t>(voxels.get_voxel(Vector3i(), channel)));
		return true;
	}
	Span<const uint8_t> data_bytes;
	if (!voxels.get_channel_as_bytes_read_only(channel, data_bytes)) {
		return false;
	}
	const Span<const uint16_t> src = data_bytes.reinterpret_cast_to<const uint16_t>();
	if (src.size() != volume) {
		return false;
	}
	dst.resize(volume);
	memcpy(dst.data(), src.data(), volume * sizeof(uint16_t));
	return true;
}

// Picks the 4 texture layers this block will blend between: the ones carrying the most weight across
// its solid voxels. Air voxels are excluded because nothing renders there, so letting them vote would
// let a layer that only exists in the empty space above the ground displace one that is actually
// visible.
void select_block_material_layers(Cache &cache) {
	uint64_t layer_weight_sums[16] = {};
	const size_t volume = cache.sdf.size();
	for (size_t i = 0; i < volume; ++i) {
		if (cache.sdf[i] >= 0.f) {
			continue;
		}
		const FixedArray<uint8_t, 4> layers = mixel4::decode_indices_from_packed_u16(cache.material_indices[i]);
		const FixedArray<uint8_t, 4> weights = mixel4::decode_weights_from_packed_u16(cache.material_weights[i]);
		for (unsigned int j = 0; j < 4; ++j) {
			layer_weight_sums[layers[j]] += weights[j];
		}
	}

	// Top 4 by total weight, ties broken by layer index so two blocks with the same content always
	// agree -- otherwise the choice could differ across a block boundary and reintroduce the seam
	// this is meant to avoid.
	uint8_t chosen[4];
	bool taken[16] = {};
	for (unsigned int slot = 0; slot < 4; ++slot) {
		unsigned int best = 0;
		bool found = false;
		for (unsigned int layer = 0; layer < 16; ++layer) {
			if (taken[layer]) {
				continue;
			}
			if (!found || layer_weight_sums[layer] > layer_weight_sums[best]) {
				best = layer;
				found = true;
			}
		}
		taken[best] = true;
		chosen[slot] = static_cast<uint8_t>(best);
	}
	// Ascending, so the blend is unambiguous the same way Transvoxel sorts its own selection.
	math::sort(chosen[0], chosen[1], chosen[2], chosen[3]);

	fill(cache.material_layer_to_slot, int8_t(-1));
	for (unsigned int slot = 0; slot < 4; ++slot) {
		cache.material_layer_to_slot[chosen[slot]] = static_cast<int8_t>(slot);
	}
	cache.material_packed_indices = pack_bytes(chosen[0], chosen[1], chosen[2], chosen[3]);
}

} // namespace

void fill_cache(
		const VoxelBuffer &voxels,
		const unsigned int sdf_channel,
		const bool with_materials,
		Cache &cache
) {
	ZN_PROFILE_SCOPE();
	const Vector3i size = voxels.get_size();
	cache.sdf.resize(Vector3iUtil::get_volume_u64(size));

	cache.materials_enabled = with_materials &&
			fill_u16_channel(voxels, VoxelBuffer::CHANNEL_INDICES, cache.material_indices) &&
			fill_u16_channel(voxels, VoxelBuffer::CHANNEL_WEIGHTS, cache.material_weights);
	// ponytail: one `get_voxel_f` per voxel, in the buffer's own ZXY order so the writes are linear.
	// Ceiling: still a per-voxel depth switch and decode. Upgrade path if this shows up in a profile
	// is a depth-templated bulk decode like transvoxel's `get_or_decompress_channel`.
	float *w = cache.sdf.data();
	Vector3i p;
	for (p.z = 0; p.z < size.z; ++p.z) {
		for (p.x = 0; p.x < size.x; ++p.x) {
			for (p.y = 0; p.y < size.y; ++p.y) {
				*w++ = static_cast<float>(voxels.get_voxel_f(p, sdf_channel));
			}
		}
	}

	// Needs the SDF, so it has to come after the loop above.
	if (cache.materials_enabled) {
		select_block_material_layers(cache);
	}
}

void build_regular_mesh(
		const VoxelBuffer &voxels,
		const unsigned int sdf_channel,
		const unsigned int lod_index,
		MeshArrays &output
) {
	Cache cache;
	fill_cache(voxels, sdf_channel, false, cache);
	build_regular_mesh(voxels, lod_index, cache, output);
}

void build_regular_mesh(
		const VoxelBuffer &voxels,
		const unsigned int lod_index,
		Cache &cache,
		MeshArrays &output
) {
	ZN_PROFILE_SCOPE();

	// Mesh blocks are placed with a translation-only transform, so positions must come out in LOD0
	// voxel units rather than this LOD's cell units (same as Transvoxel's `<< lod_index`).
	const float pos_scale = float(1 << lod_index);

	const Vector3i size_with_padding = voxels.get_size();
	const Vector3i block_size = size_with_padding - Vector3iUtil::create(MIN_PADDING + MAX_PADDING);

	if (block_size.x <= 0 || block_size.y <= 0 || block_size.z <= 0) {
		return;
	}
	ZN_ASSERT_RETURN(cache.sdf.size() == Vector3iUtil::get_volume_u64(size_with_padding));
	const SdfView sdf{ cache.sdf.data(), size_with_padding };

	// One extra layer of cells towards the negative axes (using the negative padding voxels), so the
	// quads owned at corner 0 can reach their tangent neighbour cells at -1.
	const Vector3i grid_dims = block_size + Vector3iUtil::create(1);
	StdVector<int32_t> &cell_vertex = cache.cell_vertex;
	cell_vertex.clear();
	cell_vertex.resize(Vector3iUtil::get_volume_u64(grid_dims), -1);

	auto grid_index = [&grid_dims](Vector3i r) {
		// r is in [-1, block_size-1]; shift by 1 to index into the dense array.
		return Vector3iUtil::get_zxy_index(r + Vector3iUtil::create(1), grid_dims);
	};

	Vector3i r;
	for (r.z = -1; r.z < block_size.z; ++r.z) {
		for (r.x = -1; r.x < block_size.x; ++r.x) {
			for (r.y = -1; r.y < block_size.y; ++r.y) {
				const Vector3i corner_pos = r + Vector3iUtil::create(MIN_PADDING);
				CellSolveResult result;
				if (solve_cell_vertex(sdf, corner_pos, 1, to_vec3f(r), pos_scale, result)) {
					cell_vertex[grid_index(r)] = push_vertex(output, cache, result);
				}
			}
		}
	}

	// A quad belongs to the block containing its edge's corner, so `r` ranges over [0, block_size)
	// only -- same ownership Transvoxel uses (its cell loop is min_pos..max_pos). Emitting for
	// corners at -1 as well would duplicate the neighbour's boundary quads and, worse, hang geometry
	// outside this block's own volume, which at coarse LODs visibly intrudes into finer-LOD blocks.
	// The tangent neighbours (r - step_b etc.) still reach -1, which the vertex pass above covers.
	for (r.z = 0; r.z < block_size.z; ++r.z) {
		for (r.x = 0; r.x < block_size.x; ++r.x) {
			for (r.y = 0; r.y < block_size.y; ++r.y) {
				const int32_t v_here = cell_vertex[grid_index(r)];
				if (v_here < 0) {
					continue;
				}
				const Vector3i corner_pos = r + Vector3iUtil::create(MIN_PADDING);

				for (unsigned int axis = 0; axis < 3; ++axis) {
					Vector3i step_a;
					step_a[axis] = 1;
					const float s0 = sdf.at(corner_pos);
					const float s1 = sdf.at(corner_pos + step_a);
					const bool solid0 = s0 < 0.f;
					if (solid0 == (s1 < 0.f)) {
						continue;
					}

					const unsigned int axis_b = (axis + 1) % 3;
					const unsigned int axis_d = (axis + 2) % 3;

					Vector3i step_b;
					step_b[axis_b] = 1;
					Vector3i step_d;
					step_d[axis_d] = 1;

					const int32_t v_b = cell_vertex[grid_index(r - step_b)];
					const int32_t v_d = cell_vertex[grid_index(r - step_d)];
					const int32_t v_bd = cell_vertex[grid_index(r - step_b - step_d)];
					if (v_b < 0 || v_d < 0 || v_bd < 0) {
						// One of the 4 cells sharing this edge has no surface; skip (can happen at
						// non-manifold-looking configurations, naive Surface Nets doesn't handle
						// every case perfectly).
						continue;
					}

					emit_quad(output, solid0, v_here, v_b, v_bd, v_d);
				}
			}
		}
	}
}

void build_skirts(MeshArrays &output, const unsigned int lod_index, const float depth_in_cells) {
	ZN_PROFILE_SCOPE();

	if (depth_in_cells <= 0.f) {
		return;
	}
	const size_t triangle_count = output.indices.size() / 3;
	if (triangle_count == 0) {
		return;
	}
	// Positions are in LOD0 voxel units (see build_regular_mesh), so a depth expressed in this LOD's
	// cells has to be scaled the same way -- the seam it hides grows with the LOD too.
	const float depth = depth_in_cells * float(1 << lod_index);

	struct BoundaryEdge {
		int32_t a;
		int32_t b;
		uint32_t use_count;
	};

	// Undirected edge key -> slot in `edges`. Kept as a vector so the emitted geometry is in a
	// deterministic order rather than whatever the hash map happens to iterate in.
	StdUnorderedMap<uint64_t, uint32_t> edge_slots;
	StdVector<BoundaryEdge> edges;
	edge_slots.reserve(output.indices.size());
	edges.reserve(output.indices.size());

	for (size_t t = 0; t < triangle_count; ++t) {
		for (unsigned int e = 0; e < 3; ++e) {
			const int32_t a = output.indices[t * 3 + e];
			const int32_t b = output.indices[t * 3 + (e + 1) % 3];
			if (a == b) {
				// Degenerate triangles have no boundary to speak of.
				continue;
			}
			const uint32_t lo = static_cast<uint32_t>(math::min(a, b));
			const uint32_t hi = static_cast<uint32_t>(math::max(a, b));
			const uint64_t key = (static_cast<uint64_t>(lo) << 32) | hi;
			const auto it = edge_slots.find(key);
			if (it == edge_slots.end()) {
				edge_slots.insert({ key, static_cast<uint32_t>(edges.size()) });
				// Keep the direction this triangle saw it in, so the skirt quad inherits its winding.
				edges.push_back(BoundaryEdge{ a, b, 1 });
			} else {
				++edges[it->second].use_count;
			}
		}
	}

	// A vertex on the rim is shared by two boundary edges, so its extruded twin is created once and
	// reused -- otherwise the curtain would be a set of disconnected quads with cracks between them.
	StdUnorderedMap<int32_t, int32_t> skirt_twins;
	auto get_skirt_twin = [&output, &skirt_twins, depth](const int32_t v) -> int32_t {
		const auto it = skirt_twins.find(v);
		if (it != skirt_twins.end()) {
			return it->second;
		}
		const int32_t twin = static_cast<int32_t>(output.vertices.size());
		// Extrude along -normal, i.e. into the matter the surface encloses, so the curtain hangs
		// beneath the surface and is hidden whenever it isn't needed.
		const Vector3f position = output.vertices[v] - output.normals[v] * depth;
		const Vector3f normal = output.normals[v];
		output.vertices.push_back(position);
		output.normals.push_back(normal);
		output.transition_tag.push_back(0);
		if (output.texturing_data.size() > 0) {
			output.texturing_data.push_back(output.texturing_data[v]);
		}
		skirt_twins.insert({ v, twin });
		return twin;
	};

	for (const BoundaryEdge &edge : edges) {
		if (edge.use_count != 1) {
			continue;
		}
		const int32_t a2 = get_skirt_twin(edge.a);
		const int32_t b2 = get_skirt_twin(edge.b);
		output.indices.push_back(edge.a);
		output.indices.push_back(edge.b);
		output.indices.push_back(b2);
		output.indices.push_back(edge.a);
		output.indices.push_back(b2);
		output.indices.push_back(a2);
	}
}

} // namespace zylann::voxel::surface_nets
