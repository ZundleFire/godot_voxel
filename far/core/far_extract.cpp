#include "far_extract.h"

#include <vector>

namespace zylann::voxel::far {

namespace {

// Central-difference gradient of the SDF at a sample, clamped at the slab
// borders to a one-sided difference.
inline Vec3f sample_gradient(const SdfSlab &slab, unsigned int x, unsigned int y, unsigned int z) {
	const unsigned int max_xz = slab.res_xz - 1;
	const unsigned int max_y = slab.height - 1;

	const unsigned int x0 = (x > 0) ? (x - 1) : x;
	const unsigned int x1 = (x < max_xz) ? (x + 1) : x;
	const unsigned int y0 = (y > 0) ? (y - 1) : y;
	const unsigned int y1 = (y < max_y) ? (y + 1) : y;
	const unsigned int z0 = (z > 0) ? (z - 1) : z;
	const unsigned int z1 = (z < max_xz) ? (z + 1) : z;

	// Divide by the actual span so the gradient stays correctly scaled where
	// the difference was one-sided.
	const float dx = (slab.at(x1, y, z) - slab.at(x0, y, z)) / static_cast<float>(x1 - x0 ? (x1 - x0) : 1);
	const float dy = (slab.at(x, y1, z) - slab.at(x, y0, z)) / static_cast<float>(y1 - y0 ? (y1 - y0) : 1);
	const float dz = (slab.at(x, y, z1) - slab.at(x, y, z0)) / static_cast<float>(z1 - z0 ? (z1 - z0) : 1);

	return Vec3f(dx, dy, dz);
}

// The SDF increases outward (negative inside matter), so the outward surface
// normal is the normalized gradient. Interpolated between the two samples that
// bracket the crossing, weighted by where the crossing fell.
inline uint32_t crossing_normal(
		const SdfSlab &slab,
		unsigned int x,
		unsigned int z,
		unsigned int y_lo,
		unsigned int y_hi,
		float t
) {
	const Vec3f g_lo = sample_gradient(slab, x, y_lo, z);
	const Vec3f g_hi = sample_gradient(slab, x, y_hi, z);
	Vec3f g(lerpf(g_lo.x, g_hi.x, t), lerpf(g_lo.y, g_hi.y, t), lerpf(g_lo.z, g_hi.z, t));
	if (g.length_squared() < 1e-16f) {
		g = Vec3f(0.f, 1.f, 0.f);
	}
	return pack_normal_oct16(g.normalized());
}

inline uint16_t sample_material(const SdfSlab &slab, unsigned int x, unsigned int y, unsigned int z) {
	return slab.material ? slab.material[slab.index(x, y, z)] : 0;
}

} // namespace

unsigned int extract_columns_from_sdf(const SdfSlab &slab, const ExtractSettings &settings, ColumnStack *out_stacks) {
	if (!slab.is_valid() || out_stacks == nullptr) {
		return 0;
	}

	const unsigned int res = slab.res_xz;
	const unsigned int height = slab.height;
	const unsigned int top_y = height - 1;

	unsigned int non_empty = 0;

	for (unsigned int z = 0; z < res; ++z) {
		for (unsigned int x = 0; x < res; ++x) {
			ColumnStack &stack = out_stacks[z * res + x];
			stack.clear();

			const float *column = slab.sdf + slab.index(x, 0, z);

			// Walk downward, which produces layers already sorted top-down.
			//
			// `open_top` is the world Y of the top of the solid span currently
			// being accumulated; the span is closed when the field turns
			// positive again.
			bool inside = column[top_y] < 0.f;
			float open_top = 0.f;
			uint32_t open_normal = pack_normal_oct16(Vec3f(0.f, 1.f, 0.f));
			uint16_t open_material = 0;
			uint8_t open_flags = 0;

			if (inside) {
				// The column is already solid at the very top of the sampled
				// range, so its top is a clip plane rather than a surface. The
				// mesher must not cap it -- that would draw a flat lid across
				// the top of the world.
				open_top = slab.world_y(top_y);
				open_normal = pack_normal_oct16(Vec3f(0.f, 1.f, 0.f));
				open_material = sample_material(slab, x, top_y, z);
				open_flags = 0;
			}

			for (unsigned int i = top_y; i > 0; --i) {
				const unsigned int y_hi = i;
				const unsigned int y_lo = i - 1;
				const float s_hi = column[y_hi];
				const float s_lo = column[y_lo];

				const bool solid_hi = s_hi < 0.f;
				const bool solid_lo = s_lo < 0.f;

				if (solid_hi == solid_lo) {
					continue;
				}

				// Linear interpolation of the zero crossing between the two
				// samples. `t` is measured from y_lo toward y_hi.
				const float denom = s_hi - s_lo;
				const float t = (std::abs(denom) > 1e-20f) ? clampf(-s_lo / denom, 0.f, 1.f) : 0.5f;
				const float crossing_y = lerpf(slab.world_y(y_lo), slab.world_y(y_hi), t);
				const uint32_t normal = crossing_normal(slab, x, z, y_lo, y_hi, t);

				if (!solid_hi && solid_lo) {
					// Air above, matter below: this is a top surface. Open a
					// new span.
					inside = true;
					open_top = crossing_y;
					open_normal = normal;
					open_material = sample_material(slab, x, y_lo, z);
					open_flags = LAYER_FLAG_TOP_IS_SURFACE;
				} else {
					// Matter above, air below: the span being accumulated ends.
					if (inside) {
						ColumnLayer layer;
						layer.top = open_top;
						layer.bottom = crossing_y;
						layer.normal = open_normal;
						layer.material = open_material;
						layer.flags = open_flags | LAYER_FLAG_BOTTOM_IS_SURFACE;
						layer.ao = 255;
						stack.push(layer);
					}
					inside = false;
				}
			}

			if (inside) {
				// Solid all the way down to the bottom of the sampled range.
				// Same reasoning as the top clip: no bottom surface here.
				ColumnLayer layer;
				layer.top = open_top;
				layer.bottom = slab.world_y(0);
				layer.normal = open_normal;
				layer.material = open_material;
				layer.flags = open_flags;
				layer.ao = 255;
				stack.push(layer);
			}

			stack.sort_top_down();
			stack.simplify(settings.min_gap, settings.min_thickness);

			if (stack.count > 0) {
				++non_empty;
			}
		}
	}

	if (settings.compute_ao) {
		compute_column_ao(out_stacks, res, slab.step, settings.ao_strength);
	}

	return non_empty;
}

// ---------------------------------------------------------------------------

void compute_column_ao(ColumnStack *stacks, unsigned int res_xz, float cell_size, float strength) {
	if (stacks == nullptr || res_xz == 0 || strength <= 0.f) {
		return;
	}

	// Cache the top surface of every column once. Reading it back out of the
	// stacks inside the 8-neighbour loop would be 8x the pointer chasing.
	std::vector<float> tops(static_cast<size_t>(res_xz) * res_xz);
	std::vector<uint8_t> has_top(static_cast<size_t>(res_xz) * res_xz, 0);

	for (unsigned int i = 0; i < res_xz * res_xz; ++i) {
		if (stacks[i].count > 0) {
			tops[i] = stacks[i].layers[0].top;
			has_top[i] = 1;
		}
	}

	// A neighbour one cell away and `cell_size` higher subtends roughly 45
	// degrees, which is a reasonable "fully occluded in that direction" point.
	// Beyond that the term saturates.
	const float horizon = cell_size * 2.f;
	const float inv_horizon = (horizon > 1e-9f) ? (1.f / horizon) : 0.f;

	static const int OFFSETS[8][2] = { { -1, -1 }, { 0, -1 }, { 1, -1 }, { -1, 0 },
									   { 1, 0 },   { -1, 1 }, { 0, 1 },  { 1, 1 } };

	for (unsigned int z = 0; z < res_xz; ++z) {
		for (unsigned int x = 0; x < res_xz; ++x) {
			const unsigned int i = z * res_xz + x;
			if (!has_top[i]) {
				continue;
			}
			const float my_top = tops[i];

			float occlusion = 0.f;
			float samples = 0.f;

			for (const auto &off : OFFSETS) {
				const int nx = static_cast<int>(x) + off[0];
				const int nz = static_cast<int>(z) + off[1];
				if (nx < 0 || nz < 0 || nx >= static_cast<int>(res_xz) || nz >= static_cast<int>(res_xz)) {
					// Border columns see fewer neighbours. Skipping them rather
					// than assuming open sky avoids a bright seam around every
					// sector; the average over the remaining samples is close
					// enough and continuous across the boundary.
					continue;
				}
				const unsigned int ni = static_cast<unsigned int>(nz) * res_xz + static_cast<unsigned int>(nx);
				samples += 1.f;
				if (!has_top[ni]) {
					continue;
				}
				const float rise = tops[ni] - my_top;
				if (rise > 0.f) {
					occlusion += clampf(rise * inv_horizon, 0.f, 1.f);
				}
			}

			if (samples <= 0.f) {
				continue;
			}

			const float ao = clampf(1.f - (occlusion / samples) * strength, 0.f, 1.f);
			stacks[i].layers[0].ao = static_cast<uint8_t>(std::lround(ao * 255.f));
		}
	}
}

// ---------------------------------------------------------------------------
// Downsampling
// ---------------------------------------------------------------------------

namespace {

struct LayerCluster {
	float top_sum = 0.f;
	float weight_sum = 0.f;
	float bottom_min = 0.f;
	float top_max = 0.f;
	uint32_t normals[4] = { 0, 0, 0, 0 };
	float normal_weights[4] = { 0.f, 0.f, 0.f, 0.f };
	unsigned int normal_count = 0;
	// Materials vote by weight rather than the first one winning. A coarse
	// column is reduced from up to nine fine ones, and taking whichever reached
	// the cluster first takes the corner of the neighbourhood instead of the
	// bulk of it -- one stray column could repaint a whole coarse column, and at
	// distance the materials drifted. Four candidates is plenty: a far-field
	// cell covering more than four distinct materials has no single right
	// answer anyway.
	uint16_t materials[4] = { 0, 0, 0, 0 };
	float material_weights[4] = { 0.f, 0.f, 0.f, 0.f };
	unsigned int material_count = 0;
	float ao_sum = 0.f;
	uint8_t flags = 0;
	unsigned int member_count = 0;
};

} // namespace

// Maximum neighbourhood the downsampler will consider (a 3x3 tent filter).
static constexpr unsigned int MAX_DOWNSAMPLE_GROUP = 9;

void downsample_column_group(
		const ColumnStack *stacks,
		const float *weights,
		unsigned int count,
		float merge_tolerance,
		ColumnStack &out_stack
) {
	out_stack.clear();

	if (count == 0 || count > MAX_DOWNSAMPLE_GROUP) {
		return;
	}

	// Gather every layer of the neighbourhood, tallest first, carrying each
	// source column's filter weight along with it.
	ColumnLayer gathered[MAX_DOWNSAMPLE_GROUP * MAX_LAYERS_PER_COLUMN];
	float gathered_weights[MAX_DOWNSAMPLE_GROUP * MAX_LAYERS_PER_COLUMN];
	unsigned int gathered_count = 0;

	for (unsigned int q = 0; q < count; ++q) {
		if (weights[q] <= 0.f) {
			continue;
		}
		const ColumnStack &src = stacks[q];
		for (unsigned int i = 0; i < src.count; ++i) {
			gathered[gathered_count] = src.layers[i];
			gathered_weights[gathered_count] = weights[q];
			++gathered_count;
		}
	}

	if (gathered_count == 0) {
		return;
	}

	// Insertion sort, top-down, carrying the weights along. At most 72 elements
	// in the worst case and typically under a dozen.
	for (unsigned int i = 1; i < gathered_count; ++i) {
		const ColumnLayer key = gathered[i];
		const float key_weight = gathered_weights[i];
		int j = static_cast<int>(i) - 1;
		while (j >= 0 && gathered[j].top < key.top) {
			gathered[j + 1] = gathered[j];
			gathered_weights[j + 1] = gathered_weights[j];
			--j;
		}
		gathered[j + 1] = key;
		gathered_weights[j + 1] = key_weight;
	}

	// Cluster layers whose tops agree within tolerance. Layers belonging to the
	// same surface across the neighbourhood collapse into one; a cliff running
	// through it produces two clusters, and the resulting column keeps both --
	// which is what lets the mesher still draw a wall there instead of smearing
	// the cliff into a ramp.
	LayerCluster clusters[MAX_DOWNSAMPLE_GROUP * MAX_LAYERS_PER_COLUMN];
	unsigned int cluster_count = 0;

	for (unsigned int i = 0; i < gathered_count; ++i) {
		const ColumnLayer &layer = gathered[i];
		const float filter_weight = gathered_weights[i];

		LayerCluster *target = nullptr;
		if (cluster_count > 0) {
			LayerCluster &last = clusters[cluster_count - 1];
			const float last_mean_top = last.top_sum / last.weight_sum;
			// Overlapping spans belong together regardless of tolerance; that
			// is the case where two source columns describe one thick slab.
			const bool overlaps = layer.top >= last.bottom_min && layer.bottom <= last.top_max;
			if (overlaps || (last_mean_top - layer.top) <= merge_tolerance) {
				target = &last;
			}
		}

		if (target == nullptr) {
			LayerCluster &fresh = clusters[cluster_count++];
			fresh = LayerCluster();
			fresh.bottom_min = layer.bottom;
			fresh.top_max = layer.top;
			target = &fresh;
		}

		// Two weights multiply here. The filter weight says how much this
		// source column counts (the tent kernel). The thickness says how much
		// this particular span counts within its column, so a thick slab
		// dominates the averaged height over a sliver that happened to land
		// nearby rather than both pulling equally.
		const float weight = filter_weight * std::max(layer.thickness(), 1e-3f);
		target->top_sum += layer.top * weight;
		target->weight_sum += weight;
		target->bottom_min = std::min(target->bottom_min, layer.bottom);
		target->top_max = std::max(target->top_max, layer.top);
		target->ao_sum += static_cast<float>(layer.ao) * weight;
		target->flags |= layer.flags;
		++target->member_count;

		// Material vote. An existing candidate accumulates; a new one takes a
		// free slot; past four, the weight goes to the current leader, which
		// keeps the winner stable rather than letting a late arrival displace
		// an established majority.
		{
			unsigned int slot = 4;
			for (unsigned int k = 0; k < target->material_count; ++k) {
				if (target->materials[k] == layer.material) {
					slot = k;
					break;
				}
			}
			if (slot == 4 && target->material_count < 4) {
				slot = target->material_count++;
				target->materials[slot] = layer.material;
			}
			if (slot == 4) {
				unsigned int heaviest = 0;
				for (unsigned int k = 1; k < 4; ++k) {
					if (target->material_weights[k] > target->material_weights[heaviest]) {
						heaviest = k;
					}
				}
				slot = heaviest;
			}
			target->material_weights[slot] += weight;
		}

		if (target->normal_count < 4) {
			target->normals[target->normal_count] = layer.normal;
			target->normal_weights[target->normal_count] = weight;
			++target->normal_count;
		} else {
			// Fold extra normals into the heaviest existing slot rather than
			// dropping them.
			unsigned int heaviest = 0;
			for (unsigned int k = 1; k < 4; ++k) {
				if (target->normal_weights[k] > target->normal_weights[heaviest]) {
					heaviest = k;
				}
			}
			const uint32_t pair[2] = { target->normals[heaviest], layer.normal };
			const float pair_w[2] = { target->normal_weights[heaviest], weight };
			target->normals[heaviest] = blend_normals_oct16(pair, pair_w, 2);
			target->normal_weights[heaviest] += weight;
		}
	}

	for (unsigned int i = 0; i < cluster_count; ++i) {
		const LayerCluster &cluster = clusters[i];
		if (cluster.weight_sum <= 0.f) {
			continue;
		}

		ColumnLayer layer;
		layer.top = cluster.top_sum / cluster.weight_sum;
		layer.bottom = cluster.bottom_min;
		// Averaging can push the mean top below the lowest contributing bottom
		// in pathological cases; keep the span non-degenerate.
		if (layer.top < layer.bottom) {
			layer.top = layer.bottom;
		}
		layer.normal = blend_normals_oct16(cluster.normals, cluster.normal_weights, cluster.normal_count);
		unsigned int winner = 0;
		for (unsigned int k = 1; k < cluster.material_count; ++k) {
			if (cluster.material_weights[k] > cluster.material_weights[winner]) {
				winner = k;
			}
		}
		layer.material = cluster.material_count > 0 ? cluster.materials[winner] : 0;
		layer.ao = static_cast<uint8_t>(clampf(cluster.ao_sum / cluster.weight_sum, 0.f, 255.f));
		layer.flags = cluster.flags;
		if (cluster.member_count > 1) {
			layer.flags |= LAYER_FLAG_MERGED;
		}
		out_stack.push(layer);
	}

	out_stack.sort_top_down();
}

namespace {

// The four child sectors together form one point grid of
// (2 * SECTOR_CELLS + 1) columns per side, with the shared edges counted once.
// This reads a column out of that combined grid by global coordinate.
//
// A global coordinate exactly on the shared edge (== SECTOR_CELLS) is resolved
// to the *low* quadrant. Either choice is correct -- the two quadrants hold the
// same values there -- but it has to be consistent, or a null low quadrant and
// a present high one would silently produce different results depending on
// which way the tie broke.
bool read_group_column(const ColumnField *const *src_fields, unsigned int gx, unsigned int gz, ColumnStack &out_stack) {
	const unsigned int qx = (gx <= SECTOR_CELLS) ? 0u : 1u;
	const unsigned int qz = (gz <= SECTOR_CELLS) ? 0u : 1u;
	const ColumnField *src = src_fields[qz * 2 + qx];
	if (src == nullptr) {
		out_stack.clear();
		return false;
	}
	const unsigned int lx = gx - qx * SECTOR_CELLS;
	const unsigned int lz = gz - qz * SECTOR_CELLS;
	if (lx >= SECTOR_RES || lz >= SECTOR_RES) {
		out_stack.clear();
		return false;
	}
	src->read_column(lx, lz, out_stack);
	return true;
}

} // namespace

void downsample_field(const ColumnField *const *src_fields, float cell_size, ColumnField &out_field) {
	std::vector<ColumnStack> out_stacks(SECTOR_COLUMN_COUNT);

	// What counts as "the same surface" has to scale with how far apart the
	// columns now are. At one cell of tolerance, a 45-degree slope stays a
	// single connected surface through the merge, while anything steeper is
	// kept as separate layers and continues to be drawn as a wall.
	const float merge_tolerance = cell_size;

	// 3x3 tent kernel. Normalisation is irrelevant -- the cluster averaging
	// divides by the accumulated weight -- so the integer form is used as-is.
	static const float TENT[9] = { 1.f, 2.f, 1.f, 2.f, 4.f, 2.f, 1.f, 2.f, 1.f };

	ColumnStack group[MAX_DOWNSAMPLE_GROUP];
	float weights[MAX_DOWNSAMPLE_GROUP];

	const unsigned int group_max = 2 * SECTOR_CELLS; // highest valid global index

	for (unsigned int dz = 0; dz < SECTOR_RES; ++dz) {
		for (unsigned int dx = 0; dx < SECTOR_RES; ++dx) {
			ColumnStack &out_stack = out_stacks[dz * SECTOR_RES + dx];
			out_stack.clear();

			// Destination column (dx, dz) lands exactly on source column
			// (dx * 2, dz * 2) of the combined grid. No half-cell shift, so
			// levels stay registered with each other.
			const unsigned int gx = dx * 2;
			const unsigned int gz = dz * 2;

			const bool on_border = (dx == 0 || dx == SECTOR_CELLS || dz == 0 || dz == SECTOR_CELLS);

			if (on_border) {
				// Point sample. The sector next door computes this same column
				// from a different set of source sectors; only an unfiltered
				// read is guaranteed to give both the same answer.
				if (!read_group_column(src_fields, gx, gz, group[0])) {
					continue;
				}
				weights[0] = 1.f;
				downsample_column_group(group, weights, 1, merge_tolerance, out_stack);
			} else {
				unsigned int count = 0;
				for (int oz = -1; oz <= 1; ++oz) {
					for (int ox = -1; ox <= 1; ++ox) {
						const int sx = static_cast<int>(gx) + ox;
						const int sz = static_cast<int>(gz) + oz;
						if (sx < 0 || sz < 0 || sx > static_cast<int>(group_max) ||
								sz > static_cast<int>(group_max)) {
							continue;
						}
						if (!read_group_column(
									src_fields,
									static_cast<unsigned int>(sx),
									static_cast<unsigned int>(sz),
									group[count]
							)) {
							continue;
						}
						weights[count] = TENT[(oz + 1) * 3 + (ox + 1)];
						++count;
					}
				}
				if (count == 0) {
					continue;
				}
				downsample_column_group(group, weights, count, merge_tolerance, out_stack);
			}

			out_stack.simplify(cell_size * 0.5f, cell_size * 0.25f);
		}
	}

	out_field.build_from_stacks(out_stacks.data());
}

} // namespace zylann::voxel::far
