#ifndef VOXEL_FAR_COLUMN_DATA_H
#define VOXEL_FAR_COLUMN_DATA_H

#include "far_math.h"

#include <cstring>
#include <vector>

namespace zylann::voxel::far {

// ---------------------------------------------------------------------------
// The far-field representation: a multi-layer column field.
//
// Why not just keep voxels at a coarse LOD?
//
//   A 3D voxel volume costs O(n^3). Far terrain is overwhelmingly empty above
//   the surface and overwhelmingly solid below it, so nearly all of that cube
//   carries no information. What actually varies is *where the surface is*, and
//   that is a 2D field. Collapsing the vertical axis into a short run-list per
//   column turns the cost into O(n^2 * k) where k is the number of solid spans
//   in that column -- typically 1, occasionally 2-4 where there are caves,
//   overhangs or floating geometry.
//
//   For a 32x32x384 region that is ~393k voxels versus ~1024 columns * ~1.4
//   layers = ~1400 records. Two to three orders of magnitude, which is what
//   makes a horizon-scale view distance affordable at all.
//
// Adapting it to smooth (SDF) terrain:
//
//   With cube voxels a layer boundary lands on an integer block face. With an
//   SDF it lands wherever the field crosses zero, so `top`/`bottom` are
//   continuous values obtained by interpolating the crossing, and each layer
//   carries the surface normal sampled from the SDF gradient at that crossing.
//
//   Storing the normal is the single most important quality decision in this
//   format. If you discard it and recompute normals from the coarse heights at
//   mesh time, distant terrain reads as faceted low-poly scenery, because the
//   only gradient information left is the difference between samples that are
//   `1 << lod` units apart. Keeping the normal from the resolution the data was
//   *captured* at preserves the shading of the original surface even after the
//   positions have been decimated hard.
// ---------------------------------------------------------------------------

// Sectors are square in XZ and are sampled on a *point grid with one column of
// padding*: SECTOR_CELLS quads per side require SECTOR_CELLS + 1 sample columns.
//
// The padding is what makes sectors tile exactly. Column SECTOR_CELLS of one
// sector occupies the same world position as column 0 of its neighbour, and
// both are produced by evaluating the same generator at the same point, so they
// agree bit for bit. Meshing a sector therefore never has to read its
// neighbours, which is what allows every sector to be generated and meshed on
// its own thread with no synchronisation. The only seams left to deal with are
// the ones between *different LOD levels*, which no amount of padding can fix
// and which the skirts in the mesher handle instead.
//
// 32 cells keeps a sector's mesh near the point where a single mesh resource is
// still a useful unit of frustum culling, without making the sector count (and
// per-sector overhead) explode.
static constexpr unsigned int SECTOR_CELLS = 32;
static constexpr unsigned int SECTOR_RES = SECTOR_CELLS + 1;
static constexpr unsigned int SECTOR_COLUMN_COUNT = SECTOR_RES * SECTOR_RES;

// Hard cap on solid spans kept per column. Anything beyond this is merged into
// the nearest existing span during extraction. Deep cave systems can exceed it
// at LOD 0, but by the time a column is 16+ units wide the thin stuff has been
// merged away anyway.
static constexpr unsigned int MAX_LAYERS_PER_COLUMN = 8;

static constexpr unsigned int MAX_LOD_COUNT = 24;

enum ColumnLayerFlags : uint8_t {
	// The top surface of this span is a real surface (the field crossed zero
	// there). When false, the span was clipped by the top of the sampled range
	// and its `top` is not a surface -- do not emit a cap for it.
	LAYER_FLAG_TOP_IS_SURFACE = 1 << 0,
	// Same for the bottom.
	LAYER_FLAG_BOTTOM_IS_SURFACE = 1 << 1,
	// Set by the downsampler when several distinct spans were merged into this
	// one. The mesher widens its matching tolerance for such layers, because
	// their `top` is an average rather than a measured surface.
	LAYER_FLAG_MERGED = 1 << 2,
};

// 16 bytes. Keep it that way -- this is the hot array.
struct ColumnLayer {
	// World-space Y of the top and bottom of a solid span.
	float top = 0.f;
	float bottom = 0.f;
	// Octahedral-packed normal of the top surface.
	uint32_t normal = 0;
	// Material / biome index, resolved by the host into a palette lookup.
	uint16_t material = 0;
	uint8_t flags = 0;
	// Ambient occlusion of the top surface, 0 = fully occluded, 255 = open sky.
	// Computed at extraction time where the neighbourhood is still available at
	// full resolution; recovering it at mesh time from decimated data is both
	// more expensive and much less accurate.
	uint8_t ao = 255;

	inline float thickness() const {
		return top - bottom;
	}
	inline bool top_is_surface() const {
		return (flags & LAYER_FLAG_TOP_IS_SURFACE) != 0;
	}
	inline bool bottom_is_surface() const {
		return (flags & LAYER_FLAG_BOTTOM_IS_SURFACE) != 0;
	}
};

static_assert(sizeof(ColumnLayer) == 16, "ColumnLayer is expected to stay 16 bytes");

// Scratch structure used while extracting or merging a single column, before it
// is compacted into the field's flat storage. Fixed capacity so extraction never
// allocates per column.
struct ColumnStack {
	ColumnLayer layers[MAX_LAYERS_PER_COLUMN];
	uint8_t count = 0;

	inline void clear() {
		count = 0;
	}

	// Appends a span. If the column is already full, the two vertically closest
	// adjacent spans are merged to make room, so the tallest / most significant
	// features survive and hairline detail is what gets dropped.
	void push(const ColumnLayer &layer);

	// Merges spans separated by a gap smaller than `min_gap`, and drops spans
	// thinner than `min_thickness`. Called after extraction and after
	// downsampling, where `min_gap`/`min_thickness` scale with the cell size.
	void simplify(float min_gap, float min_thickness);

	// Sorts spans from highest to lowest. Extraction produces them in order
	// already; downsampling does not.
	void sort_top_down();
};

// ---------------------------------------------------------------------------
// ColumnField: the per-sector payload.
//
// Layers are stored in one flat array with a CSR-style per-column offset table,
// rather than a vector-per-column. One allocation, cache-friendly iteration, and
// serialization is a straight memcpy of two buffers.
// ---------------------------------------------------------------------------
class ColumnField {
public:
	ColumnField();

	void clear();

	// Builds the field from `SECTOR_COLUMN_COUNT` stacks in raster order
	// (index = z * SECTOR_RES + x).
	void build_from_stacks(const ColumnStack *stacks);

	inline bool is_empty() const {
		return _layers.empty();
	}

	inline unsigned int get_layer_count() const {
		return static_cast<unsigned int>(_layers.size());
	}

	inline unsigned int get_column_layer_count(unsigned int x, unsigned int z) const {
		const unsigned int i = z * SECTOR_RES + x;
		return _offsets[i + 1] - _offsets[i];
	}

	// Index of a column's first layer within the flat layer array. The mesher
	// uses these flat indices as stable node ids when labelling surface sheets,
	// so it never has to build a side table mapping (column, layer) to an id.
	inline uint32_t get_column_offset(unsigned int x, unsigned int z) const {
		return _offsets[z * SECTOR_RES + x];
	}

	inline const ColumnLayer *get_layers_data() const {
		return _layers.data();
	}

	inline const ColumnLayer *get_column(unsigned int x, unsigned int z, unsigned int &out_count) const {
		const unsigned int i = z * SECTOR_RES + x;
		const uint32_t begin = _offsets[i];
		out_count = _offsets[i + 1] - begin;
		return out_count ? (_layers.data() + begin) : nullptr;
	}

	// Copies a column back out into a stack, for downsampling and for neighbour
	// queries at sector borders.
	void read_column(unsigned int x, unsigned int z, ColumnStack &out_stack) const;

	// Returns the topmost surface height in a column, or `fallback` if the
	// column has no solid span at all. Used by the mesher for skirt depths and
	// by the quadtree for vertical culling bounds.
	float get_top_surface(unsigned int x, unsigned int z, float fallback) const;

	inline float get_min_y() const {
		return _min_y;
	}
	inline float get_max_y() const {
		return _max_y;
	}

	// Approximate resident size, for the memory budget in the streaming layer.
	inline size_t get_size_in_bytes() const {
		return _layers.capacity() * sizeof(ColumnLayer) + _offsets.capacity() * sizeof(uint32_t);
	}

	// -- Serialization ------------------------------------------------------
	//
	// On disk, heights are quantized to 16 bits across the sector's own Y range
	// rather than stored as floats. A sector spans at most a few thousand units
	// vertically, so a 16-bit fixed-point value resolves to well under a
	// millimetre at LOD 0 and is still far finer than the horizontal cell size
	// at any LOD. That halves the height cost and makes the blob compress well,
	// because neighbouring columns share high bytes.

	void serialize(std::vector<uint8_t> &out_bytes) const;
	bool deserialize(const uint8_t *bytes, size_t size);

	static constexpr uint32_t SERIAL_MAGIC = 0x464c4443u; // 'CDLF'
	static constexpr uint16_t SERIAL_VERSION = 1;

private:
	std::vector<ColumnLayer> _layers;
	// SECTOR_COLUMN_COUNT + 1 entries.
	std::vector<uint32_t> _offsets;
	float _min_y = 0.f;
	float _max_y = 0.f;
};

} // namespace zylann::voxel::far

#endif // VOXEL_FAR_COLUMN_DATA_H
