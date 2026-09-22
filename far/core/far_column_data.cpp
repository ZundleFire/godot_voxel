#include "far_column_data.h"

namespace zylann::voxel::far {

namespace {

inline void merge_into(ColumnLayer &dst, const ColumnLayer &src) {
	// The merged span covers the union of both. The surviving top surface is
	// whichever was higher, so the visible silhouette is preserved; the same
	// logic applies downward for the bottom.
	if (src.top > dst.top) {
		dst.top = src.top;
		dst.normal = src.normal;
		dst.ao = src.ao;
		dst.material = src.material;
		dst.flags = (dst.flags & ~LAYER_FLAG_TOP_IS_SURFACE) | (src.flags & LAYER_FLAG_TOP_IS_SURFACE);
	}
	if (src.bottom < dst.bottom) {
		dst.bottom = src.bottom;
		dst.flags = (dst.flags & ~LAYER_FLAG_BOTTOM_IS_SURFACE) | (src.flags & LAYER_FLAG_BOTTOM_IS_SURFACE);
	}
	dst.flags |= LAYER_FLAG_MERGED;
}

} // namespace

// ---------------------------------------------------------------------------
// ColumnStack
// ---------------------------------------------------------------------------

void ColumnStack::push(const ColumnLayer &layer) {
	if (count < MAX_LAYERS_PER_COLUMN) {
		layers[count++] = layer;
		return;
	}

	// Full. Rather than dropping the incoming span outright, collapse the pair
	// of adjacent spans separated by the smallest gap -- that pair is the one
	// whose merger changes the silhouette least -- and use the freed slot.
	sort_top_down();

	unsigned int best_i = 0;
	float best_gap = layers[0].bottom - layers[1].top;
	for (unsigned int i = 1; i + 1 < count; ++i) {
		const float gap = layers[i].bottom - layers[i + 1].top;
		if (gap < best_gap) {
			best_gap = gap;
			best_i = i;
		}
	}

	// Would the incoming span itself be the cheapest thing to fold away? If it
	// sits below the lowest kept span and the gap there is larger, prefer
	// merging it into the bottom span instead of disturbing the visible stack.
	const float incoming_gap = layers[count - 1].bottom - layer.top;
	if (incoming_gap >= 0.f && incoming_gap < best_gap) {
		merge_into(layers[count - 1], layer);
		return;
	}

	merge_into(layers[best_i], layers[best_i + 1]);
	for (unsigned int i = best_i + 1; i + 1 < count; ++i) {
		layers[i] = layers[i + 1];
	}
	--count;
	layers[count++] = layer;
	sort_top_down();
}

void ColumnStack::sort_top_down() {
	// Insertion sort: `count` is at most 8, so this beats anything fancier.
	for (unsigned int i = 1; i < count; ++i) {
		const ColumnLayer key = layers[i];
		int j = static_cast<int>(i) - 1;
		while (j >= 0 && layers[j].top < key.top) {
			layers[j + 1] = layers[j];
			--j;
		}
		layers[j + 1] = key;
	}
}

void ColumnStack::simplify(float min_gap, float min_thickness) {
	if (count == 0) {
		return;
	}
	sort_top_down();

	// Pass 1: absorb spans separated by a negligible vertical gap. At high LOD
	// a two-unit cave ceiling is smaller than one cell, so representing it costs
	// a layer and buys nothing visible.
	unsigned int write = 0;
	for (unsigned int read = 0; read < count; ++read) {
		if (write > 0 && (layers[write - 1].bottom - layers[read].top) <= min_gap) {
			merge_into(layers[write - 1], layers[read]);
		} else {
			layers[write++] = layers[read];
		}
	}
	count = static_cast<uint8_t>(write);

	// Pass 2: drop spans too thin to be worth drawing. The topmost span is kept
	// unconditionally -- dropping it would punch a hole in the horizon, which
	// is far more noticeable than an overly thin slab.
	write = 0;
	for (unsigned int read = 0; read < count; ++read) {
		if (read == 0 || layers[read].thickness() >= min_thickness) {
			layers[write++] = layers[read];
		}
	}
	count = static_cast<uint8_t>(write);
}

// ---------------------------------------------------------------------------
// ColumnField
// ---------------------------------------------------------------------------

ColumnField::ColumnField() {
	_offsets.assign(SECTOR_COLUMN_COUNT + 1, 0);
}

void ColumnField::clear() {
	_layers.clear();
	_offsets.assign(SECTOR_COLUMN_COUNT + 1, 0);
	_min_y = 0.f;
	_max_y = 0.f;
}

void ColumnField::build_from_stacks(const ColumnStack *stacks) {
	unsigned int total = 0;
	for (unsigned int i = 0; i < SECTOR_COLUMN_COUNT; ++i) {
		total += stacks[i].count;
	}

	_layers.clear();
	_layers.reserve(total);
	_offsets.resize(SECTOR_COLUMN_COUNT + 1);

	bool any = false;
	float min_y = 0.f;
	float max_y = 0.f;

	uint32_t cursor = 0;
	for (unsigned int i = 0; i < SECTOR_COLUMN_COUNT; ++i) {
		_offsets[i] = cursor;
		const ColumnStack &stack = stacks[i];
		for (unsigned int j = 0; j < stack.count; ++j) {
			const ColumnLayer &layer = stack.layers[j];
			_layers.push_back(layer);
			if (!any) {
				min_y = layer.bottom;
				max_y = layer.top;
				any = true;
			} else {
				min_y = std::min(min_y, layer.bottom);
				max_y = std::max(max_y, layer.top);
			}
		}
		cursor += stack.count;
	}
	_offsets[SECTOR_COLUMN_COUNT] = cursor;

	_min_y = min_y;
	_max_y = max_y;
}

void ColumnField::read_column(unsigned int x, unsigned int z, ColumnStack &out_stack) const {
	out_stack.clear();
	unsigned int count = 0;
	const ColumnLayer *layers = get_column(x, z, count);
	const unsigned int n = std::min(count, MAX_LAYERS_PER_COLUMN);
	for (unsigned int i = 0; i < n; ++i) {
		out_stack.layers[i] = layers[i];
	}
	out_stack.count = static_cast<uint8_t>(n);
}

float ColumnField::get_top_surface(unsigned int x, unsigned int z, float fallback) const {
	unsigned int count = 0;
	const ColumnLayer *layers = get_column(x, z, count);
	if (count == 0) {
		return fallback;
	}
	// Layers are stored top-down, so index 0 is the highest.
	return layers[0].top;
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

namespace {

template <typename T>
inline void write_pod(std::vector<uint8_t> &out, const T &value) {
	const size_t offset = out.size();
	out.resize(offset + sizeof(T));
	std::memcpy(out.data() + offset, &value, sizeof(T));
}

template <typename T>
inline bool read_pod(const uint8_t *bytes, size_t size, size_t &cursor, T &out_value) {
	if (cursor + sizeof(T) > size) {
		return false;
	}
	std::memcpy(&out_value, bytes + cursor, sizeof(T));
	cursor += sizeof(T);
	return true;
}

// Fixed-point height codec. Both directions must agree exactly, so they live
// next to each other.
inline uint16_t quantize_height(float y, float min_y, float inv_range) {
	const float t = clampf((y - min_y) * inv_range, 0.f, 1.f);
	return static_cast<uint16_t>(std::lround(t * 65535.f));
}

inline float dequantize_height(uint16_t q, float min_y, float range) {
	return min_y + (static_cast<float>(q) / 65535.f) * range;
}

} // namespace

void ColumnField::serialize(std::vector<uint8_t> &out_bytes) const {
	out_bytes.clear();

	const uint32_t layer_count = static_cast<uint32_t>(_layers.size());

	write_pod(out_bytes, SERIAL_MAGIC);
	write_pod(out_bytes, SERIAL_VERSION);
	write_pod(out_bytes, static_cast<uint16_t>(SECTOR_RES));
	write_pod(out_bytes, layer_count);
	write_pod(out_bytes, _min_y);
	write_pod(out_bytes, _max_y);

	if (layer_count == 0) {
		return;
	}

	// Per-column counts as bytes rather than the 32-bit offset table: counts are
	// bounded by MAX_LAYERS_PER_COLUMN, and the offsets are a prefix sum that
	// the reader can rebuild. Saves 3 bytes per column, and a run of identical
	// small values is exactly what a general-purpose compressor eats best.
	out_bytes.reserve(out_bytes.size() + SECTOR_COLUMN_COUNT + layer_count * 10);
	for (unsigned int i = 0; i < SECTOR_COLUMN_COUNT; ++i) {
		out_bytes.push_back(static_cast<uint8_t>(_offsets[i + 1] - _offsets[i]));
	}

	const float range = _max_y - _min_y;
	const float inv_range = (range > 1e-9f) ? (1.f / range) : 0.f;

	for (const ColumnLayer &layer : _layers) {
		write_pod(out_bytes, quantize_height(layer.top, _min_y, inv_range));
		write_pod(out_bytes, quantize_height(layer.bottom, _min_y, inv_range));
		write_pod(out_bytes, layer.normal);
		write_pod(out_bytes, layer.material);
		write_pod(out_bytes, layer.flags);
		write_pod(out_bytes, layer.ao);
	}
}

bool ColumnField::deserialize(const uint8_t *bytes, size_t size) {
	clear();

	size_t cursor = 0;
	uint32_t magic = 0;
	uint16_t version = 0;
	uint16_t res = 0;
	uint32_t layer_count = 0;

	if (!read_pod(bytes, size, cursor, magic) || magic != SERIAL_MAGIC) {
		return false;
	}
	if (!read_pod(bytes, size, cursor, version) || version != SERIAL_VERSION) {
		return false;
	}
	if (!read_pod(bytes, size, cursor, res) || res != SECTOR_RES) {
		// A sector resolution change invalidates the cache. Reject rather than
		// misread; the caller treats this as a miss and regenerates.
		return false;
	}
	if (!read_pod(bytes, size, cursor, layer_count)) {
		return false;
	}
	if (!read_pod(bytes, size, cursor, _min_y) || !read_pod(bytes, size, cursor, _max_y)) {
		return false;
	}

	if (layer_count == 0) {
		return true;
	}
	if (layer_count > SECTOR_COLUMN_COUNT * MAX_LAYERS_PER_COLUMN) {
		return false;
	}
	if (cursor + SECTOR_COLUMN_COUNT > size) {
		return false;
	}

	_offsets.resize(SECTOR_COLUMN_COUNT + 1);
	uint32_t running = 0;
	for (unsigned int i = 0; i < SECTOR_COLUMN_COUNT; ++i) {
		_offsets[i] = running;
		running += bytes[cursor + i];
	}
	_offsets[SECTOR_COLUMN_COUNT] = running;
	cursor += SECTOR_COLUMN_COUNT;

	if (running != layer_count) {
		return false;
	}

	const float range = _max_y - _min_y;

	_layers.resize(layer_count);
	for (uint32_t i = 0; i < layer_count; ++i) {
		uint16_t qtop = 0;
		uint16_t qbottom = 0;
		ColumnLayer &layer = _layers[i];
		if (!read_pod(bytes, size, cursor, qtop) || !read_pod(bytes, size, cursor, qbottom) ||
				!read_pod(bytes, size, cursor, layer.normal) || !read_pod(bytes, size, cursor, layer.material) ||
				!read_pod(bytes, size, cursor, layer.flags) || !read_pod(bytes, size, cursor, layer.ao)) {
			clear();
			return false;
		}
		layer.top = dequantize_height(qtop, _min_y, range);
		layer.bottom = dequantize_height(qbottom, _min_y, range);
	}

	return true;
}

} // namespace zylann::voxel::far
