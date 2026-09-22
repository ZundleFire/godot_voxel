#pragma once

// No Godot headers: compiles standalone against gpu_driven/tests/.

#include <cstdint>
#include <map>

namespace zylann::voxel::gpu_driven {

// Sub-allocates ranges of a fixed-capacity pool (in arbitrary units, e.g. vertices).
// ponytail: first-fit over a std::map of free ranges, O(free ranges) per allocation. Swap for size-segregated
// free lists if profiling shows it (thousands of fragmented chunks).
class RangeAllocator {
public:
	static constexpr uint32_t INVALID = 0xffffffffu;

	// Returns the offset, or INVALID if no free range is large enough.
	uint32_t allocate(uint32_t size);
	void free(uint32_t offset, uint32_t size);
	// Only grows. The added space is appended at the end and merged with a trailing free range.
	void grow(uint32_t new_capacity);

	uint32_t get_capacity() const {
		return _capacity;
	}
	uint32_t get_used() const {
		return _used;
	}
	uint32_t get_free_range_count() const {
		return static_cast<uint32_t>(_free.size());
	}

private:
	void insert_free(uint32_t offset, uint32_t size);

	// offset -> size
	std::map<uint32_t, uint32_t> _free;
	uint32_t _capacity = 0;
	uint32_t _used = 0;
};

} // namespace zylann::voxel::gpu_driven
