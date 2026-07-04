#ifndef REGION_FILE_MMAP_H
#define REGION_FILE_MMAP_H

#include <cstdint>
#include <cstddef>

namespace zylann::voxel {

// Read-only memory-mapped view of a region file.
// Provides zero-copy access to block data within the file.
// On Windows: uses CreateFileMapping / MapViewOfFile.
// On Linux/macOS: uses mmap() / munmap().
//
// This is NOT thread-safe by itself. The caller must synchronize access
// (the owning RegionFile or VoxelStreamRegionFiles already does this).
//
class RegionFileMMap {
public:
	RegionFileMMap();
	~RegionFileMMap();

	// Open a read-only memory mapping for the given file.
	// Returns true on success.
	bool open(const char *filepath);

	// Close the memory mapping and release resources.
	void close();

	bool is_open() const;

	// Get the total mapped file size in bytes.
	size_t get_file_size() const;

	// Get a read-only pointer to data at the given offset.
	// Returns nullptr if offset + length exceeds file size.
	const uint8_t *get_data(size_t offset, size_t length) const;

	// Get the base pointer to the entire mapped region.
	const uint8_t *get_base() const;

	// Re-map the file (e.g. after the file grew due to a write).
	// Returns true on success.
	bool remap(const char *filepath);

private:
	// Platform-specific handles are stored as opaque values to avoid
	// leaking <windows.h> / POSIX headers into every includer.
#ifdef _WIN32
	void *_file_handle;   // HANDLE — initialised in ctor
	void *_mapping_handle; // HANDLE — initialised in ctor
#else
	int _fd = -1;
#endif
	const uint8_t *_base = nullptr;
	size_t _file_size = 0;
};

} // namespace zylann::voxel

#endif // REGION_FILE_MMAP_H
