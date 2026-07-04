#include "region_file_mmap.h"
#include "../../util/errors.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace zylann::voxel {

RegionFileMMap::RegionFileMMap()
#ifdef _WIN32
	: _file_handle(INVALID_HANDLE_VALUE), _mapping_handle(nullptr)
#endif
{
}

RegionFileMMap::~RegionFileMMap() {
	close();
}

#ifdef _WIN32

// ===================== Windows implementation =====================

bool RegionFileMMap::open(const char *filepath) {
	close();

	HANDLE fh = CreateFileA(
			filepath,
			GENERIC_READ,
			FILE_SHARE_READ | FILE_SHARE_WRITE,
			nullptr,
			OPEN_EXISTING,
			FILE_ATTRIBUTE_NORMAL,
			nullptr
	);

	if (fh == INVALID_HANDLE_VALUE) {
		return false;
	}

	LARGE_INTEGER file_size;
	if (!GetFileSizeEx(fh, &file_size)) {
		CloseHandle(fh);
		return false;
	}

	_file_size = static_cast<size_t>(file_size.QuadPart);

	if (_file_size == 0) {
		CloseHandle(fh);
		return false;
	}

	HANDLE mh = CreateFileMappingA(
			fh,
			nullptr,
			PAGE_READONLY,
			file_size.HighPart,
			file_size.LowPart,
			nullptr
	);

	if (mh == nullptr) {
		CloseHandle(fh);
		return false;
	}

	_base = static_cast<const uint8_t *>(
			MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0)
	);

	if (_base == nullptr) {
		CloseHandle(mh);
		CloseHandle(fh);
		return false;
	}

	_file_handle = static_cast<void *>(fh);
	_mapping_handle = static_cast<void *>(mh);
	return true;
}

void RegionFileMMap::close() {
	if (_base != nullptr) {
		UnmapViewOfFile(_base);
		_base = nullptr;
	}
	if (_mapping_handle != nullptr) {
		CloseHandle(static_cast<HANDLE>(_mapping_handle));
		_mapping_handle = nullptr;
	}
	if (_file_handle != INVALID_HANDLE_VALUE) {
		CloseHandle(static_cast<HANDLE>(_file_handle));
		_file_handle = INVALID_HANDLE_VALUE;
	}
	_file_size = 0;
}

bool RegionFileMMap::remap(const char *filepath) {
	close();
	return open(filepath);
}

#else

// ===================== POSIX implementation =====================

bool RegionFileMMap::open(const char *filepath) {
	close();

	_fd = ::open(filepath, O_RDONLY);
	if (_fd < 0) {
		return false;
	}

	struct stat st;
	if (fstat(_fd, &st) != 0) {
		::close(_fd);
		_fd = -1;
		return false;
	}

	_file_size = static_cast<size_t>(st.st_size);

	if (_file_size == 0) {
		::close(_fd);
		_fd = -1;
		return false;
	}

	void *mapped = mmap(nullptr, _file_size, PROT_READ, MAP_PRIVATE, _fd, 0);
	if (mapped == MAP_FAILED) {
		::close(_fd);
		_fd = -1;
		return false;
	}

	_base = static_cast<const uint8_t *>(mapped);
	return true;
}

void RegionFileMMap::close() {
	if (_base != nullptr) {
		munmap(const_cast<uint8_t *>(_base), _file_size);
		_base = nullptr;
	}
	if (_fd >= 0) {
		::close(_fd);
		_fd = -1;
	}
	_file_size = 0;
}

bool RegionFileMMap::remap(const char *filepath) {
	close();
	return open(filepath);
}

#endif // _WIN32

// ===================== Common methods =====================

bool RegionFileMMap::is_open() const {
	return _base != nullptr;
}

size_t RegionFileMMap::get_file_size() const {
	return _file_size;
}

const uint8_t *RegionFileMMap::get_data(size_t offset, size_t length) const {
	if (_base == nullptr || offset + length > _file_size) {
		return nullptr;
	}
	return _base + offset;
}

const uint8_t *RegionFileMMap::get_base() const {
	return _base;
}

} // namespace zylann::voxel
