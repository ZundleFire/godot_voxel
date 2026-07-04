#ifndef VOXEL_COMPUTE_SHADER_H
#define VOXEL_COMPUTE_SHADER_H

#include "../../util/godot/core/rid.h"
#include "../../util/godot/core/string.h"
#include "../../util/memory/memory.h"

#include <atomic>

ZN_GODOT_FORWARD_DECLARE(class RenderingDevice)

namespace zylann::voxel {

struct ComputeShaderInternal {
	RID rid;
#if DEBUG_ENABLED
	String debug_name;
#endif

	void clear(RenderingDevice &rd);
	void load_from_glsl(RenderingDevice &rd, String source_text, String name);

	// An invalid instance means the shader failed to compile
	inline bool is_valid() const {
		return rid.is_valid();
	}
};

class ComputeShader;

// See ComputeShaderResourceFactory
struct ComputeShaderFactory {
	ComputeShaderFactory() = delete;

	[[nodiscard]]
	static std::shared_ptr<ComputeShader> create_from_glsl(String source_text, String name);

	[[nodiscard]]
	static std::shared_ptr<ComputeShader> create_invalid();
};

// Thin RAII wrapper around compute shaders created with the `RenderingDevice` held inside `VoxelEngine`.
// If the source can change at runtime, it may be passed around using shared pointers and a new instance may be created,
// rather than clearing the old shader anytime, for thread-safety. A reference should be kept as long as a dispatch of
// this shader is running on the graphics card.
class ComputeShader {
public:
	friend struct ComputeShaderFactory;

	~ComputeShader();

	// Only use on GPU task thread
	RID get_rid() const;

	// Thread-safe: returns true once the GPU thread has finished compiling (whether it succeeded or failed).
	bool is_compilation_complete() const {
		return _compilation_complete.load(std::memory_order_acquire);
	}

	// Thread-safe: returns true if compilation completed AND the shader RID is valid.
	// Always returns false before compilation is complete.
	bool is_valid() const {
		return _compilation_succeeded.load(std::memory_order_acquire);
	}

private:
	ComputeShaderInternal _internal;
	std::atomic<bool> _compilation_complete{ false };
	std::atomic<bool> _compilation_succeeded{ false };
};

} // namespace zylann::voxel

#endif // VOXEL_COMPUTE_SHADER_H
