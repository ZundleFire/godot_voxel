#ifndef VOXEL_GENERATOR_SCRIPT_H
#define VOXEL_GENERATOR_SCRIPT_H

#include "../util/godot/core/gdvirtual.h"
#include "voxel_generator.h"

#ifdef ZN_GODOT_EXTENSION
// GodotCpp wants the full definition of the class in GDVIRTUAL
#include "../storage/voxel_buffer_gd.h"
#endif

namespace zylann::voxel {

// Generator based on a script, like GDScript, C# or NativeScript.
// The script is expected to properly handle multithreading.
class VoxelGeneratorScript : public VoxelGenerator {
	GDCLASS(VoxelGeneratorScript, VoxelGenerator)
public:
	VoxelGeneratorScript();

	Result generate_block(VoxelGenerator::VoxelQueryData input) override;
	bool generate_broad_block(VoxelGenerator::VoxelQueryData input) override;
	int get_used_channels_mask() const override;

#ifdef VOXEL_ENABLE_GPU
	bool supports_shaders() const override;
	bool get_shader_source(ShaderSourceData &out_data) const override;
#endif

	// GPU SDF support: if _get_sdf_compute_shader returns a non-empty string,
	// the engine can use GPU acceleration for SDF generation.
	bool has_sdf_compute_shader() const;
	String get_sdf_compute_shader_source() const;
	Dictionary get_sdf_compute_params() const;
	void generate_materials(VoxelGenerator::VoxelQueryData input);

	bool is_runnable() const override;

protected:
	GDVIRTUAL3(_generate_block, Ref<godot::VoxelBuffer>, Vector3i, int)
	GDVIRTUAL3R(bool, _generate_broad_block, Ref<godot::VoxelBuffer>, Vector3i, int)
	GDVIRTUAL0RC(int, _get_used_channels_mask) // I think `C` means `const`?

	// Optional GPU SDF virtuals
	GDVIRTUAL0RC(String, _get_sdf_compute_shader)
	GDVIRTUAL0RC(Dictionary, _get_sdf_compute_params)
	// Called after GPU has filled the SDF channel — user fills material channels only
	GDVIRTUAL3(_generate_materials, Ref<godot::VoxelBuffer>, Vector3i, int)
	// Optional: return Array of [String name, Image texture] pairs for compute shader bindings
	GDVIRTUAL0RC(Array, _get_sdf_shader_textures)

private:
	static void _bind_methods();
};

} // namespace zylann::voxel

#endif // VOXEL_GENERATOR_SCRIPT_H
