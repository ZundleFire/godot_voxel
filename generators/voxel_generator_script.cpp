#include "voxel_generator_script.h"
#include "../constants/voxel_string_names.h"
#include "../storage/voxel_buffer_gd.h"
#include "../util/godot/check_ref_ownership.h"
#include "../util/godot/classes/engine.h"
#include "../util/godot/classes/script.h"

#ifdef VOXEL_ENABLE_GPU
#include "../engine/gpu/compute_shader.h"
#include "../engine/gpu/compute_shader_parameters.h"
#include "../engine/gpu/compute_shader_resource.h"
#include "../util/godot/classes/image.h"
// Declared in shaders/fast_noise_lite_shader.h — use extern to avoid duplicate definition
extern const char *g_fast_noise_lite_shader[];
#endif

#ifdef ZN_GODOT
#include "../util/godot/core/class_db.h"
#endif

namespace zylann::voxel {

VoxelGeneratorScript::VoxelGeneratorScript() {}

VoxelGenerator::Result VoxelGeneratorScript::generate_block(VoxelGenerator::VoxelQueryData input) {
	Result result;

	// Create a temporary wrapper so Godot can pass it to scripts
	Ref<godot::VoxelBuffer> buffer_wrapper(
			memnew(godot::VoxelBuffer(static_cast<godot::VoxelBuffer::Allocator>(input.voxel_buffer.get_allocator())))
	);
	buffer_wrapper.instantiate();
	buffer_wrapper->get_buffer().copy_format(input.voxel_buffer);
	buffer_wrapper->get_buffer().create(input.voxel_buffer.get_size());

	{
		ZN_GODOT_CHECK_REF_COUNT_DOES_NOT_CHANGE(buffer_wrapper);
		if (!GDVIRTUAL_CALL(_generate_block, buffer_wrapper, input.origin_in_voxels, input.lod)) {
			WARN_PRINT_ONCE("VoxelGeneratorScript::_generate_block is unimplemented!");
		}
	}

	// The wrapper is discarded
	buffer_wrapper->get_buffer().move_to(input.voxel_buffer);

	// We may expose this to scripts the day it actually gets used
	// if (ret.get_type() == Variant::DICTIONARY) {
	// 	Dictionary d = ret;
	// 	result.max_lod_hint = d.get("max_lod_hint", false);
	// }

	return result;
}

int VoxelGeneratorScript::get_used_channels_mask() const {
	int mask = 0;
	if (!GDVIRTUAL_CALL(_get_used_channels_mask, mask)) {
		WARN_PRINT_ONCE("VoxelGeneratorScript::_get_used_channels_mask is unimplemented!");
	}
	return mask;
}

bool VoxelGeneratorScript::generate_broad_block(VoxelGenerator::VoxelQueryData input) {
	if (!GDVIRTUAL_IS_OVERRIDDEN(_generate_broad_block)) {
		return false;
	}
	Ref<godot::VoxelBuffer> buffer_wrapper(
			memnew(godot::VoxelBuffer(static_cast<godot::VoxelBuffer::Allocator>(input.voxel_buffer.get_allocator())))
	);
	buffer_wrapper.instantiate();
	buffer_wrapper->get_buffer().copy_format(input.voxel_buffer);
	buffer_wrapper->get_buffer().create(input.voxel_buffer.get_size());

	bool handled = false;
	{
		ZN_GODOT_CHECK_REF_COUNT_DOES_NOT_CHANGE(buffer_wrapper);
		GDVIRTUAL_CALL(_generate_broad_block, buffer_wrapper, input.origin_in_voxels, input.lod, handled);
	}

	if (handled) {
		buffer_wrapper->get_buffer().move_to(input.voxel_buffer);
	}
	return handled;
}

bool VoxelGeneratorScript::has_sdf_compute_shader() const {
	String source;
	if (GDVIRTUAL_CALL(_get_sdf_compute_shader, source)) {
		return !source.is_empty();
	}
	return false;
}

#ifdef VOXEL_ENABLE_GPU

bool VoxelGeneratorScript::supports_shaders() const {
	return has_sdf_compute_shader();
}

bool VoxelGeneratorScript::get_shader_source(ShaderSourceData &out_data) const {
	String source;
	if (!GDVIRTUAL_CALL(_get_sdf_compute_shader, source) || source.is_empty()) {
		return false;
	}

	// The user's GLSL must provide a `generate(vec3 pos, out float out_sd)` function
	// compatible with the block generator shader template.
	// Auto-prepend FastNoiseLite GLSL library so users can call fnlCreateState/fnlGetNoise*
	String fnl_lib;
	for (int i = 0; g_fast_noise_lite_shader[i] != nullptr; ++i) {
		fnl_lib += g_fast_noise_lite_shader[i];
	}
	out_data.glsl = fnl_lib + "\n" + source;

	// Single SDF output
	ShaderOutput sdf_out;
	sdf_out.type = ShaderOutput::TYPE_SDF;
	out_data.outputs.push_back(sdf_out);

	// Process texture and buffer resources from _get_sdf_shader_textures()
	// Returns Array of [String name, Image texture] or [String name, PackedByteArray buffer] pairs
	out_data.parameters.clear();
	Array textures;
	if (GDVIRTUAL_CALL(_get_sdf_shader_textures, textures)) {
		for (int i = 0; i < textures.size(); ++i) {
			Array pair = textures[i];
			if (pair.size() >= 2) {
				String tex_name = pair[0];
				if (tex_name.is_empty()) {
					continue;
				}
				Variant resource_var = pair[1];
				if (resource_var.get_type() == Variant::OBJECT) {
					// Image → texture2D binding
					Ref<Image> image = resource_var;
					if (image.is_valid()) {
						ShaderParameter param;
						param.name = tex_name;
						param.resource = ComputeShaderResourceFactory::create_texture_2d(image);
						out_data.parameters.push_back(param);
					}
				} else if (resource_var.get_type() == Variant::PACKED_BYTE_ARRAY) {
					// PackedByteArray → storage buffer binding
					PackedByteArray buffer_data = resource_var;
					if (buffer_data.size() > 0) {
						ShaderParameter param;
						param.name = tex_name;
						param.resource = ComputeShaderResourceFactory::create_storage_buffer(buffer_data);
						out_data.parameters.push_back(param);
					}
				}
			}
		}
	}

	return true;
}

#endif // VOXEL_ENABLE_GPU

String VoxelGeneratorScript::get_sdf_compute_shader_source() const {
	String source;
	if (!GDVIRTUAL_CALL(_get_sdf_compute_shader, source)) {
		return String();
	}
	return source;
}

Dictionary VoxelGeneratorScript::get_sdf_compute_params() const {
	Dictionary params;
	if (!GDVIRTUAL_CALL(_get_sdf_compute_params, params)) {
		return Dictionary();
	}
	return params;
}

void VoxelGeneratorScript::generate_materials(VoxelGenerator::VoxelQueryData input) {
	Ref<godot::VoxelBuffer> buffer_wrapper(
			memnew(godot::VoxelBuffer(static_cast<godot::VoxelBuffer::Allocator>(input.voxel_buffer.get_allocator())))
	);
	buffer_wrapper.instantiate();
	buffer_wrapper->get_buffer().copy_format(input.voxel_buffer);
	buffer_wrapper->get_buffer().create(input.voxel_buffer.get_size());
	// Copy the SDF channel from input (already filled by GPU)
	input.voxel_buffer.copy_to(buffer_wrapper->get_buffer(), true);

	{
		ZN_GODOT_CHECK_REF_COUNT_DOES_NOT_CHANGE(buffer_wrapper);
		if (!GDVIRTUAL_CALL(_generate_materials, buffer_wrapper, input.origin_in_voxels, input.lod)) {
			// If _generate_materials is not implemented, fall back to full _generate_block
			GDVIRTUAL_CALL(_generate_block, buffer_wrapper, input.origin_in_voxels, input.lod);
		}
	}

	buffer_wrapper->get_buffer().move_to(input.voxel_buffer);
}

bool VoxelGeneratorScript::is_runnable() const {
	Ref<Script> my_script = get_script();
	if (my_script.is_null()) {
		return false;
	}
	if (Engine::get_singleton()->is_editor_hint()) {
		return my_script->is_tool();
	}
	return true;
}

void VoxelGeneratorScript::_bind_methods() {
	GDVIRTUAL_BIND(_generate_block, "out_buffer", "origin_in_voxels", "lod");
	GDVIRTUAL_BIND(_generate_broad_block, "out_buffer", "origin_in_voxels", "lod");
	GDVIRTUAL_BIND(_get_used_channels_mask);
	GDVIRTUAL_BIND(_get_sdf_compute_shader);
	GDVIRTUAL_BIND(_get_sdf_compute_params);
	GDVIRTUAL_BIND(_get_sdf_shader_textures);
	GDVIRTUAL_BIND(_generate_materials, "out_buffer", "origin_in_voxels", "lod");
}

} // namespace zylann::voxel
