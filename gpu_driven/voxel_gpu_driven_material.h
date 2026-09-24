#pragma once

#include "voxel_gpu_driven_renderer.h"

#ifdef VOXEL_ENABLE_GPU_DRIVEN_RENDERING

#include "servers/rendering/renderer_rd/storage_rd/material_storage.h"
#include "servers/rendering/shader_compiler.h"

#include <memory>

class ShaderMaterial;

namespace zylann::voxel::gpu_driven {

// A terrain ShaderMaterial's spatial shader, translated to run in the GPU-driven renderer.
//
// The shader goes through Godot's own ShaderCompiler (the one Forward+ uses), and its vertex() and fragment() code is
// spliced into templates that feed it the same inputs as the traditional path: VERTEX, NORMAL and CUSTOM0..2 rebuilt
// from the packed vertex (Transvoxel secondary position and flags, MIXEL4 indices/weights, surface data), MODEL_MATRIX
// per chunk, u_transition_mask per chunk. Its outputs (ALBEDO, ROUGHNESS, METALLIC, SPECULAR, EMISSION, AO, NORMAL,
// NORMAL_MAP...) are lit like the built-in path: first directional light, ambient and sky radiance, fog. Uniforms and
// textures come from the material through Godot's MaterialStorage, so hints (source_color, filters, defaults) behave
// the same.
//
// Not supported, the shader is rejected with an error and the built-in shading is used instead: screen, depth and
// normal-roughness textures, instance uniforms, global texture uniforms. Ignored: light() (lighting is built in),
// blend and depth render modes (the terrain is opaque), shadows.
struct MaterialProgram {
	Vector<uint8_t> vertex_spirv;
	Vector<uint8_t> fragment_spirv;
	HashMap<StringName, ShaderLanguage::ShaderNode::Uniform> uniforms;
	Vector<uint32_t> uniform_offsets;
	Vector<ShaderCompiler::GeneratedCode::Texture> texture_uniforms;
	uint32_t ubo_size = 0;
	int cull_mode = 0; // RenderingDevice::PolygonCullMode
};

// Any thread. Returns null and sets r_error when the shader can't run in the GPU-driven renderer.
std::shared_ptr<MaterialProgram> compile_material_program(const String &p_code, const String &p_path, String &r_error);
// Module uninitialization
void free_material_compiler();

// Main thread. The material's current values in the form MaterialStorage expects (textures as RIDs), and the
// shader's default textures.
void get_material_parameters(
		const ShaderMaterial &p_material,
		const MaterialProgram &p_program,
		HashMap<StringName, Variant> &r_params,
		HashMap<StringName, HashMap<int, RID>> &r_default_textures
);

// Render thread. Owns the material's uniform buffer and uniform set (set 2 of the program's shader).
class MaterialUniforms : public RendererRD::MaterialStorage::MaterialData {
public:
	RID uniform_set;

	void set_render_priority(int) override {}
	void set_next_pass(RID) override {}
	bool update_parameters(const HashMap<StringName, Variant> &, bool, bool) override {
		return false;
	}
	~MaterialUniforms() override;
};

// Set index of the material's uniforms in the program's shader
constexpr uint32_t MATERIAL_UNIFORM_SET = 2;
// First binding of Godot's default samplers in set 0
constexpr uint32_t SAMPLERS_BINDING_FIRST_INDEX = 5;
// Global shader uniforms storage buffer in set 0
constexpr uint32_t GLOBAL_UNIFORMS_BINDING = 17;

} // namespace zylann::voxel::gpu_driven

#endif // VOXEL_ENABLE_GPU_DRIVEN_RENDERING
