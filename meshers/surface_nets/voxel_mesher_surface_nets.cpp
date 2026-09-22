#include "voxel_mesher_surface_nets.h"
#include "../../shaders/surface_nets_minimal_shader.h"
#include "../../storage/voxel_buffer_gd.h"
#include "../../util/godot/classes/rendering_server.h"
#include "../../util/godot/classes/shader.h"
#include "../../util/godot/classes/shader_material.h"
#include "../../util/godot/core/packed_arrays.h"
#include "../../util/profiling.h"
#include "surface_nets.h"
#include <cstring>

using namespace zylann::godot;

namespace zylann::voxel {

namespace {
Ref<ShaderMaterial> g_minimal_shader_material;

void fill_surface_arrays(Array &arrays, const surface_nets::MeshArrays &src) {
	PackedVector3Array vertices;
	PackedVector3Array normals;
	PackedFloat32Array custom0; // 4*float32, .a = intBitsToFloat(transition_tag)
	PackedInt32Array indices;

	copy_to(vertices, to_span_const(src.vertices));
	copy_to(normals, to_span_const(src.normals));
	copy_to(indices, to_span_const(src.indices));

	PackedFloat32Array texturing_data; // 2*float32, = 4 texture indices + their 4 blend weights

	custom0.resize(src.transition_tag.size() * 4);
	{
		float *w = custom0.ptrw();
		for (size_t i = 0; i < src.transition_tag.size(); ++i) {
			w[i * 4 + 0] = 0.f;
			w[i * 4 + 1] = 0.f;
			w[i * 4 + 2] = 0.f;
			int32_t tag = src.transition_tag[i];
			memcpy(&w[i * 4 + 3], &tag, sizeof(float));
		}
	}

	arrays.resize(Mesh::ARRAY_MAX);
	arrays[Mesh::ARRAY_VERTEX] = vertices;
	arrays[Mesh::ARRAY_NORMAL] = normals;
	arrays[Mesh::ARRAY_CUSTOM0] = custom0;
	if (src.texturing_data.size() != 0) {
		texturing_data.resize(src.texturing_data.size() * 2);
		memcpy(texturing_data.ptrw(), src.texturing_data.data(), texturing_data.size() * sizeof(float));
		arrays[Mesh::ARRAY_CUSTOM1] = texturing_data;
	}
	arrays[Mesh::ARRAY_INDEX] = indices;
}

// Mixel4 needs both channels at 16 bits. Rather than silently produce a mesh with no material data,
// say so once and fall back to no texturing.
VoxelMesherSurfaceNets::TexturingMode check_texturing_mode(
		const VoxelMesherSurfaceNets::TexturingMode expected_mode,
		const VoxelBuffer &voxels
) {
	if (expected_mode != VoxelMesherSurfaceNets::TEXTURES_MIXEL4_S4) {
		return expected_mode;
	}
	if (voxels.get_channel_depth(VoxelBuffer::CHANNEL_INDICES) != VoxelBuffer::DEPTH_16_BIT ||
		voxels.get_channel_depth(VoxelBuffer::CHANNEL_WEIGHTS) != VoxelBuffer::DEPTH_16_BIT) {
		ZN_PRINT_ERROR_ONCE(
				"VoxelMesherSurfaceNets: the Indices and Weights channels must both be 16-bit to use the Mixel4 "
				"texturing mode. Material blending is disabled."
		);
		return VoxelMesherSurfaceNets::TEXTURES_NONE;
	}
	return expected_mode;
}

} // namespace

void VoxelMesherSurfaceNets::load_static_resources() {
	// Guard against module init running before RenderingServer exists (e.g. the minimal `--test`
	// doctest harness), same reasoning as VoxelMesherTransvoxel::load_static_resources.
	if (RenderingServer::get_singleton() == nullptr) {
		return;
	}
	Ref<Shader> shader;
	shader.instantiate();
	shader->set_code(g_surface_nets_minimal_shader);
	g_minimal_shader_material.instantiate();
	g_minimal_shader_material->set_shader(shader);
}

void VoxelMesherSurfaceNets::free_static_resources() {
	g_minimal_shader_material.unref();
}

VoxelMesherSurfaceNets::VoxelMesherSurfaceNets() {
	set_padding(surface_nets::MIN_PADDING, surface_nets::MAX_PADDING);
}

bool VoxelMesherSurfaceNets::is_generating_collision_surface() const {
	// Via the submesh indices set in `build`, which exclude the transition ribbons.
	return true;
}

int VoxelMesherSurfaceNets::get_used_channels_mask() const {
	uint32_t mask = 1 << VoxelBuffer::CHANNEL_SDF;
	if (_texturing_mode == TEXTURES_MIXEL4_S4) {
		mask |= (1 << VoxelBuffer::CHANNEL_INDICES) | (1 << VoxelBuffer::CHANNEL_WEIGHTS);
	}
	return mask;
}

void VoxelMesherSurfaceNets::set_texturing_mode(const TexturingMode mode) {
	ZN_ASSERT_RETURN(mode >= 0 && mode < TEXTURING_MODE_COUNT);
	if (mode != _texturing_mode) {
		_texturing_mode = mode;
		emit_changed();
	}
}

VoxelMesherSurfaceNets::TexturingMode VoxelMesherSurfaceNets::get_texturing_mode() const {
	return _texturing_mode;
}

void VoxelMesherSurfaceNets::build(VoxelMesher::Output &output, const VoxelMesher::Input &input) {
	ZN_PROFILE_SCOPE();

	const VoxelBuffer::ChannelId sdf_channel = VoxelBuffer::CHANNEL_SDF;
	const VoxelBuffer &voxels = input.voxels;

	if (voxels.is_uniform(sdf_channel)) {
		// Uniform SDF has no variations, so it can't cross the isolevel anywhere.
		return;
	}

	// `build` is called concurrently from several meshing threads, so the scratch buffers are
	// per-thread. Reusing them keeps a steady meshing load from allocating a block's worth of vectors
	// every time.
	static thread_local surface_nets::Cache tls_cache;
	static thread_local surface_nets::MeshArrays tls_mesh_arrays;
	surface_nets::MeshArrays &mesh_arrays = tls_mesh_arrays;
	mesh_arrays.clear();

	const TexturingMode texturing_mode = check_texturing_mode(_texturing_mode, voxels);
	surface_nets::fill_cache(voxels, sdf_channel, texturing_mode == TEXTURES_MIXEL4_S4, tls_cache);
	surface_nets::build_regular_mesh(voxels, input.lod_index, tls_cache, mesh_arrays);

	if (mesh_arrays.vertices.size() == 0 || mesh_arrays.indices.size() == 0) {
		return;
	}

	// Skirts are a rendering-only artifact: they hang below the surface to hide LOD seams, and would
	// otherwise become invisible collision walls players walk into. Cutting collision off here, before
	// they're appended, keeps the collider to the real surface. Same mechanism Transvoxel uses.
	output.collision_surface.submesh_vertex_end = mesh_arrays.vertices.size();
	output.collision_surface.submesh_index_end = mesh_arrays.indices.size();

	if (_skirts_enabled && input.lod_hint) {
		// Only worth it under variable LOD -- at a single LOD, neighbouring blocks agree exactly and
		// there is no seam to hide.
		surface_nets::build_skirts(mesh_arrays, input.lod_index, _skirt_depth_cells);
	}

	Array gd_arrays;
	fill_surface_arrays(gd_arrays, mesh_arrays);
	output.surfaces.push_back({ gd_arrays, 0 });
	output.primitive_type = Mesh::PRIMITIVE_TRIANGLES;
	output.mesh_flags = (RenderingServerEnums::ARRAY_CUSTOM_RGBA_FLOAT << Mesh::ARRAY_FORMAT_CUSTOM0_SHIFT);
	if (mesh_arrays.texturing_data.size() != 0) {
		output.mesh_flags |= (RenderingServerEnums::ARRAY_CUSTOM_RG_FLOAT << Mesh::ARRAY_FORMAT_CUSTOM1_SHIFT);
	}
}

void VoxelMesherSurfaceNets::set_skirts_enabled(bool enable) {
	_skirts_enabled = enable;
}

bool VoxelMesherSurfaceNets::get_skirts_enabled() const {
	return _skirts_enabled;
}

void VoxelMesherSurfaceNets::set_skirt_depth_cells(float depth) {
	_skirt_depth_cells = math::max(depth, 0.f);
}

float VoxelMesherSurfaceNets::get_skirt_depth_cells() const {
	return _skirt_depth_cells;
}

Ref<ShaderMaterial> VoxelMesherSurfaceNets::get_default_lod_material() const {
	return g_minimal_shader_material;
}

void VoxelMesherSurfaceNets::_bind_methods() {
	using Self = VoxelMesherSurfaceNets;

	ClassDB::bind_method(D_METHOD("set_skirts_enabled", "enabled"), &Self::set_skirts_enabled);
	ClassDB::bind_method(D_METHOD("get_skirts_enabled"), &Self::get_skirts_enabled);

	ClassDB::bind_method(D_METHOD("set_skirt_depth_cells", "depth"), &Self::set_skirt_depth_cells);
	ClassDB::bind_method(D_METHOD("get_skirt_depth_cells"), &Self::get_skirt_depth_cells);

	ClassDB::bind_method(D_METHOD("set_texturing_mode", "mode"), &Self::set_texturing_mode);
	ClassDB::bind_method(D_METHOD("get_texturing_mode"), &Self::get_texturing_mode);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "skirts_enabled"), "set_skirts_enabled", "get_skirts_enabled");
	ADD_PROPERTY(
			PropertyInfo(Variant::FLOAT, "skirt_depth_cells", PROPERTY_HINT_RANGE, "0.0,8.0,0.1"),
			"set_skirt_depth_cells",
			"get_skirt_depth_cells"
	);
	ADD_PROPERTY(
			PropertyInfo(Variant::INT, "texturing_mode", PROPERTY_HINT_ENUM, "None,Mixel4"),
			"set_texturing_mode",
			"get_texturing_mode"
	);

	BIND_ENUM_CONSTANT(TEXTURES_NONE);
	BIND_ENUM_CONSTANT(TEXTURES_MIXEL4_S4);
}

} // namespace zylann::voxel
