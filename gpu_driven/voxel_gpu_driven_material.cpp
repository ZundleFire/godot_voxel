#include "voxel_gpu_driven_material.h"

#ifdef VOXEL_ENABLE_GPU_DRIVEN_RENDERING

#include "core/os/mutex.h"
#include "scene/resources/material.h"
#include "servers/rendering/renderer_rd/renderer_scene_render_rd.h"
#include "servers/rendering/rendering_device.h"
#include "servers/rendering/rendering_server.h"
#include "servers/rendering/shader_preprocessor.h"

namespace zylann::voxel::gpu_driven {

namespace {

// Everything both stages declare. Params, the first fields of Scene and the radiance map match the built-in shaders
// in voxel_gpu_driven_renderer.cpp, which share the same buffers; Scene continues with what material code needs.
// Keep `Chunk` in sync with VoxelGpuDrivenRenderer::ChunkGPU.
const char *COMMON_GLSL = R"(
struct Chunk {
	vec4 origin;
	vec4 aabb_min;
	vec4 aabb_max;
	uvec4 info;
};

layout(push_constant, std430) uniform Params {
	mat4 mvp; // terrain-local position relative to the camera -> clip space
	vec4 camera_int;
	vec4 sun_dir; // terrain-local, toward the light. w = 1 if there is a sun
	vec4 sun_color;
	vec4 ambient;
} params;

layout(set = 0, binding = 4, std140) uniform Scene {
	vec4 fog_color_density;
	vec4 fog_params;
	vec4 fog_depth;
	vec4 world_y_row;
	vec4 radiance_xform0;
	vec4 radiance_xform1;
	vec4 radiance_xform2;
	vec4 ibl;
	vec4 radiance_params;
	vec4 camera_frac;
	// Material shaders only
	mat4 view_matrix;
	mat4 inv_view_matrix;
	mat4 projection_matrix;
	mat4 inv_projection_matrix;
	mat4 terrain_matrix; // terrain global transform
	mat4 terrain_to_view; // rotation only: terrain-local directions to view space
	vec4 time_viewport; // TIME, viewport width, height
} scene;

layout(set = 0, binding = 5 + 0) uniform sampler SAMPLER_NEAREST_CLAMP;
layout(set = 0, binding = 5 + 1) uniform sampler SAMPLER_LINEAR_CLAMP;
layout(set = 0, binding = 5 + 2) uniform sampler SAMPLER_NEAREST_WITH_MIPMAPS_CLAMP;
layout(set = 0, binding = 5 + 3) uniform sampler SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP;
layout(set = 0, binding = 5 + 4) uniform sampler SAMPLER_NEAREST_WITH_MIPMAPS_ANISOTROPIC_CLAMP;
layout(set = 0, binding = 5 + 5) uniform sampler SAMPLER_LINEAR_WITH_MIPMAPS_ANISOTROPIC_CLAMP;
layout(set = 0, binding = 5 + 6) uniform sampler SAMPLER_NEAREST_REPEAT;
layout(set = 0, binding = 5 + 7) uniform sampler SAMPLER_LINEAR_REPEAT;
layout(set = 0, binding = 5 + 8) uniform sampler SAMPLER_NEAREST_WITH_MIPMAPS_REPEAT;
layout(set = 0, binding = 5 + 9) uniform sampler SAMPLER_LINEAR_WITH_MIPMAPS_REPEAT;
layout(set = 0, binding = 5 + 10) uniform sampler SAMPLER_NEAREST_WITH_MIPMAPS_ANISOTROPIC_REPEAT;
layout(set = 0, binding = 5 + 11) uniform sampler SAMPLER_LINEAR_WITH_MIPMAPS_ANISOTROPIC_REPEAT;

layout(set = 0, binding = 17, std430) restrict readonly buffer GlobalShaderUniformData {
	vec4 data[];
} global_shader_uniforms;

#ifdef RADIANCE_ARRAY
layout(set = 1, binding = 0) uniform sampler2DArray radiance_octmap;
#else
layout(set = 1, binding = 0) uniform sampler2D radiance_octmap;
#endif

// Per chunk, replaces the material's u_transition_mask the way the traditional path sets it per block
int voxel_transition_mask = 0;
)";

const char *VERTEX_GLSL = R"(
layout(set = 0, binding = 0, std430) restrict readonly buffer Vertices {
	uint vertex_words[];
};
layout(set = 0, binding = 2, std430) restrict readonly buffer Chunks {
	Chunk chunks[];
};

layout(location = 0) out vec3 v_rel;
layout(location = 1) out vec3 v_normal_view;
layout(location = 2) flat out uint v_chunk_flags;
layout(location = 3) flat out vec3 v_chunk_origin;

// Keep in sync with gpu_driven::PackedVertex
const uint VERTEX_WORDS = 8u;
const uint CHUNK_FLAG_FAR = 4u;

vec3 voxel_read_vec3(uint i) {
	return uintBitsToFloat(uvec3(vertex_words[i], vertex_words[i + 1u], vertex_words[i + 2u]));
}

vec3 voxel_decode_oct(vec2 o) {
	vec3 n = vec3(o, 1.0 - abs(o.x) - abs(o.y));
	if (n.z < 0.0) {
		n.xy = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
	}
	return normalize(n);
}

#GLOBALS

void main() {
	const Chunk c = chunks[gl_InstanceIndex];
	const uint base = uint(gl_VertexIndex) * VERTEX_WORDS;
	const uint normal_flags = vertex_words[base + 3u];

	// The traditional path's vertex attributes, rebuilt from the packed vertex (see gpu_vertex_pack.cpp)
	vec3 vertex = voxel_read_vec3(base);
	vec3 normal = voxel_decode_oct(max(vec2(bitfieldExtract(int(normal_flags), 18, 7),
			bitfieldExtract(int(normal_flags), 25, 7)) / 63.0, vec2(-1.0)));
	// Transvoxel: secondary position, and the cell border / vertex border / transition side masks
	const vec3 secondary =
			vertex + vec3(unpackHalf2x16(vertex_words[base + 4u]), unpackHalf2x16(vertex_words[base + 5u]).x);
	const uint idata = (normal_flags & 63u) | (((normal_flags >> 6u) & 63u) << 8u) | (((normal_flags >> 12u) & 63u) << 16u);
	vec4 custom0 = vec4(secondary, uintBitsToFloat(idata));
	// MIXEL4 indices and 8-bit weights, packed one per byte like the Transvoxel mesher's CUSTOM1 (weights were
	// quantized to 4 bits for the GPU path)
	const uvec4 slots = (uvec4(vertex_words[base + 6u]) >> uvec4(0u, 8u, 16u, 24u)) & 0xffu;
	const uvec4 idx = slots & 0xfu;
	const uvec4 w8 = (slots >> 4u) * 17u;
	vec4 custom1 = vec4(uintBitsToFloat(idx.x | (idx.y << 8u) | (idx.z << 16u) | (idx.w << 24u)),
			uintBitsToFloat(w8.x | (w8.y << 8u) | (w8.z << 16u) | (w8.w << 24u)), 0.0, 0.0);
	// Surface data. Far sectors carry baked AO in that word instead, which is not what CUSTOM2 means.
	vec4 custom2 = (c.info.w & CHUNK_FLAG_FAR) != 0u ? vec4(0.0) : unpackUnorm4x8(vertex_words[base + 7u]);
	vec4 custom3 = vec4(0.0);

	vec3 tangent = vec3(0.0);
	vec3 binormal = vec3(0.0);
	vec2 uv = vec2(0.0);
	vec2 uv2 = vec2(0.0);
	vec4 color = vec4(1.0);
	vec4 instance_custom = vec4(0.0);
	uvec4 bone_attrib = uvec4(0u);
	vec4 weight_attrib = vec4(0.0);
	float point_size = 1.0;
	float z_clip_scale = 1.0;
	vec4 position = vec4(0.0);

	mat4 read_model_matrix = scene.terrain_matrix;
	read_model_matrix[3] = scene.terrain_matrix * vec4(c.origin.xyz, 1.0);
	mat3 model_normal_matrix = mat3(read_model_matrix);
	mat4 read_view_matrix = scene.view_matrix;
	mat4 inv_view_matrix = scene.inv_view_matrix;
	mat4 projection_matrix = scene.projection_matrix;
	mat4 inv_projection_matrix = scene.inv_projection_matrix;
	mat4 modelview = read_view_matrix * read_model_matrix;
	mat3 modelview_normal = mat3(modelview);
	vec2 read_viewport_size = scene.time_viewport.yz;
	voxel_transition_mask = int((c.info.w >> 8u) & 0xffu);

#ifdef VERTEX_WORLD_COORDS_USED
	vertex = (read_model_matrix * vec4(vertex, 1.0)).xyz;
	normal = normalize(model_normal_matrix * normal);
#endif

	{
#CODE : VERTEX
	}

#ifdef VERTEX_WORLD_COORDS_USED
	// Back to terrain-local. Loses the camera-relative precision below, fine for world_vertex_coords shaders.
	v_rel = (inverse(scene.terrain_matrix) * vec4(vertex, 1.0)).xyz - params.camera_int.xyz - scene.camera_frac.xyz;
	v_normal_view = normalize(mat3(read_view_matrix) * normal);
#else
	// Camera-relative: chunk origins and camera_int are integers, so their difference is exact at planet scale
	v_rel = (c.origin.xyz - params.camera_int.xyz) + (vertex - scene.camera_frac.xyz);
	v_normal_view = normalize(mat3(scene.terrain_to_view) * normal);
#endif
#ifdef OVERRIDE_POSITION
	gl_Position = position;
#else
	gl_Position = params.mvp * vec4(v_rel, 1.0);
#endif
	v_chunk_flags = c.info.w;
	v_chunk_origin = c.origin.xyz;
}
)";

const char *FRAGMENT_GLSL = R"(
layout(location = 0) in vec3 v_rel;
layout(location = 1) in vec3 v_normal_view;
layout(location = 2) flat in uint v_chunk_flags;
layout(location = 3) flat in vec3 v_chunk_origin;

layout(location = 0) out vec4 frag_color;

// From Godot's oct_inc.glsl
vec2 voxel_oct_wrap(vec2 v) {
	return (1.0 - abs(v.yx)) * vec2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0);
}

vec2 voxel_vec3_to_oct_with_border(vec3 n, vec2 border_size) {
	n /= (abs(n.x) + abs(n.y) + abs(n.z));
	n.xy = (n.z >= 0.0) ? n.xy : voxel_oct_wrap(n.xy);
	return (n.xy * 0.5 + 0.5) * border_size.y + border_size.x;
}

vec3 voxel_sample_radiance(vec3 dir, float lod) {
	dir = vec3(dot(scene.radiance_xform0.xyz, dir), dot(scene.radiance_xform1.xyz, dir),
			dot(scene.radiance_xform2.xyz, dir));
	const float border = scene.radiance_params.x;
	const vec2 uv = voxel_vec3_to_oct_with_border(normalize(dir), vec2(border, 1.0 - border * 2.0));
#ifdef RADIANCE_ARRAY
	const float layer = floor(lod);
	const vec3 a = textureLod(radiance_octmap, vec3(uv, layer), 0.0).rgb;
	const vec3 b = textureLod(radiance_octmap, vec3(uv, min(layer + 1.0, MAX_ROUGHNESS_LOD)), 0.0).rgb;
	return mix(a, b, lod - layer);
#else
	return textureLod(radiance_octmap, uv, lod).rgb;
#endif
}

float voxel_schlick_fresnel(float u) {
	const float m = 1.0 - u;
	const float m2 = m * m;
	return m2 * m2 * m;
}

#GLOBALS

void main() {
	// View space, like Godot's fragment(). Built from the camera-relative position, so it is precise near the camera.
	vec3 vertex = mat3(scene.terrain_to_view) * v_rel;
	vec3 view = -normalize(vertex);
	vec3 normal = v_normal_view;
#ifdef DO_SIDE_CHECK
	if (!gl_FrontFacing) {
		normal = -normal;
	}
#endif
	voxel_transition_mask = int((v_chunk_flags >> 8u) & 0xffu);

	mat4 read_model_matrix = scene.terrain_matrix;
	read_model_matrix[3] = scene.terrain_matrix * vec4(v_chunk_origin, 1.0);
	mat3 model_normal_matrix = mat3(read_model_matrix);
	mat4 read_view_matrix = scene.view_matrix;
	mat4 inv_view_matrix = scene.inv_view_matrix;
	mat4 projection_matrix = scene.projection_matrix;
	mat4 inv_projection_matrix = scene.inv_projection_matrix;
	mat4 modelview = read_view_matrix * read_model_matrix;
	mat3 modelview_normal = mat3(modelview);
	vec2 read_viewport_size = scene.time_viewport.yz;
	vec2 screen_uv = gl_FragCoord.xy / read_viewport_size;

	vec3 albedo = vec3(1.0);
	float alpha = 1.0;
	float premul_alpha = 1.0;
	float metallic = 0.0;
	float specular = 0.5;
	float roughness = 1.0;
	vec3 emission = vec3(0.0);
	float ao = 1.0;
	float ao_light_affect = 0.0;
	vec3 normal_map = vec3(0.5);
	float normal_map_depth = 1.0;
	vec3 bent_normal_map = vec3(0.5);
	float rim = 0.0;
	float rim_tint = 0.0;
	float clearcoat = 0.0;
	float clearcoat_roughness = 0.0;
	float anisotropy = 0.0;
	vec2 anisotropy_flow = vec2(1.0, 0.0);
	float sss_strength = 0.0;
	vec4 transmittance_color = vec4(0.0);
	float transmittance_depth = 0.0;
	float transmittance_boost = 0.0;
	vec3 backlight = vec3(0.0);
	float alpha_scissor_threshold = 1.0;
	float alpha_hash_scale = 1.0;
	float alpha_antialiasing_edge = 0.0;
	vec2 alpha_texture_coordinate = vec2(0.0);
	vec4 custom_fog = vec4(0.0);
	vec4 custom_radiance = vec4(0.0);
	vec4 custom_irradiance = vec4(0.0);
	vec2 uv = vec2(0.0);
	vec2 uv2 = vec2(0.0);
	vec4 color = vec4(1.0);
	vec3 tangent = vec3(0.0);
	vec3 binormal = vec3(0.0);
	vec2 point_coord = vec2(0.5);
	vec4 instance_custom = vec4(0.0);
	vec3 light_vertex = vertex;

	{
#CODE : FRAGMENT
	}

#ifdef ALPHA_SCISSOR_USED
	if (alpha < alpha_scissor_threshold) {
		discard;
	}
#endif
#ifdef NORMAL_MAP_USED
	normal_map.xy = normal_map.xy * 2.0 - 1.0;
	normal_map.z = sqrt(max(0.0, 1.0 - dot(normal_map.xy, normal_map.xy)));
	normal = normalize(mix(normal, tangent * normal_map.x + binormal * normal_map.y + normal * normal_map.z, normal_map_depth));
#endif
	normal = normalize(normal);

#ifdef MODE_UNSHADED
	vec3 out_color = albedo;
#else
	// Same lighting as the built-in GPU-driven shader, in terrain-local space (what the light and sky data are in)
	const vec3 nd = transpose(mat3(scene.terrain_to_view)) * normal;
	const vec3 v = normalize(-v_rel);
	const float ndv = clamp(dot(nd, v), 1e-4, 1.0);
	const vec3 f0 = mix(vec3(0.16 * specular * specular), albedo, metallic);
	const float f90 = clamp(50.0 * f0.g, metallic, 1.0);
	vec3 out_color = emission;

#ifndef AMBIENT_LIGHT_DISABLED
	vec3 ambient = params.ambient.rgb;
	if (scene.ibl.x > 0.0) {
		ambient = mix(ambient, voxel_sample_radiance(nd, MAX_ROUGHNESS_LOD) * scene.ibl.z, scene.ibl.w);
	}
	out_color += albedo * ambient * ao * (1.0 - metallic);
	if (scene.ibl.y > 0.0) {
		vec3 ref = reflect(-v, nd);
		ref = mix(ref, nd, roughness * roughness);
		const float horizon = min(1.0 + dot(ref, nd), 1.0);
		const vec3 reflection =
				voxel_sample_radiance(ref, sqrt(roughness) * MAX_ROUGHNESS_LOD) * scene.ibl.z * horizon * horizon;
		// Godot's environment BRDF approximation
		const vec4 r = roughness * vec4(-1.0, -0.0275, -0.572, 0.022) + vec4(1.0, 0.0425, 1.04, -0.04);
		const float a004 = min(r.x * r.x, exp2(-9.28 * ndv)) * r.x + r.y;
		const vec2 env = vec2(-1.04, 1.04) * a004 + r.zw;
		out_color += reflection * (env.x * f0 + env.y * f90) * ao;
	}
#endif

	if (params.sun_dir.w > 0.0) {
		const vec3 l = params.sun_dir.xyz;
		const float ndl = max(dot(nd, l), 0.0);
		if (ndl > 0.0) {
			const vec3 h = normalize(l + v);
			const float ldh = max(dot(l, h), 0.0);
			const float ndh = max(dot(nd, h), 0.0);
			// Godot's light energy carries a PI that cancels the 1/PI
#ifdef DIFFUSE_LAMBERT
			const float diffuse = ndl;
#else
			const float fd90_minus_1 = 2.0 * ldh * ldh * roughness - 0.5;
			const float diffuse = (1.0 + fd90_minus_1 * voxel_schlick_fresnel(ndv)) *
					(1.0 + fd90_minus_1 * voxel_schlick_fresnel(ndl)) * ndl;
#endif
			vec3 spec = vec3(0.0);
#ifndef SPECULAR_DISABLED
			const float a = roughness * roughness;
			const float a2 = a * a;
			const float dd = ndh * ndh * (a2 - 1.0) + 1.0;
			const float distribution = a2 / (3.14159265359 * dd * dd);
			const float k = a * 0.5;
			const float visibility = 0.25 / ((ndl * (1.0 - k) + k) * (ndv * (1.0 - k) + k));
			const vec3 fresnel = f0 + (f90 - f0) * voxel_schlick_fresnel(ldh);
			spec = distribution * visibility * fresnel * ndl * 3.14159265359;
#endif
			out_color += (albedo * diffuse * (1.0 - metallic) + spec) * params.sun_color.rgb * mix(1.0, ao, ao_light_affect);
		}
	}
#endif // MODE_UNSHADED

#ifndef FOG_DISABLED
	if (scene.fog_params.x > 0.0) {
		const float dist = length(v_rel);
		vec3 fog_color = scene.fog_color_density.rgb;
		if (scene.fog_params.z > 0.001 && params.sun_dir.w > 0.0) {
			fog_color += params.sun_color.rgb * pow(max(dot(v_rel / dist, params.sun_dir.xyz), 0.0), 8.0) *
					scene.fog_params.z;
		}
		float fog_amount;
		if (scene.fog_params.y > 0.5) {
			const float fog_z = smoothstep(scene.fog_depth.x, scene.fog_depth.y, dist);
			fog_amount = pow(fog_z, scene.fog_depth.z) * scene.fog_color_density.w;
		} else {
			fog_amount = 1.0 - exp(min(0.0, -dist * scene.fog_color_density.w));
		}
		if (abs(scene.fog_params.w) >= 0.0001) {
			const vec3 local = v_rel + params.camera_int.xyz + scene.camera_frac.xyz;
			const float y = dot(scene.world_y_row, vec4(local, 1.0));
			fog_amount = max(fog_amount, 1.0 - exp(min(0.0, (y - scene.fog_depth.w) * scene.fog_params.w)));
		}
		out_color = mix(out_color, fog_color, clamp(fog_amount, 0.0, 1.0));
	}
	out_color = mix(out_color, custom_fog.rgb, custom_fog.a);
#endif

	frag_color = vec4(out_color, 1.0);
}
)";

// The one ShaderCompiler. It keeps per-compile state in members, hence the lock. Created on first use and freed
// with the module: as a plain static it would build and destroy StringNames outside their lifetime.
struct CompilerState {
	ShaderCompiler compiler;
	// Written through IdentifierActions during a compile
	int cull_mode = 0;
};
Mutex g_compiler_mutex;
CompilerState *g_compiler_state = nullptr;

void initialize_compiler(CompilerState &g_compiler) {
	ShaderCompiler::DefaultIdentifierActions actions;
	HashMap<StringName, String> &r = actions.renames;

	r["MODEL_MATRIX"] = "read_model_matrix";
	r["MODEL_NORMAL_MATRIX"] = "model_normal_matrix";
	r["VIEW_MATRIX"] = "read_view_matrix";
	r["INV_VIEW_MATRIX"] = "inv_view_matrix";
	r["PROJECTION_MATRIX"] = "projection_matrix";
	r["INV_PROJECTION_MATRIX"] = "inv_projection_matrix";
	r["MODELVIEW_MATRIX"] = "modelview";
	r["MODELVIEW_NORMAL_MATRIX"] = "modelview_normal";
	r["MAIN_CAM_INV_VIEW_MATRIX"] = "inv_view_matrix";

	r["VERTEX"] = "vertex";
	r["NORMAL"] = "normal";
	r["TANGENT"] = "tangent";
	r["BINORMAL"] = "binormal";
	r["POSITION"] = "position";
	r["UV"] = "uv";
	r["UV2"] = "uv2";
	r["COLOR"] = "color";
	r["POINT_SIZE"] = "point_size";
	r["INSTANCE_ID"] = "0";
	r["VERTEX_ID"] = "gl_VertexIndex";
	r["Z_CLIP_SCALE"] = "z_clip_scale";

	r["ALPHA_SCISSOR_THRESHOLD"] = "alpha_scissor_threshold";
	r["ALPHA_HASH_SCALE"] = "alpha_hash_scale";
	r["ALPHA_ANTIALIASING_EDGE"] = "alpha_antialiasing_edge";
	r["ALPHA_TEXTURE_COORDINATE"] = "alpha_texture_coordinate";

	r["TIME"] = "scene.time_viewport.x";
	r["EXPOSURE"] = "1.0";
	r["PI"] = String::num(Math::PI);
	r["TAU"] = String::num(Math::TAU);
	r["E"] = String::num(Math::E);
	r["OUTPUT_IS_SRGB"] = "false";
	r["CLIP_SPACE_FAR"] = "0.0";
	r["IN_SHADOW_PASS"] = "false";
	r["VIEWPORT_SIZE"] = "read_viewport_size";

	r["FRAGCOORD"] = "gl_FragCoord";
	r["FRONT_FACING"] = "gl_FrontFacing";
	r["NORMAL_MAP"] = "normal_map";
	r["NORMAL_MAP_DEPTH"] = "normal_map_depth";
	r["BENT_NORMAL_MAP"] = "bent_normal_map";
	r["ALBEDO"] = "albedo";
	r["ALPHA"] = "alpha";
	r["PREMUL_ALPHA_FACTOR"] = "premul_alpha";
	r["METALLIC"] = "metallic";
	r["SPECULAR"] = "specular";
	r["ROUGHNESS"] = "roughness";
	r["RIM"] = "rim";
	r["RIM_TINT"] = "rim_tint";
	r["CLEARCOAT"] = "clearcoat";
	r["CLEARCOAT_ROUGHNESS"] = "clearcoat_roughness";
	r["ANISOTROPY"] = "anisotropy";
	r["ANISOTROPY_FLOW"] = "anisotropy_flow";
	r["SSS_STRENGTH"] = "sss_strength";
	r["SSS_TRANSMITTANCE_COLOR"] = "transmittance_color";
	r["SSS_TRANSMITTANCE_DEPTH"] = "transmittance_depth";
	r["SSS_TRANSMITTANCE_BOOST"] = "transmittance_boost";
	r["BACKLIGHT"] = "backlight";
	r["AO"] = "ao";
	r["AO_LIGHT_AFFECT"] = "ao_light_affect";
	r["EMISSION"] = "emission";
	r["POINT_COORD"] = "point_coord";
	r["INSTANCE_CUSTOM"] = "instance_custom";
	r["SCREEN_UV"] = "screen_uv";
	r["DEPTH"] = "gl_FragDepth";
	r["FOG"] = "custom_fog";
	r["RADIANCE"] = "custom_radiance";
	r["IRRADIANCE"] = "custom_irradiance";
	r["BONE_INDICES"] = "bone_attrib";
	r["BONE_WEIGHTS"] = "weight_attrib";
	r["CUSTOM0"] = "custom0";
	r["CUSTOM1"] = "custom1";
	r["CUSTOM2"] = "custom2";
	r["CUSTOM3"] = "custom3";
	r["LIGHT_VERTEX"] = "light_vertex";

	r["NODE_POSITION_WORLD"] = "read_model_matrix[3].xyz";
	r["CAMERA_POSITION_WORLD"] = "inv_view_matrix[3].xyz";
	r["CAMERA_DIRECTION_WORLD"] = "inv_view_matrix[2].xyz";
	r["CAMERA_VISIBLE_LAYERS"] = "0xffffffffu";
	r["NODE_POSITION_VIEW"] = "(read_view_matrix * read_model_matrix)[3].xyz";

	r["VIEW_INDEX"] = "0";
	r["VIEW_MONO_LEFT"] = "0";
	r["VIEW_RIGHT"] = "1";
	r["EYE_OFFSET"] = "vec3(0.0)";
	r["VIEW"] = "view";

	actions.usage_defines["NORMAL_MAP"] = "#define NORMAL_MAP_USED\n";
	actions.usage_defines["NORMAL_MAP_DEPTH"] = "@NORMAL_MAP";
	actions.usage_defines["ALPHA_SCISSOR_THRESHOLD"] = "#define ALPHA_SCISSOR_USED\n";
	actions.usage_defines["POSITION"] = "#define OVERRIDE_POSITION\n";

	actions.render_mode_defines["world_vertex_coords"] = "#define VERTEX_WORLD_COORDS_USED\n";
	actions.render_mode_defines["cull_front"] = "#define DO_SIDE_CHECK\n";
	actions.render_mode_defines["cull_disabled"] = "#define DO_SIDE_CHECK\n";
	actions.render_mode_defines["diffuse_lambert"] = "#define DIFFUSE_LAMBERT\n";
	actions.render_mode_defines["specular_disabled"] = "#define SPECULAR_DISABLED\n";
	actions.render_mode_defines["ambient_light_disabled"] = "#define AMBIENT_LIGHT_DISABLED\n";
	actions.render_mode_defines["unshaded"] = "#define MODE_UNSHADED\n";
	actions.render_mode_defines["fog_disabled"] = "#define FOG_DISABLED\n";

	actions.base_texture_binding_index = 1;
	actions.texture_layout_set = MATERIAL_UNIFORM_SET;
	actions.base_uniform_string = "material.";
	// After the template's own varyings (locations 0-3)
	actions.base_varying_index = 4;
	actions.default_filter = ShaderLanguage::FILTER_LINEAR_MIPMAP;
	actions.default_repeat = ShaderLanguage::REPEAT_ENABLE;
	actions.global_buffer_array_variable = "global_shader_uniforms.data";
	// Instance uniforms are rejected before this could be used
	actions.instance_uniform_index_variable = "0u";

	g_compiler.compiler.initialize(actions);
}

String build_stage_source(const char *p_template, const String &p_defines, const String &p_uniforms,
		const String &p_globals, const String &p_code, const char *p_code_marker) {
	const RendererSceneRenderRD &scene_render = *RendererSceneRenderRD::get_singleton();
	String header = "#version 450\n#define MAX_ROUGHNESS_LOD " + itos(scene_render.get_roughness_layers() - 1) + ".0\n";
	if (scene_render.is_using_radiance_octmap_array()) {
		header += "#define RADIANCE_ARRAY\n";
	}
	header += p_defines;
	header += COMMON_GLSL;
	if (!p_uniforms.is_empty()) {
		header += "layout(set = " + itos(MATERIAL_UNIFORM_SET) + ", binding = 0, std140) uniform MaterialUniforms {\n" +
				p_uniforms + "} material;\n";
	}
	return header + String(p_template).replace("#GLOBALS", p_globals).replace(p_code_marker, p_code);
}

// The voxel engine sets these per mesh block in the traditional path. Here they come from the chunk.
String replace_per_block_uniforms(const String &p_code) {
	return p_code.replace("material.m_u_transition_mask", "voxel_transition_mask")
			.replace("material.m_u_lod_fade", "vec2(0.0)");
}

} // namespace

std::shared_ptr<MaterialProgram> compile_material_program(const String &p_code, const String &p_path, String &r_error) {
	String code;
	{
		ShaderPreprocessor preprocessor;
		String preprocessor_error;
		if (preprocessor.preprocess(p_code, p_path, code, &preprocessor_error) != OK) {
			r_error = "preprocessor: " + preprocessor_error;
			return nullptr;
		}
	}
	if (ShaderLanguage::get_shader_type(code) != "spatial") {
		r_error = "only spatial shaders are supported";
		return nullptr;
	}

	std::shared_ptr<MaterialProgram> program = std::make_shared<MaterialProgram>();
	ShaderCompiler::GeneratedCode gen;
	{
		MutexLock lock(g_compiler_mutex);
		if (g_compiler_state == nullptr) {
			g_compiler_state = memnew(CompilerState);
			initialize_compiler(*g_compiler_state);
		}
		CompilerState &g_compiler = *g_compiler_state;
		ShaderCompiler::IdentifierActions actions;
		actions.entry_point_stages["vertex"] = ShaderCompiler::STAGE_VERTEX;
		actions.entry_point_stages["fragment"] = ShaderCompiler::STAGE_FRAGMENT;
		actions.entry_point_stages["light"] = ShaderCompiler::STAGE_FRAGMENT;
		g_compiler.cull_mode = RenderingDevice::POLYGON_CULL_BACK;
		actions.render_mode_values["cull_front"] = Pair<int *, int>(&g_compiler.cull_mode, RenderingDevice::POLYGON_CULL_FRONT);
		actions.render_mode_values["cull_disabled"] =
				Pair<int *, int>(&g_compiler.cull_mode, RenderingDevice::POLYGON_CULL_DISABLED);
		actions.uniforms = &program->uniforms;

		const Error err = g_compiler.compiler.compile(RS::SHADER_SPATIAL, code, &actions, p_path, gen);
		if (err != OK) {
			r_error = "compilation failed, see the shader's own errors";
			return nullptr;
		}
		program->cull_mode = g_compiler.cull_mode;
	}

	if (gen.uses_screen_texture || gen.uses_depth_texture || gen.uses_normal_roughness_texture) {
		r_error = "screen, depth and normal-roughness textures are not available to GPU-driven terrain";
		return nullptr;
	}
	if (gen.uses_global_textures) {
		r_error = "global texture uniforms are not supported";
		return nullptr;
	}
	for (const KeyValue<StringName, ShaderLanguage::ShaderNode::Uniform> &kv : program->uniforms) {
		if (kv.value.scope == ShaderLanguage::ShaderNode::Uniform::SCOPE_INSTANCE) {
			r_error = "instance uniforms are not supported (`" + String(kv.key) + "`)";
			return nullptr;
		}
	}
	if (gen.code.has("light") && !gen.code["light"].strip_edges().is_empty()) {
		WARN_PRINT_ONCE("GPU-driven terrain ignores the material's light() function: lighting is built in.");
	}

	String defines;
	for (const String &d : gen.defines) {
		defines += d;
	}
	const String vertex_source = build_stage_source(VERTEX_GLSL, defines, gen.uniforms,
			replace_per_block_uniforms(gen.stage_globals[ShaderCompiler::STAGE_VERTEX]),
			replace_per_block_uniforms(gen.code.has("vertex") ? gen.code["vertex"] : String()), "#CODE : VERTEX");
	const String fragment_source = build_stage_source(FRAGMENT_GLSL, defines, gen.uniforms,
			replace_per_block_uniforms(gen.stage_globals[ShaderCompiler::STAGE_FRAGMENT]),
			replace_per_block_uniforms(gen.code.has("fragment") ? gen.code["fragment"] : String()), "#CODE : FRAGMENT");

	RenderingDevice &rd = *RenderingServer::get_singleton()->get_rendering_device();
	String error;
	program->vertex_spirv = rd.shader_compile_spirv_from_source(
			RenderingDevice::SHADER_STAGE_VERTEX, vertex_source, RenderingDevice::SHADER_LANGUAGE_GLSL, &error, true);
	if (program->vertex_spirv.is_empty()) {
		r_error = "vertex stage: " + error;
		return nullptr;
	}
	program->fragment_spirv = rd.shader_compile_spirv_from_source(
			RenderingDevice::SHADER_STAGE_FRAGMENT, fragment_source, RenderingDevice::SHADER_LANGUAGE_GLSL, &error, true);
	if (program->fragment_spirv.is_empty()) {
		r_error = "fragment stage: " + error;
		return nullptr;
	}

	program->uniform_offsets = gen.uniform_offsets;
	program->texture_uniforms = gen.texture_uniforms;
	program->ubo_size = gen.uniform_total_size;
	return program;
}

void get_material_parameters(
		const ShaderMaterial &p_material,
		const MaterialProgram &p_program,
		HashMap<StringName, Variant> &r_params,
		HashMap<StringName, HashMap<int, RID>> &r_default_textures
) {
	r_params.clear();
	r_default_textures.clear();
	// Textures go to MaterialStorage as RIDs, the way ShaderMaterial hands them to the RenderingServer
	const auto to_rid = [](const Variant &v) -> Variant {
		if (v.get_type() == Variant::OBJECT) {
			const RID rid = v;
			return rid.is_valid() ? Variant(rid) : Variant();
		}
		return v;
	};
	for (const KeyValue<StringName, ShaderLanguage::ShaderNode::Uniform> &kv : p_program.uniforms) {
		const Variant v = p_material.get_shader_parameter(kv.key);
		if (v.get_type() == Variant::NIL) {
			continue; // MaterialStorage falls back to the uniform's default
		}
		if (v.get_type() == Variant::ARRAY) {
			Array a = v;
			Array converted;
			for (int i = 0; i < a.size(); ++i) {
				converted.push_back(to_rid(a[i]));
			}
			r_params.insert(kv.key, converted);
		} else {
			r_params.insert(kv.key, to_rid(v));
		}
	}
	const Ref<Shader> shader = p_material.get_shader();
	if (shader.is_null()) {
		return;
	}
	for (const ShaderCompiler::GeneratedCode::Texture &t : p_program.texture_uniforms) {
		const int count = MAX(t.array_size, 1);
		for (int i = 0; i < count; ++i) {
			const Ref<Texture> tex = shader->get_default_texture_parameter(t.name, i);
			if (tex.is_valid()) {
				r_default_textures[t.name][i] = tex->get_rid();
			}
		}
	}
}

void free_material_compiler() {
	MutexLock lock(g_compiler_mutex);
	if (g_compiler_state != nullptr) {
		memdelete(g_compiler_state);
		g_compiler_state = nullptr;
	}
}

MaterialUniforms::~MaterialUniforms() {
	free_parameters_uniform_set(uniform_set);
}

} // namespace zylann::voxel::gpu_driven

#endif // VOXEL_ENABLE_GPU_DRIVEN_RENDERING
