#include "voxel_gpu_driven_renderer.h"

#ifdef VOXEL_ENABLE_GPU_DRIVEN_RENDERING

#include "scene/resources/3d/world_3d.h"
#include "scene/resources/compositor.h"
#include "servers/rendering/renderer_rd/framebuffer_cache_rd.h"
#include "servers/rendering/renderer_rd/renderer_scene_render_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/light_storage.h"
#include "servers/rendering/renderer_rd/storage_rd/render_data_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/render_scene_buffers_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/texture_storage.h"
#include "servers/rendering/rendering_device.h"
#include "servers/rendering/rendering_server.h"
#include "servers/rendering/rendering_server_globals.h"
#include "servers/rendering/storage/camera_attributes_storage.h"
#include "servers/rendering/storage/environment_storage.h"
#include "servers/rendering/storage/render_data.h"
#include "servers/rendering/storage/render_scene_data.h"

#include <cstring>

namespace zylann::voxel {

namespace {

// Keep `Chunk` in sync with VoxelGpuDrivenRenderer::ChunkGPU.
const char *CHUNK_STRUCT_GLSL = R"(
struct Chunk {
	vec4 origin;
	vec4 aabb_min;
	vec4 aabb_max;
	uvec4 info;
};
)";

const char *CULL_SHADER_GLSL = R"(
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) restrict readonly buffer Chunks {
	Chunk chunks[];
};

// Non-indexed VkDrawIndirectCommand: vertex_count, instance_count, first_vertex, first_instance
layout(set = 0, binding = 1, std430) restrict writeonly buffer Commands {
	uvec4 commands[];
};

layout(push_constant, std430) uniform Params {
	// Terrain-local frustum planes, inside when dot(xyz, p) + w >= 0
	vec4 planes[6];
	uint chunk_count;
	uint pad0;
	uint pad1;
	uint pad2;
} params;

void main() {
	const uint i = gl_GlobalInvocationID.x;
	if (i >= params.chunk_count) {
		return;
	}
	const Chunk c = chunks[i];
	bool visible = (c.info.w & 1u) != 0u && c.info.z > 0u;
	if (visible) {
		for (int pi = 0; pi < 6; ++pi) {
			const vec4 plane = params.planes[pi];
			// Corner furthest along the plane normal
			const vec3 p = mix(c.aabb_min.xyz, c.aabb_max.xyz, greaterThan(plane.xyz, vec3(0.0)));
			if (dot(plane.xyz, p) + plane.w < 0.0) {
				visible = false;
				break;
			}
		}
	}
	// Indices are pulled in the vertex shader, so the draw is non-indexed over the chunk's index range,
	// and first_instance carries the chunk index to gl_InstanceIndex.
	commands[i] = uvec4(c.info.z, visible ? 1u : 0u, c.info.y, i);
}
)";

// Shared by the vertex and fragment stages. Keep RenderPushConstant and StyleUBO in sync.
const char *RENDER_COMMON_GLSL = R"(
layout(push_constant, std430) uniform Params {
	// Camera-relative: maps (terrain-local position - camera position) to clip space. At planet scale (~40 km)
	// terrain-local floats only have ~4 mm precision, enough for neighbor chunks to round a shared vertex apart.
	mat4 mvp;
	vec4 camera_int; // floor of the terrain-local camera position (exact in float), see Scene.camera_frac
	vec4 sun_dir; // terrain-local, toward the light. w = 1 if there is a sun
	vec4 sun_color; // linear rgb * energy
	vec4 ambient; // linear rgb * energy
} params;

// Material look, mirrors demo_eden/shaders/planet_v4.gdshader. Values come from the terrain material's shader
// parameters when it has them.
layout(set = 0, binding = 3, std140) uniform Style {
	vec4 p0; // planet_radius (0 = flat, up is +Y), vibrance, slope_rock_start, slope_rock_end
	vec4 p1; // snow_temperature, snow_blend, gully_darkening, ridge_highlight
	vec4 p2; // dryness_tint, detail_normal_strength, detail_normal_scale, unused
} style;

// Environment fog, same model as Godot's scene shader (minus aerial perspective, which needs the sky radiance map)
layout(set = 0, binding = 4, std140) uniform Scene {
	vec4 fog_color_density; // linear rgb * light energy, density
	vec4 fog_params; // enabled, depth mode, sun scatter, height density
	vec4 fog_depth; // begin, end, curve, height
	vec4 world_y_row; // row 1 of the terrain's global transform, for height fog
	// Sky radiance, same data Godot's scene shader uses for ambient and reflections
	vec4 radiance_xform0; // rows of terrain-local -> sky space
	vec4 radiance_xform1;
	vec4 radiance_xform2;
	vec4 ibl; // use ambient cubemap, use reflection cubemap, IBL exposure * bg energy, ambient color/sky mix
	vec4 radiance_params; // uv border size, unused x3
	vec4 camera_frac; // terrain-local camera position minus Params.camera_int
} scene;

#ifdef RADIANCE_ARRAY
layout(set = 1, binding = 0) uniform sampler2DArray radiance_octmap;
#else
layout(set = 1, binding = 0) uniform sampler2D radiance_octmap;
#endif

// EdenPlanetGenerator MAT_* ids: 0=grass 1=rock 2=snow 3=sand 4=dirt 5=moss 6=ocean_floor
// Linear albedos in physically plausible ranges (grass ~0.1-0.2, rock ~0.25, snow ~0.85): dark enough that sunlit
// faces don't wash out, saturated enough to read at distance. Keep in sync with
// gpu_driven/tests/godot/planet_vivid.gdshader (the traditional-path twin used for A/B checks).
vec3 mat_albedo(uint i) {
	if (i == 0u) { return vec3(0.075, 0.20, 0.045); }
	if (i == 1u) { return vec3(0.27, 0.245, 0.215); }
	if (i == 2u) { return vec3(0.82, 0.86, 0.92); }
	if (i == 3u) { return vec3(0.56, 0.44, 0.24); }
	if (i == 4u) { return vec3(0.21, 0.13, 0.065); }
	if (i == 5u) { return vec3(0.05, 0.14, 0.065); }
	// Kept bright on purpose while there is no water over the seabed (see planet_v4.gdshader)
	if (i == 6u) { return vec3(0.20, 0.30, 0.36); }
	return vec3(1.0, 0.0, 1.0);
}

float mat_roughness(uint i) {
	if (i == 2u) { return 0.42; }
	if (i == 1u) { return 0.72; }
	if (i == 3u) { return 0.78; }
	if (i == 6u) { return 0.62; }
	return 0.88;
}
)";

const char *VERTEX_SHADER_GLSL = R"(
layout(set = 0, binding = 0, std430) restrict readonly buffer Vertices {
	uint vertex_words[];
};
// Two 16-bit indices per word, or one 32-bit index per word for chunks flagged wide
layout(set = 0, binding = 1, std430) restrict readonly buffer Indices {
	uint index_words[];
};
layout(set = 0, binding = 2, std430) restrict readonly buffer Chunks {
	Chunk chunks[];
};

layout(location = 0) out vec3 v_normal;
layout(location = 1) out vec3 v_albedo;
layout(location = 2) out vec2 v_vegetation_roughness;
layout(location = 3) out vec4 v_surface;
layout(location = 4) out vec3 v_pos;
layout(location = 5) out vec3 v_rel; // position relative to the camera, precise near it

// Keep in sync with gpu_driven::PackedVertex
const uint VERTEX_WORDS = 8u;
const uint CHUNK_FLAG_WIDE_INDICES = 2u;

vec3 read_vec3(uint i) {
	return uintBitsToFloat(uvec3(vertex_words[i], vertex_words[i + 1u], vertex_words[i + 2u]));
}

vec3 decode_oct(vec2 o) {
	vec3 n = vec3(o, 1.0 - abs(o.x) - abs(o.y));
	if (n.z < 0.0) {
		n.xy = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
	}
	return normalize(n);
}

void main() {
	const Chunk c = chunks[gl_InstanceIndex];
	const uint vi = uint(gl_VertexIndex);
	uint local_index;
	if ((c.info.w & CHUNK_FLAG_WIDE_INDICES) != 0u) {
		local_index = index_words[vi];
	} else {
		local_index = (index_words[vi >> 1u] >> ((vi & 1u) * 16u)) & 0xffffu;
	}
	const uint base = (c.info.x + local_index) * VERTEX_WORDS;

	// Bits 0-5 cell border mask, 6-11 vertex border mask, 12-17 transition sides, 18-31 normal
	const uint normal_flags = vertex_words[base + 3u];

	vec3 pos = read_vec3(base);

	// Same logic as the Transvoxel/SurfaceNets default shaders: move vertices to their secondary position
	// next to lower-resolution neighbors, and collapse transition geometry of inactive sides.
	const uint transition_mask = (c.info.w >> 8u) & 0xffu;
	const uint cell_border_mask = normal_flags & 63u;
	const uint vertex_border_mask = (normal_flags >> 6u) & 63u;
	const uint itransition = (normal_flags >> 12u) & 63u;
	float secondary_factor = float((transition_mask & cell_border_mask) != 0u);
	secondary_factor *= float((vertex_border_mask & ~transition_mask) == 0u);
	if (secondary_factor > 0.0) {
		// Secondary position is stored as a half-float offset from the primary one
		const vec3 offset = vec3(unpackHalf2x16(vertex_words[base + 4u]), unpackHalf2x16(vertex_words[base + 5u]).x);
		pos += offset * secondary_factor;
	}
	pos *= float(itransition == 0u || (itransition & transition_mask) != 0u);

	// Chunk origins and camera_int are integers, so their difference is exact. What gets rounded afterwards is
	// relative to the camera, where the error is smallest.
	v_rel = (c.origin.xyz - params.camera_int.xyz) + (pos - scene.camera_frac.xyz);
	gl_Position = params.mvp * vec4(v_rel, 1.0);
	// Absolute position only drives shading (noise, planet up), where millimeters don't matter
	v_pos = c.origin.xyz + pos;
	// 7-bit snorm octahedral normal (sign-extended by bitfieldExtract)
	v_normal = decode_oct(max(vec2(bitfieldExtract(int(normal_flags), 18, 7),
			bitfieldExtract(int(normal_flags), 25, 7)) / 63.0, vec2(-1.0)));

	// MIXEL4, one byte per slot: index in the low nibble, weight 0-15 in the high nibble. Resolved per vertex then
	// interpolated: neighboring vertices can reference different material sets, so interpolating indices would band.
	const uvec4 slots = (uvec4(vertex_words[base + 6u]) >> uvec4(0u, 8u, 16u, 24u)) & 0xffu;
	const uvec4 idx = slots & 0xfu;
	const vec4 w = vec4(slots >> 4u);
	const float wsum = w.x + w.y + w.z + w.w;
	vec3 albedo = vec3(0.5);
	float roughness = 0.88;
	float vegetation = 0.0;
	if (wsum > 0.0) {
		albedo = vec3(0.0);
		roughness = 0.0;
		for (int k = 0; k < 4; ++k) {
			const float wk = w[k] / wsum;
			albedo += mat_albedo(idx[k]) * wk;
			roughness += mat_roughness(idx[k]) * wk;
			vegetation += (idx[k] == 0u || idx[k] == 5u) ? wk : 0.0;
		}
	}
	v_albedo = albedo;
	v_vegetation_roughness = vec2(vegetation, roughness);

	// Surface data (erosion, ridge, moisture, temperature). Meshes without it read neutral values.
	const uint surface_bits = vertex_words[base + 7u];
	v_surface = surface_bits == 0u ? vec4(0.5) : unpackUnorm4x8(surface_bits);
}
)";

const char *FRAGMENT_SHADER_GLSL = R"(
layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec3 v_albedo;
layout(location = 2) in vec2 v_vegetation_roughness;
layout(location = 3) in vec4 v_surface;
layout(location = 4) in vec3 v_pos;
layout(location = 5) in vec3 v_rel;

layout(location = 0) out vec4 frag_color;

const float PI = 3.14159265359;

float luma(vec3 c) {
	return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

// Saturation in a perceptual (gamma 2) space: boosting chroma in linear space and clamping turns greens neon.
vec3 saturate_color(vec3 c, float amount) {
	const vec3 p = sqrt(max(c, vec3(0.0)));
	const vec3 s = max(mix(vec3(luma(p)), p, amount), vec3(0.0));
	return s * s;
}

float hash13(vec3 p) {
	p = fract(p * 0.3183099 + 0.1);
	p *= 17.0;
	return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float value_noise(vec3 p) {
	const vec3 i = floor(p);
	vec3 f = fract(p);
	f = f * f * (3.0 - 2.0 * f);
	const float n00 = mix(hash13(i), hash13(i + vec3(1.0, 0.0, 0.0)), f.x);
	const float n10 = mix(hash13(i + vec3(0.0, 1.0, 0.0)), hash13(i + vec3(1.0, 1.0, 0.0)), f.x);
	const float n01 = mix(hash13(i + vec3(0.0, 0.0, 1.0)), hash13(i + vec3(1.0, 0.0, 1.0)), f.x);
	const float n11 = mix(hash13(i + vec3(0.0, 1.0, 1.0)), hash13(i + vec3(1.0, 1.0, 1.0)), f.x);
	return mix(mix(n00, n10, f.y), mix(n01, n11, f.y), f.z);
}

// From Godot's oct_inc.glsl
vec2 oct_wrap(vec2 v) {
	return (1.0 - abs(v.yx)) * vec2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0);
}

vec2 vec3_to_oct_with_border(vec3 n, vec2 border_size) {
	n /= (abs(n.x) + abs(n.y) + abs(n.z));
	n.xy = (n.z >= 0.0) ? n.xy : oct_wrap(n.xy);
	return (n.xy * 0.5 + 0.5) * border_size.y + border_size.x;
}

vec3 sample_radiance(vec3 dir, float lod) {
	dir = vec3(dot(scene.radiance_xform0.xyz, dir), dot(scene.radiance_xform1.xyz, dir),
			dot(scene.radiance_xform2.xyz, dir));
	const float border = scene.radiance_params.x;
	const vec2 uv = vec3_to_oct_with_border(normalize(dir), vec2(border, 1.0 - border * 2.0));
#ifdef RADIANCE_ARRAY
	// Layers are roughness levels
	const float layer = floor(lod);
	const vec3 a = textureLod(radiance_octmap, vec3(uv, layer), 0.0).rgb;
	const vec3 b = textureLod(radiance_octmap, vec3(uv, min(layer + 1.0, MAX_ROUGHNESS_LOD)), 0.0).rgb;
	return mix(a, b, lod - layer);
#else
	return textureLod(radiance_octmap, uv, lod).rgb;
#endif
}

float schlick_fresnel(float u) {
	const float m = 1.0 - u;
	const float m2 = m * m;
	return m2 * m2 * m;
}

vec3 perturb_normal(vec3 p, vec3 n, float scale, float strength) {
	const float e = scale * 0.5;
	const float h = value_noise(p / scale);
	vec3 g = vec3(
			value_noise((p + vec3(e, 0.0, 0.0)) / scale) - h,
			value_noise((p + vec3(0.0, e, 0.0)) / scale) - h,
			value_noise((p + vec3(0.0, 0.0, e)) / scale) - h);
	g -= n * dot(g, n);
	return normalize(n - g * strength);
}

void main() {
	const float planet_radius = style.p0.x;
	const float vibrance = style.p0.y;
	const float slope_rock_start = style.p0.z;
	const float slope_rock_end = style.p0.w;
	const float snow_temperature = style.p1.x;
	const float snow_blend = style.p1.y;
	const float gully_darkening = style.p1.z;
	const float ridge_highlight = style.p1.w;
	const float dryness_tint = style.p2.x;
	const float detail_normal_strength = style.p2.y;
	const float detail_normal_scale = style.p2.z;

	const vec3 n = normalize(v_normal);
	const vec3 up = planet_radius > 0.0 ? normalize(v_pos) : vec3(0.0, 1.0, 0.0);
	const float altitude = planet_radius > 0.0 ? length(v_pos) - planet_radius : v_pos.y;

	vec3 albedo = v_albedo;
	float roughness = v_vegetation_roughness.y;
	const float vegetation = v_vegetation_roughness.x;
	const float erosion = v_surface.r;
	const float ridge = v_surface.g * 2.0 - 1.0;
	const float moisture = v_surface.b;
	const float temperature = v_surface.a;

	// Climate tinting: lush greens where wet, straw where dry, desaturated where cold
	const vec3 lush = saturate_color(albedo * vec3(0.92, 1.08, 0.88), 1.12);
	const vec3 dry = mix(albedo, vec3(0.52, 0.44, 0.18), dryness_tint);
	const vec3 climate = mix(dry, lush, smoothstep(0.3, 0.7, moisture));
	albedo = mix(albedo, climate, vegetation);
	albedo = mix(albedo, saturate_color(albedo, 0.45), vegetation * (1.0 - smoothstep(0.12, 0.42, temperature)));

	const float up_dot = dot(n, up);

	// Macro variation so large areas aren't one flat color: drier/yellower patches and brightness drift on
	// vegetation, plus a fine grain everywhere
	const float macro = value_noise(v_pos / 420.0);
	const float meso = value_noise(v_pos / 65.0);
	albedo = mix(albedo, albedo * vec3(1.22, 1.08, 0.62), vegetation * smoothstep(0.45, 0.85, macro) * 0.8);
	albedo *= mix(0.82, 1.12, meso * 0.7 + macro * 0.3);

	// Erosion features: gullies read as damp sediment, ridges as bare sunlit rock
	const float gully = clamp(max(-ridge, 0.0) + max(0.5 - erosion, 0.0) * 1.5, 0.0, 1.0);
	albedo *= 1.0 - gully_darkening * gully;
	albedo = mix(albedo, albedo * vec3(0.94, 0.98, 1.02), gully * moisture);
	roughness = mix(roughness, 0.5, gully * moisture);
	const float ridge_rock = ridge_highlight * max(ridge, 0.0);

	// Rock: the slope threshold wanders with noise so cliffs don't end on a clean contour, and the rock itself gets
	// altitude strata and tone variation
	const float slope_jitter = (value_noise(v_pos / 140.0) - 0.5) * 0.16;
	const float rock_blend = max(1.0 - smoothstep(slope_rock_end, slope_rock_start, up_dot + slope_jitter), ridge_rock);
	const float strata = 0.5 + 0.5 * sin(altitude / 38.0 + meso * 5.0);
	vec3 rock = mat_albedo(1u) * mix(0.78, 1.18, value_noise(v_pos / 23.0));
	rock *= mix(vec3(1.0), vec3(1.12, 0.98, 0.84), strata * 0.6);
	albedo = mix(albedo, rock * (1.0 + ridge_rock * 0.6), rock_blend);
	roughness = mix(roughness, mat_roughness(1u), rock_blend);

	const float snow = (1.0 - smoothstep(snow_temperature - snow_blend, snow_temperature + snow_blend, temperature))
			* smoothstep(slope_rock_end, 1.0, up_dot) * step(0.0, altitude);
	albedo = mix(albedo, mat_albedo(2u), snow);
	roughness = mix(roughness, mat_roughness(2u), snow);

	albedo = saturate_color(albedo, vibrance);

	vec3 nd = n;
	if (detail_normal_strength > 0.0) {
		const float fine_fade = 1.0 - smoothstep(0.2, 0.8, length(fwidth(v_pos)) / detail_normal_scale);
		const float relief = detail_normal_strength * mix(1.0, 1.6, rock_blend) * (1.0 - 0.7 * snow);
		if (fine_fade > 0.01) {
			nd = perturb_normal(v_pos, nd, detail_normal_scale, relief * fine_fade);
			albedo *= 1.0 - 0.07 * fine_fade * value_noise(v_pos / detail_normal_scale);
		}
		nd = perturb_normal(v_pos, nd, detail_normal_scale * 5.0, relief * 0.7);
	}

	// Lighting, following Godot's scene shader (diffuse_burley, specular_schlick_ggx, dielectric F0 from SPECULAR,
	// sky radiance for ambient and reflections). No shadows.
	const float specular = 0.3 * max(gully * moisture, snow * 0.4);
	const float f0 = 0.16 * specular * specular;
	const vec3 v = normalize(-v_rel);
	const float ndv = clamp(dot(nd, v), 1e-4, 1.0);

	vec3 ambient = params.ambient.rgb;
	if (scene.ibl.x > 0.0) {
		ambient = mix(ambient, sample_radiance(nd, MAX_ROUGHNESS_LOD) * scene.ibl.z, scene.ibl.w);
	}
	vec3 color = albedo * ambient;

	if (scene.ibl.y > 0.0 && f0 > 0.0) {
		vec3 ref = reflect(-v, nd);
		ref = mix(ref, nd, roughness * roughness);
		const float horizon = min(1.0 + dot(ref, nd), 1.0);
		vec3 reflection = sample_radiance(ref, sqrt(roughness) * MAX_ROUGHNESS_LOD) * scene.ibl.z * horizon * horizon;
		// Godot's environment BRDF approximation
		const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
		const vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
		const vec4 r = roughness * c0 + c1;
		const float a004 = min(r.x * r.x, exp2(-9.28 * ndv)) * r.x + r.y;
		const vec2 env = vec2(-1.04, 1.04) * a004 + r.zw;
		color += reflection * (env.x * f0 + env.y * clamp(50.0 * f0, 0.0, 1.0));
	}

	if (params.sun_dir.w > 0.0) {
		const vec3 l = params.sun_dir.xyz;
		const float ndl = max(dot(nd, l), 0.0);
		if (ndl > 0.0) {
			const vec3 h = normalize(l + v);
			const float ldh = max(dot(l, h), 0.0);
			const float ndh = max(dot(nd, h), 0.0);
			// Burley diffuse. Godot's light energy carries a PI that cancels the 1/PI.
			const float fd90_minus_1 = 2.0 * ldh * ldh * roughness - 0.5;
			const float diffuse = (1.0 + fd90_minus_1 * schlick_fresnel(ndv)) *
					(1.0 + fd90_minus_1 * schlick_fresnel(ndl)) * ndl;
			float spec = 0.0;
			if (f0 > 0.0) {
				const float a = roughness * roughness;
				const float a2 = a * a;
				const float dd = ndh * ndh * (a2 - 1.0) + 1.0;
				const float distribution = a2 / (PI * dd * dd);
				const float k = a * 0.5;
				const float visibility = 0.25 / ((ndl * (1.0 - k) + k) * (ndv * (1.0 - k) + k));
				const float fresnel = f0 + (clamp(50.0 * f0, 0.0, 1.0) - f0) * schlick_fresnel(ldh);
				spec = distribution * visibility * fresnel * ndl * PI;
			}
			color += (albedo * diffuse + spec) * params.sun_color.rgb;
		}
	}

	if (scene.fog_params.x > 0.0) {
		const vec3 to_frag = v_rel;
		const float dist = length(to_frag);
		vec3 fog_color = scene.fog_color_density.rgb;
		if (scene.fog_params.z > 0.001 && params.sun_dir.w > 0.0) {
			fog_color += params.sun_color.rgb * pow(max(dot(to_frag / dist, params.sun_dir.xyz), 0.0), 8.0) *
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
			const float y = dot(scene.world_y_row, vec4(v_pos, 1.0));
			fog_amount = max(fog_amount, 1.0 - exp(min(0.0, (y - scene.fog_depth.w) * scene.fog_params.w)));
		}
		color = mix(color, fog_color, clamp(fog_amount, 0.0, 1.0));
	}

	frag_color = vec4(color, 1.0);
}
)";

struct RenderPushConstant {
	float mvp[16];
	float camera_int[4];
	float sun_dir[4];
	float sun_color[4];
	float ambient[4];
};
static_assert(sizeof(RenderPushConstant) == 128, "Must match shader");

struct StyleUBO {
	float p0[4];
	float p1[4];
	float p2[4];
};
static_assert(sizeof(StyleUBO) == 48, "Must match shader");

struct SceneUBO {
	float fog_color_density[4];
	float fog_params[4];
	float fog_depth[4];
	float world_y_row[4];
	float radiance_xform0[4];
	float radiance_xform1[4];
	float radiance_xform2[4];
	float ibl[4];
	float radiance_params[4];
	float camera_frac[4];
};
static_assert(sizeof(SceneUBO) == 160, "Must match shader");

struct CullPushConstant {
	float planes[6][4];
	uint32_t chunk_count;
	uint32_t pad[3];
};
static_assert(sizeof(CullPushConstant) == 112, "Must match shader");

constexpr uint32_t VERTEX_STRIDE = sizeof(gpu_driven::PackedVertex);
constexpr uint32_t INITIAL_VERTEX_CAPACITY = 256 * 1024;
// In words (two 16-bit indices each)
constexpr uint32_t INITIAL_INDEX_CAPACITY = 1024 * 1024;
// Chunk.info.w: bit 0 visible, bit 1 wide (32-bit) indices, bits 8-15 transition mask
constexpr uint32_t CHUNK_FLAG_WIDE_INDICES = 2;
constexpr uint32_t INITIAL_CHUNK_CAPACITY = 1024;
// Used when the scene has no directional light
const Vector3 FALLBACK_SUN_DIRECTION = Vector3(0.3, 1.0, 0.2).normalized();

String header_glsl(const char *body) {
	return String("#version 450\n") + CHUNK_STRUCT_GLSL + body;
}

// The sky radiance layout depends on project settings, like Godot's own scene shader variants
String render_header_glsl(const char *body) {
	const RendererSceneRenderRD &scene_render = *RendererSceneRenderRD::get_singleton();
	String defines = "#define MAX_ROUGHNESS_LOD " + itos(scene_render.get_roughness_layers() - 1) + ".0\n";
	if (scene_render.is_using_radiance_octmap_array()) {
		defines += "#define RADIANCE_ARRAY\n";
	}
	return String("#version 450\n") + defines + CHUNK_STRUCT_GLSL + RENDER_COMMON_GLSL + body;
}

Vector<uint8_t> compile_stage(RenderingDevice &rd, RenderingDevice::ShaderStage stage, const String &source) {
	String error;
	Vector<uint8_t> spirv =
			rd.shader_compile_spirv_from_source(stage, source, RenderingDevice::SHADER_LANGUAGE_GLSL, &error, true);
	if (spirv.is_empty()) {
		ERR_PRINT(String("VoxelGpuDrivenRenderer: shader compilation failed:\n") + error);
	}
	return spirv;
}

RID create_shader(RenderingDevice &rd, const Vector<RenderingDevice::ShaderStage> &stages, const Vector<String> &sources,
		const String &name) {
	Vector<RenderingDevice::ShaderStageSPIRVData> spirv_stages;
	for (int i = 0; i < stages.size(); ++i) {
		RenderingDevice::ShaderStageSPIRVData data;
		data.shader_stage = stages[i];
		data.spirv = compile_stage(rd, stages[i], sources[i]);
		if (data.spirv.is_empty()) {
			return RID();
		}
		spirv_stages.push_back(data);
	}
	return rd.shader_create_from_spirv(spirv_stages, name);
}

RenderingDevice::Uniform make_storage_uniform(int binding, RID buffer) {
	RenderingDevice::Uniform u;
	u.uniform_type = RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER;
	u.binding = binding;
	u.append_id(buffer);
	return u;
}

void free_rid_if_valid(RenderingDevice &rd, RID &rid) {
	if (rid.is_valid()) {
		rd.free_rid(rid);
		rid = RID();
	}
}

// Planes of the clip volume of `m` (Vulkan clip space: x, y in [-w, w], z in [0, w]), valid with reverse-Z.
void extract_frustum_planes(const Projection &m, float out[6][4]) {
	auto row = [&m](int r) {
		return Vector4(m.columns[0][r], m.columns[1][r], m.columns[2][r], m.columns[3][r]);
	};
	const Vector4 r0 = row(0);
	const Vector4 r1 = row(1);
	const Vector4 r2 = row(2);
	const Vector4 r3 = row(3);
	const Vector4 planes[6] = { r3 + r0, r3 - r0, r3 + r1, r3 - r1, r2, r3 - r2 };
	for (int i = 0; i < 6; ++i) {
		out[i][0] = planes[i].x;
		out[i][1] = planes[i].y;
		out[i][2] = planes[i].z;
		out[i][3] = planes[i].w;
	}
}

} // namespace

VoxelGpuDrivenRenderer::VoxelGpuDrivenRenderer() {
	RenderingServer &rs = *RenderingServer::get_singleton();
	_effect = rs.compositor_effect_create();
	rs.compositor_effect_set_callback(_effect, RenderingServer::COMPOSITOR_EFFECT_CALLBACK_TYPE_POST_OPAQUE,
			callable_mp(this, &VoxelGpuDrivenRenderer::_render_callback));
	rs.compositor_effect_set_enabled(_effect, true);
}

VoxelGpuDrivenRenderer::~VoxelGpuDrivenRenderer() {}

void VoxelGpuDrivenRenderer::destroy() {
	detach(nullptr);
	_own_compositor.unref();
	RenderingServer &rs = *RenderingServer::get_singleton();
	// Freeing the effect is queued on the render thread before our cleanup, so no callback can run after it.
	rs.free_rid(_effect);
	_effect = RID();
	rs.call_on_render_thread(callable_mp(this, &VoxelGpuDrivenRenderer::_render_thread_free));
}

void VoxelGpuDrivenRenderer::_render_thread_free() {
	RenderingDevice *rd = RenderingServer::get_singleton()->get_rendering_device();
	if (rd != nullptr) {
		for (KeyValue<int64_t, RID> &kv : _render_pipelines) {
			free_rid_if_valid(*rd, kv.value);
		}
		// Uniform sets are freed automatically with their buffers
		free_rid_if_valid(*rd, _cull_pipeline);
		free_rid_if_valid(*rd, _render_shader);
		free_rid_if_valid(*rd, _cull_shader);
		free_rid_if_valid(*rd, _vertex_buffer);
		free_rid_if_valid(*rd, _index_buffer);
		free_rid_if_valid(*rd, _chunk_buffer);
		free_rid_if_valid(*rd, _indirect_buffer);
		free_rid_if_valid(*rd, _style_buffer);
		free_rid_if_valid(*rd, _scene_buffer);
		if (_radiance_uniform_set.is_valid() && rd->uniform_set_is_valid(_radiance_uniform_set)) {
			rd->free_rid(_radiance_uniform_set);
		}
		free_rid_if_valid(*rd, _radiance_sampler);
	}
	memdelete(this);
}

void VoxelGpuDrivenRenderer::update(
		World3D &world,
		const Transform3D &terrain_transform,
		bool visible,
		const Style &style
) {
	bool needs_redraw;
	{
		MutexLock lock(_mutex);
		const bool style_changed = !(_style == style);
		needs_redraw = !_pending_ops.empty() || _terrain_transform != terrain_transform || _visible != visible ||
				style_changed;
		_terrain_transform = terrain_transform;
		_visible = visible;
		if (style_changed) {
			_style = style;
			++_style_version;
		}
	}
	if (needs_redraw) {
		// Our changes don't go through RenderingServer, so the editor (low processor mode) wouldn't redraw and
		// queued uploads would sit there. Any server write requests a redraw.
		RenderingServer::get_singleton()->compositor_effect_set_enabled(_effect, true);
	}

	// A scenario renders with a single compositor (World3D's, usually set by a WorldEnvironment). If there is none,
	// give the world our own. Either way our effect is appended server-side only, so the Compositor resource users
	// edit and save never contains it. A WorldEnvironment replacing or editing the compositor is picked up here.
	Ref<Compositor> compositor = world.get_compositor();
	if (compositor.is_null()) {
		if (_own_compositor.is_null()) {
			_own_compositor.instantiate();
		}
		world.set_compositor(_own_compositor);
		compositor = _own_compositor;
	}

	TypedArray<RID> user_effects;
	const TypedArray<CompositorEffect> effects = compositor->get_compositor_effects();
	for (int i = 0; i < effects.size(); ++i) {
		Ref<CompositorEffect> e = effects[i];
		if (e.is_valid()) {
			user_effects.push_back(e->get_rid());
		}
	}
	if (_injected_compositor == compositor && _injected_user_effects == user_effects) {
		return;
	}
	detach(nullptr);
	TypedArray<RID> all = user_effects.duplicate();
	all.push_back(_effect);
	RenderingServer::get_singleton()->compositor_set_compositor_effects(compositor->get_rid(), all);
	_injected_compositor = compositor;
	_injected_user_effects = user_effects;
}

void VoxelGpuDrivenRenderer::detach(World3D *world) {
	if (_injected_compositor.is_valid()) {
		// Restore the resource's own effect list
		RenderingServer::get_singleton()->compositor_set_compositor_effects(
				_injected_compositor->get_rid(), _injected_user_effects
		);
		_injected_compositor.unref();
		_injected_user_effects.clear();
	}
	if (world != nullptr && _own_compositor.is_valid() && world->get_compositor() == _own_compositor) {
		world->set_compositor(Ref<Compositor>());
	}
}

uint32_t VoxelGpuDrivenRenderer::create_chunk() {
	if (!_free_ids.empty()) {
		const uint32_t id = _free_ids.back();
		_free_ids.pop_back();
		return id;
	}
	return _next_id++;
}

void VoxelGpuDrivenRenderer::push_op(Op &&op) {
	MutexLock lock(_mutex);
	_pending_ops.push_back(std::move(op));
}

void VoxelGpuDrivenRenderer::set_chunk_mesh(uint32_t id, gpu_driven::PackedMesh &&mesh, Vector3 origin) {
	Op op;
	op.type = Op::SET_MESH;
	op.id = id;
	op.origin = origin;
	op.mesh = std::move(mesh);
	push_op(std::move(op));
}

void VoxelGpuDrivenRenderer::set_chunk_visible(uint32_t id, bool visible) {
	Op op;
	op.type = Op::SET_VISIBLE;
	op.id = id;
	op.visible = visible;
	push_op(std::move(op));
}

void VoxelGpuDrivenRenderer::set_chunk_transition_mask(uint32_t id, uint8_t mask) {
	Op op;
	op.type = Op::SET_TRANSITION_MASK;
	op.id = id;
	op.transition_mask = mask;
	push_op(std::move(op));
}

void VoxelGpuDrivenRenderer::remove_chunk(uint32_t id) {
	Op op;
	op.type = Op::REMOVE;
	op.id = id;
	push_op(std::move(op));
	// Safe to reuse right away: ops are applied in order on the render thread
	_free_ids.push_back(id);
}

Dictionary VoxelGpuDrivenRenderer::get_stats() const {
	MutexLock lock(_mutex);
	Dictionary d;
	d["initialized"] = _stats.initialized;
	d["failed"] = _stats.failed;
	d["chunks"] = _stats.chunk_count;
	d["chunk_slots"] = _stats.chunk_slots;
	d["vertices"] = _stats.vertex_count;
	d["index_words"] = _stats.index_count;
	d["vertex_capacity"] = _stats.vertex_capacity;
	d["index_word_capacity"] = _stats.index_capacity;
	d["vertex_stride"] = VERTEX_STRIDE;
	d["vertex_bytes_used"] = _stats.vertex_count * VERTEX_STRIDE;
	d["index_bytes_used"] = _stats.index_count * 4;
	d["vram_bytes"] = _stats.vertex_capacity * VERTEX_STRIDE + _stats.index_capacity * 4 +
			uint64_t(_stats.chunk_slots) * (sizeof(ChunkGPU) + 16);
	d["uploaded_bytes_total"] = _stats.uploaded_bytes_total;
	d["indirect_draws_total"] = _stats.indirect_draws_total;
	d["pending_ops"] = static_cast<int64_t>(_pending_ops.size());
	return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Render thread

bool VoxelGpuDrivenRenderer::_ensure_initialized() {
	if (_initialized) {
		return true;
	}
	if (_init_failed) {
		return false;
	}
	RenderingDevice *rd = RenderingServer::get_singleton()->get_rendering_device();
	if (rd == nullptr) {
		_init_failed = true;
		ERR_PRINT("VoxelGpuDrivenRenderer requires a RenderingDevice-based renderer (Forward+ or Mobile).");
		return false;
	}

	_cull_shader = create_shader(*rd, { RenderingDevice::SHADER_STAGE_COMPUTE }, { header_glsl(CULL_SHADER_GLSL) },
			"VoxelGpuDrivenCull");
	_render_shader = create_shader(*rd, { RenderingDevice::SHADER_STAGE_VERTEX, RenderingDevice::SHADER_STAGE_FRAGMENT },
			{ render_header_glsl(VERTEX_SHADER_GLSL), render_header_glsl(FRAGMENT_SHADER_GLSL) },
			"VoxelGpuDrivenRender");
	if (_cull_shader.is_null() || _render_shader.is_null()) {
		_init_failed = true;
		MutexLock lock(_mutex);
		_stats.failed = true;
		return false;
	}
	_cull_pipeline = rd->compute_pipeline_create(_cull_shader);

	_vertex_allocator.grow(INITIAL_VERTEX_CAPACITY);
	_index_allocator.grow(INITIAL_INDEX_CAPACITY);
	_vertex_buffer = rd->storage_buffer_create(INITIAL_VERTEX_CAPACITY * VERTEX_STRIDE);
	_index_buffer = rd->storage_buffer_create(INITIAL_INDEX_CAPACITY * 4);
	_style_buffer = rd->uniform_buffer_create(sizeof(StyleUBO));
	_scene_buffer = rd->uniform_buffer_create(sizeof(SceneUBO));
	_ensure_chunk_capacity(INITIAL_CHUNK_CAPACITY - 1);

	_initialized = true;
	MutexLock lock(_mutex);
	_stats.initialized = true;
	return true;
}

bool VoxelGpuDrivenRenderer::_grow_buffer(RID &buffer, uint32_t old_size, uint32_t new_size, bool indirect) {
	RenderingDevice &rd = *RenderingServer::get_singleton()->get_rendering_device();
	BitField<RenderingDevice::StorageBufferUsage> usage = 0;
	if (indirect) {
		usage.set_flag(RenderingDevice::STORAGE_BUFFER_USAGE_DISPATCH_INDIRECT);
	}
	const RID new_buffer = rd.storage_buffer_create(new_size, {}, usage);
	ERR_FAIL_COND_V_MSG(new_buffer.is_null(), false,
			vformat("VoxelGpuDrivenRenderer: failed to allocate a %d byte storage buffer", new_size));
	if (buffer.is_valid()) {
		if (old_size > 0) {
			rd.buffer_copy(buffer, new_buffer, 0, 0, old_size);
		}
		rd.free_rid(buffer);
	}
	buffer = new_buffer;
	// Uniform sets referencing the old buffer were freed with it
	_render_uniform_set = RID();
	_cull_uniform_set = RID();
	return true;
}

void VoxelGpuDrivenRenderer::_ensure_chunk_capacity(uint32_t id) {
	if (id < _chunk_capacity) {
		return;
	}
	uint32_t new_capacity = _chunk_capacity == 0 ? INITIAL_CHUNK_CAPACITY : _chunk_capacity;
	while (new_capacity <= id) {
		new_capacity *= 2;
	}
	_grow_buffer(_chunk_buffer, _chunk_capacity * sizeof(ChunkGPU), new_capacity * sizeof(ChunkGPU), false);
	// The indirect buffer is fully rewritten by the cull pass every frame, no need to copy it
	_grow_buffer(_indirect_buffer, 0, new_capacity * 16, true);

	_chunks.resize(new_capacity);
	ChunkGPU zero;
	memset(&zero, 0, sizeof(zero));
	_chunks_gpu.resize(new_capacity, zero);
	// New slots must be uploaded as empty before the cull pass reads them
	_mark_chunk_dirty(_chunk_capacity);
	_mark_chunk_dirty(new_capacity - 1);
	_chunk_capacity = new_capacity;
}

void VoxelGpuDrivenRenderer::_mark_chunk_dirty(uint32_t id) {
	_dirty_min = MIN(_dirty_min, id);
	_dirty_max = MAX(_dirty_max, id);
}

void VoxelGpuDrivenRenderer::_free_chunk_geometry(uint32_t id) {
	ChunkRecord &r = _chunks[id];
	if (r.vertex_count > 0) {
		_vertex_allocator.free(r.vertex_offset, r.vertex_count);
	}
	if (r.index_count > 0) {
		_index_allocator.free(r.index_offset, r.index_count);
	}
	r = ChunkRecord();
	ChunkGPU &g = _chunks_gpu[id];
	g.info[0] = 0;
	g.info[1] = 0;
	g.info[2] = 0;
	_mark_chunk_dirty(id);
}

void VoxelGpuDrivenRenderer::_apply_set_mesh(Op &op) {
	RenderingDevice &rd = *RenderingServer::get_singleton()->get_rendering_device();
	_ensure_chunk_capacity(op.id);
	_free_chunk_geometry(op.id);

	const gpu_driven::PackedMesh &mesh = op.mesh;
	ChunkGPU &g = _chunks_gpu[op.id];
	g.origin[0] = op.origin.x;
	g.origin[1] = op.origin.y;
	g.origin[2] = op.origin.z;
	g.origin[3] = 0.f;
	if (mesh.is_empty()) {
		return;
	}

	const uint32_t vcount = static_cast<uint32_t>(mesh.vertices.size());
	// The index pool is allocated in words: two 16-bit indices each, or one 32-bit index for wide chunks
	const uint32_t icount = static_cast<uint32_t>(mesh.index_words.size());

	uint32_t voffset = _vertex_allocator.allocate(vcount);
	if (voffset == gpu_driven::RangeAllocator::INVALID) {
		const uint32_t old_cap = _vertex_allocator.get_capacity();
		const uint32_t new_cap = MAX(old_cap * 2, old_cap + vcount);
		if (!_grow_buffer(_vertex_buffer, old_cap * VERTEX_STRIDE, new_cap * VERTEX_STRIDE, false)) {
			return;
		}
		_vertex_allocator.grow(new_cap);
		voffset = _vertex_allocator.allocate(vcount);
	}
	uint32_t ioffset = _index_allocator.allocate(icount);
	if (ioffset == gpu_driven::RangeAllocator::INVALID) {
		const uint32_t old_cap = _index_allocator.get_capacity();
		const uint32_t new_cap = MAX(old_cap * 2, old_cap + icount);
		if (!_grow_buffer(_index_buffer, old_cap * 4, new_cap * 4, false)) {
			_vertex_allocator.free(voffset, vcount);
			return;
		}
		_index_allocator.grow(new_cap);
		ioffset = _index_allocator.allocate(icount);
	}

	rd.buffer_update(_vertex_buffer, voffset * VERTEX_STRIDE, vcount * VERTEX_STRIDE, mesh.vertices.data());
	rd.buffer_update(_index_buffer, ioffset * 4, icount * 4, mesh.index_words.data());

	ChunkRecord &r = _chunks[op.id];
	r.vertex_offset = voffset;
	r.vertex_count = vcount;
	r.index_offset = ioffset;
	r.index_count = icount;

	for (int i = 0; i < 3; ++i) {
		// Transition geometry of inactive sides collapses to the chunk origin, keep it inside
		g.aabb_min[i] = g.origin[i] + MIN(mesh.aabb_min[i], 0.f);
		g.aabb_max[i] = g.origin[i] + MAX(mesh.aabb_max[i], 0.f);
	}
	g.info[0] = voffset;
	// The draw's first_vertex, in index units: gl_VertexIndex addresses indices, not words
	g.info[1] = mesh.wide_indices ? ioffset : ioffset * 2;
	g.info[2] = mesh.index_count;
	g.info[3] = (g.info[3] & ~CHUNK_FLAG_WIDE_INDICES) | (mesh.wide_indices ? CHUNK_FLAG_WIDE_INDICES : 0u);

	MutexLock lock(_mutex);
	_stats.uploaded_bytes_total += uint64_t(vcount) * VERTEX_STRIDE + uint64_t(icount) * 4;
}

void VoxelGpuDrivenRenderer::_apply_ops() {
	std::vector<Op> ops;
	{
		MutexLock lock(_mutex);
		ops.swap(_pending_ops);
	}
	for (Op &op : ops) {
		_ensure_chunk_capacity(op.id);
		_chunk_high_water = MAX(_chunk_high_water, op.id + 1);
		ChunkGPU &g = _chunks_gpu[op.id];
		switch (op.type) {
			case Op::SET_MESH:
				_apply_set_mesh(op);
				break;
			case Op::SET_VISIBLE:
				g.info[3] = (g.info[3] & ~1u) | (op.visible ? 1u : 0u);
				_mark_chunk_dirty(op.id);
				break;
			case Op::SET_TRANSITION_MASK:
				g.info[3] = (g.info[3] & 0xffu) | (uint32_t(op.transition_mask) << 8);
				_mark_chunk_dirty(op.id);
				break;
			case Op::REMOVE:
				_free_chunk_geometry(op.id);
				g.info[3] = 0;
				break;
		}
	}

	if (!ops.empty()) {
		uint32_t chunk_count = 0;
		for (uint32_t i = 0; i < _chunk_high_water; ++i) {
			chunk_count += _chunks[i].index_count > 0 ? 1 : 0;
		}
		MutexLock lock(_mutex);
		_stats.chunk_count = chunk_count;
		_stats.chunk_slots = _chunk_capacity;
		_stats.vertex_count = _vertex_allocator.get_used();
		_stats.index_count = _index_allocator.get_used();
		_stats.vertex_capacity = _vertex_allocator.get_capacity();
		_stats.index_capacity = _index_allocator.get_capacity();
	}
}

void VoxelGpuDrivenRenderer::_upload_dirty_chunks() {
	if (_dirty_min > _dirty_max) {
		return;
	}
	RenderingDevice &rd = *RenderingServer::get_singleton()->get_rendering_device();
	const uint32_t count = _dirty_max - _dirty_min + 1;
	rd.buffer_update(_chunk_buffer, _dirty_min * sizeof(ChunkGPU), count * sizeof(ChunkGPU), &_chunks_gpu[_dirty_min]);
	_dirty_min = 0xffffffffu;
	_dirty_max = 0;
}

bool VoxelGpuDrivenRenderer::_ensure_uniform_sets() {
	RenderingDevice &rd = *RenderingServer::get_singleton()->get_rendering_device();
	if (_render_uniform_set.is_null() || !rd.uniform_set_is_valid(_render_uniform_set)) {
		Vector<RenderingDevice::Uniform> uniforms;
		uniforms.push_back(make_storage_uniform(0, _vertex_buffer));
		uniforms.push_back(make_storage_uniform(1, _index_buffer));
		uniforms.push_back(make_storage_uniform(2, _chunk_buffer));
		RenderingDevice::Uniform style_uniform;
		style_uniform.uniform_type = RenderingDevice::UNIFORM_TYPE_UNIFORM_BUFFER;
		style_uniform.binding = 3;
		style_uniform.append_id(_style_buffer);
		uniforms.push_back(style_uniform);
		RenderingDevice::Uniform scene_uniform;
		scene_uniform.uniform_type = RenderingDevice::UNIFORM_TYPE_UNIFORM_BUFFER;
		scene_uniform.binding = 4;
		scene_uniform.append_id(_scene_buffer);
		uniforms.push_back(scene_uniform);
		_render_uniform_set = rd.uniform_set_create(uniforms, _render_shader, 0);
	}
	if (_cull_uniform_set.is_null() || !rd.uniform_set_is_valid(_cull_uniform_set)) {
		Vector<RenderingDevice::Uniform> uniforms;
		uniforms.push_back(make_storage_uniform(0, _chunk_buffer));
		uniforms.push_back(make_storage_uniform(1, _indirect_buffer));
		_cull_uniform_set = rd.uniform_set_create(uniforms, _cull_shader, 0);
	}
	return _render_uniform_set.is_valid() && _cull_uniform_set.is_valid();
}

// ponytail: single pass, no depth prepass. A prepass (depth-only, then EQUAL) was measured slower on the planet vista
// (GPU 4.47 ms vs 4.32 ms, GTX 750 Ti): vertex pulling twice costs more than the overdraw it saves. Revisit with
// front-to-back chunk ordering if fragment cost grows.
RID VoxelGpuDrivenRenderer::_get_render_pipeline(int64_t framebuffer_format, int samples) {
	if (RID *existing = _render_pipelines.getptr(framebuffer_format)) {
		return *existing;
	}
	RenderingDevice &rd = *RenderingServer::get_singleton()->get_rendering_device();

	RenderingDevice::PipelineRasterizationState raster;
	raster.cull_mode = RenderingDevice::POLYGON_CULL_BACK;

	RenderingDevice::PipelineMultisampleState multisample;
	multisample.sample_count = static_cast<RenderingDevice::TextureSamples>(samples);

	RenderingDevice::PipelineDepthStencilState depth;
	depth.enable_depth_test = true;
	depth.enable_depth_write = true;
	// Godot uses reverse-Z
	depth.depth_compare_operator = RenderingDevice::COMPARE_OP_GREATER_OR_EQUAL;

	const RID pipeline = rd.render_pipeline_create(_render_shader, framebuffer_format, RenderingDevice::INVALID_ID,
			RenderingDevice::RENDER_PRIMITIVE_TRIANGLES, raster, multisample, depth,
			RenderingDevice::PipelineColorBlendState::create_disabled(1));
	_render_pipelines.insert(framebuffer_format, pipeline);
	return pipeline;
}

RID VoxelGpuDrivenRenderer::_get_radiance_uniform_set(RID radiance) {
	RenderingDevice &rd = *RenderingServer::get_singleton()->get_rendering_device();
	if (radiance.is_null()) {
		RendererRD::TextureStorage &ts = *RendererRD::TextureStorage::get_singleton();
		radiance = ts.texture_rd_get_default(
				RendererSceneRenderRD::get_singleton()->is_using_radiance_octmap_array()
						? RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_2D_ARRAY_BLACK
						: RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_BLACK
		);
	}
	const bool set_valid = _radiance_uniform_set.is_valid() && rd.uniform_set_is_valid(_radiance_uniform_set);
	if (set_valid && _radiance_uniform_set_texture == radiance) {
		return _radiance_uniform_set;
	}
	if (set_valid) {
		rd.free_rid(_radiance_uniform_set);
	}
	if (_radiance_sampler.is_null()) {
		RenderingDevice::SamplerState ss;
		ss.mag_filter = RenderingDevice::SAMPLER_FILTER_LINEAR;
		ss.min_filter = RenderingDevice::SAMPLER_FILTER_LINEAR;
		ss.mip_filter = RenderingDevice::SAMPLER_FILTER_LINEAR;
		ss.repeat_u = RenderingDevice::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
		ss.repeat_v = RenderingDevice::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
		ss.repeat_w = RenderingDevice::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
		_radiance_sampler = rd.sampler_create(ss);
	}
	RenderingDevice::Uniform u;
	u.uniform_type = RenderingDevice::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
	u.binding = 0;
	u.append_id(_radiance_sampler);
	u.append_id(radiance);
	Vector<RenderingDevice::Uniform> uniforms;
	uniforms.push_back(u);
	_radiance_uniform_set = rd.uniform_set_create(uniforms, _render_shader, 1);
	_radiance_uniform_set_texture = radiance;
	return _radiance_uniform_set;
}

namespace {

struct SceneLighting {
	bool has_sun = false;
	Vector3 sun_direction; // world space, toward the light
	Color sun; // linear, energy applied
	Color ambient; // linear, energy applied (the non-sky part of Godot's ambient)
	// Sky radiance
	bool use_ambient_cubemap = false;
	bool use_reflection_cubemap = false;
	float ibl_energy = 1.f; // IBL exposure normalization * bg energy multiplier
	float ambient_sky_mix = 1.f;
	RID radiance;
	float radiance_border = 0.f;
	Basis sky_orientation;
	// Fog
	bool fog = false;
	bool fog_depth_mode = false;
	Color fog_color; // linear, energy applied
	float fog_density = 0.f;
	float fog_sun_scatter = 0.f;
	float fog_height = 0.f;
	float fog_height_density = 0.f;
	float fog_depth_begin = 0.f;
	float fog_depth_end = 0.f;
	float fog_depth_curve = 1.f;
};

// Same inputs Godot's scene shader gets (see RenderSceneDataRD::update_ubo): the first directional light of this
// render, and the environment's ambient, sky radiance and fog.
SceneLighting get_scene_lighting(const RenderData *p_render_data, float luminance_multiplier) {
	SceneLighting sl;
	float exposure = 1.f;
	const RID camera_attributes = p_render_data->get_camera_attributes();
	if (camera_attributes.is_valid()) {
		exposure = RSG::camera_attributes->camera_attributes_get_exposure_normalization_factor(camera_attributes);
	}

	const RenderDataRD *rdd = Object::cast_to<RenderDataRD>(p_render_data);
	if (rdd != nullptr && rdd->lights != nullptr) {
		RendererRD::LightStorage &ls = *RendererRD::LightStorage::get_singleton();
		for (uint32_t i = 0; i < rdd->lights->size(); ++i) {
			const RID light_instance = (*rdd->lights)[i];
			const RID light = ls.light_instance_get_base_light(light_instance);
			if (ls.light_get_type(light) != RS::LIGHT_DIRECTIONAL) {
				continue;
			}
			sl.sun_direction = ls.light_instance_get_base_transform(light_instance).basis.xform(Vector3(0, 0, 1));
			sl.sun_direction.normalize();
			float energy = ls.light_get_param(light, RS::LIGHT_PARAM_ENERGY) * exposure;
			if (RendererSceneRenderRD::get_singleton()->is_using_physical_light_units()) {
				energy *= ls.light_get_param(light, RS::LIGHT_PARAM_INTENSITY) / Math::PI;
			}
			sl.sun = ls.light_get_color(light).srgb_to_linear() * energy;
			sl.has_sun = true;
			break;
		}
	}

	const RID env = p_render_data->get_environment();
	RendererEnvironmentStorage *es = RendererEnvironmentStorage::get_singleton();
	if (!env.is_valid() || !es->is_environment(env)) {
		sl.ambient = RSG::texture_storage->get_default_clear_color().srgb_to_linear();
		return sl;
	}

	const RS::EnvironmentBG bg = es->environment_get_background(env);
	const RS::EnvironmentAmbientSource ambient_source = es->environment_get_ambient_source(env);
	const float bg_energy = es->environment_get_bg_energy_multiplier(env);

	if (ambient_source == RS::ENV_AMBIENT_SOURCE_BG && (bg == RS::ENV_BG_CLEAR_COLOR || bg == RS::ENV_BG_COLOR)) {
		const Color c = bg == RS::ENV_BG_CLEAR_COLOR ? RSG::texture_storage->get_default_clear_color()
													 : es->environment_get_bg_color(env);
		sl.ambient = c.srgb_to_linear() * bg_energy;
	} else {
		sl.use_ambient_cubemap = (ambient_source == RS::ENV_AMBIENT_SOURCE_BG && bg == RS::ENV_BG_SKY) ||
				ambient_source == RS::ENV_AMBIENT_SOURCE_SKY;
		const bool use_ambient = sl.use_ambient_cubemap || ambient_source == RS::ENV_AMBIENT_SOURCE_COLOR;
		sl.ambient = use_ambient
				? es->environment_get_ambient_light(env).srgb_to_linear() * es->environment_get_ambient_light_energy(env)
				: Color(0, 0, 0);
	}
	const RS::EnvironmentReflectionSource reflection_source = es->environment_get_reflection_source(env);
	sl.use_reflection_cubemap = (reflection_source == RS::ENV_REFLECTION_SOURCE_BG && bg == RS::ENV_BG_SKY) ||
			reflection_source == RS::ENV_REFLECTION_SOURCE_SKY;
	sl.ambient_sky_mix = es->environment_get_ambient_sky_contribution(env);

	const RID sky = es->environment_get_sky(env);
	if (sky.is_valid() && (sl.use_ambient_cubemap || sl.use_reflection_cubemap)) {
		RendererRD::SkyRD &sky_rd = *RendererSceneRenderRD::get_singleton()->get_sky();
		sl.radiance = sky_rd.sky_get_radiance_texture_rd(sky);
		sl.radiance_border = sky_rd.sky_get_uv_border_size(sky);
		sl.sky_orientation = es->environment_get_sky_orientation(env);
		float ibl_exposure = 1.f;
		if (camera_attributes.is_valid()) {
			ibl_exposure = exposure * es->environment_get_bg_intensity(env) / luminance_multiplier /
					MAX(0.001f, sky_rd.sky_get_baked_exposure(sky));
		}
		sl.ibl_energy = ibl_exposure * bg_energy;
	}
	if (sl.radiance.is_null()) {
		sl.use_ambient_cubemap = false;
		sl.use_reflection_cubemap = false;
	}

	if (es->environment_get_fog_enabled(env)) {
		sl.fog = true;
		sl.fog_depth_mode = es->environment_get_fog_mode(env) == RS::ENV_FOG_MODE_DEPTH;
		sl.fog_color =
				es->environment_get_fog_light_color(env).srgb_to_linear() * es->environment_get_fog_light_energy(env);
		sl.fog_density = es->environment_get_fog_density(env);
		sl.fog_sun_scatter = es->environment_get_fog_sun_scatter(env);
		sl.fog_height = es->environment_get_fog_height(env);
		sl.fog_height_density = es->environment_get_fog_height_density(env);
		sl.fog_depth_begin = es->environment_get_fog_depth_begin(env);
		sl.fog_depth_end = es->environment_get_fog_depth_end(env);
		sl.fog_depth_curve = es->environment_get_fog_depth_curve(env);
	}
	return sl;
}

} // namespace

void VoxelGpuDrivenRenderer::_render_callback(int p_callback_type, RenderData *p_render_data) {
	if (p_callback_type != RenderingServer::COMPOSITOR_EFFECT_CALLBACK_TYPE_POST_OPAQUE || p_render_data == nullptr) {
		return;
	}
	if (!_ensure_initialized()) {
		return;
	}

	Transform3D terrain_transform;
	bool visible;
	Style style;
	uint32_t style_version;
	{
		MutexLock lock(_mutex);
		terrain_transform = _terrain_transform;
		visible = _visible;
		style = _style;
		style_version = _style_version;
	}

	// Always consume pending work, even when hidden, so memory reflects the terrain state
	_apply_ops();
	_upload_dirty_chunks();

	if (_uploaded_style_version != style_version) {
		const StyleUBO ubo = {
			{ style.planet_radius, style.vibrance, style.slope_rock_start, style.slope_rock_end },
			{ style.snow_temperature, style.snow_blend, style.gully_darkening, style.ridge_highlight },
			{ style.dryness_tint, style.detail_normal_strength, style.detail_normal_scale, 0.f },
		};
		RenderingServer::get_singleton()->get_rendering_device()->buffer_update(
				_style_buffer, 0, sizeof(ubo), &ubo
		);
		_uploaded_style_version = style_version;
	}

	if (!visible || _chunk_high_water == 0) {
		return;
	}

	Ref<RenderSceneBuffersRD> rb = p_render_data->get_render_scene_buffers();
	RenderSceneData *scene_data = p_render_data->get_render_scene_data();
	if (rb.is_null() || scene_data == nullptr || rb->get_view_count() != 1) {
		return;
	}
	if (!_ensure_uniform_sets()) {
		return;
	}

	RenderingDevice &rd = *RenderingServer::get_singleton()->get_rendering_device();

	// Projection already includes Godot's depth correction (reverse-Z, Y flip) and TAA jitter
	const Projection view_projection =
			scene_data->get_cam_projection() * Projection(scene_data->get_cam_transform().affine_inverse());
	const Projection mvp = view_projection * Projection(terrain_transform);

	// Cull
	{
		CullPushConstant pc;
		extract_frustum_planes(mvp, pc.planes);
		pc.chunk_count = _chunk_high_water;
		pc.pad[0] = pc.pad[1] = pc.pad[2] = 0;

		RenderingDevice::ComputeListID cl = rd.compute_list_begin();
		rd.compute_list_bind_compute_pipeline(cl, _cull_pipeline);
		rd.compute_list_bind_uniform_set(cl, _cull_uniform_set, 0);
		rd.compute_list_set_push_constant(cl, &pc, sizeof(pc));
		rd.compute_list_dispatch(cl, (_chunk_high_water + 63) / 64, 1, 1);
		rd.compute_list_end();
	}

	// Draw
	const bool msaa = rb->get_msaa_3d() != RenderingServer::VIEWPORT_MSAA_DISABLED;
	const RID color = msaa ? rb->get_color_msaa() : rb->get_internal_texture();
	const RID depth_tex = msaa ? rb->get_depth_msaa() : rb->get_depth_texture();
	if (color.is_null() || depth_tex.is_null()) {
		return;
	}
	const RID framebuffer = FramebufferCacheRD::get_singleton()->get_cache_multiview(1, color, depth_tex);
	const RID pipeline = _get_render_pipeline(rd.framebuffer_get_format(framebuffer), rb->get_texture_samples());
	if (pipeline.is_null()) {
		return;
	}

	// Camera-relative projection: the view * terrain translation cancels out for (local - camera_local), leaving
	// only rotations, so no large numbers reach the GPU. See `camera_int` in the shader.
	const Transform3D view = scene_data->get_cam_transform().affine_inverse();
	const Projection mvp_rel =
			scene_data->get_cam_projection() * Projection(Transform3D(view.basis * terrain_transform.basis, Vector3()));
	RenderPushConstant pc;
	for (int c = 0; c < 4; ++c) {
		for (int r = 0; r < 4; ++r) {
			pc.mvp[c * 4 + r] = mvp_rel.columns[c][r];
		}
	}
	const Vector3 camera_local = terrain_transform.affine_inverse().xform(scene_data->get_cam_transform().origin);
	const Vector3 camera_int = camera_local.floor();
	const Vector3 camera_frac = camera_local - camera_int;
	pc.camera_int[0] = camera_int.x;
	pc.camera_int[1] = camera_int.y;
	pc.camera_int[2] = camera_int.z;
	pc.camera_int[3] = 0.f;

	const SceneLighting lighting = get_scene_lighting(p_render_data, rb->get_luminance_multiplier());
	const Vector3 sun_local = terrain_transform.basis
									  .xform_inv(lighting.has_sun ? lighting.sun_direction : FALLBACK_SUN_DIRECTION)
									  .normalized();
	const Color sun = lighting.has_sun ? lighting.sun : Color(1, 1, 1);
	pc.sun_dir[0] = sun_local.x;
	pc.sun_dir[1] = sun_local.y;
	pc.sun_dir[2] = sun_local.z;
	pc.sun_dir[3] = 1.f;
	pc.sun_color[0] = sun.r;
	pc.sun_color[1] = sun.g;
	pc.sun_color[2] = sun.b;
	pc.sun_color[3] = 0.f;
	pc.ambient[0] = lighting.ambient.r;
	pc.ambient[1] = lighting.ambient.g;
	pc.ambient[2] = lighting.ambient.b;
	pc.ambient[3] = 0.f;

	{
		const Transform3D &t = terrain_transform;
		// Terrain-local directions to sky space: to world with the terrain basis, then into the sky's orientation
		const Basis rx = lighting.sky_orientation.inverse() * t.basis;
		const SceneUBO scene = {
			{ lighting.fog_color.r, lighting.fog_color.g, lighting.fog_color.b, lighting.fog_density },
			{ lighting.fog ? 1.f : 0.f, lighting.fog_depth_mode ? 1.f : 0.f, lighting.fog_sun_scatter,
					lighting.fog_height_density },
			{ lighting.fog_depth_begin, lighting.fog_depth_end, lighting.fog_depth_curve, lighting.fog_height },
			{ static_cast<float>(t.basis.rows[1].x), static_cast<float>(t.basis.rows[1].y),
					static_cast<float>(t.basis.rows[1].z), static_cast<float>(t.origin.y) },
			{ float(rx.rows[0].x), float(rx.rows[0].y), float(rx.rows[0].z), 0.f },
			{ float(rx.rows[1].x), float(rx.rows[1].y), float(rx.rows[1].z), 0.f },
			{ float(rx.rows[2].x), float(rx.rows[2].y), float(rx.rows[2].z), 0.f },
			{ lighting.use_ambient_cubemap ? 1.f : 0.f, lighting.use_reflection_cubemap ? 1.f : 0.f,
					lighting.ibl_energy, lighting.ambient_sky_mix },
			{ lighting.radiance_border, 0.f, 0.f, 0.f },
			{ float(camera_frac.x), float(camera_frac.y), float(camera_frac.z), 0.f },
		};
		rd.buffer_update(_scene_buffer, 0, sizeof(scene), &scene);
	}

	const RID radiance_set = _get_radiance_uniform_set(lighting.radiance);
	if (radiance_set.is_null()) {
		return;
	}

	RenderingDevice::DrawListID dl = rd.draw_list_begin(framebuffer);
	rd.draw_list_bind_render_pipeline(dl, pipeline);
	rd.draw_list_bind_uniform_set(dl, _render_uniform_set, 0);
	rd.draw_list_bind_uniform_set(dl, radiance_set, 1);
	rd.draw_list_set_push_constant(dl, &pc, sizeof(pc));
	rd.draw_list_draw_indirect(dl, false, _indirect_buffer, 0, _chunk_high_water, 16);
	rd.draw_list_end();

	MutexLock lock(_mutex);
	++_stats.indirect_draws_total;
}

} // namespace zylann::voxel

#endif // VOXEL_ENABLE_GPU_DRIVEN_RENDERING
