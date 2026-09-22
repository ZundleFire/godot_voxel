shader_type spatial;
render_mode blend_mix, depth_draw_opaque, cull_back, diffuse_burley, specular_schlick_ggx;

// demo_eden/shaders/planet_v4.gdshader with the look changes made to the GPU-driven shader
// (voxel_gpu_driven_renderer.cpp), so both render paths shade identically in A/B checks:
//   - vibrance applied in a perceptual (gamma 2) space instead of linear + clamp (greens went neon)
//   - darker, physically plausible palette (ocean floor kept bright while there is no water)
//   - macro/meso color variation, noisy rock slope threshold, rock strata and tone variation
// Parameters are unchanged.

uniform int u_transition_mask;
uniform vec2 u_lod_fade = vec2(0.0, 0.0);

uniform float u_planet_radius = 40000.0;
// 1=erosion 2=ridge 3=moisture 4=temperature 5=world normal (unlit)
uniform int u_debug_mode = 0;

uniform float vibrance : hint_range(0.0, 2.0) = 1.18;
uniform float slope_rock_start : hint_range(0.0, 1.0) = 0.6;
uniform float slope_rock_end : hint_range(0.0, 1.0) = 0.38;
uniform float snow_temperature : hint_range(0.0, 1.0) = 0.22;
uniform float snow_blend : hint_range(0.001, 0.5) = 0.06;
uniform float gully_darkening : hint_range(0.0, 1.0) = 0.3;
uniform float ridge_highlight : hint_range(0.0, 1.0) = 0.25;
uniform float dryness_tint : hint_range(0.0, 1.0) = 0.6;
uniform float detail_normal_strength : hint_range(0.0, 3.0) = 0.0;
uniform float detail_normal_scale = 6.0;

varying vec3 v_albedo;
varying vec2 v_vegetation_roughness;
varying vec4 v_surface;
varying vec3 v_normal_world;
varying vec3 v_pos_world;

vec4 decode_8bit_vec4(float v) {
	uint i = floatBitsToUint(v);
	return vec4(float(i & 0xffu), float((i >> 8u) & 0xffu), float((i >> 16u) & 0xffu), float((i >> 24u) & 0xffu));
}

vec3 mat_albedo(int i) {
	if (i == 0) return vec3(0.075, 0.20, 0.045); // grass
	if (i == 1) return vec3(0.27, 0.245, 0.215); // rock
	if (i == 2) return vec3(0.82, 0.86, 0.92); // snow
	if (i == 3) return vec3(0.56, 0.44, 0.24); // sand
	if (i == 4) return vec3(0.21, 0.13, 0.065); // dirt
	if (i == 5) return vec3(0.05, 0.14, 0.065); // moss
	if (i == 6) return vec3(0.20, 0.30, 0.36); // ocean floor (bright until water exists)
	return vec3(1.0, 0.0, 1.0);
}

float mat_roughness(int i) {
	if (i == 2) return 0.42;
	if (i == 1) return 0.72;
	if (i == 3) return 0.78;
	if (i == 6) return 0.62;
	return 0.88;
}

float luma(vec3 c) {
	return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

vec3 saturate_color(vec3 c, float amount) {
	vec3 p = sqrt(max(c, vec3(0.0)));
	vec3 s = max(mix(vec3(luma(p)), p, amount), vec3(0.0));
	return s * s;
}

float hash13(vec3 p) {
	p = fract(p * 0.3183099 + 0.1);
	p *= 17.0;
	return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float value_noise(vec3 p) {
	vec3 i = floor(p);
	vec3 f = fract(p);
	f = f * f * (3.0 - 2.0 * f);
	float n00 = mix(hash13(i), hash13(i + vec3(1.0, 0.0, 0.0)), f.x);
	float n10 = mix(hash13(i + vec3(0.0, 1.0, 0.0)), hash13(i + vec3(1.0, 1.0, 0.0)), f.x);
	float n01 = mix(hash13(i + vec3(0.0, 0.0, 1.0)), hash13(i + vec3(1.0, 0.0, 1.0)), f.x);
	float n11 = mix(hash13(i + vec3(0.0, 1.0, 1.0)), hash13(i + vec3(1.0, 1.0, 1.0)), f.x);
	return mix(mix(n00, n10, f.y), mix(n01, n11, f.y), f.z);
}

vec3 perturb_normal(vec3 p, vec3 n, float scale, float strength) {
	float e = scale * 0.5;
	float h = value_noise(p / scale);
	vec3 g = vec3(
			value_noise((p + vec3(e, 0.0, 0.0)) / scale) - h,
			value_noise((p + vec3(0.0, e, 0.0)) / scale) - h,
			value_noise((p + vec3(0.0, 0.0, e)) / scale) - h);
	g -= n * dot(g, n);
	return normalize(n - g * strength);
}

vec3 get_transvoxel_position(vec3 vertex_pos, vec4 fdata) {
	int idata = floatBitsToInt(fdata.a);
	int transition_mask = u_transition_mask & 0xff;
	int cell_border_mask = idata & 63;
	int vertex_border_mask = (idata >> 8) & 63;
	float t = float((transition_mask & cell_border_mask) != 0);
	t *= float((vertex_border_mask & ~transition_mask) == 0);
	vec3 pos = mix(vertex_pos, fdata.xyz, t);
	int itransition = (idata >> 16) & 0xff;
	return pos * float(itransition == 0 || (itransition & u_transition_mask) != 0);
}

float get_hash(vec2 c) {
	return fract(sin(dot(c.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

void vertex() {
	VERTEX = get_transvoxel_position(VERTEX, CUSTOM0);
	v_pos_world = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
	v_normal_world = (MODEL_MATRIX * vec4(NORMAL, 0.0)).xyz;
	// MIXEL4 resolved per vertex, then interpolated (neighbors can reference different material sets)
	vec4 idx = decode_8bit_vec4(CUSTOM1.x);
	vec4 w = decode_8bit_vec4(CUSTOM1.y);
	float wsum = w.x + w.y + w.z + w.w;
	vec3 albedo = vec3(0.5);
	float roughness = 0.88;
	float vegetation = 0.0;
	if (wsum > 0.0) {
		albedo = vec3(0.0);
		roughness = 0.0;
		for (int k = 0; k < 4; ++k) {
			int mi = int(idx[k]);
			float wk = w[k] / wsum;
			albedo += mat_albedo(mi) * wk;
			roughness += mat_roughness(mi) * wk;
			vegetation += (mi == 0 || mi == 5) ? wk : 0.0;
		}
	}
	v_albedo = albedo;
	v_vegetation_roughness = vec2(vegetation, roughness);
	v_surface = dot(CUSTOM2, CUSTOM2) == 0.0 ? vec4(0.5) : CUSTOM2;
}

void fragment() {
	vec3 n_world = normalize(v_normal_world);
	vec3 albedo = v_albedo;
	float roughness = v_vegetation_roughness.y;
	float vegetation = v_vegetation_roughness.x;

	float erosion = v_surface.r;
	float ridge = v_surface.g * 2.0 - 1.0;
	float moisture = v_surface.b;
	float temperature = v_surface.a;

	vec3 lush = saturate_color(albedo * vec3(0.92, 1.08, 0.88), 1.12);
	vec3 dry = mix(albedo, vec3(0.52, 0.44, 0.18), dryness_tint);
	vec3 climate = mix(dry, lush, smoothstep(0.3, 0.7, moisture));
	albedo = mix(albedo, climate, vegetation);
	albedo = mix(albedo, saturate_color(albedo, 0.45), vegetation * (1.0 - smoothstep(0.12, 0.42, temperature)));

	vec3 true_up = normalize(v_pos_world);
	float up_dot = dot(n_world, true_up);
	float altitude = length(v_pos_world) - u_planet_radius;

	float macro = value_noise(v_pos_world / 420.0);
	float meso = value_noise(v_pos_world / 65.0);
	albedo = mix(albedo, albedo * vec3(1.22, 1.08, 0.62), vegetation * smoothstep(0.45, 0.85, macro) * 0.8);
	albedo *= mix(0.82, 1.12, meso * 0.7 + macro * 0.3);

	float gully = clamp(max(-ridge, 0.0) + max(0.5 - erosion, 0.0) * 1.5, 0.0, 1.0);
	albedo *= 1.0 - gully_darkening * gully;
	albedo = mix(albedo, albedo * vec3(0.94, 0.98, 1.02), gully * moisture);
	roughness = mix(roughness, 0.5, gully * moisture);
	float ridge_rock = ridge_highlight * max(ridge, 0.0);

	float slope_jitter = (value_noise(v_pos_world / 140.0) - 0.5) * 0.16;
	float rock_blend = max(1.0 - smoothstep(slope_rock_end, slope_rock_start, up_dot + slope_jitter), ridge_rock);
	float strata = 0.5 + 0.5 * sin(altitude / 38.0 + meso * 5.0);
	vec3 rock = mat_albedo(1) * mix(0.78, 1.18, value_noise(v_pos_world / 23.0));
	rock *= mix(vec3(1.0), vec3(1.12, 0.98, 0.84), strata * 0.6);
	albedo = mix(albedo, rock * (1.0 + ridge_rock * 0.6), rock_blend);
	roughness = mix(roughness, mat_roughness(1), rock_blend);

	float snow = (1.0 - smoothstep(snow_temperature - snow_blend, snow_temperature + snow_blend, temperature))
			* smoothstep(slope_rock_end, 1.0, up_dot) * step(0.0, altitude);
	albedo = mix(albedo, mat_albedo(2), snow);
	roughness = mix(roughness, mat_roughness(2), snow);

	albedo = saturate_color(albedo, vibrance);

	vec3 n_detail = n_world;
	if (detail_normal_strength > 0.0) {
		float fine_fade = 1.0 - smoothstep(0.2, 0.8, length(fwidth(v_pos_world)) / detail_normal_scale);
		float relief = detail_normal_strength * mix(1.0, 1.6, rock_blend) * (1.0 - 0.7 * snow);
		if (fine_fade > 0.01) {
			n_detail = perturb_normal(v_pos_world, n_detail, detail_normal_scale, relief * fine_fade);
			albedo *= 1.0 - 0.07 * fine_fade * value_noise(v_pos_world / detail_normal_scale);
		}
		n_detail = perturb_normal(v_pos_world, n_detail, detail_normal_scale * 5.0, relief * 0.7);
	}
	NORMAL = normalize((VIEW_MATRIX * vec4(n_detail, 0.0)).xyz);

	if (u_debug_mode > 0) {
		vec3 d = u_debug_mode == 1 ? vec3(erosion)
				: u_debug_mode == 2 ? vec3(v_surface.g)
				: u_debug_mode == 3 ? vec3(moisture)
				: u_debug_mode == 4 ? vec3(temperature)
				: n_detail * 0.5 + 0.5;
		ALBEDO = vec3(0.0);
		EMISSION = d;
	} else {
		ALBEDO = albedo;
	}
	ROUGHNESS = roughness;
	METALLIC = 0.0;
	SPECULAR = 0.3 * max(gully * moisture, snow * 0.4);

	if (u_lod_fade.x > 0.0 || u_lod_fade.y > 0.0) {
		float fh = get_hash(SCREEN_UV);
		if (u_lod_fade.y > 0.5) {
			if (u_lod_fade.x < fh) { discard; }
		} else {
			if (u_lod_fade.x > fh) { discard; }
		}
	}
}
