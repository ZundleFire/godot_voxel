#ifndef VOXEL_EROSION_FILTER_GLSL_H
#define VOXEL_EROSION_FILTER_GLSL_H

// GLSL twin of voxel_erosion_filter_impl.h for voxel graph compute shaders — keep in sync.
//
// Phacelle Noise and Advanced Terrain Erosion Filter
// Copyright (c) 2025 Rune Skovbo Johansen
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

namespace zylann::voxel {

// Split in chunks: some compilers limit string literal length
static const char *g_erosion_filter_shader[] = {
	R"GLSL(
struct VgErosionParams {
	int seed; float planet_radius; float tile_size; float triplanar_sharpness;
	float height_frequency; float height_amp; int height_octaves; float height_lacunarity; float height_gain;
	float scale; float strength; float gully_weight; float detail;
	float rounding_ridge; float rounding_crease; float rounding_input_mult; float rounding_octave_mult;
	float onset_initial; float onset_octave; float ridge_onset_initial; float ridge_onset_octave;
	float assumed_slope; float assumed_slope_blend; float cell_scale; float normalization;
	int octaves; float lacunarity; float gain; float height_offset; float height_offset_fade_blend;
};

float vg_er_u01(uint v) {
	return float(int(v)) * (1.0 / 4294967296.0) + 0.5;
}

vec2 vg_er_hash22(uint seed, vec2 p) {
	uint h = floatBitsToUint(p.x * 141421356.0) ^ floatBitsToUint(p.y * 2718281828.0) ^ seed;
	return vec2(vg_er_u01(h * 3141592653u), vg_er_u01(h * 1618033988u));
}

float vg_er_perlin_deriv(uint seed, vec2 p, out vec2 grad) {
	vec2 c = floor(p);
	vec2 f = p - c;
	vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
	vec2 du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
	vec2 ga = vg_er_hash22(seed, c + vec2(0.0, 0.0)) * 2.0 - 1.0;
	vec2 gb = vg_er_hash22(seed, c + vec2(1.0, 0.0)) * 2.0 - 1.0;
	vec2 gc = vg_er_hash22(seed, c + vec2(0.0, 1.0)) * 2.0 - 1.0;
	vec2 gd = vg_er_hash22(seed, c + vec2(1.0, 1.0)) * 2.0 - 1.0;
	float va = dot(ga, f - vec2(0.0, 0.0));
	float vb = dot(gb, f - vec2(1.0, 0.0));
	float vc = dot(gc, f - vec2(0.0, 1.0));
	float vd = dot(gd, f - vec2(1.0, 1.0));
	grad = ga + u.x * (gb - ga) + u.y * (gc - ga) + u.x * u.y * (ga - gb - gc + gd) +
			du * (u.yx * (va - vb - vc + vd) + vec2(vb, vc) - va);
	return va + u.x * (vb - va) + u.y * (vc - va) + u.x * u.y * (va - vb - vc + vd);
}

float vg_er_clamp01(float x) { return clamp(x, 0.0, 1.0); }
float vg_er_pow_inv(float t, float power) { return 1.0 - pow(1.0 - vg_er_clamp01(t), power); }
float vg_er_ease_out(float t) { float v = 1.0 - vg_er_clamp01(t); return 1.0 - v * v; }
float vg_er_smooth_start(float t, float s) {
	return t >= s ? t - 0.5 * s : 0.5 * t * t / max(s, 1e-10);
}
vec2 vg_er_safe_normalize(vec2 n) {
	float l = length(n);
	return l > 1e-10 ? n / l : n;
}
)GLSL",
	R"GLSL(
vec2 vg_er_phacelle(uint seed, vec2 p, vec2 norm_dir, float freq, float offset, float normalization, out vec2 side_dir) {
	side_dir = vec2(-norm_dir.y, norm_dir.x) * (freq * 6.28318530718);
	float offset_rad = offset * 6.28318530718;
	vec2 pi = floor(p);
	vec2 pf = p - pi;
	vec2 phase_dir = vec2(0.0);
	float weight_sum = 0.0;
	for (int i = -1; i <= 2; i++) {
		for (int j = -1; j <= 2; j++) {
			vec2 go = vec2(float(i), float(j));
			vec2 v = pf - go - (vg_er_hash22(seed, pi + go) - 0.5);
			float w = max(0.0, exp(-dot(v, v) * 2.0) - 0.01111);
			weight_sum += w;
			float wave = dot(v, side_dir) + offset_rad;
			phase_dir += vec2(cos(wave), sin(wave)) * w;
		}
	}
	vec2 interpolated = phase_dir / max(weight_sum, 1e-6);
	float magnitude = max(max(1.0 - normalization, length(interpolated)), 1e-10);
	return interpolated / magnitude;
}

// Returns (delta, magnitude, ridge_map)
vec3 vg_er_erosion_filter(uint seed, vec2 p, vec3 in_hs, float in_fade_target, VgErosionParams prm) {
	float strength = prm.strength * prm.scale;
	float fade_target = clamp(in_fade_target, -1.0, 1.0);
	float height = in_hs.x;
	float freq = 1.0 / (prm.scale * prm.cell_scale);
	float rounding_mult = 1.0;
	float magnitude = 0.0;
	vec2 in_slope = in_hs.yz;
	float slope_length = max(length(in_slope), 1e-10);
	float rounding_for_input = mix(prm.rounding_crease, prm.rounding_ridge, vg_er_clamp01(fade_target + 0.5)) * prm.rounding_input_mult;
	float combi_mask = vg_er_ease_out(vg_er_smooth_start(slope_length * prm.onset_initial, rounding_for_input * prm.onset_initial));
	float ridge_combi_mask = vg_er_ease_out(slope_length * prm.ridge_onset_initial);
	float ridge_fade_target = fade_target;
	vec2 gully_slope = in_slope + (in_slope * (prm.assumed_slope / slope_length) - in_slope) * prm.assumed_slope_blend;
	for (int i = 0; i < prm.octaves; i++) {
		vec2 side_dir;
		vec2 ph = vg_er_phacelle(seed, p * freq, vg_er_safe_normalize(gully_slope), prm.cell_scale, 0.25, prm.normalization, side_dir);
		side_dir *= -freq;
		float sloping = abs(ph.y);
		gully_slope += side_dir * (sign(ph.y) * strength * prm.gully_weight);
		float faded = mix(fade_target, ph.x * prm.gully_weight, combi_mask);
		height += faded * strength;
		magnitude += strength;
		fade_target = faded;
		float rounding_for_octave = mix(prm.rounding_crease, prm.rounding_ridge, vg_er_clamp01(ph.x + 0.5)) * rounding_mult;
		float new_mask = vg_er_ease_out(vg_er_smooth_start(sloping * prm.onset_octave, rounding_for_octave * prm.onset_octave));
		combi_mask = vg_er_pow_inv(combi_mask, prm.detail) * new_mask;
		ridge_fade_target = mix(ridge_fade_target, ph.x, ridge_combi_mask);
		ridge_combi_mask *= vg_er_ease_out(sloping * prm.ridge_onset_octave);
		strength *= prm.gain;
		freq *= prm.lacunarity;
		rounding_mult *= prm.rounding_octave_mult;
	}
	return vec3(height - in_hs.x, magnitude, ridge_fade_target * (1.0 - ridge_combi_mask));
}
)GLSL",
	R"GLSL(
// Returns (height, ridge, erosion) in filter units
vec3 vg_er_eroded_terrain_2d(vec2 p, VgErosionParams prm) {
	float n = 0.0;
	vec2 slope = vec2(0.0);
	float freq = prm.height_frequency;
	float amp = 1.0;
	int octave_seed = prm.seed;
	for (int i = 0; i < prm.height_octaves; i++) {
		vec2 g;
		n += vg_er_perlin_deriv(uint(octave_seed), p * freq, g) * amp;
		slope += g * (amp * freq);
		amp *= prm.height_gain;
		freq *= prm.height_lacunarity;
		octave_seed = octave_seed * 196314165 + 907633515;
	}
	n *= prm.height_amp;
	slope *= prm.height_amp;
	float fade_target = clamp(n / max(prm.height_amp * 0.6, 1e-6), -1.0, 1.0);
	vec3 hs = vec3(n, slope) * 0.5;
	vec3 e = vg_er_erosion_filter(uint(prm.seed) ^ 0x2545F491u, p, hs, fade_target, prm);
	float offset = mix(prm.height_offset, -fade_target, prm.height_offset_fade_blend) * e.y;
	return vec3(hs.x + e.x + offset, e.z, vg_er_clamp01(e.x / max(e.y, 1e-10) * 0.5 + 0.5));
}

// Returns (height in world units, ridge, erosion)
vec3 vg_planet_erosion(vec3 pos, VgErosionParams prm) {
	vec3 dir = pos / max(length(pos), 1e-6);
	vec3 s = dir * (prm.planet_radius / prm.tile_size);
	vec3 w = pow(abs(dir), vec3(prm.triplanar_sharpness));
	w = max(w / max(w.x + w.y + w.z, 1e-6) - 0.02, vec3(0.0));
	w /= max(w.x + w.y + w.z, 1e-6);
	vec3 r = vec3(0.0);
	if (w.x > 0.0) {
		r += vg_er_eroded_terrain_2d(vec2(s.y + (dir.x < 0.0 ? 517.3 : 0.0), s.z), prm) * w.x;
	}
	if (w.y > 0.0) {
		r += vg_er_eroded_terrain_2d(vec2(s.z + (dir.y < 0.0 ? 311.9 : 0.0), s.x), prm) * w.y;
	}
	if (w.z > 0.0) {
		r += vg_er_eroded_terrain_2d(vec2(s.x + (dir.z < 0.0 ? 743.1 : 0.0), s.y), prm) * w.z;
	}
	r.x *= prm.tile_size;
	return r;
}
)GLSL",
	nullptr
};

} // namespace zylann::voxel

#endif // VOXEL_EROSION_FILTER_GLSL_H
