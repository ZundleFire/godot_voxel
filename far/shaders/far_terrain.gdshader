shader_type spatial;

// Far terrain surface shader for VoxelFarTerrain.
//
// Assign to the node's `material` as a ShaderMaterial.
//
// Vertex attribute contract, set by VoxelFarTerrain::build_mesh_resource:
//   VERTEX   sector-local position (sector world offset lives in the transform)
//   NORMAL   normal captured at the resolution the data came from
//   COLOR.r  baked ambient occlusion, 0 = occluded, 1 = open
//   COLOR.g  material index / 255
//
// The atmosphere model here is the part that matters. A single distance
// smoothstep -- the obvious approach, and what this shader used to do -- is not
// enough at horizon scale, for a reason that only shows up once the view
// distance is tens of kilometres: fog that depends on distance alone makes
// mountains 40 km away the same colour as the valley floor 40 km away, so all
// depth cue is lost and the scene flattens into a painted backdrop.
//
// What fixes it is two independent fog terms that are then mixed:
//
//   * DISTANCE fog, which handles aerial perspective with distance.
//   * HEIGHT fog, which puts haze in the valleys and leaves peaks clear.
//
// Height fog is what restores the depth cue, because it varies with altitude
// rather than range. Terrain silhouettes separate again, and the classic look
// of distant ranges stacked in progressively paler layers falls out on its own.
//
// Both terms carry their own falloff curve, and the mixing mode matters as much
// as the terms do: MAX keeps whichever is denser (crisp peaks above a fog
// layer), ADD compounds them (thick weather), MULTIPLY lets height fog thin out
// the distance fog (clear high-altitude air).

render_mode cull_back, depth_draw_opaque, diffuse_burley, specular_schlick_ggx;

group_uniforms Handover;
// Where the near terrain ends. Bracket VoxelFarTerrain.near_clip_radius.
uniform float near_fade_start = 400.0;
uniform float near_fade_end = 520.0;
// Pushes far terrain very slightly away from the camera so that where it
// coincides with near terrain, near terrain wins. Far cheaper and steadier than
// depth bias, because the offset is in world units and so does not vary with
// depth precision across a 200 km range.
uniform float micro_depth_offset = 0.02;

group_uniforms Curvature;
// Bends distant terrain down as if over a sphere. At a 40 km view distance a
// flat world shows a hard straight horizon edge with sky under it; curving the
// surface away hides that edge and reads as genuine scale.
// 0 disables. Earth is ~6371000. Smaller values exaggerate.
uniform float planet_radius = 0.0;

group_uniforms Surface;
uniform vec3 low_color : source_color = vec3(0.24, 0.34, 0.18);
uniform vec3 high_color : source_color = vec3(0.52, 0.54, 0.56);
uniform vec3 peak_color : source_color = vec3(0.92, 0.94, 0.97);
uniform float low_height = 0.0;
uniform float high_height = 220.0;
uniform float peak_height = 420.0;
uniform vec3 cliff_color : source_color = vec3(0.35, 0.32, 0.29);
uniform float cliff_slope_start = 0.72;
uniform float cliff_slope_end = 0.45;
// Material palette.
//
// The mesher stores one material index per layer, packed into vertex colour
// green. `material_blend` mixes the palette over the height/slope colouring
// rather than replacing it, so a partly-filled palette still reads sensibly.
// Off by default: a generator with no material channel reports 0 everywhere,
// and tinting the whole world with palette entry 0 would look like a bug.
uniform float material_blend : hint_range(0.0, 1.0) = 0.0;
uniform vec3 material_colors[8];

uniform float ao_strength : hint_range(0.0, 1.0) = 0.75;
uniform float roughness_value : hint_range(0.0, 1.0) = 0.95;

group_uniforms Detail_Noise;
// Decimated terrain is too clean: large flat facets read as obviously low
// detail. A little quantised noise restores the *impression* of detail for a
// few ALU ops, with no extra geometry and no texture fetch.
//
// Quantised rather than smooth on purpose. Banding into a handful of steps
// reads as terrain strata and material variation; smooth noise just looks like
// the surface is dirty.
uniform bool noise_enabled = true;
uniform float noise_intensity : hint_range(0.0, 1.0) = 0.06;
uniform int noise_steps : hint_range(1, 16) = 4;
// Distance at which noise has fully faded out. Near the camera the real
// geometry already carries detail, so noise there only adds visual grain.
uniform float noise_dropoff = 12000.0;

group_uniforms Distance_Fog;
uniform vec3 fog_color : source_color = vec3(0.55, 0.68, 0.85);
uniform float fog_start = 8000.0;
uniform float fog_end = 90000.0;
uniform float fog_density : hint_range(0.0, 4.0) = 1.0;
uniform float fog_min : hint_range(0.0, 1.0) = 0.0;
uniform float fog_max : hint_range(0.0, 1.0) = 1.0;
// 0 linear, 1 exponential, 2 exponential squared
uniform int fog_falloff : hint_range(0, 2) = 1;
// Spherical fog uses true range, so fog is even in every direction. Cylindrical
// ignores the vertical component, so flying up does not push the horizon fog
// away -- usually what you want for terrain.
uniform bool fog_spherical = false;

group_uniforms Height_Fog;
uniform bool height_fog_enabled = true;
uniform vec3 height_fog_color : source_color = vec3(0.62, 0.72, 0.86);
uniform float height_fog_base = 60.0;
// Thickness of the layer above/below the base height.
uniform float height_fog_length = 400.0;
uniform float height_fog_density : hint_range(0.0, 4.0) = 1.0;
uniform float height_fog_min : hint_range(0.0, 1.0) = 0.0;
uniform float height_fog_max : hint_range(0.0, 1.0) = 1.0;
uniform int height_fog_falloff : hint_range(0, 2) = 1;
uniform bool height_fog_applies_down = true;
uniform bool height_fog_applies_up = false;
// Anchor the layer to the camera instead of to world zero, so the haze follows
// you rather than being something you climb out of once and never see again.
uniform bool height_fog_follows_camera = false;
// 0 max, 1 add, 2 multiply, 3 height fog only
uniform int fog_mixing_mode : hint_range(0, 3) = 0;

varying float v_world_height;
varying float v_view_distance;
varying float v_flat_distance;
varying float v_ao;
// `flat` on purpose. The material is an index, not a quantity: interpolating it
// across a triangle and rounding in the fragment shader produces a band of
// wrong materials along every edge where two materials meet, which reads as
// speckle.
// The palette colour for this vertex's material, not the material index.
//
// The index is looked up in the vertex shader, where it is exact, and what gets
// interpolated is the resulting colour. Two wrong ways to do this:
//
//   interpolate the index and round in the fragment shader -- produces a band of
//   whatever material happens to lie between the two along every edge where two
//   materials meet, which reads as speckle;
//
//   mark the index `flat` -- correct, but then a whole triangle takes the
//   provoking vertex's material and every material boundary becomes a visible
//   facet.
//
// Interpolating the colour has neither problem, and the palette stays a uniform
// rather than being baked into the mesh.
varying vec3 v_material_color;
varying vec3 v_world_position;

float apply_falloff(float t, int mode) {
	t = clamp(t, 0.0, 1.0);
	if (mode == 1) {
		// Exponential: thickens early, which matches how haze actually
		// accumulates and avoids the obviously linear ramp.
		return 1.0 - exp(-3.0 * t);
	} else if (mode == 2) {
		return 1.0 - exp(-3.0 * t * t);
	}
	return t;
}

void vertex() {
	vec3 world_position = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;

	vec3 to_camera = world_position - CAMERA_POSITION_WORLD;
	float flat_distance = length(to_camera.xz);

	// Planet curvature. Dropping by d^2 / 2R is the sagitta of a circle of
	// radius R over a chord of length d -- accurate to well under a metre at
	// any distance this renders, and one multiply-add instead of a trig call.
	float curve_drop = 0.0;
	if (planet_radius > 0.0) {
		curve_drop = (flat_distance * flat_distance) / (2.0 * planet_radius);
	}

	// Curvature and the depth nudge are both world-space displacements, so they
	// are summed and converted back once. Calling inverse() per displacement
	// would pay for a full 4x4 inversion twice per vertex, which at this vertex
	// count is not free.
	vec3 view_dir = normalize(to_camera);
	vec3 world_displacement = view_dir * micro_depth_offset - vec3(0.0, curve_drop, 0.0);

	if (planet_radius > 0.0 || micro_depth_offset != 0.0) {
		VERTEX += (inverse(MODEL_MATRIX) * vec4(world_displacement, 0.0)).xyz;
		world_position += world_displacement;
	}

	v_world_position = world_position;
	v_world_height = world_position.y;
	v_view_distance = length(world_position - CAMERA_POSITION_WORLD);
	v_flat_distance = flat_distance;
	v_ao = COLOR.r;
	// Vertex colours are quantised to 8 bits by ARRAY_FLAG_COMPRESS_ATTRIBUTES,
	// and the index was written as `material / 255`, so this round-trips exactly.
	v_material_color = material_colors[int(clamp(COLOR.g * 255.0 + 0.5, 0.0, 7.0))];
}

// Ordered dithering, rotated per frame. Screen-space rather than real alpha
// blending so the far terrain stays in the opaque pass -- depth-sorting
// thousands of transparent sectors is not worth it for a fade band.
//
// The per-frame rotation matters: a static Bayer pattern is a visible fixed
// grid, while rotating it turns the artifact into temporal noise that TAA
// resolves into a smooth gradient.
float dither_threshold(vec2 frag, int frame) {
	const mat4 bayer = mat4(
		vec4( 0.0,  8.0,  2.0, 10.0),
		vec4(12.0,  4.0, 14.0,  6.0),
		vec4( 3.0, 11.0,  1.0,  9.0),
		vec4(15.0,  7.0, 13.0,  5.0)
	);
	ivec2 p = ivec2(mod(frag + vec2(float(frame) * 3.0, float(frame) * 5.0), 4.0));
	return (bayer[p.y][p.x] + 0.5) / 16.0;
}

float hash13(vec3 p) {
	p = fract(p * 0.1031);
	p += dot(p, p.zyx + 31.32);
	return fract((p.x + p.y) * p.z);
}

float value_noise(vec3 p) {
	vec3 i = floor(p);
	vec3 f = fract(p);
	f = f * f * (3.0 - 2.0 * f);
	float n000 = hash13(i + vec3(0.0, 0.0, 0.0));
	float n100 = hash13(i + vec3(1.0, 0.0, 0.0));
	float n010 = hash13(i + vec3(0.0, 1.0, 0.0));
	float n110 = hash13(i + vec3(1.0, 1.0, 0.0));
	float n001 = hash13(i + vec3(0.0, 0.0, 1.0));
	float n101 = hash13(i + vec3(1.0, 0.0, 1.0));
	float n011 = hash13(i + vec3(0.0, 1.0, 1.0));
	float n111 = hash13(i + vec3(1.0, 1.0, 1.0));
	return mix(
		mix(mix(n000, n100, f.x), mix(n010, n110, f.x), f.y),
		mix(mix(n001, n101, f.x), mix(n011, n111, f.x), f.y),
		f.z
	);
}

void fragment() {
	int frame = int(mod(TIME * 60.0, 8.0));

	float near_fade = smoothstep(near_fade_start, near_fade_end, v_view_distance);
	if (near_fade < dither_threshold(FRAGCOORD.xy, frame)) {
		discard;
	}

	vec3 world_normal = normalize(NORMAL);

	vec3 albedo = mix(low_color, high_color, smoothstep(low_height, high_height, v_world_height));
	albedo = mix(albedo, peak_color, smoothstep(high_height, peak_height, v_world_height));

	// Slope override. The stored normal is what makes this work at distance: it
	// was captured at the source resolution, so a cliff reads as a cliff even
	// after the geometry under it has decimated to a single quad.
	float rockiness = 1.0 - smoothstep(cliff_slope_end, cliff_slope_start, world_normal.y);
	albedo = mix(albedo, cliff_color, rockiness);

	if (material_blend > 0.0) {
		albedo = mix(albedo, v_material_color, material_blend);
	}

	if (noise_enabled) {
		float n = value_noise(v_world_position * 0.05);
		// Quantise into bands so it reads as strata rather than grain.
		n = floor(n * float(noise_steps)) / float(noise_steps);
		float fade = 1.0 - smoothstep(0.0, noise_dropoff, v_view_distance);
		albedo *= 1.0 + (n - 0.5) * 2.0 * noise_intensity * fade;
	}


	// -- Atmosphere --------------------------------------------------------

	float fog_distance = fog_spherical ? v_view_distance : v_flat_distance;
	float dist_t = (fog_distance - fog_start) / max(fog_end - fog_start, 0.001);
	float distance_fog = apply_falloff(dist_t, fog_falloff) * fog_density;
	distance_fog = clamp(distance_fog, fog_min, fog_max);

	float height_fog = 0.0;
	if (height_fog_enabled) {
		float base = height_fog_follows_camera ? (CAMERA_POSITION_WORLD.y + height_fog_base) : height_fog_base;
		float delta = base - v_world_height;

		// Only accumulate on the sides the layer is configured to fill. Down is
		// the usual case: haze pools in valleys and peaks stand clear of it.
		float depth_in_layer = 0.0;
		if (delta > 0.0 && height_fog_applies_down) {
			depth_in_layer = delta;
		} else if (delta < 0.0 && height_fog_applies_up) {
			depth_in_layer = -delta;
		}

		float height_t = depth_in_layer / max(height_fog_length, 0.001);
		height_fog = apply_falloff(height_t, height_fog_falloff) * height_fog_density;
		height_fog = clamp(height_fog, height_fog_min, height_fog_max);

		// Height fog still has to respect range, or terrain directly below the
		// camera would be fogged as heavily as the horizon.
		height_fog *= clamp(dist_t, 0.0, 1.0);
	}

	float total_fog;
	vec3 mixed_fog_color;
	if (fog_mixing_mode == 1) {
		total_fog = clamp(distance_fog + height_fog, 0.0, 1.0);
		mixed_fog_color = mix(fog_color, height_fog_color, height_fog / max(total_fog, 0.001));
	} else if (fog_mixing_mode == 2) {
		total_fog = clamp(distance_fog * height_fog, 0.0, 1.0);
		mixed_fog_color = mix(fog_color, height_fog_color, 0.5);
	} else if (fog_mixing_mode == 3) {
		total_fog = height_fog;
		mixed_fog_color = height_fog_color;
	} else {
		// MAX: whichever term is denser wins, and the colour follows it. This
		// is what gives peaks standing above a valley haze layer.
		total_fog = max(distance_fog, height_fog);
		mixed_fog_color = (height_fog > distance_fog) ? height_fog_color : fog_color;
	}

	// Fog washes out lighting as it thickens; leaving the surface fully lit
	// under dense fog is what makes distant terrain look like a cardboard
	// cutout pasted over the sky. Mixing the fog into ALBEDO is not enough --
	// the result is then multiplied by sun and ambient, so a fully fogged
	// surface comes out as the fog colour *lit*, never the fog colour itself,
	// and the horizon can never meet the sky. Fading ALBEDO out while driving
	// the same amount through EMISSION takes the fog out of the lit path
	// entirely: at total_fog == 1 the fragment is exactly mixed_fog_color.
	float fog_amount = clamp(total_fog, 0.0, 1.0);

	ALBEDO = albedo * (1.0 - fog_amount);
	EMISSION = mixed_fog_color * fog_amount;
	ROUGHNESS = mix(roughness_value, 1.0, fog_amount);
	SPECULAR = 0.05 * (1.0 - fog_amount);
	NORMAL = (VIEW_MATRIX * vec4(world_normal, 0.0)).xyz;
	AO = mix(1.0, v_ao, ao_strength);
	AO_LIGHT_AFFECT = 0.4;
}
