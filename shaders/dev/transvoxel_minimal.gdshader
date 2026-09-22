shader_type spatial;

// Default material for VoxelMesherTransvoxel (returned by get_default_lod_material() when no
// material_override is set on the terrain node). Renders the MIXEL4 per-voxel material blend
// (CHANNEL_INDICES/CHANNEL_WEIGHTS) via a fixed palette. Not a uniform array: this shader
// compiler doesn't accept default values on array uniforms (tried both brace-init and the
// vec4[16](...) constructor form, both failed to parse), so the palette is a plain function
// instead -- not Inspector-editable, but this is only ever the no-material-assigned fallback.
// Indices must match EdenPlanetGeneratorV1's MAT_* enum: 0=grass 1=rock 2=snow 3=sand 4=dirt
// 5=moss 6=ocean_floor.
vec4 get_material_color(int i) {
	if (i == 0) {
		return vec4(0.30, 0.55, 0.20, 1.0); // grass
	} else if (i == 1) {
		return vec4(0.45, 0.42, 0.40, 1.0); // rock
	} else if (i == 2) {
		return vec4(0.95, 0.97, 1.00, 1.0); // snow
	} else if (i == 3) {
		return vec4(0.85, 0.75, 0.45, 1.0); // sand
	} else if (i == 4) {
		return vec4(0.35, 0.25, 0.15, 1.0); // dirt
	} else if (i == 5) {
		return vec4(0.25, 0.40, 0.22, 1.0); // moss
	} else if (i == 6) {
		return vec4(0.30, 0.30, 0.35, 1.0); // ocean floor
	}
	return vec4(1.0, 0.0, 1.0, 1.0); // unknown index -- obvious magenta, not a silent guess
}
uniform float u_roughness = 0.9;

// From Voxel Tools API
uniform int u_transition_mask;

float get_transvoxel_secondary_factor(int idata) {
	int transition_mask = u_transition_mask & 0xff;

	int cell_border_mask = idata & 63; // Which sides the cell is touching
	int vertex_border_mask = (idata >> 8) & 63; // Which sides the vertex is touching
	// If the vertex is near a side where there is a low-resolution neighbor,
	// move it to secondary position
	int m = transition_mask & cell_border_mask;
	float t = float(m != 0);
	// If the vertex lies on one or more sides, and at least one side has no low-resolution neighbor,
	// don't move the vertex.
	t *= float((vertex_border_mask & ~transition_mask) == 0);

	// Debugging
	//t *= 0.5 + 0.5 * sin(TIME * 4.0);
	//t *= 2.0;

	return t;
}

vec3 get_transvoxel_position(vec3 vertex_pos, vec4 fdata) {
	int idata = floatBitsToInt(fdata.a);

	// Move vertices to smooth transitions
	float secondary_factor = get_transvoxel_secondary_factor(idata);
	vec3 secondary_position = fdata.xyz;
	vec3 pos = mix(vertex_pos, secondary_position, secondary_factor);

	// If the mesh combines transitions and the vertex belongs to a transition,
	// when that transition isn't active we change the position of the vertices so
	// all triangles will be degenerate and won't be visible.
	// This is an alternative to rendering them separately,
	// which has less draw calls and less mesh resources to create in Godot.
	// Ideally I would tweak the index buffer like LOD does but Godot does not
	// expose anything to use it that way.
	int itransition = (idata >> 16) & 0xff; // Is the vertex on a transition mesh?
	float transition_cull = float(itransition == 0 || (itransition & u_transition_mask) != 0);
	pos *= transition_cull;

	return pos;
}

// CUSTOM0/CUSTOM1 are only valid inside vertex() -- capture what fragment() needs into varyings.
// Not marked flat: matches planet_biome_debug.gdshader's approach of decoding this per-fragment
// after ordinary interpolation, rather than risking another compiler quirk with that qualifier.
varying vec2 v_custom1;

void vertex() {
	VERTEX = get_transvoxel_position(VERTEX, CUSTOM0);
	v_custom1 = CUSTOM1.xy;
}

// v_custom1 = MIXEL4 material data: x = packed 4x8 material indices, y = packed 4x8 weights
// (see transvoxel_materials_common.h add_4i8_4w8_texture_data / transvoxel_materials_mixel4.h).
void fragment() {
	uint pi = floatBitsToUint(v_custom1.x);
	uint pw = floatBitsToUint(v_custom1.y);

	vec4 idx = vec4(float(pi & 0xffu), float((pi >> 8u) & 0xffu), float((pi >> 16u) & 0xffu), float((pi >> 24u) & 0xffu));
	vec4 w = vec4(float(pw & 0xffu), float((pw >> 8u) & 0xffu), float((pw >> 16u) & 0xffu), float((pw >> 24u) & 0xffu));
	float wsum = max(w.x + w.y + w.z + w.w, 0.0001);

	vec3 col = get_material_color(int(idx.x)).rgb * w.x;
	col += get_material_color(int(idx.y)).rgb * w.y;
	col += get_material_color(int(idx.z)).rgb * w.z;
	col += get_material_color(int(idx.w)).rgb * w.w;
	col /= wsum;

	ALBEDO = col;
	ROUGHNESS = u_roughness;
}
