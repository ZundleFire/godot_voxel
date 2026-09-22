shader_type spatial;

// Traditional-path material for the A/B checks: Transvoxel seam handling (same as the module's default
// Transvoxel shader) with a flat albedo, so both render paths show the same silhouettes.
uniform int u_transition_mask;

void vertex() {
	int idata = floatBitsToInt(CUSTOM0.a);
	int transition_mask = u_transition_mask & 0xff;
	int cell_border_mask = idata & 63;
	int vertex_border_mask = (idata >> 8) & 63;
	float t = float((transition_mask & cell_border_mask) != 0);
	t *= float((vertex_border_mask & ~transition_mask) == 0);
	VERTEX = mix(VERTEX, CUSTOM0.xyz, t);
	int itransition = (idata >> 16) & 0xff;
	VERTEX *= float(itransition == 0 || (itransition & u_transition_mask) != 0);
}

void fragment() {
	ALBEDO = vec3(0.6);
	ROUGHNESS = 0.9;
}
