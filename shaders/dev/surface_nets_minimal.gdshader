shader_type spatial;

// Default material for VoxelMesherSurfaceNets (returned by get_default_lod_material() when no
// material_override is set on the terrain node). Its only job is the transition-strip discard
// trick: transition geometry is combined into the main mesh (see surface_nets.cpp's
// `transition_tag` / voxel_mesher_surface_nets.cpp), and this shader collapses it to a degenerate
// triangle when the corresponding side isn't actually a coarser neighbor at render time.
uniform float u_roughness = 0.9;
uniform vec3 u_albedo = vec3(0.6, 0.6, 0.6);

// From Voxel Tools API
uniform int u_transition_mask;

void vertex() {
	int tag = floatBitsToInt(CUSTOM0.a);
	// tag == 0: regular mesh vertex, always shown.
	// tag != 0: belongs to the transition strip for one side (1 << Cube::Side); only shown while
	// that side is flagged active in u_transition_mask.
	float cull = float(tag == 0 || (tag & u_transition_mask) != 0);
	VERTEX *= cull;
}

void fragment() {
	ALBEDO = u_albedo;
	ROUGHNESS = u_roughness;
}
