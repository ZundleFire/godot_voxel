// Generated file

// clang-format off
const char *g_surface_nets_minimal_shader =
"shader_type spatial;\n"
"\n"
"// Default material for VoxelMesherSurfaceNets (returned by get_default_lod_material() when no\n"
"// material_override is set on the terrain node). Its only job is the transition-strip discard\n"
"// trick: transition geometry is combined into the main mesh (see surface_nets.cpp's\n"
"// `transition_tag` / voxel_mesher_surface_nets.cpp), and this shader collapses it to a degenerate\n"
"// triangle when the corresponding side isn't actually a coarser neighbor at render time.\n"
"uniform float u_roughness = 0.9;\n"
"uniform vec3 u_albedo = vec3(0.6, 0.6, 0.6);\n"
"\n"
"// From Voxel Tools API\n"
"uniform int u_transition_mask;\n"
"\n"
"void vertex() {\n"
"	int tag = floatBitsToInt(CUSTOM0.a);\n"
"	// tag == 0: regular mesh vertex, always shown.\n"
"	// tag != 0: belongs to the transition strip for one side (1 << Cube::Side); only shown while\n"
"	// that side is flagged active in u_transition_mask.\n"
"	float cull = float(tag == 0 || (tag & u_transition_mask) != 0);\n"
"	VERTEX *= cull;\n"
"}\n"
"\n"
"void fragment() {\n"
"	ALBEDO = u_albedo;\n"
"	ROUGHNESS = u_roughness;\n"
"}\n";
// clang-format on
