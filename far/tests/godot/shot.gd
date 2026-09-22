extends SceneTree

# HANDOFF.md Step 2.5 — put the shader through a real compiler and look at a frame.
#   --script shot.gd -- <out.png> [shader|standard] [cam_y] [pitch_deg]

const MAX_FRAMES := 3000


func _init() -> void:
	_run.call_deferred()


func _make_terrain() -> Node:
	var far: Node = ClassDB.instantiate("VoxelLodTerrain")
	far.view_distance = 1024
	far.lod_count = 4
	far.far_enabled = true
	var g: Object = ClassDB.instantiate("VoxelGeneratorNoise2D")
	var n := FastNoiseLite.new()
	n.noise_type = FastNoiseLite.TYPE_SIMPLEX
	n.frequency = 0.0008
	n.fractal_octaves = 5
	g.noise = n
	g.channel = 1 # SDF
	g.height_start = -200.0
	g.height_range = 900.0

	far.generator = g
	far.far_first_lod = 4
	far.far_lod_count = 6
	far.far_ring_radius = 4
	far.far_near_clip_radius = -1.0  # off: this shot is the far field alone
	far.far_vertical_min = -256.0
	far.far_vertical_max = 1024.0
	far.far_cache_enabled = false
	far.far_max_uploads_per_frame = 64
	return far


func _run() -> void:
	var args: PackedStringArray = OS.get_cmdline_user_args()
	var out_path: String = args[0] if args.size() > 0 else "shot.png"
	var mat_kind: String = args[1] if args.size() > 1 else "shader"
	var cam_y: float = float(args[2]) if args.size() > 2 else 1400.0
	var pitch: float = float(args[3]) if args.size() > 3 else -12.0

	var world := Node3D.new()
	root.add_child(world)

	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-35.0, 40.0, 0.0)
	sun.light_energy = 1.2
	world.add_child(sun)

	var env := WorldEnvironment.new()
	var e := Environment.new()
	e.background_mode = Environment.BG_SKY
	e.sky = Sky.new()
	e.sky.sky_material = ProceduralSkyMaterial.new()
	e.ambient_light_source = Environment.AMBIENT_SOURCE_SKY
	e.ambient_light_energy = 0.6
	env.environment = e
	world.add_child(env)

	var cam := Camera3D.new()
	cam.near = 5.0
	cam.far = 300000.0
	cam.fov = 70.0
	cam.position = Vector3(0.0, cam_y, 0.0)
	cam.rotation_degrees = Vector3(pitch, 0.0, 0.0)
	world.add_child(cam)
	cam.current = true

	var far := _make_terrain()
	if mat_kind == "shader" or mat_kind == "fog":
		var sh: Shader = load("res://far_terrain.gdshader")
		if sh == null:
			print("SHADER_LOAD=FAILED")
			quit(1)
			return
		print("SHADER_LOAD=OK  code_len=", sh.code.length())
		var sm := ShaderMaterial.new()
		sm.shader = sh
		# Bracket the near fade around the clip radius, per HANDOFF step 5.
		sm.set_shader_parameter("near_fade_start", 0.0)
		sm.set_shader_parameter("near_fade_end", 0.0)
		sm.set_shader_parameter("fog_start", 20000.0)
		sm.set_shader_parameter("fog_end", 160000.0)
		if mat_kind == "fog":
			# Distance fog plus a height layer: haze pools below the peaks and the
			# ranges stack in progressively paler bands (HANDOFF 4.1).
			sm.set_shader_parameter("fog_color", Color(0.62, 0.70, 0.80))
			sm.set_shader_parameter("height_fog_color", Color(0.72, 0.78, 0.86))
			sm.set_shader_parameter("fog_start", 6000.0)
			sm.set_shader_parameter("fog_end", 90000.0)
			sm.set_shader_parameter("fog_density", 1.0)
			sm.set_shader_parameter("height_fog_enabled", true)
			sm.set_shader_parameter("height_fog_base", 460.0)
			sm.set_shader_parameter("height_fog_length", 420.0)
			sm.set_shader_parameter("height_fog_density", 2.0)
			sm.set_shader_parameter("height_fog_applies_down", true)
			sm.set_shader_parameter("fog_mixing_mode", 0)
		far.far_material = sm
	else:
		var sm := StandardMaterial3D.new()
		sm.vertex_color_use_as_albedo = false
		sm.albedo_color = Color(0.45, 0.5, 0.4)
		far.far_material = sm
	world.add_child(far)

	var frames := 0
	while frames < MAX_FRAMES:
		await process_frame
		frames += 1
		var s: Dictionary = far.get_far_statistics()
		if frames > 4 and s.tasks_in_flight == 0 and s.pending_uploads == 0 \
				and s.rendered_sectors == s.resident_sectors and s.resident_sectors > 0:
			break
	for i in 10:
		await process_frame

	var s2: Dictionary = far.get_far_statistics()
	print("settled_frames=", frames, " sectors=", s2.rendered_sectors,
			" verts=", s2.total_vertices, " aabb=", s2.world_aabb)

	await RenderingServer.frame_post_draw
	var img: Image = root.get_texture().get_image()
	var err := img.save_png(out_path)
	print("SAVED=", out_path, " err=", err, " size=", img.get_width(), "x", img.get_height())
	quit(0)
