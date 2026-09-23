extends SceneTree

# GPU-driven vs traditional on the real planet setup: EdenPlanetGeneratorV4 (40 km), Transvoxel MIXEL4 + surface
# data, block size 32, 12 LODs, planet_v4.gdshader for the traditional path (same config as
# demo_eden/_ocean_editor_probe.tscn, far field off).
#   $GODOT --path . --script planet_check.gd --resolution 1280x720 -- <surface|orbit|vista> [out_dir]
# vista: temperate land, procedural sky, sun, fog and filmic tonemap, closest to an in-game shot.
# Checks silhouettes and holes like gpu_check.gd, then benchmarks both modes.

const RADIUS := 40000.0
# Twin of the GPU-driven shader (planet_v4.gdshader + the same look changes), so both paths shade identically
const PLANET_SHADER := "planet_vivid.gdshader"
const MAX_SETTLE_FRAMES := 20000
const MASK_W := 320
const MASK_H := 180
const TRADITIONAL := 0
const GPU := 1

var _out_dir := "."
var _world: Node3D
var _cam: Camera3D
var _background: Image
var _failures := 0
var _viewer: Node


func _init() -> void:
	_run.call_deferred()


func _check(cond: bool, what: String) -> void:
	print(("  ok    " if cond else "  FAIL  ") + what)
	if not cond:
		_failures += 1


func _make_material() -> ShaderMaterial:
	var sh := Shader.new()
	sh.code = FileAccess.get_file_as_string(ProjectSettings.globalize_path("res://").path_join(PLANET_SHADER))
	var sm := ShaderMaterial.new()
	sm.shader = sh
	sm.set_shader_parameter("u_planet_radius", RADIUS)
	sm.set_shader_parameter("vibrance", 1.38)
	sm.set_shader_parameter("slope_rock_start", 0.6)
	sm.set_shader_parameter("slope_rock_end", 0.38)
	sm.set_shader_parameter("snow_temperature", 0.22)
	sm.set_shader_parameter("snow_blend", 0.06)
	sm.set_shader_parameter("gully_darkening", 0.605)
	sm.set_shader_parameter("ridge_highlight", 0.534)
	sm.set_shader_parameter("dryness_tint", 0.281)
	return sm


func _make_generator() -> Object:
	var g: Object = ClassDB.instantiate("EdenPlanetGeneratorV4")
	g.planet_radius = RADIUS
	g.bake_ocean_water = false
	g.terrain_feature_scale = 10642.0
	g.terrain_aesthetic_bias = 0.84
	g.continent_scale = 80420.0
	g.island_bias = 0.0
	return g


# Lush temperate land above sea level, so the vista shows climate tinting rather than polar snow or seabed
func _find_vista_direction() -> Vector3:
	var g := _make_generator()
	var best := Vector3.RIGHT
	var best_score := -INF
	var n := 4000
	for i in n:
		# Fibonacci sphere
		var y := 1.0 - 2.0 * (i + 0.5) / n
		var r := sqrt(1.0 - y * y)
		var a := i * 2.399963
		var d := Vector3(cos(a) * r, y, sin(a) * r)
		var s: Dictionary = g.sample_surface(d)
		if s.height < 150.0 or s.height > 1800.0:
			continue
		var score: float = s.moisture - abs(s.temperature - 0.6) * 2.0 + s.height / 3000.0
		if score > best_score:
			best_score = score
			best = d
	var s2: Dictionary = g.sample_surface(best)
	print("  vista site ", best, " ", s2)
	_vista_height = s2.height

	# Look toward the most land (no seabed across the frame): score headings by terrain height 4-16 km out
	var east := best.cross(Vector3.UP).normalized() if abs(best.y) < 0.95 else best.cross(Vector3.RIGHT).normalized()
	var north := east.cross(best).normalized()
	var best_heading_score := -INF
	for k in 24:
		var ang := TAU * k / 24.0
		var heading := (north * cos(ang) + east * sin(ang)).normalized()
		var score := 0.0
		for dist in [4000.0, 8000.0, 12000.0, 16000.0]:
			var p: Vector3 = (best * RADIUS + heading * dist).normalized()
			var h: float = g.sample_surface(p).height
			score += h if h > 0.0 else h * 3.0
		if score > best_heading_score:
			best_heading_score = score
			_vista_heading = heading
	return best


var _vista_height := 0.0
var _vista_heading := Vector3.ZERO


var _mesh_optimization := true
var _edge_clamp_margin := 0.5
var _lod_distance := 128.0


func _make_terrain() -> Node:
	var g := _make_generator()

	var m: Object = ClassDB.instantiate("VoxelMesherTransvoxel")
	m.texturing_mode = 1 # MIXEL4
	m.surface_data_enabled = true
	m.mesh_optimization_enabled = _mesh_optimization
	m.edge_clamp_margin = _edge_clamp_margin

	var t: Node = ClassDB.instantiate("VoxelLodTerrain")
	# Surface data (erosion/ridge/moisture/temperature) only reaches CUSTOM2 with DATA6 at 32 bits
	var fmt: Object = ClassDB.instantiate("VoxelFormat")
	fmt.set_channel_depth(VoxelBuffer.CHANNEL_DATA6, VoxelBuffer.DEPTH_32_BIT)
	t.format = fmt
	t.generator = g
	t.mesher = m
	t.mesh_block_size = 32
	t.view_distance = _view_distance
	t.lod_count = _lod_count
	if _far_enabled:
		# Far field takes over past the near terrain's view distance (see far/ and far_check.gd)
		t.far_enabled = true
		t.far_planet_radius = RADIUS
		t.far_first_lod = 6
		t.far_lod_count = 6
		t.far_ring_radius = 4
		t.far_vertical_min = -3000.0
		t.far_vertical_max = 4000.0
		t.far_near_clip_radius = _far_near_clip
		t.far_antialias = false
		t.far_cache_enabled = false
		t.far_max_uploads_per_frame = 64
	t.lod_distance = _lod_distance
	t.secondary_lod_distance = 256.0
	t.generate_collisions = false
	t.material = _make_material()
	return t


func _make_scene(view: String) -> void:
	_world = Node3D.new()
	root.add_child(_world)

	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-50.0, 30.0, 0.0)
	_world.add_child(sun)

	var we := WorldEnvironment.new()
	var e := Environment.new()
	e.background_mode = Environment.BG_COLOR
	e.background_color = Color(0.2, 0.3, 0.6)
	e.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	e.ambient_light_color = Color(0.5, 0.5, 0.5)
	we.environment = e
	_world.add_child(we)

	_cam = Camera3D.new()
	_cam.near = 1.0
	_cam.far = 300000.0
	_cam.fov = 70.0
	_world.add_child(_cam)
	if view == "vista":
		# A game-like shot: sky ambient, a sun 35 degrees above the local horizon, light haze, filmic tonemap.
		# Both render paths read these same Environment/light settings.
		var d := _find_vista_direction()
		var north := _vista_heading
		var east := north.cross(d).normalized()
		# Sun from the side and slightly ahead: long shading across the relief
		var sun_dir := (d * sin(deg_to_rad(32.0)) + (east * 0.85 + north * 0.5).normalized() * cos(deg_to_rad(32.0)))
		sun.basis = Basis.looking_at(-sun_dir.normalized(), d)
		sun.light_energy = 1.3
		sun.light_color = Color(1.0, 0.96, 0.88)
		e.background_mode = Environment.BG_SKY
		# The sky's up is world +Y, the planet's up here is `d`
		e.sky_rotation = Basis(Quaternion(Vector3.UP, d)).get_euler()
		e.sky = Sky.new()
		var psky := ProceduralSkyMaterial.new()
		# On a small planet the real horizon dips below local horizontal: show haze there, not ground color
		psky.ground_horizon_color = psky.sky_horizon_color
		psky.ground_bottom_color = psky.sky_horizon_color
		e.sky.sky_material = psky
		e.ambient_light_source = Environment.AMBIENT_SOURCE_SKY
		e.ambient_light_energy = 1.0
		e.tonemap_mode = Environment.TONE_MAPPER_ACES
		e.tonemap_exposure = 1.1
		e.fog_enabled = true
		e.fog_light_color = Color(0.62, 0.72, 0.86)
		e.fog_density = 0.000012
		e.fog_sun_scatter = 0.1
		e.fog_sky_affect = 0.0
		var ground := d * (RADIUS + _vista_height)
		_cam.position = ground + d * 900.0 - north * 2500.0
		_cam.look_at(ground + north * 7000.0 - d * 300.0, d)
	elif view == "orbit":
		_cam.position = Vector3(0.0, 0.0, RADIUS * 2.5)
		_cam.look_at(Vector3.ZERO, Vector3.UP)
	else:
		# 4 km above the reference radius, looking at the horizon (~18 km away)
		_cam.position = Vector3(0.0, RADIUS + 4000.0, 0.0)
		_cam.rotation_degrees = Vector3(-15.0, 0.0, 0.0)
	_cam.current = true

	# A real viewer node, positioned like the camera. The far field streams around this, not the camera.
	_viewer = ClassDB.instantiate("VoxelViewer")
	_viewer.view_distance = 80000
	_viewer.requires_visuals = true
	_viewer.requires_collisions = false
	_world.add_child(_viewer)
	_viewer.global_position = _cam.global_position


func _settle(t: Node) -> int:
	var engine: Object = Engine.get_singleton("VoxelEngine")
	var quiet := 0
	var frames := 0
	var t0 := Time.get_ticks_msec()
	while frames < MAX_SETTLE_FRAMES:
		await process_frame
		frames += 1
		var tasks: Dictionary = engine.get_stats().tasks
		var busy: int = tasks.streaming + tasks.generation + tasks.meshing + tasks.main_thread
		var g: Dictionary = t.get_gpu_driven_statistics()
		if not g.is_empty():
			busy += int(g.pending_ops)
		quiet = quiet + 1 if busy == 0 else 0
		if quiet >= 60 and frames > 10:
			break
	print("    settled in ", frames, " frames, ", (Time.get_ticks_msec() - t0) / 1000.0, " s, mesh blocks ",
			t.debug_get_mesh_block_count())
	for i in 3:
		await process_frame
	return frames


func _capture(name: String) -> Image:
	await RenderingServer.frame_post_draw
	var img: Image = root.get_texture().get_image()
	img.save_png(_out_dir.path_join(name + ".png"))
	return img


func _mask(img: Image) -> PackedByteArray:
	var a := img.duplicate() as Image
	a.resize(MASK_W, MASK_H, Image.INTERPOLATE_NEAREST)
	var b := _background.duplicate() as Image
	b.resize(MASK_W, MASK_H, Image.INTERPOLATE_NEAREST)
	var m := PackedByteArray()
	m.resize(MASK_W * MASK_H)
	for y in MASK_H:
		for x in MASK_W:
			var ca := a.get_pixel(x, y)
			var cb := b.get_pixel(x, y)
			m[y * MASK_W + x] = 1 if abs(ca.r - cb.r) + abs(ca.g - cb.g) + abs(ca.b - cb.b) > 0.03 else 0
	return m


func _iou(a: PackedByteArray, b: PackedByteArray) -> float:
	var inter := 0
	var uni := 0
	for i in a.size():
		inter += a[i] & b[i]
		uni += a[i] | b[i]
	return 1.0 if uni == 0 else float(inter) / uni


func _coverage(m: PackedByteArray) -> float:
	var c := 0
	for v in m:
		c += v
	return float(c) / m.size()


# Isolated background pixels surrounded by terrain (cracks), full resolution. Compared between paths rather than
# required to be zero: the meshes themselves can have hairline cracks, which then show in both.
func _pinholes(img: Image) -> int:
	var a := img.duplicate() as Image
	var bg := _background.duplicate() as Image
	a.convert(Image.FORMAT_RGB8)
	bg.convert(Image.FORMAT_RGB8)
	var w := a.get_width()
	var h := a.get_height()
	var da := a.get_data()
	var dg := bg.get_data()
	# 0 = background, 1 = ambiguous (fogged terrain close to the sky color), 2 = clearly terrain
	var cls := PackedByteArray()
	cls.resize(w * h)
	for p in w * h:
		var i := p * 3
		var diff: int = abs(da[i] - dg[i]) + abs(da[i + 1] - dg[i + 1]) + abs(da[i + 2] - dg[i + 2])
		cls[p] = 0 if diff < 8 else (1 if diff < 24 else 2)
	var count := 0
	for y in range(1, h - 1):
		for x in range(1, w - 1):
			var p := y * w + x
			if cls[p] == 0 and cls[p - 1] == 2 and cls[p + 1] == 2 and cls[p - w] == 2 and cls[p + w] == 2:
				count += 1
	return count


func _measure(frames: int) -> Dictionary:
	var vp := root.get_viewport_rid()
	RenderingServer.viewport_set_measure_render_time(vp, true)
	var cpu := 0.0
	var gpu := 0.0
	var draws := 0.0
	var prims := 0.0
	var base_rot := _cam.rotation_degrees
	var t0 := Time.get_ticks_usec()
	for i in frames:
		# Small yaw sweep: keeps culling busy without moving the viewer (no streaming)
		_cam.rotation_degrees = base_rot + Vector3(0.0, 20.0 * sin(TAU * i / frames), 0.0)
		await process_frame
		cpu += RenderingServer.viewport_get_measured_render_time_cpu(vp)
		gpu += RenderingServer.viewport_get_measured_render_time_gpu(vp)
		draws += Performance.get_monitor(Performance.RENDER_TOTAL_DRAW_CALLS_IN_FRAME)
		prims += Performance.get_monitor(Performance.RENDER_TOTAL_PRIMITIVES_IN_FRAME)
	_cam.rotation_degrees = base_rot
	return {
		"frame_ms": snappedf((Time.get_ticks_usec() - t0) / 1000.0 / frames, 0.001),
		"render_cpu_ms": snappedf(cpu / frames, 0.001),
		"render_gpu_ms": snappedf(gpu / frames, 0.001),
		"scene_draw_calls": int(draws / frames),
		"scene_primitives": int(prims / frames),
	}


var _far_enabled := false
var _far_near_clip := 0.0
var _view_distance := 80000
var _lod_count := 12


func _make_far_material(t: Node) -> ShaderMaterial:
	var sh := Shader.new()
	sh.code = FileAccess.get_file_as_string(
			ProjectSettings.globalize_path("res://").path_join("../../../far/shaders/far_terrain.gdshader"))
	var sm := ShaderMaterial.new()
	sm.shader = sh
	var clip: float = t.get_far_statistics()["near_clip_radius"]
	sm.set_shader_parameter("planet_radius", RADIUS)
	sm.set_shader_parameter("near_fade_start", clip * 0.85)
	sm.set_shader_parameter("near_fade_end", clip * 1.15)
	sm.set_shader_parameter("fog_color", Color(0.62, 0.70, 0.80))
	sm.set_shader_parameter("fog_start", 6000.0)
	sm.set_shader_parameter("fog_end", 90000.0)
	return sm


# Why does the far field stream nothing here when far_check's project works? Vary one thing at a time.
func _stage_far_diag() -> void:
	var variants := [
		{ "name": "block 16 (default)", "mesher": true, "format": true, "material": true, "block": 16 },
		{ "name": "block 32 (scene)", "mesher": true, "format": true, "material": true, "block": 32 },
		{ "name": "block 32, lod_count 7, view 4096", "mesher": true, "format": true, "material": true,
				"block": 32, "lods": 7, "view": 4096 },
		{ "name": "block 32, lod_count 12, view 80000", "mesher": true, "format": true, "material": true,
				"block": 32, "lods": 12, "view": 80000 },
		{ "name": "block 32 + lod_distance 128", "mesher": true, "format": true, "material": true,
				"block": 32, "lod_distance": 128.0 },
		{ "name": "block 32 + secondary_lod_distance 256", "mesher": true, "format": true, "material": true,
				"block": 32, "secondary": 256.0 },
	]
	# Same construction the benchmark stage uses, to compare against the variants below
	_far_enabled = true
	_far_near_clip = -1.0
	_view_distance = 1024
	_lod_count = 4
	var tt := _make_terrain()
	_world.add_child(tt)
	for i in 900:
		await process_frame
	var ss: Dictionary = tt.get_far_statistics()
	print("  via _make_terrain -> resident %d, rendered %d, built %d, inflight %d" % [
			ss.resident_sectors, ss.rendered_sectors, ss.sectors_built, ss.tasks_in_flight])
	tt.queue_free()
	for i in 10:
		await process_frame

	for v in variants:
		var g := _make_generator()
		var t: Node = ClassDB.instantiate("VoxelLodTerrain")
		if v.format:
			var fmt: Object = ClassDB.instantiate("VoxelFormat")
			fmt.set_channel_depth(VoxelBuffer.CHANNEL_DATA6, VoxelBuffer.DEPTH_32_BIT)
			t.format = fmt
		t.generator = g
		if v.mesher:
			var m: Object = ClassDB.instantiate("VoxelMesherTransvoxel")
			m.texturing_mode = 1
			m.surface_data_enabled = true
			t.mesher = m
		if v.material:
			t.material = _make_material()
		t.mesh_block_size = v.get("block", 16)
		t.view_distance = v.get("view", 1024)
		t.lod_count = v.get("lods", 4)
		if v.has("lod_distance"):
			t.lod_distance = v.lod_distance
		if v.has("secondary"):
			t.secondary_lod_distance = v.secondary
		t.generate_collisions = false
		t.far_enabled = true
		t.far_planet_radius = RADIUS
		t.far_first_lod = 6
		t.far_lod_count = 6
		t.far_ring_radius = 4
		t.far_vertical_min = -3000.0
		t.far_vertical_max = 4000.0
		t.far_near_clip_radius = -1.0
		t.far_antialias = false
		t.far_cache_enabled = false
		t.far_max_uploads_per_frame = 64
		_world.add_child(t)
		for i in 900:
			await process_frame
		var s: Dictionary = t.get_far_statistics()
		print("  %s -> resident %d, rendered %d, built %d, inflight %d, verts %d" % [v.name,
				s.resident_sectors, s.rendered_sectors, s.sectors_built, s.tasks_in_flight, s.total_vertices])
		t.queue_free()
		for i in 10:
			await process_frame


# Traditional vs GPU-driven, with and without the far field, on the same view
func _stage_compare_all() -> void:
	var rows := []
	var configs := [
		{ "name": "near only, traditional", "mode": TRADITIONAL, "far": false, "view": 80000, "lods": 12 },
		{ "name": "near only, gpu-driven", "mode": GPU, "far": false, "view": 80000, "lods": 12 },
		{ "name": "near + far, traditional", "mode": TRADITIONAL, "far": true, "view": 4096, "lods": 7 },
		{ "name": "near + far, gpu-driven", "mode": GPU, "far": true, "view": 4096, "lods": 7 },
	]
	for cfg in configs:
		_far_enabled = cfg.far
		_view_distance = cfg.view
		_lod_count = cfg.lods
		_far_near_clip = cfg.get("clip", 0.0)
		var t := _make_terrain()
		t.render_mode = cfg.mode
		_world.add_child(t)
		if cfg.far:
			t.far_material = _make_far_material(t)
		var frames := await _settle(t)
		if cfg.far:
			print("  far stats: ", t.get_far_statistics())
		await _measure(90)
		var m := await _measure(600)
		var g: Dictionary = t.get_gpu_driven_statistics()
		if cfg.mode == GPU:
			# Without this a broken renderer just reports a very fast empty frame
			_check(not g.is_empty() and g.get("initialized", false) and not g.get("failed", true),
					"%s: gpu renderer running" % cfg.name)
			_check(int(g.get("visible_chunks", 0)) > 0, "%s: chunks drawn (%d)" % [cfg.name, int(g.get("visible_chunks", 0))])
		var far_stats: Dictionary = t.get_far_statistics() if cfg.far else {}
		if cfg.far:
			_check(int(far_stats.get("rendered_sectors", 0)) > 0,
					"%s: far sectors built (%d)" % [cfg.name, int(far_stats.get("rendered_sectors", 0))])
		var row := {
			"name": cfg.name,
			"frame_ms": m.frame_ms,
			"render_cpu_ms": m.render_cpu_ms,
			"render_gpu_ms": m.render_gpu_ms,
			"scene_draw_calls": m.scene_draw_calls,
			"scene_primitives": m.scene_primitives,
			"mesh_blocks": t.debug_get_mesh_block_count(),
			"settle_frames": frames,
			"gpu_vram_mb": snappedf(float(g.get("vram_bytes", 0)) / 1048576.0, 0.1),
			"gpu_visible_chunks": int(g.get("visible_chunks", 0)),
			"far_sectors": int(far_stats.get("rendered_sectors", 0)),
			"far_vertices": int(far_stats.get("total_vertices", 0)),
		}
		rows.append(row)
		print("  ", row)
		await _capture("compare_" + cfg.name.replace(" ", "_").replace(",", ""))
		t.queue_free()
		for i in 8:
			await process_frame
	print("RESULTS=", rows)


# Hunts holes in the mesh itself. Magenta background, no fog and no sky, so any gap is unmistakable, sampled over
# several headings. Reports both render paths for each mesher configuration.
func _stage_cracks() -> void:
	var configs := [
		{ "name": "clamp 0.02, lod_distance 128 (scene)", "opt": true, "clamp": 0.02, "lod_distance": 128.0 },
	]
	var d := _find_vista_direction()
	var north := _vista_heading
	var ground := d * (RADIUS + _vista_height)

	var we: WorldEnvironment = _world.get_child(1)
	we.environment.background_mode = Environment.BG_COLOR
	we.environment.background_color = Color(1, 0, 1)
	we.environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	we.environment.ambient_light_color = Color(0.6, 0.6, 0.6)
	we.environment.fog_enabled = false
	we.environment.tonemap_mode = Environment.TONE_MAPPER_LINEAR
	_cam.position = ground + d * 900.0 - north * 2500.0
	_cam.look_at(ground + north * 7000.0 - d * 300.0, d)
	for i in 5:
		await process_frame
	_background = await _capture("cracks_background")

	for cfg in configs:
		_mesh_optimization = cfg.opt
		_edge_clamp_margin = cfg.clamp
		if cfg.has("lod_distance"):
			_lod_distance = cfg.lod_distance
		var t := _make_terrain()
		_world.add_child(t)
		await _settle(t)
		var counts := { "traditional": 0, "gpu": 0 }
		for mode in [TRADITIONAL, GPU]:
			t.render_mode = mode
			await _settle(t)
			var total := 0
			var base := _cam.rotation_degrees
			for k in 4:
				_cam.rotation_degrees = base + Vector3(0.0, k * 12.0 - 18.0, 0.0)
				for i in 3:
					await process_frame
				var img := await _capture("cracks_%s_%d" % ["gpu" if mode == GPU else "trad", k])
				total += _pinholes(img)
			_cam.rotation_degrees = base
			counts["gpu" if mode == GPU else "traditional"] = total
		print("  %s: traditional %d, gpu %d (holes over 4 views)" % [cfg.name, counts.traditional, counts.gpu])
		t.queue_free()
		for i in 5:
			await process_frame


# Frustum culling: how many chunks survive as the camera turns away, and what it costs
func _stage_cull(t: Node) -> void:
	var base := _cam.rotation_degrees
	var results := {}
	for entry in [["ahead", 0.0], ["turned 90", 90.0], ["turned away", 180.0], ["pitched up", -1.0]]:
		var label: String = entry[0]
		if label == "pitched up":
			_cam.rotation_degrees = base + Vector3(80.0, 0.0, 0.0)
		else:
			_cam.rotation_degrees = base + Vector3(0.0, float(entry[1]), 0.0)
		for i in 5:
			await process_frame
		# get_gpu_driven_statistics asks the render thread, the answer lands a frame later
		t.get_gpu_driven_statistics()
		await process_frame
		await process_frame
		var g: Dictionary = t.get_gpu_driven_statistics()
		var m := await _measure(120)
		results[label] = { "visible": int(g.visible_chunks), "gpu_ms": m.render_gpu_ms }
	_cam.rotation_degrees = base
	print("  culling: ", results)
	var total: int = int(t.get_gpu_driven_statistics().chunks)
	_check(results["ahead"].visible > 0 and results["ahead"].visible < total,
			"some chunks culled while looking at the terrain (%d of %d visible)" % [results["ahead"].visible, total])
	_check(results["turned away"].visible < results["ahead"].visible / 4,
			"turning away culls nearly everything (%d visible)" % results["turned away"].visible)
	_check(results["turned away"].gpu_ms < results["ahead"].gpu_ms,
			"culling saves GPU time (%.2f ms vs %.2f ms)" % [results["turned away"].gpu_ms, results["ahead"].gpu_ms])


func _run() -> void:
	var args: PackedStringArray = OS.get_cmdline_user_args()
	var view: String = args[0] if args.size() > 0 else "surface"
	if args.size() > 1:
		_out_dir = args[1]
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	Engine.max_fps = 0
	_make_scene("vista" if view == "cracks" else view)
	print("planet view ", view)
	if view == "cracks":
		await _stage_cracks()
		print("FAILURES=", _failures)
		quit(1 if _failures > 0 else 0)
		return
	if view == "fardiag":
		var dd := _find_vista_direction()
		_viewer.global_position = dd * (RADIUS + _vista_height + 900.0)
		_cam.position = _viewer.global_position
		await _stage_far_diag()
		quit(0)
		return
	if view == "compare":
		DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
		Engine.max_fps = 0
		# Same viewpoint as the vista shots: standing on land, looking across it
		var d := _find_vista_direction()
		var ground := d * (RADIUS + _vista_height)
		_cam.position = ground + d * 900.0 - _vista_heading * 2500.0
		_cam.look_at(ground + _vista_heading * 7000.0 - d * 300.0, d)
		_viewer.global_position = _cam.global_position
		await _stage_compare_all()
		print("FAILURES=", _failures)
		quit(1 if _failures > 0 else 0)
		return

	for i in 5:
		await process_frame
	_background = await _capture("planet_%s_background" % view)

	var t := _make_terrain()
	_world.add_child(t)
	print("  traditional")
	await _settle(t)
	var trad_img := await _capture("planet_%s_traditional" % view)
	var trad_mask := _mask(trad_img)
	_check(_coverage(trad_mask) > 0.1, "planet visible (coverage %.3f)" % _coverage(trad_mask))
	await _measure(60)
	var trad_bench := await _measure(600)

	t.render_mode = 1
	print("  gpu_driven")
	await _settle(t)
	var gpu_img := await _capture("planet_%s_gpu" % view)
	var g: Dictionary = t.get_gpu_driven_statistics()
	print("  gpu stats: ", g)
	_check(g.get("initialized", false) and not g.get("failed", true), "renderer initialized")
	var blocks: int = t.debug_get_mesh_block_count()
	_check(int(g.get("chunks", 0)) > 0 and int(g.get("chunks", 0)) <= blocks,
			"chunks uploaded, none leaked (%d chunks, %d blocks)" % [int(g.get("chunks", 0)), blocks])
	var iou := _iou(trad_mask, _mask(gpu_img))
	_check(iou >= 0.97, "gpu silhouette matches traditional (IoU %.4f)" % iou)
	var gpu_holes := _pinholes(gpu_img)
	var trad_holes := _pinholes(trad_img)
	_check(gpu_holes <= trad_holes, "no more cracks than traditional (gpu %d, traditional %d)" % [gpu_holes, trad_holes])
	await _measure(60)
	var gpu_bench := await _measure(600)
	await _stage_cull(t)

	print("  bench traditional: ", trad_bench)
	print("  bench gpu_driven:  ", gpu_bench)
	print("FAILURES=", _failures)
	quit(1 if _failures > 0 else 0)
