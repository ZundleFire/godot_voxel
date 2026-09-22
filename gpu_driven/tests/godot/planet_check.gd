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

var _out_dir := "."
var _world: Node3D
var _cam: Camera3D
var _background: Image
var _failures := 0


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


func _make_terrain() -> Node:
	var g := _make_generator()

	var m: Object = ClassDB.instantiate("VoxelMesherTransvoxel")
	m.texturing_mode = 1 # MIXEL4
	m.surface_data_enabled = true
	m.mesh_optimization_enabled = true
	m.edge_clamp_margin = 0.5

	var t: Node = ClassDB.instantiate("VoxelLodTerrain")
	# Surface data (erosion/ridge/moisture/temperature) only reaches CUSTOM2 with DATA6 at 32 bits
	var fmt: Object = ClassDB.instantiate("VoxelFormat")
	fmt.set_channel_depth(VoxelBuffer.CHANNEL_DATA6, VoxelBuffer.DEPTH_32_BIT)
	t.format = fmt
	t.generator = g
	t.mesher = m
	t.mesh_block_size = 32
	t.view_distance = 80000
	t.lod_count = 12
	t.lod_distance = 128.0
	t.secondary_lod_distance = 256.0
	t.generate_collisions = false
	t.far_enabled = false
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

	var viewer: Node = ClassDB.instantiate("VoxelViewer")
	viewer.view_distance = 80000
	viewer.requires_collisions = false
	_cam.add_child(viewer)


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


func _run() -> void:
	var args: PackedStringArray = OS.get_cmdline_user_args()
	var view: String = args[0] if args.size() > 0 else "surface"
	if args.size() > 1:
		_out_dir = args[1]
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	Engine.max_fps = 0
	_make_scene(view)
	print("planet view ", view)

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

	print("  bench traditional: ", trad_bench)
	print("  bench gpu_driven:  ", gpu_bench)
	print("FAILURES=", _failures)
	quit(1 if _failures > 0 else 0)
