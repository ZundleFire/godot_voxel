extends SceneTree

# Engine-level checks for the far field, now that it is a streaming system
# inside VoxelLodTerrain rather than its own node.
#
#   --script far_check.gd -- <stage> [clear]
#
# Stages 1-6 mirror HANDOFF.md step 2. "move" walks the viewer. "handover"
# adds a near mesher so the two terrains are drawn together, which needs a
# real window; every other stage runs headless, because without a mesher the
# near terrain builds no meshes and so never touches the dummy renderer.

const MAX_FRAMES := 1500
const PLANET_RADIUS := 40000.0
const V4_RADIUS := 40000.0


func _init() -> void:
	_run.call_deferred()


func _make_generator(kind: String) -> Object:
	if kind == "flat":
		var g: Object = ClassDB.instantiate("VoxelGeneratorFlat")
		g.height = 40.0
		g.channel = 1 # SDF
		return g
	var g2: Object = ClassDB.instantiate("VoxelGeneratorNoise2D")
	var n := FastNoiseLite.new()
	n.noise_type = FastNoiseLite.TYPE_SIMPLEX
	n.frequency = 0.0008
	n.fractal_octaves = 5
	g2.noise = n
	g2.channel = 1 # SDF
	g2.height_start = -200.0
	g2.height_range = 900.0
	return g2


# A planet as a graph, because VoxelGeneratorGraph is one of the two generators
# that implement generate_series -- which is what radial column sampling needs.
# See HANDOFF 9.3.
func _make_planet_generator() -> Object:
	var g: Object = ClassDB.instantiate("VoxelGeneratorGraph")
	var f: Object = g.get_main_function()

	var sphere: int = f.create_node(VoxelGraphFunction.NODE_SDF_SPHERE, Vector2(0, 0))
	f.set_node_default_input(sphere, 3, PLANET_RADIUS)

	var noise: int = f.create_node(VoxelGraphFunction.NODE_FAST_NOISE_3D, Vector2(0, 200))
	var n := ZN_FastNoiseLite.new()
	# Continent-scale features on a 40 km planet.
	n.noise_type = ZN_FastNoiseLite.TYPE_OPEN_SIMPLEX_2
	n.fractal_type = ZN_FastNoiseLite.FRACTAL_FBM
	n.period = 9000.0
	n.fractal_octaves = 5
	f.set_node_param(noise, 0, n)

	# The noise is roughly [-1, 1]; scale it into metres of relief, otherwise a
	# unit of displacement is invisible against a 40 km radius.
	var scale: int = f.create_node(VoxelGraphFunction.NODE_MULTIPLY, Vector2(200, 200))
	f.set_node_default_input(scale, 1, 1400.0)

	var add: int = f.create_node(VoxelGraphFunction.NODE_ADD, Vector2(400, 0))
	var out: int = f.create_node(VoxelGraphFunction.NODE_OUTPUT_SDF, Vector2(600, 0))

	f.add_connection(noise, 0, scale, 0)
	f.add_connection(sphere, 0, add, 0)
	f.add_connection(scale, 0, add, 1)
	f.add_connection(add, 0, out, 0)

	var err = g.compile()
	if err != null and not err.success:
		push_error("graph compile failed: " + str(err.message))
		return null
	return g


func _make_v4_generator() -> Object:
	var g: Object = ClassDB.instantiate("EdenPlanetGeneratorV4")
	g.planet_radius = V4_RADIUS
	g.seed = 12345
	g.bake_ocean_water = false
	return g


func _make_terrain(stage: String) -> Node:
	var t: Node = ClassDB.instantiate("VoxelLodTerrain")
	t.view_distance = 1024
	t.lod_count = 4
	t.far_enabled = true
	t.far_cache_enabled = false
	t.far_vertical_min = -256.0
	t.far_vertical_max = 1024.0
	t.far_max_uploads_per_frame = 64
	# 0 derives the clip from view_distance. Stages that are not about the clip
	# switch it off so their sector counts are the full ring.
	t.far_near_clip_radius = -1.0

	match stage:
		"1_one_sector":
			t.generator = _make_generator("flat")
			t.far_first_lod = 6
			t.far_lod_count = 1
			t.far_ring_radius = 1
		"2_full_ring":
			t.generator = _make_generator("noise")
			t.far_first_lod = 6
			t.far_lod_count = 1
			t.far_ring_radius = 4
		"3_multi_level", "6_cache", "move":
			t.generator = _make_generator("noise")
			t.far_first_lod = 4
			t.far_lod_count = 6
			t.far_ring_radius = 4
			if stage == "6_cache":
				t.far_lod_count = 4
				t.far_ring_radius = 3
				t.far_cache_enabled = true
				t.far_cache_directory = "user://far_cache_probe"
		"planet_v4", "planet_v4_shot":
			# The project's real planet generator, which is the case that
			# matters. It only became usable here once it implemented
			# generate_series -- before that every radial sample cost a
			# generate_block call. See HANDOFF 9.3.
			t.generator = _make_v4_generator()
			t.far_planet_radius = V4_RADIUS
			t.far_first_lod = 6
			t.far_lod_count = 6
			t.far_ring_radius = 4
			t.far_vertical_min = -3000.0
			t.far_vertical_max = 4000.0
			t.far_near_clip_radius = -1.0
			t.far_antialias = false
		"planet_aa":
			# Antialiasing samples the four children one level finer and
			# reduces. On a sphere a child is a cube-face sub-patch, so it goes
			# through the same geometry builder -- untested until now.
			t.generator = _make_planet_generator()
			t.far_planet_radius = PLANET_RADIUS
			t.far_first_lod = 7
			t.far_lod_count = 4
			t.far_ring_radius = 4
			t.far_vertical_min = -1500.0
			t.far_vertical_max = 2500.0
			t.far_near_clip_radius = -1.0
			t.far_antialias = true
		"cache_v4":
			# Materials have to survive the cache, or they are right on a cold
			# run and silently gone on every run after it.
			t.generator = _make_v4_generator()
			t.far_planet_radius = PLANET_RADIUS
			t.far_first_lod = 8
			t.far_lod_count = 2
			t.far_ring_radius = 3
			t.far_vertical_min = -3000.0
			t.far_vertical_max = 4000.0
			t.far_near_clip_radius = -1.0
			t.far_antialias = false
			t.far_cache_enabled = true
			t.far_cache_directory = "user://far_cache_v4"
		"flat_v4":
			# The flat path, over the top of V4's planet. Contrived on purpose:
			# it is the only way to exercise the *flat* branch of the MIXEL4
			# material decode, since the simple generators write no material
			# channel at all and V4 is the only one here that does.
			t.generator = _make_v4_generator()
			t.far_planet_radius = 0.0   # flat: columns run along Y
			t.far_first_lod = 6
			t.far_lod_count = 1
			t.far_ring_radius = 1
			t.far_vertical_min = 36000.0
			t.far_vertical_max = 43000.0
			t.far_near_clip_radius = -1.0
			t.far_antialias = false
		"planet_handover":
			# Near terrain and far field on the same planet. `far_first_lod` is
			# low enough that one far sector is comparable to the near view
			# distance -- otherwise a whole sector never fits inside the clip
			# radius and the clip never bites. That is the planetary version of
			# the `first_lod` tuning note in Step 3.
			t.generator = _make_v4_generator()
			t.mesher = ClassDB.instantiate("VoxelMesherTransvoxel")
			t.far_planet_radius = PLANET_RADIUS
			t.far_first_lod = 4
			t.far_lod_count = 7
			t.far_ring_radius = 4
			t.far_vertical_min = -1500.0
			t.far_vertical_max = 2500.0
			t.far_near_clip_radius = 0.0
			t.far_antialias = false
		"planet_move":
			t.generator = _make_planet_generator()
			t.far_planet_radius = PLANET_RADIUS
			t.far_first_lod = 7
			t.far_lod_count = 4
			t.far_ring_radius = 4
			t.far_vertical_min = -1500.0
			t.far_vertical_max = 2500.0
			# Derived from view_distance, so the clip is actually exercised.
			t.far_near_clip_radius = 0.0
			t.far_antialias = false
		"planet", "planet_shot":
			# A 40 km planet: sphere SDF with a bit of relief, matching the scale
			# of eden_planet_gen's defaults. VoxelGeneratorGraph is used because
			# it supports generate_series, which radial columns need.
			t.generator = _make_planet_generator()
			t.far_planet_radius = PLANET_RADIUS
			t.far_first_lod = 6
			t.far_lod_count = 6
			t.far_ring_radius = 4
			t.far_vertical_min = -1500.0
			t.far_vertical_max = 2500.0
			t.far_near_clip_radius = -1.0
			t.far_antialias = false
		"4_near_clip", "handover":
			t.generator = _make_generator("noise")
			t.far_first_lod = 4
			t.far_lod_count = 6
			t.far_ring_radius = 4
			# Derived from view_distance -- the whole point of integrating.
			t.far_near_clip_radius = 0.0
			if stage == "handover":
				t.mesher = ClassDB.instantiate("VoxelMesherTransvoxel")
		_:
			push_error("unknown stage " + stage)
			return null
	return t


func _settle(t: Node) -> Dictionary:
	var frames := 0
	while frames < MAX_FRAMES:
		await process_frame
		frames += 1
		var s: Dictionary = t.get_far_statistics()
		if frames > 8 and s.tasks_in_flight == 0 and s.pending_uploads == 0 \
				and s.resident_sectors > 0 and s.rendered_sectors == s.resident_sectors:
			break
	for i in 10:
		await process_frame
	var out: Dictionary = t.get_far_statistics()
	out["frames"] = frames
	return out


func _report(s: Dictionary) -> void:
	for k in ["frames", "resident_sectors", "rendered_sectors", "sectors_built", "empty_sectors",
			"clipped_sectors", "stale_results_discarded", "total_vertices", "total_triangles",
			"max_resident_bound", "near_clip_radius", "max_material", "tasks_in_flight", "pending_uploads", "world_aabb"]:
		print("  ", k, "=", s[k])
	if s.has("cache"):
		print("  cache=", s["cache"])


func _run() -> void:
	var args: PackedStringArray = OS.get_cmdline_user_args()
	var stage: String = args[0] if args.size() > 0 else "1_one_sector"
	var clear_cache := args.size() > 1 and args[1] == "clear"

	print("=== stage ", stage)
	var t: Node = _make_terrain(stage)
	if t == null:
		quit(1)
		return
	root.add_child(t)

	# A real VoxelViewer, not the terrain node's own position. Without one,
	# VoxelLodTerrain::get_local_viewer_pos() falls back to a position it
	# already stored in local space and transforms it into local space again
	# every frame -- so with a non-identity terrain transform the viewer drifts
	# without bound. See HANDOFF 3.8.
	var viewer: Node = ClassDB.instantiate("VoxelViewer")
	viewer.view_distance = 1024
	viewer.requires_visuals = true
	root.add_child(viewer)
	# On a planet the origin is the planet centre, so the flat default would put
	# the viewer thousands of kilometres underground.
	var viewer_pos := Vector3(0.0, 900.0, 0.0)
	if stage == "flat_v4":
		# Above the north pole, where a flat XZ grid crosses the planet surface.
		viewer_pos = Vector3(0.0, V4_RADIUS + 3000.0, 0.0)
	elif stage.begins_with("planet"):
		var dir := Vector3(0.0, 0.0, 1.0)
		var alt := 2400.0
		if t.generator.has_method("sample_surface"):
			# Stand 120 m above the actual ground. Altitude eats the clip
			# radius: from 2.4 km up, a near terrain that only reaches 1 km is
			# not drawing ground at all, so nothing can be clipped and the far
			# field correctly covers everything. That is the planetary form of
			# the `first_lod` tuning note in Step 3, and it is easy to mistake
			# for a broken clip.
			alt = float(t.generator.sample_surface(dir)["height"]) + 120.0
		viewer_pos = dir * (PLANET_RADIUS + alt)
	viewer.global_position = viewer_pos

	if clear_cache:
		t.far_clear_cache()

	var failures: Array[String] = []

	if stage == "planet_move":
		# Orbit the viewer around the planet. This is where face crossings
		# actually happen: the ring walks off one cube face and onto another,
		# and over a full orbit it visits all six plus the corners between them.
		for step in 12:
			var ang := float(step) * 0.52
			var d := Vector3(cos(ang), sin(ang * 0.8), sin(ang)).normalized()
			viewer.global_position = d * (PLANET_RADIUS + 2000.0)
			var sm: Dictionary = await _settle(t)
			var missing: int = sm.resident_sectors - sm.rendered_sectors
			print("  step ", step, " resident=", sm.resident_sectors,
					" rendered=", sm.rendered_sectors, " clipped=", sm.clipped_sectors,
					" stale=", sm.stale_results_discarded, " missing=", missing)
			if missing != 0:
				failures.append("step %d left %d resident sectors unbuilt" % [step, missing])
			if sm.stale_results_discarded != 0:
				failures.append("step %d discarded stale results" % step)

	if stage == "move":
		# Walks the viewer with the clip on and checks nothing resident is left
		# unbuilt -- the regression from HANDOFF 3.5.
		t.far_near_clip_radius = 1500.0
		for step in 6:
			viewer.global_position = Vector3(step * 700.0, 0.0, step * 300.0)
			var sm: Dictionary = await _settle(t)
			var missing: int = sm.resident_sectors - sm.rendered_sectors
			print("  step ", step, " resident=", sm.resident_sectors, " rendered=", sm.rendered_sectors,
					" built=", sm.sectors_built, " stale=", sm.stale_results_discarded,
					" inflight=", sm.tasks_in_flight, " pend=", sm.pending_uploads,
					" clipped=", sm.clipped_sectors, " missing=", missing, " frames=", sm.frames)
			if missing != 0:
				failures.append("step %d left %d resident sectors unbuilt" % [step, missing])
	else:
		var s: Dictionary = await _settle(t)
		_report(s)
		if s.sectors_built == 0 or s.total_vertices == 0:
			failures.append("no geometry")
		if s.stale_results_discarded != 0:
			failures.append("stale results discarded")
		if s.rendered_sectors != s.resident_sectors:
			failures.append("resident sectors left unbuilt")
		if stage == "4_near_clip" and s.near_clip_radius <= 0.0:
			failures.append("near clip was not derived from view_distance")
		if stage == "cache_v4" and s.max_material == 0:
			failures.append("no materials after a cache round trip")
		if stage == "flat_v4" and s.max_material == 0:
			failures.append("flat path read no materials from a MIXEL4 generator")

	if stage.begins_with("planet_v4"):
		# Cross-check the far field's geometry against the generator's own
		# surface sampler. They share compute_heights() but reach it by
		# different routes -- generate_series here, a direct call there -- so
		# agreement means the series path is wired up correctly.
		var gen: Object = t.generator
		var lo := 1e30
		var hi := -1e30
		for i in 400:
			var y := 1.0 - (float(i) / 399.0) * 2.0
			var rr: float = sqrt(max(0.0, 1.0 - y * y))
			var th: float = 2.399963 * float(i)
			var d := Vector3(cos(th) * rr, y, sin(th) * rr)
			var h: float = gen.sample_surface(d)["height"]
			lo = min(lo, h)
			hi = max(hi, h)
		print("  sample_surface height range: ", int(lo), " .. ", int(hi))

		var stats: Dictionary = t.get_far_statistics()
		print("  max_material=", stats.max_material)
		var ab: AABB = stats["world_aabb"]
		var centre := ab.position + ab.size * 0.5
		var half := maxf(maxf(ab.size.x, ab.size.y), ab.size.z) * 0.5
		print("  far aabb centre=", centre.round(), " half_extent=", int(half),
				" expected~", int(V4_RADIUS + hi))

		# V4 writes MIXEL4 materials, so the far field must be decoding them.
		if stats.max_material == 0:
			failures.append("no materials came through from the generator")

		if centre.length() > 2000.0:
			failures.append("far field is not centred on the planet (%s)" % centre)
		# The heights differ a little by construction: sample_surface evaluates
		# the height field on the reference sphere, while the real surface is
		# where the SDF crosses zero at the actual radius, and the field is 3D.
		# A generous band still catches a wrong radius, a missing base-height
		# offset or a sign error.
		var expected := V4_RADIUS + hi
		if absf(half - expected) > maxf(1500.0, absf(hi - lo) * 0.6):
			failures.append("far surface radius %d does not match generator %d" % [int(half), int(expected)])

	if stage == "planet_handover":
		var hs: Dictionary = t.get_far_statistics()
		print("  near_clip=", hs.near_clip_radius, " clipped=", hs.clipped_sectors,
				" rendered=", hs.rendered_sectors)
		if hs.near_clip_radius <= 0.0:
			failures.append("near clip was not derived from view_distance")
		await _shoot_planet_handover(t, viewer_pos)

	if stage == "planet_v4_shot":
		await _shoot_planet(t, viewer)

	if stage == "planet_shot":
		await _shoot_planet(t, viewer)

	if stage == "handover":
		# Needs a real window. Sets up lights, sky, camera and the far shader so
		# the seam between near and far can actually be looked at.
		await _shoot(t, viewer_pos)

	print("RESULT=", "PASS" if failures.is_empty() else "FAIL " + str(failures))
	quit(0 if failures.is_empty() else 1)


func _shoot(t: Node, viewer_pos: Vector3) -> void:
	var sh: Shader = load("res://far_terrain.gdshader")
	if sh == null:
		print("SHADER_LOAD=FAILED")
		return
	var sm := ShaderMaterial.new()
	sm.shader = sh
	var clip: float = t.get_far_statistics()["near_clip_radius"]
	sm.set_shader_parameter("near_fade_start", clip * 0.85)
	sm.set_shader_parameter("near_fade_end", clip * 1.15)
	sm.set_shader_parameter("fog_color", Color(0.62, 0.70, 0.80))
	sm.set_shader_parameter("height_fog_color", Color(0.72, 0.78, 0.86))
	sm.set_shader_parameter("fog_start", 6000.0)
	sm.set_shader_parameter("fog_end", 90000.0)
	t.far_material = sm

	var near_mat := StandardMaterial3D.new()
	near_mat.albedo_color = Color(0.42, 0.40, 0.36)
	near_mat.roughness = 0.95
	t.set_material(near_mat)

	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-35.0, 40.0, 0.0)
	sun.light_energy = 1.2
	root.add_child(sun)

	var env := WorldEnvironment.new()
	var e := Environment.new()
	e.background_mode = Environment.BG_SKY
	e.sky = Sky.new()
	e.sky.sky_material = ProceduralSkyMaterial.new()
	e.ambient_light_source = Environment.AMBIENT_SOURCE_SKY
	e.ambient_light_energy = 0.6
	env.environment = e
	root.add_child(env)

	var cam := Camera3D.new()
	cam.near = 1.0
	cam.far = 300000.0
	cam.fov = 70.0
	root.add_child(cam)
	cam.global_position = viewer_pos
	cam.rotation_degrees = Vector3(-18.0, 35.0, 0.0)
	cam.current = true
	root.use_taa = true

	for i in 90:
		await process_frame
	await RenderingServer.frame_post_draw
	var img: Image = root.get_texture().get_image()
	print("SAVED=shot_handover.png err=", img.save_png("res://shot_handover.png"))


# Two views of the planet: the globe from orbit, and the horizon from just above
# the surface. The first proves the sphere closes; the second proves the far
# field still reads as terrain rather than a faceted ball.
func _shoot_planet(t: Node, viewer: Node) -> void:
	var sh: Shader = load("res://far_terrain.gdshader")
	var sm := ShaderMaterial.new()
	sm.shader = sh
	# Layer heights are radii from the planet centre here, so the colour ramp
	# has to be moved out to the planet radius with them.
	sm.set_shader_parameter("near_fade_start", 0.0)
	sm.set_shader_parameter("near_fade_end", 0.0)
	sm.set_shader_parameter("low_height", PLANET_RADIUS - 400.0)
	sm.set_shader_parameter("high_height", PLANET_RADIUS + 700.0)
	sm.set_shader_parameter("peak_height", PLANET_RADIUS + 1600.0)
	sm.set_shader_parameter("fog_start", 20000.0)
	sm.set_shader_parameter("fog_end", 400000.0)
	sm.set_shader_parameter("noise_enabled", false)
	# EdenPlanetGeneratorV4's material ids, in order: grass, rock, snow, sand,
	# dirt, moss, ocean floor. Slot 7 is unused.
	sm.set_shader_parameter("material_blend", 0.85)
	sm.set_shader_parameter("material_colors", PackedVector3Array([
		Vector3(0.26, 0.42, 0.16),  # grass
		Vector3(0.42, 0.40, 0.37),  # rock
		Vector3(0.92, 0.94, 0.97),  # snow
		Vector3(0.80, 0.72, 0.50),  # sand
		Vector3(0.36, 0.29, 0.21),  # dirt
		Vector3(0.24, 0.36, 0.22),  # moss
		Vector3(0.20, 0.26, 0.30),  # ocean floor
		Vector3(1.00, 0.00, 1.00),  # unused, deliberately obvious
	]))
	t.far_material = sm

	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-25.0, 35.0, 0.0)
	sun.light_energy = 1.4
	root.add_child(sun)

	var env := WorldEnvironment.new()
	var e := Environment.new()
	e.background_mode = Environment.BG_COLOR
	e.background_color = Color(0.02, 0.02, 0.05)
	e.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	e.ambient_light_color = Color(0.3, 0.35, 0.45)
	e.ambient_light_energy = 0.5
	env.environment = e
	root.add_child(env)

	var cam := Camera3D.new()
	cam.near = 10.0
	cam.far = 1000000.0
	cam.fov = 55.0
	root.add_child(cam)
	cam.current = true

	# 1) From orbit.
	cam.global_position = Vector3(0.0, 30000.0, 130000.0)
	cam.look_at(Vector3.ZERO, Vector3.UP)
	for i in 30:
		await process_frame
	await RenderingServer.frame_post_draw
	print("SAVED=planet_orbit.png err=", root.get_texture().get_image().save_png("res://planet_orbit.png"))

	# 2) Just above the surface, looking along the horizon. The viewer has to
	# move with the camera or the far field stays centred where it was.
	var surface := Vector3(0.0, 0.0, 1.0) * (PLANET_RADIUS + 2400.0)
	viewer.global_position = surface
	cam.global_position = surface
	cam.look_at(surface + Vector3(0.0, 1.0, -0.16), Vector3(0.0, 0.0, 1.0))
	var s2: Dictionary = await _settle(t)
	print("  surface view: sectors=", s2.rendered_sectors, " verts=", s2.total_vertices)
	for i in 20:
		await process_frame
	await RenderingServer.frame_post_draw
	print("SAVED=planet_surface.png err=", root.get_texture().get_image().save_png("res://planet_surface.png"))


# Near terrain and far field together on a planet, looking along the seam.
func _shoot_planet_handover(t: Node, viewer_pos: Vector3) -> void:
	var sh: Shader = load("res://far_terrain.gdshader")
	var sm := ShaderMaterial.new()
	sm.shader = sh
	var clip: float = t.get_far_statistics()["near_clip_radius"]
	sm.set_shader_parameter("near_fade_start", clip * 0.85)
	sm.set_shader_parameter("near_fade_end", clip * 1.15)
	sm.set_shader_parameter("low_height", PLANET_RADIUS - 400.0)
	sm.set_shader_parameter("high_height", PLANET_RADIUS + 700.0)
	sm.set_shader_parameter("peak_height", PLANET_RADIUS + 1600.0)
	sm.set_shader_parameter("fog_start", 8000.0)
	sm.set_shader_parameter("fog_end", 120000.0)
	t.far_material = sm

	# Roughly matched to the far field's low/high ramp so the seam is judged on
	# geometry rather than on a colour mismatch.
	var near_mat := StandardMaterial3D.new()
	near_mat.albedo_color = Color(0.40, 0.44, 0.34)
	near_mat.roughness = 0.95
	t.set_material(near_mat)

	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-30.0, 30.0, 0.0)
	sun.light_energy = 1.3
	root.add_child(sun)

	var env := WorldEnvironment.new()
	var e := Environment.new()
	e.background_mode = Environment.BG_COLOR
	e.background_color = Color(0.05, 0.07, 0.12)
	e.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	e.ambient_light_color = Color(0.35, 0.40, 0.50)
	e.ambient_light_energy = 0.55
	env.environment = e
	root.add_child(env)

	var cam := Camera3D.new()
	cam.near = 1.0
	cam.far = 500000.0
	cam.fov = 65.0
	root.add_child(cam)
	cam.global_position = viewer_pos
	# Up is radial here, so look along a tangent and keep radial as up.
	cam.look_at(viewer_pos + Vector3(0.0, 1.0, -0.12), viewer_pos.normalized())
	cam.current = true
	root.use_taa = true

	for i in 60:
		await process_frame
	await RenderingServer.frame_post_draw
	print("SAVED=planet_handover.png err=",
			root.get_texture().get_image().save_png("res://planet_handover.png"))
