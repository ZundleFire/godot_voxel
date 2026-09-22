extends SceneTree

# Engine-level checks for VoxelLodTerrain.render_mode (needs a real window, the GPU path draws through
# RenderingDevice). Run from this directory:
#   GODOT=../../../../../bin/godot.windows.editor.x86_64.console.exe
#   $GODOT --path . --script gpu_check.gd --resolution 1280x720 -- <stage> [out_dir]
# Stages:
#   compare  traditional -> gpu -> traditional in one run. Checks the terrain silhouettes match (IoU of
#            coverage masks), that switching leaves nothing behind, and GPU stats sanity. Exits 1 on failure.
#   hide     gpu mode, hides then shows the terrain.
#   free     gpu mode, frees the terrain mid-run (renderer teardown path).
#   move     gpu mode, flies the viewer (chunk churn), then compares with traditional at the end position.
#   bench    frame/render times, draw calls, objects for both modes, camera spinning.

const MAX_SETTLE_FRAMES := 4000
const MASK_W := 320
const MASK_H := 180
const MIN_IOU := 0.97

const TRADITIONAL := 0
const GPU := 1

var _out_dir := "."
var _cam: Camera3D
var _world: Node3D
var _background: Image
var _failures := 0


func _init() -> void:
	_run.call_deferred()


func _check(cond: bool, what: String) -> void:
	if cond:
		print("  ok    ", what)
	else:
		print("  FAIL  ", what)
		_failures += 1


func _make_scene() -> void:
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
	_cam.near = 0.5
	_cam.far = 4000.0
	_cam.fov = 70.0
	_cam.position = Vector3(0.0, 110.0, 0.0)
	_cam.rotation_degrees = Vector3(-25.0, 20.0, 0.0)
	_world.add_child(_cam)
	_cam.current = true

	var viewer: Node = ClassDB.instantiate("VoxelViewer")
	viewer.view_distance = 1024
	_cam.add_child(viewer)


func _make_terrain(mode: int) -> Node:
	var t: Node = ClassDB.instantiate("VoxelLodTerrain")
	t.mesher = ClassDB.instantiate("VoxelMesherTransvoxel")
	var g: Object = ClassDB.instantiate("VoxelGeneratorNoise2D")
	var n := FastNoiseLite.new()
	n.frequency = 0.003
	n.fractal_octaves = 4
	g.noise = n
	g.channel = 1 # SDF
	g.height_start = -60.0
	g.height_range = 160.0
	t.generator = g
	t.view_distance = 1024
	t.lod_count = 6
	var sm := ShaderMaterial.new()
	sm.shader = load("res://traditional.gdshader")
	t.material = sm
	t.render_mode = mode
	return t


func _settle(t: Node) -> int:
	var engine: Object = Engine.get_singleton("VoxelEngine")
	var quiet := 0
	var frames := 0
	while frames < MAX_SETTLE_FRAMES:
		await process_frame
		frames += 1
		var tasks: Dictionary = engine.get_stats().tasks
		var busy: int = tasks.streaming + tasks.generation + tasks.meshing + tasks.main_thread
		var g: Dictionary = t.get_gpu_driven_statistics() if t != null else {}
		if not g.is_empty():
			busy += int(g.pending_ops)
		quiet = quiet + 1 if busy == 0 else 0
		if quiet >= 30 and frames > 10:
			break
	return frames


func _capture(name: String) -> Image:
	await RenderingServer.frame_post_draw
	var img: Image = root.get_texture().get_image()
	var path := _out_dir.path_join(name + ".png")
	img.save_png(path)
	print("  saved ", path)
	return img


# 1 where the pixel differs from the terrain-less background frame
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
			var d: float = abs(ca.r - cb.r) + abs(ca.g - cb.g) + abs(ca.b - cb.b)
			m[y * MASK_W + x] = 1 if d > 0.03 else 0
	return m


# Full resolution: pixels covered by terrain in `ref` that show the background in `img` (cracks, missing chunks)
func _holes(ref: Image, img: Image) -> int:
	var a := ref.duplicate() as Image
	var b := img.duplicate() as Image
	var bg := _background.duplicate() as Image
	for im in [a, b, bg]:
		im.convert(Image.FORMAT_RGB8)
	var da := a.get_data()
	var db := b.get_data()
	var dg := bg.get_data()
	var count := 0
	var i := 0
	while i < dg.size():
		var ref_is_bg: bool = abs(da[i] - dg[i]) + abs(da[i + 1] - dg[i + 1]) + abs(da[i + 2] - dg[i + 2]) < 8
		var img_is_bg: bool = abs(db[i] - dg[i]) + abs(db[i + 1] - dg[i + 1]) + abs(db[i + 2] - dg[i + 2]) < 8
		if img_is_bg and not ref_is_bg:
			count += 1
		i += 3
	return count


func _coverage(m: PackedByteArray) -> float:
	var c := 0
	for v in m:
		c += v
	return float(c) / m.size()


func _iou(a: PackedByteArray, b: PackedByteArray) -> float:
	var inter := 0
	var uni := 0
	for i in a.size():
		inter += a[i] & b[i]
		uni += a[i] | b[i]
	return 1.0 if uni == 0 else float(inter) / uni


func _capture_background() -> void:
	for i in 5:
		await process_frame
	_background = await _capture("background")


func _print_gpu_stats(t: Node) -> Dictionary:
	var g: Dictionary = t.get_gpu_driven_statistics()
	print("  gpu stats: ", g)
	return g


func _stage_compare() -> void:
	await _capture_background()
	var t := _make_terrain(TRADITIONAL)
	_world.add_child(t)
	print("traditional: settled in ", await _settle(t), " frames, mesh blocks ", t.debug_get_mesh_block_count())
	var trad_img := await _capture("traditional")
	var trad_mask := _mask(trad_img)
	var trad_cov := _coverage(trad_mask)
	_check(trad_cov > 0.2, "traditional terrain covers the view (%.3f)" % trad_cov)
	_check(t.get_gpu_driven_statistics().is_empty(), "no GPU renderer in traditional mode")

	t.render_mode = GPU
	print("gpu: settled in ", await _settle(t), " frames, mesh blocks ", t.debug_get_mesh_block_count())
	# The draw happens on the render thread, give it a couple of frames past the last op
	for i in 3:
		await process_frame
	var gpu_img := await _capture("gpu")
	var gpu_mask := _mask(gpu_img)
	var g := _print_gpu_stats(t)
	_check(g.get("initialized", false) and not g.get("failed", true), "renderer initialized")
	_check(int(g.get("chunks", 0)) > 0, "chunks uploaded (%d)" % int(g.get("chunks", 0)))
	_check(int(g.get("indirect_draws_total", 0)) > 0, "indirect draws issued")
	var iou := _iou(trad_mask, gpu_mask)
	_check(iou >= MIN_IOU, "gpu silhouette matches traditional (IoU %.4f)" % iou)
	var holes := _holes(trad_img, gpu_img)
	_check(holes == 0, "no background showing through the gpu terrain (%d pixels)" % holes)

	t.render_mode = TRADITIONAL
	print("back to traditional: settled in ", await _settle(t), " frames")
	var back_mask := _mask(await _capture("traditional_again"))
	_check(t.get_gpu_driven_statistics().is_empty(), "GPU renderer destroyed after switching back")
	var iou_back := _iou(trad_mask, back_mask)
	_check(iou_back >= MIN_IOU, "switching back restores the same image (IoU %.4f)" % iou_back)


func _stage_hide() -> void:
	await _capture_background()
	var t := _make_terrain(GPU)
	_world.add_child(t)
	await _settle(t)
	var shown := _coverage(_mask(await _capture("hide_shown")))
	_check(shown > 0.2, "visible before hiding (%.3f)" % shown)
	t.visible = false
	for i in 3:
		await process_frame
	var hidden := _coverage(_mask(await _capture("hide_hidden")))
	_check(hidden < 0.001, "nothing drawn while hidden (%.4f)" % hidden)
	t.visible = true
	for i in 3:
		await process_frame
	var again := _coverage(_mask(await _capture("hide_shown_again")))
	_check(abs(again - shown) < 0.01, "same coverage after showing again (%.3f)" % again)


func _stage_free() -> void:
	await _capture_background()
	var t := _make_terrain(GPU)
	_world.add_child(t)
	await _settle(t)
	var shown := _coverage(_mask(await _capture("free_before")))
	_check(shown > 0.2, "visible before freeing (%.3f)" % shown)
	t.queue_free()
	for i in 5:
		await process_frame
	var after := _coverage(_mask(await _capture("free_after")))
	_check(after < 0.001, "nothing drawn after the terrain is freed (%.4f)" % after)
	# A second terrain must be able to install its own effect again
	var t2 := _make_terrain(GPU)
	_world.add_child(t2)
	await _settle(t2)
	var again := _coverage(_mask(await _capture("free_new_terrain")))
	_check(abs(again - shown) < 0.01, "a new GPU terrain renders the same (%.3f)" % again)


# Streaming churn: chunks get removed, slots and buffer ranges reused. The end state must match the traditional path.
func _stage_move() -> void:
	await _capture_background()
	var t := _make_terrain(GPU)
	_world.add_child(t)
	await _settle(t)
	var start := _cam.position
	for i in 600:
		_cam.position = start + Vector3(i * 2.0, 0.0, i * 1.5)
		await process_frame
	print("gpu after move: settled in ", await _settle(t), " frames")
	for i in 3:
		await process_frame
	var gpu_img := await _capture("move_gpu")
	var g := _print_gpu_stats(t)
	var blocks: int = t.debug_get_mesh_block_count()
	_check(int(g.chunks) <= blocks, "no leaked chunks (%d chunks, %d mesh blocks)" % [int(g.chunks), blocks])
	t.render_mode = TRADITIONAL
	await _settle(t)
	var trad_img := await _capture("move_traditional")
	var iou := _iou(_mask(trad_img), _mask(gpu_img))
	_check(iou >= MIN_IOU, "after moving, gpu matches traditional (IoU %.4f)" % iou)
	var holes := _holes(trad_img, gpu_img)
	_check(holes == 0, "after moving, no holes (%d pixels)" % holes)
	_check(int(g.chunks) > 0 and int(g.pending_ops) == 0, "gpu state settled")


func _measure(frames: int) -> Dictionary:
	var vp := root.get_viewport_rid()
	RenderingServer.viewport_set_measure_render_time(vp, true)
	var cpu := 0.0
	var gpu := 0.0
	var draws := 0.0
	var objects := 0.0
	var prims := 0.0
	var t0 := Time.get_ticks_usec()
	for i in frames:
		_cam.rotation_degrees.y += 360.0 / frames
		await process_frame
		cpu += RenderingServer.viewport_get_measured_render_time_cpu(vp)
		gpu += RenderingServer.viewport_get_measured_render_time_gpu(vp)
		draws += Performance.get_monitor(Performance.RENDER_TOTAL_DRAW_CALLS_IN_FRAME)
		objects += Performance.get_monitor(Performance.RENDER_TOTAL_OBJECTS_IN_FRAME)
		prims += Performance.get_monitor(Performance.RENDER_TOTAL_PRIMITIVES_IN_FRAME)
	var frame_ms := (Time.get_ticks_usec() - t0) / 1000.0 / frames
	return {
		"frame_ms": snappedf(frame_ms, 0.001),
		"render_cpu_ms": snappedf(cpu / frames, 0.001),
		"render_gpu_ms": snappedf(gpu / frames, 0.001),
		"scene_draw_calls": int(draws / frames),
		"scene_objects": int(objects / frames),
		"scene_primitives": int(prims / frames),
	}


func _stage_bench() -> void:
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	Engine.max_fps = 0
	var t := _make_terrain(TRADITIONAL)
	_world.add_child(t)
	await _settle(t)
	print("mesh blocks: ", t.debug_get_mesh_block_count())
	await _measure(60) # warm up
	var trad := await _measure(600)
	t.render_mode = GPU
	await _settle(t)
	await _measure(60)
	var gpu := await _measure(600)
	print("  traditional: ", trad)
	print("  gpu_driven:  ", gpu, "  (+1 indirect multi-draw per view, not counted by Godot)")
	print("  gpu stats: ", t.get_gpu_driven_statistics())


func _run() -> void:
	var args: PackedStringArray = OS.get_cmdline_user_args()
	var stage: String = args[0] if args.size() > 0 else "compare"
	if args.size() > 1:
		_out_dir = args[1]
	_make_scene()
	print("stage ", stage)
	match stage:
		"compare":
			await _stage_compare()
		"hide":
			await _stage_hide()
		"free":
			await _stage_free()
		"move":
			await _stage_move()
		"bench":
			await _stage_bench()
		_:
			print("unknown stage ", stage)
			quit(2)
			return
	print("FAILURES=", _failures)
	quit(1 if _failures > 0 else 0)
