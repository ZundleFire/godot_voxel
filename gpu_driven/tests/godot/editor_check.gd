@tool
extends Node3D

# Renders a GPU-driven terrain in the editor's own 3D viewport (editor camera, no running game) and compares it with
# the traditional path. Inert unless the editor is started with the "capture" user arg:
#   $GODOT --path . -e res://editor_check.tscn -- capture
# Quits the editor with exit code 0 (pass) or 1 (fail).

const MASK_W := 320
const MASK_H := 180

var _active := false
var _stage := 0
var _frames := 0
var _quiet := 0
var _terrain: Node
var _background: Image
var _gpu_img: Image
var _failures := 0


func _ready() -> void:
	_active = Engine.is_editor_hint() and "capture" in OS.get_cmdline_user_args()


func _check(cond: bool, what: String) -> void:
	print(("  ok    " if cond else "  FAIL  ") + what)
	if not cond:
		_failures += 1


func _viewport_image() -> Image:
	return EditorInterface.get_editor_viewport_3d(0).get_texture().get_image()


func _settled() -> bool:
	var tasks: Dictionary = Engine.get_singleton("VoxelEngine").get_stats().tasks
	var busy: int = tasks.streaming + tasks.generation + tasks.meshing + tasks.main_thread
	var g: Dictionary = _terrain.get_gpu_driven_statistics()
	if not g.is_empty():
		busy += int(g.pending_ops)
	_quiet = _quiet + 1 if busy == 0 else 0
	return _quiet > 60


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


func _build_environment() -> void:
	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-50.0, 30.0, 0.0)
	add_child(sun)

	var we := WorldEnvironment.new()
	var e := Environment.new()
	e.background_mode = Environment.BG_COLOR
	e.background_color = Color(0.2, 0.3, 0.6)
	we.environment = e
	add_child(we)


func _build_terrain() -> void:
	var viewer: Node = ClassDB.instantiate("VoxelViewer")
	viewer.view_distance = 1024
	viewer.enabled_in_editor = true
	add_child(viewer)

	_terrain = ClassDB.instantiate("VoxelLodTerrain")
	_terrain.mesher = ClassDB.instantiate("VoxelMesherTransvoxel")
	var g: Object = ClassDB.instantiate("VoxelGeneratorNoise2D")
	var n := FastNoiseLite.new()
	n.frequency = 0.003
	n.fractal_octaves = 4
	g.noise = n
	g.channel = 1
	g.height_start = -60.0
	g.height_range = 160.0
	_terrain.generator = g
	_terrain.view_distance = 1024
	_terrain.lod_count = 6
	var sm := ShaderMaterial.new()
	sm.shader = load("res://traditional.gdshader")
	_terrain.material = sm
	_terrain.render_mode = 1
	# Below the default editor camera, which sits near the origin looking down
	_terrain.position = Vector3(0.0, -120.0, 0.0)
	add_child(_terrain)


func _process(_delta: float) -> void:
	if not _active:
		return
	_frames += 1
	match _stage:
		0:
			if _frames == 30:
				EditorInterface.set_main_screen_editor("3D")
			if _frames == 60:
				_build_environment()
			if _frames > 120:
				# Environment and editor gizmos, no terrain yet
				_background = _viewport_image()
				_background.save_png("res://editor_background.png")
				_build_terrain()
				_stage = 1
				_frames = 0
		1:
			if _frames > 120 and _settled():
				_gpu_img = _viewport_image()
				_gpu_img.save_png("res://editor_gpu.png")
				var g: Dictionary = _terrain.get_gpu_driven_statistics()
				print("  gpu stats: ", g)
				_check(int(g.get("chunks", 0)) > 0, "chunks uploaded in the editor")
				_check(int(g.get("indirect_draws_total", 0)) > 0, "indirect draws issued for the editor viewport")
				var cov := 0
				for v in _mask(_gpu_img):
					cov += v
				_check(cov > MASK_W * MASK_H / 5 and cov < MASK_W * MASK_H * 0.95,
						"terrain covers part of the editor viewport (%d / %d px)" % [cov, MASK_W * MASK_H])
				_terrain.render_mode = 0
				_stage = 2
				_frames = 0
				_quiet = 0
		2:
			if _frames > 120 and _settled():
				var trad := _viewport_image()
				trad.save_png("res://editor_traditional.png")
				var a := _mask(_gpu_img)
				var b := _mask(trad)
				var inter := 0
				var uni := 0
				for i in a.size():
					inter += a[i] & b[i]
					uni += a[i] | b[i]
				var iou := 1.0 if uni == 0 else float(inter) / uni
				_check(iou > 0.97, "editor gpu silhouette matches traditional (IoU %.4f)" % iou)
				print("FAILURES=", _failures)
				get_tree().quit(1 if _failures > 0 else 0)
				_stage = 3
