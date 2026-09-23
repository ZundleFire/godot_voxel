extends "res://planet_check.gd"

# Load + render benchmark for one render path from a cold start, one process per run so memory numbers don't mix:
#   $GODOT --path . --script planet_bench.gd --resolution 1280x720 -- <orbit|surface> <0 traditional|1 gpu> [out_dir]
# Prints one BENCH_JSON= line. run_bench.py runs all four and builds the report.

const LOAD_TIMEOUT_S := 900.0
const MEASURE_FRAMES := 600


func _mem() -> Dictionary:
	var pools: Dictionary = Engine.get_singleton("VoxelEngine").get_stats().memory_pools
	return {
		"static_mb": snappedf(Performance.get_monitor(Performance.MEMORY_STATIC) / 1048576.0, 0.1),
		"static_peak_mb": snappedf(Performance.get_monitor(Performance.MEMORY_STATIC_MAX) / 1048576.0, 0.1),
		"video_mb": snappedf(Performance.get_monitor(Performance.RENDER_VIDEO_MEM_USED) / 1048576.0, 0.1),
		"buffer_mb": snappedf(Performance.get_monitor(Performance.RENDER_BUFFER_MEM_USED) / 1048576.0, 0.1),
		# Pool total is a high-water mark (it never shrinks), used is what live blocks hold
		"voxel_pool_mb": snappedf(float(pools.voxel_total) / 1048576.0, 0.1),
		"voxel_used_mb": snappedf(float(pools.voxel_used) / 1048576.0, 0.1),
		"voxel_blocks": int(pools.block_count),
	}


# Cold load: time until the engine has no voxel work left for 60 frames straight
func _load(t: Node) -> Dictionary:
	var engine: Object = Engine.get_singleton("VoxelEngine")
	var t0 := Time.get_ticks_usec()
	var first_block_s := -1.0
	var frames := 0
	var quiet := 0
	var worst_ms := 0.0
	var worst_at_s := 0.0
	var hitches := 0
	var last := t0
	var settled := false
	while (Time.get_ticks_usec() - t0) / 1e6 < LOAD_TIMEOUT_S:
		await process_frame
		frames += 1
		var now := Time.get_ticks_usec()
		var dt := (now - last) / 1000.0
		if dt > worst_ms:
			worst_ms = dt
			worst_at_s = (now - t0) / 1e6
		if dt > 33.3:
			hitches += 1
		last = now
		if first_block_s < 0.0 and t.debug_get_mesh_block_count() > 0:
			first_block_s = (now - t0) / 1e6
		var tasks: Dictionary = engine.get_stats().tasks
		var busy: int = tasks.streaming + tasks.generation + tasks.meshing + tasks.main_thread
		busy += int(t.get_gpu_driven_statistics().get("pending_ops", 0))
		quiet = quiet + 1 if busy == 0 else 0
		if quiet >= 60:
			settled = true
			break
	# The 60 quiet frames are not load time
	var total_s := (Time.get_ticks_usec() - t0) / 1e6
	return {
		"settled": settled,
		"load_s": snappedf(total_s, 0.01),
		"first_block_s": snappedf(first_block_s, 0.01),
		"load_frames": frames,
		"load_avg_fps": snappedf(frames / total_s, 0.1),
		"load_worst_frame_ms": snappedf(worst_ms, 0.1),
		"load_worst_frame_at_s": snappedf(worst_at_s, 0.01),
		"load_hitches_over_33ms": hitches,
		"mesh_blocks": t.debug_get_mesh_block_count(),
	}


func _steady(frames: int) -> Dictionary:
	var vp := root.get_viewport_rid()
	RenderingServer.viewport_set_measure_render_time(vp, true)
	var times := PackedFloat64Array()
	var cpu := 0.0
	var gpu := 0.0
	var draws := 0.0
	var prims := 0.0
	var base_rot := _cam.rotation_degrees
	var last := Time.get_ticks_usec()
	for i in frames:
		# Small yaw sweep: keeps culling busy without moving the viewer (no streaming)
		_cam.rotation_degrees = base_rot + Vector3(0.0, 20.0 * sin(TAU * i / frames), 0.0)
		await process_frame
		var now := Time.get_ticks_usec()
		times.append((now - last) / 1000.0)
		last = now
		cpu += RenderingServer.viewport_get_measured_render_time_cpu(vp)
		gpu += RenderingServer.viewport_get_measured_render_time_gpu(vp)
		draws += Performance.get_monitor(Performance.RENDER_TOTAL_DRAW_CALLS_IN_FRAME)
		prims += Performance.get_monitor(Performance.RENDER_TOTAL_PRIMITIVES_IN_FRAME)
	_cam.rotation_degrees = base_rot
	var sorted := times.duplicate()
	sorted.sort()
	var total := 0.0
	for x in times:
		total += x
	var avg := total / frames
	# 1% low: average FPS over the slowest 1% of frames
	var n_low := maxi(1, frames / 100)
	var low_total := 0.0
	for i in n_low:
		low_total += sorted[frames - 1 - i]
	return {
		"avg_fps": snappedf(1000.0 / avg, 0.1),
		"low1_fps": snappedf(1000.0 / (low_total / n_low), 0.1),
		"frame_ms": snappedf(avg, 0.001),
		"frame_ms_p50": snappedf(sorted[frames / 2], 0.001),
		"frame_ms_p99": snappedf(sorted[int(frames * 0.99)], 0.001),
		"render_cpu_ms": snappedf(cpu / frames, 0.001),
		"render_gpu_ms": snappedf(gpu / frames, 0.001),
		"draw_calls": int(draws / frames),
		"primitives": int(prims / frames),
	}


func _run() -> void:
	var args: PackedStringArray = OS.get_cmdline_user_args()
	var view: String = args[0] if args.size() > 0 else "surface"
	var mode: int = int(args[1]) if args.size() > 1 else TRADITIONAL
	if args.size() > 2:
		_out_dir = args[2]
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	Engine.max_fps = 0
	_make_scene(view)
	for i in 5:
		await process_frame
	var mem_before := _mem()

	var t := _make_terrain()
	t.render_mode = mode
	_world.add_child(t)
	var load := await _load(t)
	for i in 3:
		await process_frame
	var label := "gpu" if mode == GPU else "traditional"
	await _capture("bench_%s_%s" % [view, label])
	await _steady(60)
	var steady := await _steady(MEASURE_FRAMES)
	var g: Dictionary = t.get_gpu_driven_statistics()
	var result := {
		"view": view,
		"path": label,
		"load": load,
		"steady": steady,
		"mem_before": mem_before,
		"mem_after": _mem(),
		"gpu_renderer": {
			"chunks": int(g.get("chunks", 0)),
			"visible_chunks": int(g.get("visible_chunks", 0)),
			"visible_triangles": int(g.get("visible_triangles", 0)),
			"vertices": int(g.get("vertices", 0)),
			"vram_mb": snappedf(float(g.get("vram_bytes", 0)) / 1048576.0, 0.1),
		} if mode == GPU else {},
		"adapter": RenderingServer.get_video_adapter_name(),
	}
	print("BENCH_JSON=", JSON.stringify(result))
	quit(0 if load.settled else 2)
