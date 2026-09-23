extends "res://planet_check.gd"

# EdenPlanetGeneratorV4's lattice path: per-block cost, SDF error vs exact per-direction heights (sample_surface), and
# seam equality between neighboring blocks. Exit 1 on a seam sign mismatch or error over half a voxel.
#   $GODOT --headless --path . --script lattice_check.gd
func _make_buf() -> VoxelBuffer:
	var buf := VoxelBuffer.new()
	buf.set_channel_depth(VoxelBuffer.CHANNEL_SDF, VoxelBuffer.DEPTH_32_BIT)
	buf.set_channel_depth(VoxelBuffer.CHANNEL_DATA6, VoxelBuffer.DEPTH_32_BIT)
	buf.create(35, 35, 35)
	return buf


func _run() -> void:
	var g := _make_generator()
	var worst := 0.0
	var seam_mismatch := 0
	var sign_flips := 0
	var seam_diff := 0.0
	for lod: int in [0, 3, 6]:
		var s: int = 1 << lod
		var gen_us := 0
		var n := 12
		var err_sum := 0.0
		var err_n := 0
		var lod_worst := 0.0
		for i in n:
			var dir := Vector3(0.1 * i, 1.0, 0.05 * i).normalized()
			var h: float = g.sample_surface(dir).height
			var o: Vector3i = Vector3i(dir * (RADIUS + h)) - Vector3i(16, 16, 16) * s
			o = (o / (32 * s)) * (32 * s)
			var buf := _make_buf()
			var t0 := Time.get_ticks_usec()
			g.generate_block(buf, o, lod)
			gen_us += Time.get_ticks_usec() - t0
			# Error near the surface, where the mesh comes from
			for z in range(0, 35, 3):
				for y in range(0, 35, 3):
					for x in range(0, 35, 3):
						var sdf := buf.get_voxel_f(x, y, z, VoxelBuffer.CHANNEL_SDF)
						var p := Vector3(o) + Vector3(x, y, z) * s + Vector3.ONE * (s * 0.5)
						var exact: float = p.length() - RADIUS - g.sample_surface(p.normalized()).height
						# Sign may differ inside the interpolation error, i.e. right at the surface
						if signf(sdf) != signf(exact) and absf(exact) > 0.5 * s:
							sign_flips += 1
						if absf(exact) > 2.0 * s:
							continue
						var e := absf(sdf - exact) / s
						err_sum += e
						err_n += 1
						lod_worst = maxf(lod_worst, e)
			# Seam: the next block along x shares voxels 32..34 of this one
			var nb := _make_buf()
			g.generate_block(nb, o + Vector3i(32 * s, 0, 0), lod)
			for z in 35:
				for y in 35:
					for k in 3:
						var a := buf.get_voxel_f(32 + k, y, z, VoxelBuffer.CHANNEL_SDF)
						var b := nb.get_voxel_f(k, y, z, VoxelBuffer.CHANNEL_SDF)
						# Early-out blocks store +-100 away from the surface: only sign and near-surface values matter
						if signf(a) != signf(b):
							seam_mismatch += 1
						elif absf(a) != 100.0 and absf(b) != 100.0:
							seam_diff = maxf(seam_diff, absf(a - b))
		worst = maxf(worst, lod_worst)
		print("lod %d: generate %.2f ms/block, near-surface SDF error mean %.4f / max %.4f voxels (%d samples)" % [
				lod, gen_us / 1000.0 / n, err_sum / maxi(err_n, 1), lod_worst, err_n])
	print("seam sign mismatches: ", seam_mismatch, "  seam max value diff: ", seam_diff, "  sign flips vs exact: ", sign_flips)
	quit(0 if seam_mismatch == 0 and sign_flips == 0 and worst < 0.5 else 1)
