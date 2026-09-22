# Engine-level checks

`../` holds host tests for `far/core/`, which needs no engine. These cover what only exists once
the far field is linked into Godot: the streaming system, the task pipeline, the cache, and the
shader.

Run from this directory with a built editor binary:

```sh
GODOT=../../../../../bin/godot.windows.editor.x86_64.console.exe

# HANDOFF.md step 2, one subsystem at a time. Exits non-zero on failure.
$GODOT --headless --path . --script far_check.gd -- 1_one_sector
$GODOT --headless --path . --script far_check.gd -- 2_full_ring
$GODOT --headless --path . --script far_check.gd -- 3_multi_level
$GODOT --headless --path . --script far_check.gd -- 4_near_clip     # clip derived from view_distance
$GODOT --headless --path . --script far_check.gd -- 6_cache clear   # cold: saves, no hits
$GODOT --headless --path . --script far_check.gd -- 6_cache         # warm: all hits, settles faster
$GODOT --headless --path . --script far_check.gd -- move            # walks a VoxelViewer

# A 40 km planet. Headless checks, then renders.
$GODOT --headless --path . --script far_check.gd -- planet        # simple graph generator
$GODOT --headless --path . --script far_check.gd -- planet_v4     # EdenPlanetGeneratorV4
$GODOT --headless --path . --script far_check.gd -- planet_aa     # antialiasing on a sphere
$GODOT --headless --path . --script far_check.gd -- planet_move   # orbits, crossing cube faces
$GODOT --headless --path . --script far_check.gd -- flat_v4       # flat path, MIXEL4 materials
$GODOT --headless --path . --script far_check.gd -- cache_v4 clear   # materials survive the cache
$GODOT --headless --path . --script far_check.gd -- cache_v4
$GODOT --path . --script far_check.gd --resolution 1280x720 -- planet_shot
$GODOT --path . --script far_check.gd --resolution 1280x720 -- planet_v4_shot
$GODOT --path . --script far_check.gd --resolution 1280x720 -- planet_handover

# Needs a real window: adds a near mesher and saves a frame of the seam.
$GODOT --path . --script far_check.gd --resolution 1280x720 -- handover

# Far field on its own, for looking at the shader.
#   shot.gd -- <out.png> <shader|fog|standard> [camera_y] [pitch_deg]
$GODOT --path . --script shot.gd --resolution 1280x720 -- out.png fog 3000 -11
```

Everything except `handover` runs headless, because without a mesher the near terrain builds no
meshes and so never touches the dummy renderer — `VoxelLodTerrain` otherwise dies there with
"Attempting to initialize the wrong RID".

`far_check.gd` waits for `tasks_in_flight` and `pending_uploads` to reach zero and for
`rendered_sectors` to equal `resident_sectors`. That last equality is the useful one: it is what
caught sectors being marked resident without ever being built, twice (HANDOFF.md §3.5 and §8.4).

Every stage puts a real `VoxelViewer` in the scene, and `move` drives that rather than the terrain
node's transform. See HANDOFF.md §3.8 for why that distinction matters.

`far_terrain.gdshader` here is a copy of `../../shaders/far_terrain.gdshader`, since a Godot project
cannot reach outside its own directory. Re-copy it after editing the original.
