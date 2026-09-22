# far — findings and continuation plan

Handoff notes for picking this up locally with the full Distant Horizons repo and its submodules.

`README.md` beside this file is the user-facing doc; this file is the working record: what was
decided and why, what broke, what was never verified, and what to do next.

Last updated: 2026-09-21 (planets)

---

## 1. Where this stands

Renders smooth (SDF) terrain from the edge of a `VoxelLodTerrain`'s view distance out to the
horizon. It lives **inside** `modules/voxel` as a third streaming system, alongside octree and
clipbox streaming — see §8.

It began as a separate `voxel_far` module, deliberately a companion using public APIs only so that
upstream `godot_voxel` could move without rebasing. That constraint was dropped on 2026-09-21:
`modules/voxel` here is a fork, so there was nothing to stay compatible with, and staying outside
cost more than it saved. §8 has the reasoning and what changed.

| Layer | Files | Status |
|---|---|---|
| `core/` — column format, extraction, downsampling, LOD rings, mesher, cubesphere | 12 | **Compiled and tested.** 71,754 assertions, clean under ASan. Flat and planetary. |
| `tasks/` — threaded pipeline, region-file cache | 4 | **Running.** Threaded generation and the region cache both exercised end to end; cold run saves, warm run is all hits. |
| `far_renderer` + far streaming | 4 | **Running and rendering.** Streams, clips, and settles; survives a moving viewer. Driven by `VoxelLodTerrain`, not its own node. |
| `shaders/far_terrain.gdshader` | 1 | **Compiles and renders.** Terrain, slope/height colouring, AO, both fog systems seen working. |
| `register_types`, `SCsub`, `config.py` | 4 | **Build clean.** |

Now a streaming system inside `VoxelLodTerrain` (§8), and it works on planets as well as flat
worlds (§9). First light is done, including the
handover against a real near terrain: see §5 Step 2. Four
real bugs came out of it, all fixed — §3.4 to §3.7. The core submodule has been read against this
design: see §5 Step 4. It endorses the format and the rings, and says to take their storage layer.

~6,100 lines of C++/GLSL plus ~1,000 lines of tests.

`core/` deliberately includes **no Godot headers**. That is the single most useful structural
decision in the project: the hard algorithms can be compiled and tested in about two seconds with a
plain `g++`, without launching an engine. §3.1 and §3.2 were both found that way, and neither would
have been obvious from looking at a rendered frame. The counterpart lesson from first light is in
§3, above the bug list: the bugs that survived to that point all lived in the seams between
components, where no single unit test was looking.

---

## 2. The design, and why

### 2.1 The representation is the whole problem

Storing voxels at a coarse LOD costs `O(n³)`, and almost none of that cube carries information — it
is empty above the surface and solid below it. The only thing that varies is *where the surface
is*, which is a 2D field.

So the far field is a **multi-layer column field**: per column, a short list of solid spans.

```
column (x, z):
    ┌─ top    = 412.7   normal = (0.12, 0.97, -0.21)   material = 3   ao = 0.82
    └─ bottom = 380.1
    ┌─ top    = 204.0        <- cave ceiling
    └─ bottom = 196.5
    ┌─ top    = 150.2        <- floor beneath it
    └─ bottom = -256.0       <- clipped by sampled range, NOT a real surface
```

Usually one span per column; two to four where there are caves, overhangs or floating geometry.
Measured in the test suite: **~10× smaller than the equivalent dense volume in memory, ~15× on
disk**, for eventful terrain at LOD 0. It improves with coarseness.

Every sector is a fixed 33×33 grid regardless of LOD, so **each level costs the same to store and
mesh while covering four times the area**. Total cost grows logarithmically with view distance
instead of quadratically. That is the entire economic argument and there is a test pinning it.

### 2.2 Three places a cube-world design would have failed on SDF

**Layer boundaries are continuous and carry normals.** With cubes a boundary lands on an integer
block face; with an SDF it lands wherever the field crosses zero, so `top`/`bottom` are
interpolated and each layer stores the normal from the SDF gradient at that crossing.

Storing the normal is the most important quality decision in the format. Recompute normals from
coarse heights at mesh time instead and distant terrain reads as faceted low-poly scenery — the
only gradient information left is the difference between samples `1 << lod` apart. Keeping the
normal from the resolution the data was *captured* at preserves the original shading after
positions have been decimated hard, and it is what lets a near-vertical cliff still read as a cliff
after its geometry collapses to one quad.

**Steepness must not decide where walls go.** See bug 3.1 — this is the big one.

**Sectors are padded point grids, not cell grids.** 33 columns cover 32 cells. Column 32 of one
sector is the same world position as column 0 of its neighbour, and both come from evaluating the
same generator at the same point, so they agree exactly. This is what allows every sector to be
generated and meshed on its own thread with **zero synchronisation** — meshing never reads a
neighbour.

### 2.3 Concentric rings, not a quadtree

A quadtree is the obvious structure and it is a trap for this workload. It is a stateful graph that
must stay consistent while nodes load asynchronously: a node that splits while its children are
still generating leaves a hole, and so does a node that merges while its parent is still
generating. Every split and merge becomes a small transaction with a pending state.

Rings get the same result with no state machine. Per level, keep the sectors within a fixed radius
of the viewer, minus whatever the finer level already covers. **The required set is a pure function
of viewer position** — recomputable from scratch any frame and diffed against what is resident.
Nothing can be inconsistent because there is no incremental state.

Cost: each level keeps a fixed ring rather than adapting its shape to the terrain. At a few hundred
sectors per level that is worth paying, and it makes the memory ceiling exactly predictable, which
a quadtree's is not.

---

## 3. Bugs found, and what they teach

§3.1 and §3.2 were found by the host test suite before the module had ever been built, and both are
the kind that would be maddening to debug from a rendered frame. §3.4 to §3.7 came out of first
light, and §3.8 out of moving the far field into `modules/voxel`. Worth knowing all of them exist, because a reorganisation could reintroduce any of them.

A pattern across the second batch: none of them were algorithm bugs. Every one was two pieces of
code disagreeing about who was responsible — the node and the engine over task cleanup, the node and
the LOD tree over what "resident" means, the shader's comment and its code over fog, and
`clear_cache()` and `restart()` over when the cache exists. The hard parts were already right.

### 3.1 Slope-driven walls make mountainsides disappear

**The trap.** The natural design for the mesher: connect neighbouring spans into a surface when the
slope between them is gentle, and draw a vertical wall where it is too steep. This is correct for
cube worlds, where every surface really is either flat or a cliff.

**What actually happens on an SDF.** On a *sustained* steep slope, no adjacent pair passes the
threshold. Every span becomes a sheet of its own. No cell ever gets four corners agreeing on a
sheet. The mesher emits **nothing** — an entire mountainside renders as a hole with a skirt around
it. It does not degrade; it vanishes.

**The fix.** Sheet connection is deliberately permissive (`max_connect_gap_cells = 64`, a guard
against connecting across a genuine void, not a slope limit). A steep-but-continuous surface stays
one sheet and is drawn as a steep quad spanning one cell — the correct reading of the data, and one
cell is the resolution limit at that LOD anyway. The stored normals keep it shading like a cliff.

Walls are now reserved for discontinuities in **layer structure**: the lip of an overhang, the edge
of an island, a cave mouth — places where a sheet stops and there is nothing on the far side.

**Guarded by** `test_sustained_steep_slope_has_no_holes` (slopes 0.5 → 40.0, all must give full
coverage) and `test_island_edge_produces_a_wall`.

### 3.2 Unsnapped LOD rings cannot tile

**The trap.** A ring of `2R+1` sectors has odd length, so its edges fall in the middle of the parent
level's sectors. Whichever way you round the conversion you get either a gap (nothing drawn) or a
double-covered band (two levels drawing the same ground, z-fighting along every boundary). Rounding
conservatively does not fix it — it just picks overlap over gaps.

**First attempt** produced 3,058 double-covered points out of 3,721 sampled.

**The fix.** `snapped_ring()` widens each level's range to start on an even coordinate and end on an
odd one. Such a range always maps to a whole number of parent sectors, so the hole is exact and
levels meet edge to edge.

Knock-on: the hysteresis band needs `+1` as well, because snapping shifts a level's range by two
sectors when the viewer crosses one boundary. Without it, a single step across a boundary unloads
two full edges of every level.

**Guarded by** `test_coverage_has_no_gaps_or_overlaps` across on-grid, off-grid and negative viewer
positions.

### 3.3 Godot's triangle winding, settled empirically

Not a bug, but it was going to be a coin flip. Derived from `godot_voxel`'s own cube tables
(`constants/cube_tables.cpp`, `g_side_corners` + `g_side_quad_triangles` for `SIDE_TOP`) rather than
guessed:

> **Godot front faces are clockwise from the front**, so the right-hand-rule normal of an emitted
> triangle points **opposite** the shading normal.

Caps use hardcoded orders derived from that. Walls and skirts go through `emit_quad_oriented()`,
which picks the order from the desired normal so it cannot be got wrong. `test_mesh_is_well_formed`
checks every cap triangle.

**Confirmed in a frame (2026-09-21).** Surfaces are lit from above and nothing is inside-out, so the
derivation was right and no cap order needed flipping.

### 3.4 Nothing pumped VoxelEngine, so no task was ever destroyed

Found on the very first headless run: all 16 sectors built and rendered, yet `tasks_in_flight`
stayed pinned at 16 forever and the engine printed "General tasks remain on module cleanup" once
per task at exit.

`VoxelEngine` only collects and deletes finished tasks when something calls into it every frame,
and the only thing that does is `VoxelEngineUpdater`, a hidden node the near terrains create for
themselves. The old standalone `VoxelFarTerrain` never created one. Tasks ran, pushed results and
were then simply leaked — the counter only decrements in `~FarBuildSectorTask`.

**The fix.** `VoxelEngineUpdater::ensure_existence(get_tree())` in `NOTIFICATION_INTERNAL_PROCESS`,
same placement as `VoxelTerrain` uses. It costs nothing when a near terrain already made one.

Worth knowing this hides itself: with a `VoxelLodTerrain` sibling in the scene the updater already
exists and nothing looks wrong. It only bites when the node runs alone, which is exactly how you
will first test it.

**Guarded by** `tests/godot/step.gd`, which will not settle unless `tasks_in_flight` reaches zero.

### 3.5 Sectors hidden under the near terrain went missing for good

The worst of the three, and invisible until the viewer moves.

**The trap.** A sector lying entirely within `near_clip_radius` is never drawn, and because rings
tile rather than nest, nothing else is built from it either — so generating it is pure waste, and
the most expensive kind, since these are the finest levels. `schedule_loads()` therefore skipped it.

But `FarLodTree::update()` had already recorded it as **resident**. Residency is what suppresses a
reload request. So the sector sat marked as present while nothing had ever built it, and the moment
the viewer moved and it came out from under the near terrain it was never offered again. A hole
that only appears after moving, never heals, and does not exist at the position you started from.

Observed directly: 475 resident, 443 rendered, and the gap never closed.

**The fix.** Residency now means "built or building", never "asked for". `FarLodTree` withholds
clipped sectors in a separate `_deferred` set and re-offers them the moment they become visible;
the node builds everything it is handed. One authority instead of two disagreeing ones.

**Guarded by** `test_near_clip_withholds_sectors` (nothing offered may be clipped) and
`test_withheld_sectors_are_built_when_they_come_into_view` (walks the viewer and checks every
required, unclipped sector is built). Both fail against the old code. `tests/godot/move.gd` is the
end-to-end version: it walks the viewer and asserts `resident == rendered` at every step.

### 3.6 Fog mixed into ALBEDO gets lit, so the horizon can never meet the sky

The shader mixed the fog colour into `albedo` and assigned the result to `ALBEDO`. That is then
multiplied by sun and ambient, so a fully fogged surface comes out as the fog colour *lit* — never
the fog colour itself. Terrain relief and sun shading stayed visible through solid fog, and the far
field sat on top of the sky as a brighter, hard-edged cutout instead of blending into it.

The shader's own comment describes exactly this failure ("leaving the surface fully lit under dense
fog is what makes distant terrain look like a cardboard cutout pasted over the sky") — the code
implementing it was simply never written.

**The fix.** Take the fog out of the lit path: fade `ALBEDO` toward black by the fog amount and
drive the same amount through `EMISSION`, so at `total_fog == 1` the fragment is exactly
`mixed_fog_color`. `ROUGHNESS` and `SPECULAR` follow.

**How to see it.** Set `fog_color` to something saturated and look at the far band: it must be flat
and featureless. If relief shows through, the fog is being lit again.

Two things made this hard to spot with sensible settings. Snow-capped terrain against pale blue fog
barely changes colour, and from a camera at ground level everything past a few kilometres is
compressed into the top few pixels of the screen. Use a high camera and a violent fog colour when
checking this — `tests/godot/shot.gd -- out.png fog 9000 -14`.

### 3.7 `clear_cache()` before the first frame did nothing

Minor, but it makes a cold-versus-warm measurement lie, which is the one thing the cache needs you
to be able to measure.

`clear_cache()` early-returned when `_cache` was null. The cache is only opened by `restart()`,
which does not run until the node's first `process()` — so calling it from a setup script, right
after adding the node to the tree, silently did nothing and then loaded the very files it had been
asked to delete. It now defers the clear to the restart that opens the cache.

Caught because the "cold" run reported 208 hits and 0 saves.

---

### 3.8 VoxelLodTerrain's viewer position drifts without bound when the node is moved

Not this code's bug, but it will waste a day if you hit it, and the far field is what exposed it.

`VoxelLodTerrain::get_local_viewer_pos()` defaults to
`state.lods[0].last_viewer_data_block_pos << get_data_block_size_pow2()` when no `VoxelViewer`
exists, and then transforms that through `get_global_transform().affine_inverse()`. But
`last_viewer_data_block_pos` is **already in local space**, so with a non-identity terrain transform
the offset is subtracted again every frame. The viewer runs away from the terrain without bound.

It is invisible while the terrain sits at the origin, which is why every stage but the moving one
passed. Moving the terrain node with no viewer in the scene made the far field thrash: ~750 sectors
resident, 97,000 tasks in flight, and the same sectors built 11,000 times over.

**Not fixed here** — it is upstream code, it also affects near streaming, and the right fix depends
on whether that fallback is meant to be local or global. Work around it by putting a real
`VoxelViewer` in the scene, which any real project has anyway. `tests/godot/far_check.gd` does.

---

## 4. What the Distant Horizons zip did and didn't contain

**The LOD engine is not in it.** `coreSubProjects` is a git submodule
(`https://gitlab.com/jeseibel/distant-horizons-core.git`) and it is empty in the archive. What
shipped is the Minecraft platform layer — 298 Java files of OpenGL rendering, mixins and wrappers.
The column format, the quadtree, the meshing and the world-gen queues are all in the missing
submodule.

So the data structures here remain independently derived. The rendering layer *was* present, and it
is where the first pass was weakest.

### 4.1 Fog was the biggest miss

The first shader had one distance smoothstep into a horizon colour. Theirs runs **two independent
fog systems** — distance and height — each with its own start, length, density, min/max clamp and
falloff curve, plus an explicit **mixing mode**, and a spherical-vs-cylindrical choice for the
distance term. All applied as a screen-space pass over the depth buffer.

That looks like over-engineering until you work out why it exists. Past ~10 km, fog keyed on range
alone gives a mountain 40 km out the same colour as the valley floor 40 km out. Every depth cue
dies and the scene flattens into a painted backdrop.

**Height fog is what restores it**, because it varies with altitude instead of range — haze pools in
valleys, peaks stand clear, and distant ranges stack in progressively paler layers on their own.
The mixing mode carries real weight too: `MAX` is what puts crisp peaks above a fog layer, `ADD`
compounds for heavy weather, `MULTIPLY` lets height fog thin the distance fog for clear
high-altitude air.

Observed uniform set on their fog shader, which is effectively the spec:
`uFarFogStart/Length/Range/Min/Density/FalloffType`, `uUseSphericalFog`,
`uHeightFogBaseHeight/Length/Range/Min/Density/FalloffType`, `uHeightFogAppliesUp/Down`,
`uHeightFogMixingMode`, `uHeightBasedOnCamera`, `uFogScale`, `uFogVerticalScale`.

All of this is now implemented in `far_terrain.gdshader` as a per-fragment model (Godot puts near
and far terrain in the same depth buffer, so a separate composite pass isn't needed), and both
systems have been seen working: haze pools below the peaks, ranges stack in paler bands, and the
horizon blends into the sky instead of ending at it. Getting there needed §3.6 — fog has to be kept
out of the lit path or none of this reads as atmosphere.

### 4.2 Banded noise stops terrain looking decimated

Their noise is driven by `intensity`, `steps` and a distance `dropoff`. The `steps` part is the
insight: the noise is **quantised into bands**, not smooth. Banded noise reads as strata and
material variation; smooth noise just looks like a dirty surface. It restores the *impression* of
detail on large flat facets for a few ALU ops — no geometry, no texture fetch — and fades out near
the camera where real geometry already carries detail.

### 4.3 Micro depth offset beats depth bias

A fixed ~0.01-block world-space nudge away from the camera (`uMircoOffset`, their typo), so near
terrain wins wherever the two surfaces coincide. Steadier than depth bias, which varies with depth
precision — and across a 200 km range that varies enormously.

### 4.4 Temporal dither, not static

They rotate the dither pattern per frame (`uFrameMod8`) and run TAA. A static Bayer pattern is a
visible fixed grid; rotating it converts the artifact into temporal noise that TAA resolves into a
smooth gradient. Now driven from `TIME`, so no CPU plumbing.

### 4.5 Taken, or noted for later

- **Planet curvature** (`uEarthRadius`). Bends distant terrain down over a sphere. At tens of km a
  flat world shows a hard straight horizon with sky beneath it; curving it away hides that edge.
  Implemented as `planet_radius`, off by default. One multiply-add per vertex — the cheapest
  "this world is enormous" cue there is.
- **SSAO and far-fade as screen-space post-passes** over a depth texture. Not needed in Godot:
  SSAO comes from `WorldEnvironment`, and the near/far handover works off the shared depth buffer
  plus the micro offset. Their compositing exists because Minecraft's renderer is a separate pass.
- **TAA.** Godot has it built in; pair it with the temporal dither.
- **16-byte vertices** — 16-bit positions, byte colour/light, normal as an *index*. The normal index
  only works in a cube world with six normals, so octahedral stays here. But the principle applies:
  `ARRAY_FLAG_COMPRESS_ATTRIBUTES` is now set on every surface, safe precisely because vertices are
  sector-local rather than world-space.

---

## 5. Going forward

### Step 0 — get the full repo — **done (2026-09-21)**

`Distant Horizon Source/coreSubProjects/` is populated (561 Java files, cloned from
`https://gitlab.com/jeseibel/distant-horizons-core.git`; the zip's submodule was empty).

The interesting tree is `coreSubProjects/core/src/main/java/com/seibel/distanthorizons/core/`.

### Step 1 — build the module — **done (2026-09-21)**

```
godot/modules/voxel/        # github.com/Zylann/godot_voxel
godot/modules/voxel/far/    # this
```

Builds into Godot 4.6.1 with `build_eden.bat` (MSVC 2022, `platform=windows target=editor`).
`config.py` returns `can_build() == False` if `modules/voxel` is missing, so a tree without it still
compiles.

The eight predicted breakages below all turned out to be **correct as written** — `godot_voxel`'s
API was read accurately. Only three things actually failed:

1. `SCsub` had one `os.path.dirname` too many when locating `modules/voxel`, so the module
   disabled itself. Now `os.path.dirname(Dir(".").srcnode().abspath)`.
2. `register_types.h` included `core/object/class_db.h`, which does not declare
   `ModuleInitializationLevel`. That lives in `modules/register_module_types.h`.
3. `far_cache.cpp` called two helpers that do not exist in `zylann::godot`:
   `file_exists()` → `FileAccess::exists()`, and `erase_directory_contents_recursive(dir)` →
   `dir->erase_contents_recursive()`.

Predicted-but-fine, kept for the record: `Mesh::ARRAY_FLAG_COMPRESS_ATTRIBUTES`,
`add_surface_from_arrays` arity, the shadow-casting enum namespacing, `String::num_int64`,
`open_file`/`get_buffer`/`store_buffer`, `VoxelEngine::for_each_viewer`, `ZN_NEW`, and the
`Dictionary`/`AABB`/`Transform3D` includes.

### Step 2 — first light — **done (2026-09-21)**

All six stages pass, automated in `tests/godot/` (see its README). Keep them: each isolates one
subsystem, and stages 1, 4 and 5 each turned up a bug.

1. **One sector, no LOD.** 16 sectors (not 9 — ring snapping widens `ring_radius = 1` to 4×4),
   25,616 vertices, settles in 7 frames. Bounds `x,z ∈ ±4096` — exactly 4 × 2048-unit level-6
   sectors — and against `VoxelGeneratorFlat(height = 40)` the surface lands at y = 40 with the
   skirt at y = −88, which is 2 cells × 64 units below it. Right shape, right place, right way up.
   **Found §3.4**: tasks completed but were never destroyed.
2. **One level, full ring.** 100 sectors, 230,400 triangles, no cracks.
3. **Multiple levels.** `first_lod = 4`, `lod_count = 6`: 475 sectors, 1.09 M triangles, reaching
   ±81,920 units, against a `max_resident_bound` of 1014. From directly overhead the levels show as
   concentric square bands of decreasing detail with no gaps and no z-fighting — §3.2 holds.
4. **Near clip.** **Found §3.5**, the serious one: resident sectors that nothing had built.
5. **The shader.** Compiled first try, no typos. **Found §3.6**: fog was being lit.
6. **Cache on.** Cold run 208 saves / 0 hits / 2,948,816 bytes written, settling in 41 frames; warm
   run 208 hits / 0 saves / the same byte count read, settling in 8. Round-tripped heights differ in
   the third decimal, which is the quantisation error §7's serialisation test already bounds.

**The handover, against a real `VoxelLodTerrain`** (`tests/godot/handover.gd`). Both nodes sharing
one generator, a `VoxelViewer` 7 km from the origin, near view distance 1024, `near_clip_radius` 960.

- The far field **follows the `VoxelViewer`**, not the node's own position — so the fallback in
  `update_viewer_position()` is not masking a broken path.
- Surfaces **line up across the seam**: no ring of missing world, no double-drawn ground, and no
  z-fighting where the two coincide, so §4.3's micro depth offset is doing its job.
- The clip withholds 4 sectors here (471 resident against 475 with the clip off). Note that
  `clipped_sectors` stays 0 for a stationary viewer — withheld sectors are never built, so there is
  nothing to hide. Rebuild with the clip off and diff the counts instead, which is what the script
  does.
- **What does not look right is the fade band**, and it is a materials problem, not a fade problem:
  see §5 Step 5 item 1.
- `VoxelLodTerrain` cannot run under `--headless`; its meshes hit the dummy renderer and it fails
  with "Attempting to initialize the wrong RID". This module is fine headless. Run `handover.gd`
  windowed.

### Step 3 — tuning order

`first_lod` is the setting that matters most. Every level whose ring fits inside the near terrain's
radius is generated and then never drawn, and the fine levels are by far the most expensive — a
level-0 sector spanning the full vertical range needs 64× the Y samples of a level-6 one.

Rule of thumb: pick `first_lod` so that
`ring_radius × 32 × 2^first_lod` lands just under the near view distance.

Then, in order: `vertical_min`/`vertical_max` (tight to your actual world), `lod_count` for reach,
`antialias` off if generation is too slow, `generate_bottoms` off if your world has few overhangs.

### Step 4 — read against the core submodule — **done (2026-09-21)**

Answers to the questions this section used to ask, from
`coreSubProjects/core/src/main/java/com/seibel/distanthorizons/core/`. Read for *why*, not for
lines — see §7 Licensing.

The headline: **the data-format and LOD decisions here hold up.** Two are directly validated by what
they do differently and why. The places they are ahead are storage, layer eviction, and seams.

#### Data format — one 64-bit long per layer

`util/RenderDataPointUtil.java` packs a layer into a single `long`:

| Field | Bits |
|---|---|
| block material id | 4 |
| colour A / R / G / B | 4 + 8 + 8 + 8 |
| max Y (top) | 12 |
| min Y (bottom) | 12 |
| block light | 4 |
| sky light | 4 |

**No normal, anywhere.** §2.2's claim was that storing one is essential for SDF and pointless in a
cube world, and that is exactly the split: they spend those bits on a baked colour and two light
values instead. Nothing here suggests changing the octahedral normal.

Heights are 12-bit **absolute** integers against a world range of `[-2032, 2031]` plus height
(`MAX_WORLD_Y_SIZE = 6095`), not relative to a region's own Y extent as here. Absolute is simpler and
they can afford it because Minecraft's world height is fixed and small. A Godot world has no such
bound, so the relative quantisation here stays.

Baked per-layer lighting is the same trick as baking AO here, taken further.

#### Layer overflow — their eviction policy is better than ours

`util/RenderDataPointReducingList.java`. Nodes live in two parallel `long[]`s and are kept in **two
sort orders at once** — lowest-to-highest and smallest-to-biggest — via four 16-bit links packed
into one long (bigger, smaller, higher, lower). `reduce(target)` is then a cascade:

1. `mergeVerySmallConnectedSegments`
2. `mergeConnectedSegments`
3. `removeLeastImportantSegments`
4. `forceBottomToMerge`

This module caps at 8 and merges the closest-gap pair, which is their step 2 alone. **The idea worth
taking is the size ordering.** Merging by closest gap can annihilate a thin but tall feature that
happens to sit near its neighbour, while sparing a fat one; merging the *smallest* segment first
targets what is least visually significant. Changing `far_column_data.cpp` to pick the smallest span
rather than the closest gap is a small change with a real quality argument behind it, and
`test_column_with_too_many_spans_degrades_gracefully` already exists to extend.

Their two-order intrusive list is what makes each stage cheap at their layer counts. At a cap of 8 it
would be over-engineering here — a linear scan over 8 spans is nothing.

#### Seams — they stitch, and they do not skirt

There is no "skirt" in the core submodule at all. `dataObjects/render/bufferBuilding/ColumnBox.java`
takes `ColumnRenderView[] adjData` — the four compass neighbours — plus a parallel
`boolean[] isAdjDataSameDetailLevel`, and builds side faces by walking the neighbour's spans
(`makeAdjVerticalQuad`). A null neighbour, meaning "not generated yet", emits a full side face.

So the answer to "how do they handle a neighbour whose level changes while meshing is in flight" is:
they accept a temporarily visible wall rather than a hole, which is the same bargain a skirt makes —
but they pay for it by having the mesher **read its neighbours**. That forfeits §2.2's zero-
synchronisation property, which is the single thing that lets every sector here generate and mesh on
its own thread with no coordination at all. Given that, exact stitching is not worth adopting: it
would buy a slightly better-looking boundary in exchange for the module's cleanest invariant.

They have a second mechanism worth knowing about, and it is the one to steal if boundaries ever
misbehave — a deliberate **border overdraw** rule (`ColumnBox.java`, the `onBorder` test):

> The following logic is done to provide a little bit of overdraw to prevent holes when low detail
> LODs are replaced by higher-detail ones when moving. If not done higher quality LODs can cause
> holes due to not covering the whole face like the lower detail LODs they replaced.

That is a transition artifact, not a static one, and it is precisely the failure mode skirts here are
meant to cover. Their fix is "do not cull faces on a section border"; ours is `skirt_depth_cells`.

#### LOD management — quadtree, and section size

`DhSectionPos.SECTION_MINIMUM_DETAIL_LEVEL = 6` and `ColumnRenderSource.WIDTH = 64`: sections are
**64×64 columns**, not the 512×512 this section previously guessed. Against 33×33 here that is 2×
wider and 4× the area — the same order, so nothing argues for changing `SECTOR_CELLS`.

They do use a quadtree (`render/QuadTree/LodQuadTree.java`), and §2.3's async split/merge
consistency problem is real for them: they carry a `sortedMissingPosList`, gate regeneration on
whether any high-detail request is outstanding (`setCanRegenerate`), and hold per-node tick state in
`QuadTreeTickNodeHolder`. That is the machinery rings exist to avoid, and none of it looks like a
clean answer worth importing. **Keep the rings.**

#### Downsampling — a union of spans, not a filter

`FullDataSourceV2.updateFromOneBelowDetailLevel` → `mergeInputTwoByTwoDataColumn`: it collects every
Y transition across the four child columns into a `yTransitions` list, then emits one datapoint per
resulting band. It is a **union with a per-band representative**, with no averaging — because you
cannot average a block id.

This module reduces with a 3×3 tent filter and point-samples the shared border ring. That is the
right answer for a continuous height field and theirs is the right answer for a discrete one;
neither transfers. Another point for the format, not a change to make.

The other direction, `downsampleFromOneAboveDetailLevel`, is a plain nearest-neighbour replicate
(one coarse column copied into a 2×2 fine block), tagged `EDhApiWorldGenerationStep.DOWN_SAMPLED` so
it can be replaced once real data arrives. A placeholder, not a quality path.

#### Propagation — lazy and driven from the database

Coarse levels are not rebuilt eagerly. Rows carry `ApplyToParent` / `ApplyToChildren` dirty bits
(with an index on `ApplyToParent`, migration `0050`), and `FullDataUpdatePropagatorV2` walks them.
A merge decides whether anything changed by comparing datapoints **one by one**, with an explicit
note that a hash "has too high a chance of showing different data as the same" — and propagation
stops the moment nothing changed. That bounded cascade is what makes lazy propagation affordable.

Nothing to do here yet: this module regenerates each level independently from the generator, which is
correct for a world with no edits. It becomes the reference the day terrain becomes editable.

#### Storage — adopt this

`core/src/main/resources/sqlScripts/0020-sqlite-createFullDataSourceV2Tables.sql`:

```sql
CREATE TABLE FullData (
     DetailLevel TINYINT NOT NULL
    ,PosX INT NOT NULL
    ,PosZ INT NOT NULL
    ,MinY INT NOT NULL
    ,DataChecksum INT NOT NULL
    ,Data BLOB NULL
    ,ColumnGenerationStep BLOB NULL
    ,ColumnWorldCompressionMode BLOB NULL
    ,Mapping BLOB NULL
    ,DataFormatVersion TINYINT NULL
    ,CompressionMode TINYINT NULL
    ,ApplyToParent BIT NULL
    ,LastModifiedUnixDateTime BIGINT NOT NULL
    ,CreatedUnixDateTime BIGINT NOT NULL
    ,PRIMARY KEY (DetailLevel, PosX, PosZ)
);
```

Plus `pragma journal_mode = WAL; pragma synchronous = NORMAL;` (migration `0031`) and a numbered
migration runner (`sql/DatabaseUpdater.java`, scripts `0010`–`0090`).

This is strictly better than `far_cache.cpp` and it is the clearest thing to take from the whole
submodule:

- **One row per sector, no region grouping.** The in-place-if-the-size-matches / append-and-mark-waste
  / compact-later machinery in `far_cache.cpp` — the untested part flagged in §6 — simply stops
  existing. SQLite is the thing that already solved it.
- **`DataFormatVersion` and `CompressionMode` per row.** A format bump degrades to a miss on the
  affected rows instead of invalidating the file.
- **`DataChecksum` per row**, rather than trusting the blob.
- **WAL**, so a crash mid-write cannot corrupt the store — which the current buffered-in-memory
  compaction very much can.

`godot_voxel` already bundles SQLite, so this costs a dependency of zero.

#### Overdraw and transparency

Far-versus-near occlusion is not a separate system for them the way §4.5 assumed: the cull happens
per column against adjacent data (`FullDataOcclusionCuller`, and the `caveCullingMaxY` test in
`ColumnBox`), which is finer-grained than this module's radius test plus dithered fade, but is also
inseparable from their block model.

Water and transparency are carried in the datapoint's own alpha (`ColorUtil.getAlpha(...) < 255`)
under an `EDhApiTransparency` setting, with adjacent-transparent faces taking their light from the
span below. There is no separate water type: transparency is a property of a layer. That is a
lighter design than the "separate span type plus a transparent pass" §5 assumes, and it is probably
the right one here too — a flag on `ColumnLayer` plus a second surface on the mesh, rather than a
parallel field.

#### What this changes, in order

1. **Replace the region-file cache with SQLite.** The one clear win. Deletes the riskiest untested
   code in the module (§6) and costs no new dependency.
2. **Merge the smallest span, not the closest-gap pair,** on layer overflow.
3. **Water as an alpha flag on a layer**, not a separate span type, when §5's water item comes up.
4. **Leave the mesher, the rings, the tent filter and the stored normal alone.** Each is a
   considered difference that the comparison supports rather than undermines.

### Step 5 — roadmap beyond that

Reordered after Step 4 and first light.

1. **Match the near terrain's material.** Promoted to the top by the handover shot: the two surfaces
   line up geometrically, but the dithered fade band is only invisible when near and far shade
   alike. With a plain grey near terrain against the far shader's height and slope colouring, the
   fade reads as a hatched patch of one material laid over the other — not a fade. Nothing is wrong
   with the fade; it has no chance until the colours agree. This is the same work as the palette
   item below, which is why they have merged.
2. **Material palette.** Currently one index tinted by height and slope in the shader. A palette
   texture plus per-layer blend weights would make it look like a real world — and is what lets (1)
   actually match.
3. **SQLite cache.** Confirmed worth adopting: see Step 4, *Storage*. This also deletes the module's
   riskiest untested code, the region-file rewrite and compaction path in §6.
4. **Water**, as an alpha flag on a layer plus a second mesh surface — not a separate span type.
   Step 4, *Overdraw and transparency*, argues for the lighter design.
5. **Smallest-span-first layer eviction** instead of closest-gap. Small change, real quality
   argument; Step 4, *Layer overflow*.
6. ~~**Exact LOD stitching.**~~ Dropped. They stitch by reading the four neighbouring columns, which
   costs the zero-synchronisation property in §2.2 — the module's cleanest invariant — for a
   marginally better boundary. If boundaries do misbehave during LOD transitions, take their
   border-overdraw rule instead; it is cheaper and keeps the invariant.
7. **Blocky variant.** The column format suits cube worlds even better, but the mesher would want
   terraced caps and a slope-driven wall rule — the *opposite* of the SDF choice in §3.1. Best done
   as a second mesher behind the same data format, not by parameterising this one.

---

## 6. Risk areas in this code

Where to look first if something misbehaves. Honest assessment, most to least likely.

| Area | File | Why it is risky |
|---|---|---|
| Generator LOD convention | `far_tasks.cpp` | `origin_in_voxels` is LOD-0 world coords with stride `1<<lod`. Verified against `VoxelGeneratorFlat`, but a custom generator could interpret it differently and you would get wrong-scale terrain. |
| Vertical range snapping | `far_tasks.cpp` | `min_y` is snapped to the sample stride so levels stay registered. Get this wrong and levels disagree vertically. |
| Region file rewrites | `far_cache.cpp` | In-place rewrite only when the blob size matches exactly; otherwise append and mark waste. Compaction is buffered in memory. Exercised now (2.9 MB over 16 regions, byte-exact on read-back), but never with a blob that changed size, so the append-and-compact path is still untravelled. |
| Task lifetime | `far_tasks.cpp` / node | Tasks hold `shared_ptr<FarSharedState>`, never a node pointer, and check a generation counter. Now run under real concurrency, including a viewer that moves and cancels work in flight; `stale_results_discarded` stayed at 0 throughout. But see §3.4 — nothing deleted a finished task at all until `VoxelEngineUpdater` was wired in, so this path is younger than it looks. |
| Mesh upload budget | `far_renderer.cpp` | `far_max_uploads_per_frame` trades streaming speed against frame time. Default 4 is a guess. |
| Shader `inverse(MODEL_MATRIX)` | `far_terrain.gdshader` | One 4×4 inversion per vertex when curvature or micro offset is on. Fine at this vertex count, but if profiling says otherwise, pass the inverse as a uniform. |

---

## 7. Reference

### Tests

On Windows, `tests\run_tests.bat` builds and runs both suites under MSVC + ASan (no UBSan there).
Last green run: 26,025 + 10,751 checks, 0 failures.

`tests/godot/` holds the engine-level checks that need a built binary — the task pipeline, the
cache, the node and the shader. Its README has the invocations.

Elsewhere:

```sh
cd modules/voxel/far

g++ -std=c++17 -O1 -g -fsanitize=address,undefined \
    tests/test_far_core.cpp core/far_column_data.cpp core/far_extract.cpp core/far_mesher.cpp \
    -o /tmp/test_far && /tmp/test_far

g++ -std=c++17 -O1 -g -fsanitize=address,undefined \
    tests/test_far_lod_tree.cpp core/far_column_data.cpp core/far_lod_tree.cpp \
    -o /tmp/test_lod && /tmp/test_lod
```

Run these before and after touching anything in `core/`. They take seconds and they are the reason
the two bugs in §3 were caught at all.

### Files

| File | Role |
|---|---|
| `core/far_math.h` | Vectors, floor-division, octahedral normal packing |
| `core/far_column_data.*` | Column format, CSR storage, quantised serialisation |
| `core/far_extract.*` | SDF → columns, AO baking, downsampling |
| `core/far_lod_tree.*` | Which sectors exist at which LOD, flat rings or sphere rings |
| `core/far_sphere.*` | Cubesphere: face mapping, surface frames, crossing a face edge |
| `core/far_geometry.*` | Where a sector's columns are in the world. The only place that knows flat from planetary |
| `core/far_mesher.*` | Sheet labelling, caps, walls, skirts |
| `tasks/far_cache.*` | Region-file persistence |
| `tasks/far_tasks.*` | Threaded pipeline |
| `far_renderer.*` | Main-thread mesh instances for far sectors |
| `../terrain/variable_lod/voxel_lod_terrain_update_far_streaming.*` | The streaming system itself |
| `shaders/far_terrain.gdshader` | Surface shader, fog, noise, curvature |
| `tests/run_tests.bat` | Builds and runs both host suites under MSVC + ASan |
| `tests/godot/` | Engine-level checks: streaming, cache, clipping, shader. See its README |

### Key constants

| Constant | Value | Where |
|---|---|---|
| `SECTOR_CELLS` | 32 | `far_column_data.h` |
| `SECTOR_RES` | 33 (padded point grid) | `far_column_data.h` |
| `MAX_LAYERS_PER_COLUMN` | 8 | `far_column_data.h` |
| `max_connect_gap_cells` | 64 (a void guard, **not** a slope limit) | `far_mesher.h` |
| `skirt_depth_cells` | 2.0 | `far_mesher.h` |
| `COMPACT_WASTE_RATIO` | 0.4 | `far_cache.h` |

### Licensing

Distant Horizons is LGPL-3.0. A line-by-line translation would be a derivative work and the licence
would follow it — a bad outcome for a Godot **engine module** specifically, since modules link
statically into the engine binary and LGPL-3.0's relinking requirement means shipping object files
or a relink mechanism with your game.

Techniques and architecture are not copyrightable; only expression is. Everything in `far/` is
written from scratch, so it is yours to license however you want.

Keep it that way when you have the submodule in front of you: read it for **why** a thing is done,
then implement it your own way. The fog model in §4.1 is a good example — the valuable part was the
insight that height fog is what restores depth perception at range, not any particular line of GLSL.

---

## 8. Inside VoxelLodTerrain (2026-09-21)

### 8.1 What moved, and why the companion rule was dropped

The far field was a separate module with its own node, on the rule in §1: public APIs only, no fork,
so upstream `godot_voxel` could move without rebasing. That rule bought nothing here — this tree's
`modules/voxel` is a fork — and it cost a real bug (§3.4) plus a setting that had to be kept in step
by hand (`near_clip_radius` against `view_distance`).

```
modules/voxel/far/core/          engine-agnostic, no Godot headers (unchanged)
modules/voxel/far/tasks/         threaded pipeline + region cache (unchanged)
modules/voxel/far/far_renderer.* main-thread mesh instances          <- was in the node
modules/voxel/terrain/variable_lod/voxel_lod_terrain_update_far_streaming.*   <- new
```

`modules/voxel_far/` and `VoxelFarTerrain` are gone. `far/core/` keeps its no-Godot-headers rule and
still builds standalone against `far/tests/` — that property is worth more than its directory
location, and §1 explains why.

### 8.2 It is a peer of octree and clipbox, not an alternative

`VoxelLodTerrainUpdateData::State` now holds `far_streaming` beside `octree_streaming` and
`clipbox_streaming`, and `process_update` calls `process_far_streaming()` **after** whichever near
system is selected, unconditionally:

```cpp
if (settings.streaming_system == STREAMING_SYSTEM_LEGACY_OCTREE) {
    process_octree_streaming(...);
} else {
    process_clipbox_streaming(...);
}
// Runs in addition, not instead: it covers what is past view_distance,
// which neither near system looks at.
process_far_streaming(state, settings, _viewer_pos, generator);
```

That is the whole shape. The far field streams *sectors*, not voxels, so it competes with nothing:
`far_enabled` can be toggled without touching near streaming, and either near system works under it.

It inherits the update task, the viewer positions, the generator and the volume transform. The
threading split is the one VoxelLodTerrain already uses — selection on the update thread, mesh
building on the main thread in `far::FarRenderer`.

### 8.3 The setting that stopped being a trap

`far_near_clip_radius` now defaults to 0, meaning **derive from `view_distance`**:

```cpp
constexpr float NEAR_EDGE_MARGIN = 0.94f;
return static_cast<float>(settings.view_distance_voxels) * NEAR_EDGE_MARGIN;
```

Measured: `view_distance = 1024` gives a clip of 962.56 and withholds 4 sectors, with no number
typed anywhere. Getting this wrong by hand gave either a ring of ground neither terrain drew or a
band both drew. Positive is still an explicit radius; negative turns clipping off, which is only
useful when looking at the far field on its own.

### 8.4 Residency is reconciled, not replayed

The first cut had the update thread push unloaded sector ids to a queue that the main thread
drained. That is wrong for the same reason §3.5 was wrong: a sector can be dropped and re-required
between two main-thread frames, and replaying the drop destroys a mesh that is resident again —
which nothing rebuilds, because residency is what suppresses a reload request.

`FarRenderer::reconcile()` instead walks its own sectors each frame and destroys any the LOD tree no
longer holds resident. A pure function of current state, like the required set in §2.3, and it
deleted the queue and its mutex rather than fixing them.

### 8.5 Equivalence after the move

Same generator, same settings, before and after — the numbers are identical, which is the useful
property to check after a move this size.

| Stage | Sectors | Vertices | Bounds |
|---|---|---|---|
| One sector, LOD 6, ring 1 | 16 | 25,616 | x,z ±4096, y −88..40 |
| Full ring, LOD 6, ring 4 | 100 | 160,100 | — |
| Six levels from LOD 4 | 475 | 760,475 | — |
| Cache cold | 208 saves, 0 hits | 2,948,816 B written | — |
| Cache warm | 0 saves, 208 hits | 2,948,816 B read | — |

Moving viewer: `missing = 0` at every step, `stale = 0`, `clipped` rising 0 → 17 as it advances.
Handover with a near `VoxelMesherTransvoxel`: 471 sectors, clip 962.56, surfaces meeting with no
hole and no z-fighting.

### 8.6 Testing got narrower, and one thing got worse

`far/tests/godot/far_check.gd` replaces the three old scripts: with no standalone node there is
always a `VoxelLodTerrain`, so the stages converged.

Everything still runs headless **as long as the near terrain has no mesher** — without one it builds
no meshes and so never touches the dummy renderer, which `VoxelLodTerrain` otherwise crashes on
("Attempting to initialize the wrong RID"). Only the `handover` stage needs a window, and only
because it deliberately draws the near terrain.

Put a real `VoxelViewer` in the scene. See §3.8 for what happens otherwise.

### 8.7 Open: none of this is planetary — **done, see §9**

The LOD structure question was answered "whichever works best for a planet in 3D space", and the
honest answer is that **neither rings nor an octree is the thing standing in the way** — the column
field itself is 2.5D:

- spans are measured along **Y**, from `vertical_min` to `vertical_max`
- sectors tile an **X/Z** grid
- `SectorId::get_world_x/z` and the mesher's caps, walls and skirts all assume a Y-up world

On a sphere there is no single up, and an X/Z grid does not tile one. Rings versus quadtree is a
detail by comparison.

What the format gets right is that **a column is direction-agnostic**: a short list of `(top,
bottom)` scalars along *some* axis, each carrying the normal captured at source resolution. Make the
axis **radial** and the format survives intact. Then the 2D parameterisation has to tile a sphere,
and this project already has that: `modules/eden_icosphere`'s `IcosphereMapper` gives
`dir_to_face_uv` / `face_uv_to_dir`, face neighbours, and edge adjacency with reversal flags — with
shared vertices along edges, which is exactly the §2.2 padding invariant one dimension up.

So the shape of planetary support is:

1. **Radial columns.** `vertical_min/max` become `radial_min/max` about a planet centre. `far_extract`
   samples along a ray from the centre instead of a Y column.
2. **Icosahedral tiles.** `SectorId(x, z, lod)` becomes `(face, u, v, lod)`. Rings become angular
   rings in face-local space, still a pure function of viewer direction — §2.3's argument survives
   the move to a sphere unchanged, and Step 4 found Distant Horizons paying the quadtree cost this
   avoids. Face crossings at icosahedron edges are the new work.
3. **A point-batch sampling path.** This is the blocker to check first. `far_extract` currently fills
   a `VoxelBuffer` via `generate_block`, which is an axis-aligned box. Radial columns need arbitrary
   points. `VoxelGenerator::generate_series(positions_x, positions_y, positions_z, ...)` is exactly
   that API — but `supports_series_generation()` is only implemented by `VoxelGeneratorGraph` and
   `VoxelGeneratorNoise2D`, and **`EdenPlanetGeneratorV4` does not implement it**. Either add it
   there (it already runs `_series` kernels internally) or fall back to `generate_single` per point,
   which is correct for any generator and slow.

Worth knowing before starting: on a 40 km planet with ~4 km of relief, the shell is about 10% of the
radius, so the column field's economics (§2.1) are better on a sphere than on a flat world, not
worse. Nothing found so far argues against the representation — only against its current axis.

---

## 9. Planets (2026-09-21)

Set `far_planet_radius` on the terrain and the far field wraps a sphere instead
of a plane. `far_vertical_min`/`max` become heights relative to that radius.

```gdscript
$VoxelLodTerrain.far_enabled       = true
$VoxelLodTerrain.far_planet_radius = 40000.0
$VoxelLodTerrain.far_vertical_min  = -1500.0   # relative to the radius
$VoxelLodTerrain.far_vertical_max  =  2500.0
```

### 9.1 Why the question was not rings vs quadtree

The LOD structure was never what stood in the way. Rings and quadtrees are both
2D schemes, and the far field's problem on a planet is that its *representation*
was 2.5D: spans measured along Y, sectors tiling an XZ grid, and a mesher whose
caps, walls and skirts all assumed Y-up. On a sphere there is no single up.

What saves it is that **a column is direction-agnostic**. It is a short list of
`(top, bottom)` scalars along *some* axis, each carrying the normal captured at
source resolution. Make the axis radial and the format survives untouched. So
what actually changed is the geometry, in one place:

| | flat | planet |
|---|---|---|
| column axis | +Y | radial from `planet_centre` |
| sector grid | unbounded XZ plane | one cube face, per-face grids |
| stored `top`/`bottom` | world Y | radius from the centre |
| sector world size | `32 * 2^lod` | arc length at the reference radius |

`far/core/far_geometry.h` holds that mapping and nothing else knows about it.
The column format, the extractor, the mesher, the cache and the ring selection
are the same code on both.

### 9.2 Cubesphere, not the icosphere

`modules/eden_icosphere` has a better-conditioned parameterisation and this does
not use it, deliberately. Its tiles are triangles with barycentric grids, and the
mesher walks a **square** grid of columns emitting quads from four corners.
Triangular tiles would mean rewriting it -- including the sheet-labelling rules
in §3.1, the part most expensive to get wrong and the part with the most tests
behind it. A cube face keeps the grid square and changes nothing but coordinates.

Two properties make it work:

**Faces meet exactly.** Both faces either side of a cube edge subdivide that edge
into the same angular steps, because the tangent warp is symmetric and depends
only on the parameter along the edge. A column on a shared edge is therefore at
the same world position from either side -- §2.2's padding invariant, carried onto
the sphere. Sectors still tile with no seam and meshing still never reads a
neighbour, so the zero-synchronisation property survives intact. Measured worst
disagreement along a shared edge: 8.4e-8 radians.

**The tangent warp is worth its tanf.** Spacing grid lines evenly on the face
*plane* makes corner cells subtend 2.07x the angle of centre cells; spacing them
evenly in *angle* brings that to 1.414 (which is sqrt(2), the known floor for a
warped cube face). Cell size is the unit the entire LOD scheme is expressed in,
so that ratio propagates into every distance decision.

`sphere_root_lod` ties the two numbering schemes together: it is chosen so LOD 0
cells are about one unit across, which keeps `first_lod` meaning what it means on
a flat world and carries the tuning advice in Step 3 over unchanged.

### 9.3 Sampling was the real blocker

A radial column is a line of arbitrary 3D points. `generate_block()` fills an
axis-aligned box and cannot express that, so §8.7 flagged this as the thing to
check first. It was right to.

- `generate_series()` takes arbitrary point batches, which is exactly the shape
  needed. `VoxelGeneratorGraph`, `VoxelGeneratorNoise2D` and — since this work —
  **`EdenPlanetGeneratorV4`** implement it.
- Everything else falls back to one `generate_block()` call per sample on a
  reused 1x1x1 buffer. Correct for any generator and roughly two orders of
  magnitude slower. (`VoxelGenerator::generate_single` does the same thing but
  allocates a `VoxelBuffer` *per sample*, which at ~270k samples per sector is
  not usable at all -- hence the reused buffer.)

**`EdenPlanetGeneratorV4` now implements it** (`eden_planet_generator_v4.cpp`).
It was about fifteen lines: `compute_heights()` already took point arrays, and
the SDF is the same `length(p) - planet_radius - height` that `generate_block`
uses. The only subtlety is that `Span` and `Vector3f` have to be written
`zylann::Span` / `zylann::Vector3f` there — Godot core has a `Span` of its own,
and the generator sits in the global namespace where the unqualified names
resolve to the wrong ones.

It is worth what it cost. On a 40 km planet, 534 sectors:

| | settles in |
|---|---|
| with `generate_series` | 234 frames, 1.8 s |
| without | **never** — 366 of 534 sectors after 1500 frames, 13.7 s |

Declaring support also enables detail normalmaps and slope instancing on V4,
which previously could not run at all.

Sampling is batched one row of columns at a time: the slab is column-major, so a
row of constant z is contiguous and can be written in place. Per-column batches
would multiply call overhead by 33; the whole sector at once would need a
multi-megabyte scratch per worker thread.

**Materials: see §9.6.** The first pass here concluded they could not be read on
the sphere path. That was half right and the diagnosis was wrong, which is worth
recording — the real problem was bigger and applied to the flat path too.

### 9.4 Two bugs, both found by looking at a frame

**Normals had to be rotated out of slab space.** The extractor takes its gradient
from index-space neighbours. On a flat world those axes *are* world X/Y/Z and the
gradient is already a world normal. On a sphere they are the column's surface
frame, so every normal needs rotating through it -- `rotate_stack_normals_to_world`.
Without it the terrain is lit from nowhere in particular.

**Four of the six faces were wound backwards.** This is §3.3 again, one dimension
up. Godot front faces are clockwise from the front, so a cap's right-hand normal
must point *away* from the surface. On a flat world grid u is +X, grid v is +Z,
the column axis is +Y, and X cross Z = -Y -- so every cube face must satisfy
`right x up = -normal`. Four of mine had `+normal`, so their caps were culled and
those faces rendered as holes.

It showed up as a black wedge in the first orbit render. Worth noting *why* the
tests missed it: coverage tests check which sectors are *selected*, and selection
was perfectly correct -- the sectors were there, they were just invisible.
`test_all_faces_share_the_flat_handedness` now pins it, and it fails against the
old axes.

### 9.5 What is verified

Host tests, all under ASan: 29,671 cubesphere checks and 16,058 LOD checks
(including planet mode), plus the 26,025 core checks unchanged from before, which
is the useful signal that the flat path did not move.

The properties worth knowing are pinned:

- every direction belongs to exactly one face, and round-trips
- adjacent faces agree along shared edges to 8.4e-8 rad
- the surface frame is orthonormal everywhere, including at face corners
- stepping off an edge lands on the right neighbour, and coming back returns the
  original sector
- **rings tile the sphere exactly once** -- 0 gaps and 0 overlaps over 4,000
  whole-sphere samples for three viewer positions, in the mixed configuration
  where fine levels are rings and a coarse level wraps the globe. Also checked
  with the viewer at a cube corner and on a cube edge, the awkward cases.
- residency stays bounded as the viewer orbits (845 of a 1014 bound)

End to end, a 40 km planet with 1400 m of relief: 534 sectors, 1.23 M triangles,
settled in 50 frames, world AABB +/-40,000 on every axis.

Against **`EdenPlanetGeneratorV4`**, the generator that actually matters: 534
sectors settling in 234 frames, materials 0..6 coming through, and the far
field's surface radius agreeing with the generator's own `sample_surface()` to
within 300 m on a 42 km radius. `far_check.gd -- planet_v4` checks that
automatically -- the two share `compute_heights()` but reach it by different
routes, so agreement means the series path is wired up correctly.

Also verified, and each of these found something:

- **Antialiasing on a sphere** (`planet_aa`): the four children are cube-face
  sub-patches and go through the same geometry builder.
- **An orbiting viewer** (`planet_move`): twelve positions around the planet,
  crossing every cube face and the corners between them, with no resident sector
  left unbuilt and nothing discarded as stale at any step.
- **Near and far together on a planet** (`planet_handover`): a `VoxelLodTerrain`
  with a Transvoxel mesher and the far field on one generator, meeting with no
  hole and no z-fighting. This is what turned up §9.7.
- **The flat path's MIXEL4 decode** (`flat_v4`): contrived on purpose — a flat
  far field over the top of V4's planet — because the simple generators write no
  material channel at all, so it is the only way to exercise that branch.
- **Materials through the cache** (`cache_v4`): cold run 144 saves, warm run 144
  hits, `max_material` 6 both times. A cache that dropped materials would look
  right on the first run and be wrong on every run after it. `open_regions` is 6,
  one per cube face, which also confirms the face is in the region key.

Renders: `tests/godot/planet_orbit.png`, `planet_surface.png`,
`planet_handover.png`.

### 9.6 Materials, and a wrong diagnosis worth recording

The far field read materials from `CHANNEL_TYPE`. **None of this project's planet
generators write that channel.** They write MIXEL4: four material indices and
four blend weights packed into `CHANNEL_INDICES`/`CHANNEL_WEIGHTS`. So every
layer got material 0 — not because of anything to do with spheres, but on the
flat path too, and since the beginning.

Fixed by teaching the far field both conventions (`choose_material_source()` in
`far_tasks.cpp`). MIXEL4 stores a blend and the far field stores one material per
layer, so the heaviest of the four is taken; at far-field cell sizes a blend is
not representable anyway.

On the sphere path materials cannot ride along with the SDF, because a series
call returns one channel. They are sampled separately with `generate_block` on a
reused 1x1x1 buffer — **but only at layer tops**, not at every sample. A sector
has ~1100 columns with one or two layers each against ~270k SDF samples, so this
is well under 1% of the work the sector already does.

One trap, and it cost a round trip. The first version sampled half a *cell* below
the surface. Generators commonly only compute real materials in a narrow band
around the surface and return a default elsewhere — V4's band is 28 units at
LOD 0 — and half a cell at LOD 6 is 32 units, just outside it. Every column read
back the default, and `max_material` came out as 1 (plain rock) everywhere
instead of 6. Sampling half a *voxel* down fixes it.

`get_far_statistics()["max_material"]` reports the highest index seen, which is
the quickest way to tell a working palette from a silently blank one.

The shader has a palette: `material_blend` and `material_colors[8]`, off by
default because a generator with no material channel reports 0 everywhere and
tinting the world with palette entry 0 looks like a bug.

**The palette is looked up in the vertex shader, not the fragment shader.** That
is the whole trick, and the two obvious alternatives are both wrong:

- *Interpolate the index and round in the fragment shader.* Produces a band of
  whatever material happens to lie numerically between the two along every edge
  where two materials meet. Reads as speckle — the same failure recorded against
  the biome material work.
- *Mark the index `flat`.* Correct, but a whole triangle then takes the provoking
  vertex's material and every material boundary becomes a hard facet. This was
  visibly bad at close range; `tests/godot/planet_surface_flat_before.png` is
  what it looked like.

Looking the index up per vertex, where it is exact, and interpolating the
resulting *colour* has neither problem. Transitions are smooth, nothing is ever
rounded, and the palette stays a uniform instead of being baked into the mesh.
It relies on the index surviving vertex compression: `ARRAY_FLAG_COMPRESS_ATTRIBUTES`
quantises colour to 8 bits and the index is written as `material / 255`, so it
round-trips exactly.

### 9.7 Clipping on a sphere is tangential, not 3D

The first sphere `is_clipped` measured 3D distance from the viewer to the
sector's four corners, with the corners placed at the reference radius — the
middle of the terrain's radial shell.

That never clips anything. On a shell a few thousand units thick, the gap between
that guessed radius and the real ground is far larger than a clip radius of a few
hundred, so the radial term swamps the comparison and the far field quietly draws
underneath the near terrain everywhere.

It now measures distance **along the surface**: the angle between the viewer's
direction and each corner, scaled by the reference radius. That has no dependency
on the guessed radius at all. It is also what the flat version has always done —
that one compares in XZ and ignores Y entirely.

Related, and a separate trap: **altitude eats the clip radius**. A viewer 2.4 km
above a planet whose near terrain reaches 1 km is not drawing ground at all, so
nothing should be clipped and the far field correctly covers everything. Easy to
mistake for a broken clip. `tests/godot/far_check.gd` stands the viewer 120 m
above the actual ground for this reason.

### 9.8 Downsampling voted on the wrong thing

A coarse column is reduced from up to nine fine ones by a 3x3 tent filter.
Heights are weight-averaged, normals are weight-blended, AO is weight-averaged —
and material was taken from **whichever span reached the cluster first**.

In raster order that is the corner of the neighbourhood rather than the bulk of
it, so a single stray column could repaint an entire coarse column and the
materials drifted with distance. Materials now vote by the same weight everything
else uses, keeping up to four candidates per cluster and folding the rest into
the current leader — the same shape as the existing normal blending, and four is
plenty when a far-field cell covering more than four distinct materials has no
single right answer anyway.

**Guarded by** `test_downsample_picks_the_dominant_material`, which checks both
that eight equal-weight columns outvote one that arrived first, and that one
heavily weighted column outvotes eight light ones. It fails against the old code.

### 9.9 Known gaps

- **A square ring around a cube corner names some sectors twice**, because only
  three faces meet there and the ring's four corner quadrants fold onto three.
  Deduplicated at the source rather than special-cased. Cheap and correct, but it
  means the sphere path allocates a small set per update where the flat path
  does not.
- **No sub-vertex material blending.** Each layer stores one material, so a
  vertex is 100% one thing; MIXEL4's four-way blend is reduced to its heaviest
  entry. Vertex-to-vertex interpolation covers most of the visual difference, and
  going further means widening `ColumnLayer::material` and bumping the cache
  format. Not obviously worth it.
- **`far_first_lod` has to be chosen against the planet, not just the view
  distance.** One far sector spans `(pi/2) * R / 2^(root_lod - lod)`, and if that
  is much larger than the near terrain's reach then no sector ever fits inside
  the clip radius and the clip never bites. On a 40 km planet with a 1 km near
  terrain that means `far_first_lod` around 4.
- **Water.** Still nothing, on either geometry. §5 item 4.
- **No SQLite cache yet** (§5 item 3, Step 4 *Storage*). The region-file path
  works on planets — face is in the key and the filename — but it is still the
  module's riskiest untested code.
