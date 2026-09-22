# far — horizon-scale LOD terrain

Renders smooth (SDF) terrain from the edge of `VoxelLodTerrain`'s view distance out to the horizon.

It is a **third streaming system** inside `VoxelLodTerrain`, alongside octree and clipbox streaming.
Those decide which voxel *blocks* a viewer needs out to `view_distance`; this one decides which far
*sectors* it needs past that, and runs in addition to whichever near system is selected. Turn it on
with `far_enabled` and it inherits the terrain's generator, viewers, transform and view distance.

It began life as a separate `voxel_far` module with its own `VoxelFarTerrain` node, deliberately
using public APIs only so upstream `godot_voxel` could move without rebasing. That is no longer
true; see HANDOFF.md §8 for what changed and why.

---

## Provenance and licensing — read this first

You gave me the repo zip and said it need not be a line-by-line copy. Good, because it cannot be:

**The zip does not contain the LOD engine.** `coreSubProjects` is a git submodule
(`distant-horizons-core.git`) and it is empty in the archive. What ships in the zip is the
Minecraft platform layer — 298 Java files of OpenGL rendering, mixins and MC wrappers. The column
format, the quadtree, the meshing, the world-gen queues: none of that is present.

So the data structures and algorithms here are still mine, arrived at independently. What the zip
*did* give me is the **rendering layer**, which is where my first pass was weakest, and I have used
those lessons — as techniques, reimplemented, not as transcribed code.

**On licensing.** Distant Horizons is LGPL-3.0. A line-by-line translation is a derivative work and
the licence follows it, which is a bad outcome for a Godot **engine module** specifically: modules
link statically into the engine binary, and LGPL-3.0's relinking requirement means shipping object
files or a relink mechanism with your game. Techniques and architecture are not copyrightable, only
expression. Everything here is written from scratch, so you own it outright and can license it as
you like.

## What the rendering layer taught me

Four things I got wrong or left out, now fixed.

### Fog is the whole game, and one distance term is not enough

My first shader had a single distance smoothstep into a horizon colour. Theirs has **two
independent fog systems** — distance and height — each with its own start, length, density, min/max
clamp and falloff curve, plus an explicit **mixing mode** for combining them, and a spherical vs
cylindrical choice for the distance term.

That is not over-engineering, and the reason only shows up past ~10 km: fog that depends on range
alone gives a mountain 40 km away the same colour as the valley floor 40 km away. Every depth cue
disappears and the scene flattens into a painted backdrop. **Height fog is what brings the depth
back**, because it varies with altitude instead of range — haze pools in valleys, peaks stand clear
of it, and distant ranges stack in progressively paler layers on their own.

The mixing mode carries real weight too. `MAX` keeps whichever term is denser, which is what
produces peaks above a fog layer; `ADD` compounds them for heavy weather; `MULTIPLY` lets height
fog thin the distance fog for clear high-altitude air.

The shader now implements all of this.

### Distant terrain needs noise to stop looking decimated

They drive procedural noise from `intensity`, `steps` and a distance `dropoff`. The `steps` part is
the insight — the noise is **quantised into bands**, not smooth. Banded noise reads as strata and
material variation; smooth noise just looks like a dirty surface. It restores the *impression* of
detail on large flat facets for a few ALU ops, no geometry and no texture fetch, and fades out near
the camera where real geometry already carries detail.

### A micro depth offset beats depth bias

A fixed ~0.01-block world-space nudge away from the camera, so near terrain wins wherever the two
surfaces coincide. Steadier than depth bias, which varies with depth precision — and across a 200 km
range that precision varies enormously.

### Temporal dither, not static dither

They rotate the dither pattern per frame (`frameMod8`) and run TAA. My static Bayer pattern is a
visible fixed grid; rotating it converts the artifact into temporal noise that TAA resolves into a
smooth gradient. The shader now rotates from `TIME`, so no CPU plumbing is needed.

### Also noted, and worth stealing later

- **Planet curvature** (`uEarthRadius`). Bends distant terrain down over a sphere. At tens of km a
  flat world shows a hard straight horizon with sky beneath it; curving it away hides that edge and
  reads as real scale. Implemented as `planet_radius`, off by default — it is the cheapest
  "this world is enormous" cue available, one multiply-add per vertex.
- **SSAO and the far fade as screen-space post-passes**, reconstructing position from a depth
  texture. In Godot you get both nearly free: SSAO from `WorldEnvironment`, and the near/far
  handover from the shared depth buffer plus the micro offset. Their compositing approach exists
  because Minecraft's renderer is a separate pass; a Godot module does not have that problem.
- **16-byte vertices** (16-bit positions, byte colour/light, normal as an *index*). The normal index
  only works in a cube world with six normals — SDF needs real ones, so octahedral stays. But the
  principle applies, and `ARRAY_FLAG_COMPRESS_ATTRIBUTES` is now set on every surface, which is safe
  here precisely because vertices are sector-local rather than world-space.

## The core idea, and why it works

A far-LOD system's real problem is not rendering, it is **representation**. Keeping voxels at a
coarse LOD costs `O(n³)`, and nearly all of that cube carries no information: it is empty above the
surface and solid below it. The only thing that varies is *where the surface is*, and that is a 2D
field.

So the far field is stored as a **multi-layer column field**: for each column in a 2D grid, a short
list of solid spans.

```
column (x, z):
    ┌─ top    = 412.7   normal = (0.12, 0.97, -0.21)   material = 3   ao = 0.82
    └─ bottom = 380.1
    ┌─ top    = 204.0        <- a cave ceiling
    └─ bottom = 196.5
    ┌─ top    = 150.2        <- the floor below it
    └─ bottom = -256.0       <- clipped by the sampled range, not a real surface
```

Typically one span per column, occasionally two to four where there are caves, overhangs or
floating geometry. The measured ratio against an equivalent dense voxel volume, from the test
suite, is **~10x smaller in memory and ~15x on disk** — and that is for eventful terrain at LOD 0.
It improves with altitude and coarseness.

Because each sector is a fixed 33×33 grid regardless of LOD, **every level costs the same to store
and mesh while covering four times the area**. Total cost grows logarithmically with view distance
instead of quadratically. That is the entire economic argument, and it is pinned down by a test.

## Adapting the idea to SDF terrain

This is where a direct port would have failed, and the three places it would have failed are worth
calling out because they are not obvious.

### 1. Layer boundaries are continuous, and carry normals

With cube voxels a span boundary lands on an integer block face. With an SDF it lands wherever the
field crosses zero, so `top`/`bottom` are interpolated, and **each layer stores the surface normal
sampled from the SDF gradient at that crossing**.

Storing the normal is the single most important quality decision in the format. Discard it and
recompute normals from the coarse heights at mesh time, and distant terrain reads as faceted
low-poly scenery — the only gradient information left is the difference between samples `1 << lod`
units apart. Keeping the normal from the resolution the data was *captured* at preserves the
original surface's shading after the positions have been decimated hard. It is also what makes a
near-vertical cliff still read as a cliff after its geometry has collapsed to a single quad.

### 2. Steepness must not decide where walls go

The natural design is to connect gentle neighbours into a surface and draw a vertical wall wherever
the slope exceeds a threshold. That is correct for cube worlds, where every surface really is
either flat or a cliff.

Applied to an SDF it fails, and **not gracefully**. On a *sustained* steep slope no adjacent pair
connects, so every span becomes a sheet of its own, no cell ever has four corners agreeing on a
sheet, and the mesher emits nothing at all. An entire mountainside renders as a hole.

I hit this during development — it is the failure `test_sustained_steep_slope_has_no_holes` now
guards. Connection is deliberately permissive here: a steep-but-continuous surface stays one sheet
and is drawn as a steep quad spanning one cell, which is the correct reading of the data and is the
resolution limit at that LOD anyway. Walls are reserved for discontinuities in **layer structure** —
the lip of an overhang, the edge of an island, a cave mouth — where a sheet genuinely stops and
there is nothing on the far side to connect to.

### 3. Sectors are padded point grids, not cell grids

Sectors sample 33 columns to cover 32 cells. Column 32 of one sector is the same world position as
column 0 of its neighbour, and both come from evaluating the same generator at the same point, so
they agree exactly.

This is what allows **every sector to be generated and meshed on its own thread with no
synchronisation** — meshing never reads a neighbour. `test_sectors_tile_exactly` guards it.

---

## Architecture

```
                    VoxelGenerator  (shared with your near terrain)
                            │
              ┌─────────────┴─────────────┐
              │  FarBuildSectorTask       │   worker threads
              │                           │
              │  1. cache lookup          │──► FarCache (region files)
              │  2. sample SDF slab       │
              │  3. extract columns  ─────┼──► ColumnField
              │  4. mesh             ─────┼──► FarMeshArrays
              └─────────────┬─────────────┘
                            │  FarResultQueue
                            ▼
              ┌───────────────────────────┐
              │  far::FarRenderer         │   main thread
              │  - reconcile vs residency │
              │  - ArrayMesh upload       │   bounded per frame
              │  - DirectMeshInstance     │
              └───────────────────────────┘

   Sector selection (FarLodTree, rings) runs on VoxelLodTerrain's update
   thread, in voxel_lod_terrain_update_far_streaming.cpp.
```

### Files

| File | Role |
|---|---|
| `core/far_math.h` | Vectors, floor-division, octahedral normal packing |
| `core/far_column_data.*` | The column format, CSR storage, quantised serialisation |
| `core/far_extract.*` | SDF → columns, AO baking, downsampling |
| `core/far_lod_tree.*` | Which sectors exist at which LOD, flat rings or sphere rings |
| `core/far_sphere.*` | Cubesphere: face mapping, surface frames, crossing a face edge |
| `core/far_geometry.*` | Where a sector's columns are in the world. The only place that knows flat from planetary |
| `core/far_mesher.*` | Sheet labelling, caps, walls, skirts |
| `tasks/far_cache.*` | Region-file persistence |
| `tasks/far_tasks.*` | The threaded pipeline |
| `far_renderer.*` | Main-thread mesh instances |
| `../terrain/variable_lod/voxel_lod_terrain_update_far_streaming.*` | The streaming system |
| `shaders/far_terrain.gdshader` | Surface shader with near/far fades |
| `tests/` | Host-compilable tests for everything in `core/` |

**`core/` includes no Godot headers at all.** That is deliberate: the load-bearing algorithms are
the hard part, and keeping them engine-free means they can be tested with a plain compiler in
seconds rather than by launching a game. It is how both of the bugs described above were found.

### Rings, not a quadtree

A quadtree is the obvious structure and it is a trap for this workload. It is a stateful graph that
must stay consistent while nodes are asynchronously loading: a node that split while its children
were still generating leaves a hole, and so does a node that merged while its parent was still
generating. Every split and merge becomes a small transaction.

Concentric rings get the same result with no state machine. For each level, keep the sectors within
a fixed radius of the viewer, minus whatever the finer level already covers. **The required set is
a pure function of the viewer position**, recomputable from scratch on any frame and diffed against
what is resident. Nothing can be inconsistent because there is no incremental state.

The subtlety is that rings must be **snapped to even/odd coordinates** so that a level's extent maps
to a whole number of parent sectors. An unsnapped ring has odd length, its edges fall mid-parent,
and whichever way you round you get either a gap or a double-covered band that z-fights. I got this
wrong first (3058 of 3721 sampled points were double-covered); `test_coverage_has_no_gaps_or_overlaps`
now checks it across on-grid, off-grid and negative viewer positions.

---

## Installation

Part of `modules/voxel`. Nothing to install, nothing to enable at build time.

## Usage

```gdscript
# Near terrain draws 0..1024. Far terrain draws 1024..~262 km.
$VoxelLodTerrain.generator     = preload("res://my_generator.tres")
$VoxelLodTerrain.view_distance = 1024

$VoxelLodTerrain.far_enabled      = true
$VoxelLodTerrain.far_first_lod    = 4     # level 4 sectors are 512 units across
$VoxelLodTerrain.far_lod_count    = 8
$VoxelLodTerrain.far_ring_radius  = 4
$VoxelLodTerrain.far_vertical_min = -256
$VoxelLodTerrain.far_vertical_max = 1024
$VoxelLodTerrain.far_material     = preload("res://far_terrain_material.tres")
```

The generator is shared automatically — there is no second one to keep in step.

### Planets

Set a planet radius and the far field wraps a sphere instead of a plane. Columns run radially from
the terrain's origin and sectors tile the six faces of a cubesphere; `far_vertical_min`/`max` become
heights relative to the radius.

```gdscript
$VoxelLodTerrain.far_planet_radius = 40000.0
$VoxelLodTerrain.far_vertical_min  = -1500.0   # relative to the radius, not to Y=0
$VoxelLodTerrain.far_vertical_max  =  2500.0
```

Nothing else changes: `first_lod` still means "cells about 2^first_lod units across", and the
column format, extractor, mesher and cache are the same code on both. See HANDOFF.md §9.

One thing to know before you rely on it: radial columns need the generator sampled at arbitrary
points, which means `VoxelGenerator::generate_series()`. `VoxelGeneratorGraph`,
`VoxelGeneratorNoise2D` and `EdenPlanetGeneratorV4` implement it. Anything else falls back to a path
roughly two orders of magnitude slower — slow enough that a planet does not finish streaming. If you
write your own planet generator, implement `generate_series` for `CHANNEL_SDF`; it is about fifteen
lines. HANDOFF.md §9.3.

Materials come from `CHANNEL_TYPE` or from MIXEL4 (`CHANNEL_INDICES`/`CHANNEL_WEIGHTS`), whichever
the generator declares; MIXEL4's four-way blend is reduced to its heaviest entry, since the far
field stores one material per layer. `get_far_statistics()["max_material"]` reports the highest
index seen, which is the quickest way to tell a working palette from a silently blank one.

The shader's palette is `material_blend` plus `material_colors[8]`, off by default:

```gdscript
var mat := ShaderMaterial.new()
mat.shader = preload("res://far_terrain.gdshader")
mat.set_shader_parameter("material_blend", 0.85)
mat.set_shader_parameter("material_colors", PackedVector3Array([...]))  # 8 entries
$VoxelLodTerrain.far_material = mat
```

The palette is looked up per vertex and the resulting colour interpolated, so material boundaries
are smooth and the palette stays tweakable at runtime. HANDOFF.md §9.6 explains why the two obvious
alternatives — interpolating the index, or marking it `flat` — both look wrong.

`far_near_clip_radius` defaults to 0, which **derives the clip from `view_distance`**. That is what
stops the far field drawing ground the near terrain already owns, and it used to be a number you had
to keep in step by hand. Set it positive to override, or negative to switch clipping off entirely
(only useful when looking at the far field on its own).

Set the shader's `near_fade_start` / `near_fade_end` to bracket the clip radius;
`get_far_statistics()["near_clip_radius"]` reports whatever it resolved to.

Put a real `VoxelViewer` in the scene. Without one, `VoxelLodTerrain`'s fallback viewer position
drifts whenever the terrain node is not at the origin — see HANDOFF.md §3.8.

### Tuning notes

- **`first_lod` is the most important setting.** Every level whose ring fits inside the near
  terrain's radius is generated and then not drawn, and the fine levels are by far the most
  expensive — a level-0 sector spanning the full vertical range needs 64× the Y samples of a
  level-6 one. Set it so the innermost ring lands just inside the near terrain's edge.
- **`near_clip_radius` slightly *less* than the near view distance.** A gap between the two
  terrains is a visible ring of missing world; an overlap is hidden by the near terrain drawing on
  top. The node emits a configuration warning if the innermost ring cannot reach it.
- **`antialias`** samples one level finer and reduces. Costs 4× the generator evaluations and
  removes the level-to-level popping that direct sampling produces on narrow features.
- **`generate_bottoms`** off is free triangles back if your world has few overhangs.

`get_statistics()` returns resident/rendered sector counts, tasks in flight, pending uploads, and
cache hit rates.

---

## Testing

```sh
g++ -std=c++17 -O1 -g -fsanitize=address,undefined \
    tests/test_far_core.cpp core/far_column_data.cpp core/far_extract.cpp core/far_mesher.cpp \
    -o /tmp/test_far && /tmp/test_far

g++ -std=c++17 -O1 -g -fsanitize=address,undefined \
    tests/test_far_lod_tree.cpp core/far_column_data.cpp core/far_lod_tree.cpp \
    -o /tmp/test_lod && /tmp/test_lod
```

Currently ~26,900 assertions across both suites, clean under ASan and UBSan. They cover normal
packing round-trips, extraction against analytic surfaces, cave/overhang handling, serialisation
including rejection of corrupt input, LOD registration, mesh well-formedness, triangle winding,
ring coverage, hysteresis and the resident-count bound.

Triangle winding is verified against a fact derived from `godot_voxel`'s own cube tables rather than
assumed: **Godot front faces are clockwise from the front**, so the right-hand-rule normal of an
emitted triangle points *opposite* the shading normal.

---

## What is not done

Honest list of what you will still need to do:

- **The Godot layer has never been compiled.** This sandbox has no Godot source tree, so
  `node/`, `tasks/` and `register_types.cpp` are checked for API and include correctness against
  the real `godot_voxel` headers but not built. Expect to fix some compile errors on first build.
  Everything in `core/` is fully compiled and tested.
- **The shader is untested in engine.** It is written for Godot 4.6 but has never been compiled by
  the shader compiler. Expect a typo or two on first load.
- **Materials are a single index.** The shader tints by height and slope. Real material blending
  needs a palette texture and probably a second material index per layer for transitions.
- **Water is not handled.** It needs a separate span type and a transparent pass.
- **The cache is region files, not SQLite.** `godot_voxel` already bundles SQLite, and a mature
  version should probably use it. Region files keep the module free of that build dependency; the
  trade is dead blobs, which the compaction pass handles.
- **No collision.** Far terrain is visual only, which is almost certainly what you want.
- **Blocky terrain is not targeted.** The column format suits it even better, but the mesher would
  want terraced caps and a slope-driven wall rule — the opposite of the SDF choice explained above.
