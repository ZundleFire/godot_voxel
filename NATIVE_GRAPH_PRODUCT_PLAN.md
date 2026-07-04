# Native Voxel Graph Product Plan

## Goal
Turn `VoxelGeneratorGraph` into the **project-facing procedural authoring surface** for Eden while keeping execution entirely in **native C++ with optional ISPC acceleration** and existing GPU shader generation.

> Current audit result: the runtime graph VM and node system are already native C++. The main remaining work is to expand the node catalog, move more heavy buffer ops to ISPC, and add Eden-specific tectonic/climate/biome/material nodes.

---

## Product Direction

### Authoring Layer
- Keep the existing visual graph UX and resource workflow.
- Add a curated **Eden node pack** for planet generation instead of forcing logic through large monolithic generators.

### Runtime Layer
- CPU graph execution remains in `modules/voxel/generators/graph/`.
- Node internals stay in **C++**, with hot buffer kernels moved to **ISPC**.
- GPU path remains available where shader generation is supported.

### Project-Specific Focus
The graph product should directly support:
- spherical SDF terrain
- tectonic region lookup
- climate and humidity fields
- biome classification and region biasing
- MIXEL4 material packing
- debug texture baking and runtime inspection
- worker-thread-safe block generation

---

## Architecture Workstreams

### 1. SIMD / ISPC Runtime Track
**Objective:** accelerate common buffer operations used by the graph VM.

Deliverables:
1. ISPC-backed math nodes (`Add`, `Subtract`, `Multiply`, `Divide`) ✅ **started in this patch**
2. ISPC-backed unary math (`Abs`, `Clamp`, `Min`, `Max`, `Sqrt`, `Pow`)
3. ISPC-backed noise wrappers and domain-warp nodes
4. optional runtime backend metrics/profiling per node family

Key files:
- `modules/voxel/generators/graph/voxel_graph_runtime.*`
- `modules/voxel/generators/graph/node_type_db.h`
- `modules/voxel/generators/graph/nodes/*.h`
- `modules/voxel/SCsub`

### 2. Eden Native Node Pack Track
**Objective:** expose the project’s planet-generation systems as graph nodes.

Planned node families:
- `PlanetDirection`, `SphereLatitude`, `SphereLongitude`, `AltitudeFromRadius`
- `TectonicLookup`, `PlateFalloff`, `BoundaryType`, `BoundaryDistance`
- `ClimateField`, `HumidityField`, `RainShadow`, `TemperatureField`
- `BiomeRegionLookup`, `BiomeClassifier`, `BiomeSpawnGate`
- `Mixel4Pack`, `MaterialFamily`, `DebugChannelPack`

### 3. Hybrid GPU/CPU Track
**Objective:** keep graph output customizable while letting heavy paths remain native.

Plan:
- use graph CPU VM for general fallback, materials, and unsupported nodes
- use shader generation for GPU-capable SDF subgraphs
- allow Eden-specific nodes to declare `CPU-only`, `GPU-capable`, or `baked-data` usage

### 4. Tooling / UX Track
**Objective:** make the system practical for the project rather than generic-only.

Planned editor features:
- node templates for “planet base”, “tectonics”, “biome blend”, and “material pack”
- pinned debug previews for climate / tectonics / biome families
- validation warnings for worker-unsafe or non-baked node usage

---

## Implementation Phases

## Phase 0 — Baseline and scaffolding
- audit current C++ graph VM and node registry
- establish the native/ISPC roadmap
- add the first SIMD-backed node path

**Status:** in progress

## Phase 1 — Core SIMD nodes
- move math buffer ops to ISPC
- add unit coverage for graph math correctness
- instrument profiling to compare scalar vs ISPC execution

## Phase 2 — Noise and terrain primitives
- native sphere-aware noise nodes
- domain warp and region sampling nodes
- planet-centric SDF helpers for radius/altitude workflows

## Phase 3 — Eden systems integration
- bind tectonic lookup and climate sampling as graph nodes
- expose biome-region selection and weighted biome outputs
- add native material packing outputs for `MIXEL4`

## Phase 4 — Productization
- graph presets, docs, performance tuning
- node categorization around Eden workflows
- long-range validation on planet-scale generation workloads

---

## Acceptance Criteria
A production-ready Eden graph system should satisfy all of the following:

- authoring stays graph-based and editable in the voxel editor
- runtime execution is **native C++ only**, with no GDScript dependency in generation
- hot-path nodes use ISPC where beneficial
- all nodes are worker-thread-safe or explicitly bake-only
- planet-specific systems can be built from nodes instead of a single monolithic generator
- existing debug modes can be fed directly from graph outputs or baked graph resources

---

## Immediate Next Patches
1. finish ISPC coverage for the remaining basic math nodes
2. add an `Eden` graph node category for sphere/planet helpers
3. expose tectonic/climate/biome lookups from the native planet pipeline as reusable graph nodes
4. add validation and tests for the new native node pack
