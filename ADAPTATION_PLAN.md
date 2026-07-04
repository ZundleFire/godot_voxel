# Voxel Module Optimization & Adaptation Plan

> **Source reference:** UE5 VoxelPlugin-dev (`D:\Unreal Projects\SurvivalRPGEngine\Plugins\VoxelPlugin-dev`)
> **Target module:** `F:\Dev\GodotEden\modules\voxel` (Godot 4 fork, branch `eden/main-publish`)
> **Consumer project:** `F:\Dev\Projects\planet-voxels` (40 km planet prototype)

---

## 1. Introduction and Goals

The existing Godot voxel module (Zylann's `godot_voxel`) is already a mature system with LOD octrees, clipbox streaming, GPU compute task runner, threaded task pools, region-file and SQLite persistence, and Transvoxel meshing. The goal is **not** to rewrite from scratch, but to **enhance five specific subsystems** by integrating concepts proven in the Unreal VoxelPlugin:

| # | Feature | Current State | Target State |
|---|---------|--------------|--------------|
| 1 | **LOD system** | Octree or clipbox streaming; `lod_distance` / `secondary_lod_distance` two-parameter model; `MAXIMUM_LOD_DISTANCE` capped at 8192 | Configurable per-LOD distance table; camera-distance-based LOD0 radius; on-the-fly recomputation of LOD data without full reload |
| 2 | **Chunk persistence & streaming** | `VoxelStreamSQLite` (serialized DB access), `VoxelStreamRegionFiles` (region files); LZ4 compression; in-memory `VoxelStreamCache` of 64 entries | Memory-mapped region files; larger write-behind cache; priority-aware flush; fast bulk save/load path at all LOD levels |
| 3 | **Background threading** | `ThreadedTaskRunner` (auto-detected count via `Thread::get_hardware_concurrency()`); separate I/O thread; `GPUTaskRunner` on dedicated thread | Expose thread count to GDScript; dynamic scaling; task-stealing work queues; NUMA-aware pool on large machines |
| 4 | **GPU generation** | `GenerateBlockGPUTask` dispatches GLSL compute shaders via `RenderingDevice`; graph-to-shader compiler in `VoxelGeneratorGraph`; `VoxelGeneratorScript` is CPU-only | Bridge `VoxelGeneratorScript` to GPU via an auto-generated compute fallback or a new `VoxelGeneratorComputeScript`; hybrid CPU/GPU pipeline where GPU does SDF and CPU does material assignment |
| 5 | **Voxel spacing** | Implicit 1 voxel = 1 unit; no explicit scale factor; users control world size via `planet_radius` and `voxel_bounds` | Expose `voxel_size` property on `VoxelLodTerrain`; all coordinate math respects this; configurable LOD0 resolution (1 m, 10 cm, etc.) |

---

## 2. Analysis of Existing Systems

### 2.1 Godot Voxel Module (current codebase)

**Architecture overview:**

```
VoxelEngine (singleton)
├── ThreadedTaskRunner  _general_thread_pool    ← CPU task pool (generation, meshing)
├── TimeSpreadTaskRunner                        ← main-thread micro-tasks
├── ProgressiveTaskRunner                       ← multi-frame main-thread ops
├── GPUTaskRunner        _gpu_task_runner       ← dedicated GPU compute thread
└── FileLocker                                  ← per-file locking for streams

VoxelLodTerrain (Node3D)
├── VoxelData  _data (shared_ptr)               ← per-LOD VoxelDataMap (HashMap of VoxelDataBlock)
├── VoxelLodTerrainUpdateData  _update_data     ← streaming state, per-LOD mesh map state
│   ├── OctreeStreamingState  (legacy)          ← LodOctree per grid cell
│   └── ClipboxStreamingState (new)             ← axis-aligned boxes per viewer per LOD
├── VoxelMeshMap<VoxelMeshBlockVLT> per LOD     ← mesh blocks with DirectMeshInstance
├── StreamingDependency                         ← shared_ptr holding stream+generator refs
└── MeshingDependency                           ← shared_ptr holding mesher ref
```

**Key threading model:**
- `VoxelLodTerrainUpdateTask` runs on the general thread pool and performs the streaming logic (decide which blocks to load/unload/mesh). Its outputs are deferred actions consumed on the main thread.
- `GenerateBlockTask` runs on the general pool; if `use_gpu` is true, it posts a `GenerateBlockGPUTask` and suspends.
- `MeshBlockTask` runs on the general pool; builds the Transvoxel mesh.
- `LoadBlockDataTask` / `SaveBlockDataTask` run on a dedicated I/O serial queue.
- Main thread consumes `BlockMeshOutput` and `BlockDataOutput` callbacks, applies them to scene tree.

**LOD distance computation (clipbox):**
```cpp
// get_relative_lod_distance_in_chunks()
if (lod_index == 0)
    ld = lod0_distance_in_chunks;       // ceil(lod_distance / mesh_block_size)
else
    ld = (lod0_distance_in_chunks >> lod_index) + lodn_distance_in_chunks;
// Last LOD extends to max view distance
```

This gives a two-parameter curve. The UE5 VoxelPlugin uses a more flexible approach:
- Per-chunk priority based on camera distance, with smooth LOD transitions
- Explicit octree subdivision controlled by per-node distance thresholds
- "Invokers" (equivalent to VoxelViewer) with independently configurable ranges

**GPU pipeline:**
- `VoxelGeneratorGraph` can compile its node graph to GLSL via `VoxelGraphShaderGenerator`
- The compute shader writes SDF values into a storage buffer
- `GenerateBlockGPUTask` dispatches this, then `GenerateBlockTask` collects results

**Persistence:**
- `VoxelStreamSQLite`: single file, LZ4-compressed blocks, coordinate-indexed, serialized mutex access
- `VoxelStreamRegionFiles`: directory of region files (Seed-of-Andromeda style), one file per region per LOD
- Both support batch load/save but SQLite serializes internally

### 2.2 Unreal VoxelPlugin (reference architecture)

Key concepts to adapt:

| UE5 Concept | Godot Equivalent | Gap |
|---|---|---|
| `FVoxelOctree` with per-node LOD priority | `LodOctree` | Godot octree is simpler; lacks priority-weighted subdivision |
| `FVoxelTaskGroup` with dependency graphs | `BufferedTaskScheduler` | Godot has no dependency DAG between tasks |
| `FVoxelChunkManager` with memory budget | `VoxelData` + `VoxelStreamCache` | No memory budget enforcement |
| GPU marching cubes (HLSL compute) | Transvoxel is CPU-only | Major gap — meshing is CPU bound |
| `FVoxelWorldGeneratorInstance` per-thread | `VoxelGeneratorScript` shared | GDScript generators are mutex-locked per call |
| Chunk compression with Oodle/Zstd | LZ4 only | Could add Zstd option |
| Region file with mmap | `RegionFile` with FileAccess | No mmap |

---

## 3. Proposed System Design

### 3.1 LOD System Enhancements

#### 3.1.1 Configurable Per-LOD Distance Table

**Problem:** The current two-parameter model (`lod_distance` + `secondary_lod_distance`) gives limited control. For planet-scale terrain, users need fine-grained control over where each LOD level begins and ends.

**Design:**

Add a new property `lod_distances: PackedFloat64Array` to `VoxelLodTerrain`. When set, it overrides the two-parameter formula. Each entry `lod_distances[i]` specifies the camera distance (in voxels) at which LOD level `i` ends and LOD `i+1` begins.

```
// voxel_lod_terrain.h — new members
PackedFloat64Array _custom_lod_distances; // empty = use formula
bool _use_custom_lod_distances = false;

// voxel_lod_terrain.cpp — bind
ClassDB::bind_method(D_METHOD("set_lod_distances", "distances"), &VoxelLodTerrain::set_lod_distances);
ClassDB::bind_method(D_METHOD("get_lod_distances"), &VoxelLodTerrain::get_lod_distances);
ADD_PROPERTY(PropertyInfo(Variant::PACKED_FLOAT64_ARRAY, "lod_distances"), "set_lod_distances", "get_lod_distances");
```

**Clipbox integration:** In `get_relative_lod_distance_in_chunks()`, check if custom distances are set:

```cpp
Vector3i get_relative_lod_distance_in_chunks(
    int lod_index, int lod_count,
    int lod0_dist_chunks, int lodn_dist_chunks,
    int lod_chunk_size,
    Vector3i max_vd_voxels,
    const PackedFloat64Array *custom_dists  // NEW param, nullable
) {
    int ld;
    if (custom_dists != nullptr && lod_index < custom_dists->size()) {
        // Custom per-LOD distance in voxels
        ld = math::max(
            static_cast<int>(Math::ceil((*custom_dists)[lod_index])) / lod_chunk_size,
            1
        );
    } else if (lod_index == 0) {
        ld = lod0_dist_chunks;
    } else {
        ld = (lod0_dist_chunks >> lod_index) + lodn_dist_chunks;
    }
    // ... rest unchanged
}
```

**GDScript usage:**
```gdscript
# Example: planet with 40 km radius
volume.lod_distances = PackedFloat64Array([
    128,      # LOD 0: 128 m (foot detail)
    512,      # LOD 1: 512 m
    2048,     # LOD 2: 2 km
    8192,     # LOD 3: 8 km
    20000,    # LOD 4: 20 km
    40000,    # LOD 5: 40 km (full hemisphere)
    80000,    # LOD 6: 80 km (low orbit)
])
```

#### 3.1.2 On-the-Fly LOD Data Recomputation

**Problem:** When the camera moves, blocks at the LOD boundary must transition. Currently, blocks are either generated from scratch or loaded from stream. For generated-only blocks (no edits), the generator is re-invoked, which is correct but slow for GDScript generators.

**Design:** Introduce a **LOD downsampling cache**. When a higher-resolution block exists in `VoxelData`, the lower-LOD version can be computed by downsampling rather than re-generating:

```cpp
// voxel_data.h — new method
// Downsample LOD N block from LOD N-1 data (2×2×2 blocks → 1 block)
bool try_downsample_block(Vector3i block_pos, unsigned int target_lod,
                          std::shared_ptr<VoxelBuffer> &out_buffer);
```

This mirrors the UE5 VoxelPlugin's approach of maintaining LOD pyramids from edited data, reducing generator calls when LOD boundaries shift.

**Implementation steps:**
1. Add `try_downsample_block()` to `VoxelData` that reads 8 child blocks from `lod - 1` and averages SDF/material values.
2. In `GenerateBlockTask::run()`, before calling the generator, check if all 8 child blocks exist and downsample instead.
3. Mark downsampled blocks as "derived" so they aren't saved to stream (they can be re-derived).

#### 3.1.3 LOD Transition Quality

Add smooth blending between LOD levels via the existing `lod_fade_duration` mechanism, but extend it to also interpolate SDF values at the seam (currently only mesh opacity fades). This requires passing the current LOD blend factor to the Transvoxel mesher as a uniform:

```glsl
// In planet_ground.gdshader — vertex function
uniform float u_lod_blend : hint_range(0.0, 1.0) = 1.0;
// Applied per-block by the ShaderMaterialPool
```

---

### 3.2 Chunk Persistence and Streaming

#### 3.2.1 Memory-Mapped Region Files

**Problem:** `VoxelStreamRegionFiles` uses `FileAccess` for all reads/writes, which involves syscall overhead for each block. SQLite serializes all access through a mutex.

**Design:** Add a new stream backend `VoxelStreamMMapRegion` that memory-maps region files:

```cpp
// New file: streams/mmap/voxel_stream_mmap.h

class VoxelStreamMMap : public VoxelStream {
    GDCLASS(VoxelStreamMMap, VoxelStream)
public:
    // Region file layout: fixed 4 KB header + block directory + block data
    // Block directory: sorted array of (block_key, offset, size, compression) entries
    // Memory-mapped for reads; writes go through a write-behind buffer

    struct RegionHeader {
        uint32_t magic;          // 'VXMM'
        uint32_t version;
        uint32_t block_size_po2;
        uint32_t region_size_po2;
        uint32_t block_count;
        uint32_t data_start_offset;
        uint8_t  channel_depths[VoxelBuffer::MAX_CHANNELS];
    };

    void load_voxel_blocks(Span<VoxelQueryData> p_blocks) override;
    void save_voxel_blocks(Span<VoxelQueryData> p_blocks) override;

private:
    // Platform-specific mmap handles
    struct MappedRegion {
        void *base_ptr = nullptr;
        size_t file_size = 0;
        // OS handle (HANDLE on Windows, int fd on Unix)
        // ...
    };
    StdUnorderedMap<uint64_t, MappedRegion> _mapped_regions;
};
```

**Windows implementation** uses `CreateFileMapping` / `MapViewOfFile`. On Linux, `mmap()`.

**Benefits:**
- Zero-copy reads: the OS page cache handles caching transparently
- Batch loads become pointer arithmetic instead of file seeks
- Write-behind: dirty pages are flushed by the OS or on explicit `FlushViewOfFile`

#### 3.2.2 Enhanced Write-Behind Cache

Increase `VoxelStreamCache` from 64 to a configurable size (default 256), and add **priority-based eviction**:

```cpp
// voxel_stream_cache.h modifications
struct CacheEntry {
    std::shared_ptr<VoxelBuffer> voxels;
    Vector3i position;
    uint8_t lod_index;
    uint32_t last_access_frame;
    uint8_t priority;  // NEW: 0=high (LOD0 near camera), 255=low
    bool dirty;
};

// Eviction: LRU with priority weighting
// Dirty entries must be flushed to stream before eviction
```

#### 3.2.3 Bulk Save/Load at Arbitrary LOD

Currently `load_all_blocks()` is only supported by SQLite. Extend all streams to support **LOD-filtered bulk operations**:

```cpp
// voxel_stream.h — new virtual
struct BulkLoadParams {
    Box3i region;           // In block coordinates
    uint8_t min_lod = 0;
    uint8_t max_lod = 0;
    bool decompress = true; // false = return compressed blobs
};
virtual void load_blocks_bulk(BulkLoadParams params, FullLoadingResult &result);
```

This enables loading coarse LOD data first (for distant terrain) and filling in LOD0 data on demand — exactly the pattern the UE5 VoxelPlugin uses for initial world load.

#### 3.2.4 Compression Improvements

Add Zstd as an alternative to LZ4:

```cpp
// compressed_data.h — extend enum
enum Compression {
    COMPRESSION_NONE = 0,
    COMPRESSION_LZ4 = 1,
    COMPRESSION_ZSTD = 2,  // NEW
    COMPRESSION_COUNT
};
```

Zstd provides 20–40% better compression ratio than LZ4 at comparable decompression speed, which matters for large planet worlds where saved data can reach gigabytes.

---

### 3.3 Background Threading

#### 3.3.1 Current System Analysis

The `ThreadedTaskRunner` already:
- Detects hardware threads via `Thread::get_hardware_concurrency()`
- Computes thread count as `clamp(ratio * hw_threads, minimum, hw_threads - margin)`
- Supports priority-based task scheduling with periodic re-prioritization
- Separates I/O tasks onto a serial queue

The `VoxelEngine::Config` exposes:
```cpp
int thread_count_minimum = 1;
int thread_count_margin_below_max = 1;
float thread_count_ratio_over_max = 0.5;
```

#### 3.3.2 Dynamic Thread Scaling

**Problem:** The thread count is set once at engine startup and never changes. On a 16-core machine, 8 threads are used even when the workload could benefit from more (bulk generation) or fewer (idle world).

**Design:** Add dynamic thread count adjustment:

```cpp
// threaded_task_runner.h — new methods
void set_thread_count_dynamic(uint32_t count);
uint32_t get_active_thread_count() const;
uint32_t get_pending_task_count() const;

// voxel_engine.h — expose to GDScript
void set_thread_count(uint32_t count);  // Already exists
int get_thread_count() const;           // Already exists
// NEW:
void set_thread_count_auto();           // Re-detect and set optimal count
void set_thread_scaling_enabled(bool);  // Enable/disable dynamic scaling
```

**Auto-scaling logic** (runs every 500 ms in `VoxelEngine::process()`):

```cpp
void VoxelEngine::auto_scale_threads() {
    const uint32_t pending = _general_thread_pool.get_debug_remaining_tasks();
    const uint32_t current = _general_thread_pool.get_thread_count();
    const uint32_t hw_max = Thread::get_hardware_concurrency();
    const uint32_t usable_max = hw_max - 1; // Leave 1 for main thread

    if (pending > current * 4 && current < usable_max) {
        // High load: scale up
        _general_thread_pool.set_thread_count_dynamic(
            math::min(current + 2, usable_max));
    } else if (pending == 0 && current > _config.thread_count_minimum) {
        // Idle: scale down (after cooldown)
        _general_thread_pool.set_thread_count_dynamic(
            math::max(current - 1, (uint32_t)_config.thread_count_minimum));
    }
}
```

#### 3.3.3 GDScript API Exposure

```gdscript
# In planet_scene.gd or game settings
var engine := VoxelEngine.get_singleton()
print("HW threads: ", OS.get_processor_count())
print("Voxel threads: ", engine.get_thread_count())

# Override for heavy generation phase
engine.set_thread_count(OS.get_processor_count() - 2)

# Or let the engine decide
engine.set_thread_scaling_enabled(true)
```

#### 3.3.4 Task-Stealing Work Queue

Replace the current single-mutex task queue with a **work-stealing deque** per thread:

```
Current:  [Global Queue] ←mutex→ [Thread 0..N]

Proposed: [Thread 0: local deque] [Thread 1: local deque] ... [Thread N: local deque]
          │                        │                           │
          └── steal from ──────────┴── steal from ─────────────┘
          [Global overflow queue] ← for external pushes
```

**Implementation:** Each thread has a lock-free deque (Chase-Lev). New tasks are pushed to the thread with the shortest queue (round-robin or least-loaded). When a thread's deque is empty, it steals from another thread's deque (LIFO end for cache locality).

This reduces contention on the shared mutex, which becomes a bottleneck at >8 threads when many small tasks (per-block generation) are queued.

```cpp
// New file: util/tasks/work_stealing_pool.h

template <size_t MAX_THREADS = 128>
class WorkStealingPool {
    struct alignas(64) ThreadLocal {  // cache-line aligned
        ChaseLevDeque<IThreadedTask*> deque;
        std::atomic<State> state;
        Thread thread;
    };

    FixedArray<ThreadLocal, MAX_THREADS> _threads;
    uint32_t _thread_count = 0;

    // External push: distribute to least-loaded thread
    void enqueue(IThreadedTask *task);
    // Internal: thread tries own deque, then steals
    IThreadedTask *try_get_task(uint32_t thread_index);
};
```

#### 3.3.5 Separate Generation and Meshing Pools

The UE5 VoxelPlugin separates generation and meshing into distinct thread pools, allowing independent priority and count tuning. We can replicate this:

```cpp
// voxel_engine.h — split pools
ThreadedTaskRunner _generation_thread_pool;  // SDF computation
ThreadedTaskRunner _meshing_thread_pool;     // Transvoxel meshing
ThreadedTaskRunner _io_thread_pool;          // Stream I/O (serial)
```

**Benefit:** Generation tasks (which may call GDScript) won't block meshing tasks (pure C++), and vice versa. This is especially important for `VoxelGeneratorScript` which holds the GIL.

---

### 3.4 GPU Adaptation

#### 3.4.1 Current GPU Infrastructure

The module already has:
- `GPUTaskRunner`: dedicated thread with its own `RenderingDevice`
- `ComputeShader` / `ComputeShaderParameters`: GLSL compute wrapper
- `GenerateBlockGPUTask`: dispatches generator compute, collects via shared output buffer
- `VoxelGraphShaderGenerator`: compiles `VoxelGeneratorGraph` node graphs to GLSL

**Limitation:** `VoxelGeneratorScript` (GDScript) cannot use the GPU path at all. The `generate_block` virtual is CPU-only.

#### 3.4.2 Hybrid CPU/GPU Pipeline for VoxelGeneratorScript

**Design:** Split generation into two phases that `VoxelGeneratorScript` can opt into:

```
Phase 1 (GPU): SDF field computation — runs a user-supplied compute shader
Phase 2 (CPU): Material assignment — runs _generate_block() with pre-filled SDF channel
```

**New virtual methods:**

```cpp
// voxel_generator_script.h — additions

// Optional: return GLSL source for SDF computation
GDVIRTUAL0RC(String, _get_sdf_compute_shader)
// Optional: return parameters for the compute shader
GDVIRTUAL0RC(Dictionary, _get_sdf_compute_params)
// Phase 2: called with SDF already filled, user fills material channels only
GDVIRTUAL3(_generate_materials, Ref<godot::VoxelBuffer>, Vector3i, int)
```

**GDScript usage:**

```gdscript
class_name PlanetGeneratorGPU
extends VoxelGeneratorScript

func _get_sdf_compute_shader() -> String:
    # Return GLSL compute shader source for SDF generation
    return """
    #version 450
    layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

    layout(set = 0, binding = 0) buffer Params {
        vec3 origin;
        float planet_radius;
        float max_terrain_height;
        // ... noise parameters
    };

    layout(set = 0, binding = 1) buffer Output {
        float sdf_values[];
    };

    void main() {
        ivec3 pos = ivec3(gl_GlobalInvocationID);
        vec3 world_pos = origin + vec3(pos);
        float dist = length(world_pos);
        float sdf = dist - planet_radius;
        // Add noise layers...
        sdf_values[pos.z * 32*32 + pos.y * 32 + pos.x] = sdf;
    }
    """

func _get_sdf_compute_params() -> Dictionary:
    return {
        "planet_radius": planet_radius,
        "max_terrain_height": max_terrain_height,
    }

func _generate_materials(voxels: VoxelBuffer, origin: Vector3i, lod: int) -> void:
    # SDF channel is already filled by GPU. Now assign materials.
    var size := voxels.get_size()
    for z in size.z:
        for y in size.y:
            for x in size.x:
                var sdf = voxels.get_voxel_f(x, y, z, VoxelBuffer.CHANNEL_SDF)
                if sdf < 0:
                    _assign_material(voxels, x, y, z, origin, lod)
```

**Engine-side implementation:**

```cpp
// In GenerateBlockTask::run()

if (generator->supports_gpu_sdf()) {
    // Phase 1: dispatch GPU SDF
    auto gpu_task = ZN_NEW(GenerateBlockGPUTask);
    gpu_task->consumer_task = this;
    gpu_task->generator_shader = generator->get_sdf_compute_shader();
    gpu_task->generator_shader_params = generator->get_sdf_compute_params();
    // ... fill origin, lod, etc.
    VoxelEngine::get_singleton().push_gpu_task(gpu_task);
    ctx.status = ThreadedTaskContext::STATUS_TAKEN_OUT;
    return; // Will resume when GPU completes
}

// After GPU results arrive (or if CPU-only):
void GenerateBlockTask::run_cpu_generation() {
    if (_has_gpu_sdf) {
        // SDF channel already filled — call _generate_materials only
        generator->generate_materials(_voxels, _position, _lod_index);
    } else {
        // Full CPU path (existing)
        generator->generate_block(query);
    }
}
```

#### 3.4.3 GPU Meshing (Future — Phase 2)

The UE5 VoxelPlugin performs marching cubes on the GPU. This is a larger effort but offers 10–50× speedup for meshing. The approach:

1. **Compute shader pass 1:** Count active cells and vertices per cell (prefix sum)
2. **Compute shader pass 2:** Generate vertices and indices into storage buffers
3. **Read-back or use GPU buffer directly** as mesh data

This requires changes to `VoxelMesherTransvoxel` to have a GPU backend:

```cpp
// meshers/transvoxel/transvoxel_gpu.h (future)

class TransvoxelGPUMesher {
public:
    struct GPUMeshResult {
        RID vertex_buffer;
        RID index_buffer;
        uint32_t vertex_count;
        uint32_t index_count;
    };

    // Dispatches compute shaders for transvoxel extraction
    void dispatch(GPUTaskContext &ctx,
                  RID sdf_buffer,      // SDF values as storage buffer
                  Vector3i block_size,
                  uint8_t lod_index,
                  GPUMeshResult &out_result);
};
```

**Note:** Godot 4's `RenderingDevice` API supports storage buffers and compute dispatch, but lacks direct buffer-to-mesh binding. We'd need to use `RenderingServer::mesh_create_from_raw_data()` or download the buffer and create the mesh on CPU. The download is the bottleneck — consider using **indirect rendering** if Godot's renderer supports it.

#### 3.4.4 Compute Shader for Planet SDF

For the planet-voxels project specifically, the SDF computation is the bottleneck (GDScript `_compute_sdf` is called millions of times). A dedicated compute shader:

```glsl
// shaders/planet_sdf.glsl
#version 450
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

// Uniforms matching PlanetGenerator parameters
layout(set = 0, binding = 0, std430) buffer Params {
    vec3 block_origin;
    float planet_radius;
    float max_terrain_height;
    float sea_level_bias;
    int lod_index;
    int block_size;
};

// Plate tectonic data as a texture (1024×512 RGBA8)
layout(set = 0, binding = 1) uniform sampler2D u_plate_tex;
layout(set = 0, binding = 2) uniform sampler2D u_data_tex;

// Noise textures (pre-baked or procedural)
// FastNoiseLite can be replicated in GLSL or baked into 3D textures

// Output SDF values
layout(set = 0, binding = 3, std430) buffer Output {
    float sdf[];
};

vec2 dir_to_equirect(vec3 dir) {
    float lon = atan(dir.z, dir.x);
    float lat = asin(clamp(dir.y, -1.0, 1.0));
    return vec2(lon / (2.0 * 3.14159265) + 0.5, lat / 3.14159265 + 0.5);
}

void main() {
    ivec3 local = ivec3(gl_GlobalInvocationID);
    if (any(greaterThanEqual(local, ivec3(block_size)))) return;

    int voxel_scale = 1 << lod_index;
    vec3 world_pos = block_origin + vec3(local) * float(voxel_scale);

    float dist = length(world_pos);
    vec3 dir = world_pos / max(dist, 0.001);
    vec2 uv = dir_to_equirect(dir);

    // Sample plate data
    vec4 plate_data = texture(u_plate_tex, uv);
    vec4 extra_data = texture(u_data_tex, uv);

    float falloff = plate_data.a;
    float is_oceanic = plate_data.g;
    float bnd_type = plate_data.b * 8.0;

    // Compute surface height (simplified — full version matches planet_generator.gd)
    float surface_h = planet_radius;
    // ... plate bias, noise layers via texture lookups or analytical noise ...

    float sdf_value = dist - surface_h;
    sdf[local.z * block_size * block_size + local.y * block_size + local.x] = sdf_value;
}
```

**Pre-baked 3D noise textures:** Instead of replicating FastNoiseLite in GLSL (complex), bake noise octaves into 3D textures at initialization and sample them in the compute shader. This trades memory for simplicity:

```gdscript
# In PlanetGenerator.initialize()
var noise_3d_tex := _bake_noise_to_3d_texture(_noise_base, 256, planet_radius)
# Upload as uniform to compute shader
```

---

### 3.5 Voxel Spacing Configuration

#### 3.5.1 Problem Statement

Currently, 1 voxel = 1 world unit. The `planet_scene.gd` comment confirms: *"1 voxel unit = 1 metre (planet_radius IS the voxel-space radius). There is no separate 'voxel size' setting."*

For different use cases:
- **Planet terrain:** 1 voxel = 1 m is appropriate (40,000 voxels radius)
- **Detailed architecture:** 1 voxel = 10 cm needed (0.1 m)
- **Massive worlds:** 1 voxel = 10 m could reduce memory

#### 3.5.2 Design: `voxel_size` Property

Add a `voxel_size` property to `VoxelLodTerrain` that acts as a uniform scale factor:

```cpp
// voxel_lod_terrain.h
float _voxel_size = 1.0f; // metres per voxel at LOD0

void set_voxel_size(float size);
float get_voxel_size() const;
```

**All coordinate conversions must respect this:**

```cpp
// World position → voxel position
Vector3i world_to_voxel(Vector3 world_pos) const {
    Vector3 local = get_global_transform().affine_inverse().xform(world_pos);
    return math::floor_to_int(local / _voxel_size);
}

// Voxel position → world position
Vector3 voxel_to_world(Vector3i voxel_pos) const {
    return get_global_transform().xform(Vector3(voxel_pos) * _voxel_size);
}
```

**Impact on mesh generation:** The Transvoxel mesher produces vertices in voxel coordinates. The `VoxelMeshBlockVLT` transform must scale by `voxel_size`:

```cpp
// In apply_mesh_update() — adjust block transform
Transform3D block_transform;
block_transform.origin = Vector3(mesh_block_pos) *
    (mesh_block_size << lod_index) * _voxel_size;
block_transform.basis = Basis().scaled(
    Vector3(_voxel_size, _voxel_size, _voxel_size) * (1 << lod_index));
```

**Impact on SDF values:** SDF values represent distances in voxel units. With `voxel_size = 0.1`, a voxel covers 0.1 m, so SDF quantization has 10× better precision per metre. The `QUANTIZED_SDF_16_BITS_SCALE` constant may need to be adjusted per-instance.

**Impact on LOD distances:** All distance parameters are in voxels, so `lod_distance = 128` with `voxel_size = 0.1` means 12.8 m in world space. Users must adjust accordingly, or we provide a helper:

```gdscript
# Helper: set LOD distance in world metres
func set_lod_distance_metres(metres: float) -> void:
    volume.lod_distance = metres / volume.voxel_size
```

**GDScript configuration:**

```gdscript
# 10 cm voxels for detailed terrain
volume.voxel_size = 0.1
volume.lod_distance = 1280   # 128 metres / 0.1 = 1280 voxels
volume.mesh_block_size = 32  # Each block = 32 × 0.1 = 3.2 m at LOD0
volume.voxel_bounds = AABB(
    Vector3(-1000, -1000, -1000),  # 1000 voxels = 100 m
    Vector3(2000, 2000, 2000)
)

# Or for planet (1m voxels, current behavior)
volume.voxel_size = 1.0
volume.lod_distance = 128
```

#### 3.5.3 SDF Scale Normalization

When `voxel_size` changes, the SDF scale factor must adjust so that the 16-bit quantized range covers a meaningful distance:

```cpp
float get_effective_sdf_scale() const {
    // Default: 0.002 (covers ±500 voxels = ±500m at 1m/voxel)
    // At 0.1m/voxel: should cover ±500 voxels = ±50m, which is still fine
    // At 10m/voxel: covers ±500 voxels = ±5000m, may need wider range
    return constants::QUANTIZED_SDF_16_BITS_SCALE;
    // Could adjust: return base_scale * _voxel_size;
}
```

For most cases the default scale works. Add an override if needed:

```cpp
void set_sdf_scale(float scale); // 0 = auto
float get_sdf_scale() const;
```

---

## 4. Code Examples

### 4.1 Per-LOD Distance Table (C++ patch)

```cpp
// File: terrain/variable_lod/voxel_lod_terrain.h
// Add after `float get_secondary_lod_distance() const;`

    void set_lod_distances(const PackedFloat64Array &distances);
    PackedFloat64Array get_lod_distances() const;

// File: terrain/variable_lod/voxel_lod_terrain.cpp

void VoxelLodTerrain::set_lod_distances(const PackedFloat64Array &distances) {
    _update_data->wait_for_end_of_task();
    _update_data->settings.custom_lod_distances = distances;
    _update_data->settings.use_custom_lod_distances = distances.size() > 0;
}

PackedFloat64Array VoxelLodTerrain::get_lod_distances() const {
    return _update_data->settings.custom_lod_distances;
}
```

### 4.2 Dynamic Thread Count (C++ patch)

```cpp
// File: engine/voxel_engine.cpp

void VoxelEngine::set_thread_count(uint32_t count) {
    const int hw = Thread::get_hardware_concurrency();
    count = math::clamp(count, 1u, (uint32_t)math::max(hw - 1, 1));
    _general_thread_pool.set_thread_count(count);
    ZN_PRINT_VERBOSE(format("Voxel: thread count changed to {}", count));
}

int VoxelEngine::get_thread_count() const {
    return _general_thread_pool.get_thread_count();
}
```

### 4.3 Voxel Size Property (C++ patch)

```cpp
// File: terrain/variable_lod/voxel_lod_terrain.cpp

void VoxelLodTerrain::set_voxel_size(float size) {
    ERR_FAIL_COND(size <= 0.0f);
    _voxel_size = size;
    // Recompute mesh block transforms for all existing blocks
    for (unsigned int lod = 0; lod < get_lod_count(); ++lod) {
        _mesh_maps_per_lod[lod].for_each_block([&](VoxelMeshBlockVLT &block) {
            update_block_transform(block, lod);
        });
    }
}

float VoxelLodTerrain::get_voxel_size() const {
    return _voxel_size;
}

// In _bind_methods():
ClassDB::bind_method(D_METHOD("set_voxel_size", "size"), &VoxelLodTerrain::set_voxel_size);
ClassDB::bind_method(D_METHOD("get_voxel_size"), &VoxelLodTerrain::get_voxel_size);
ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "voxel_size", PROPERTY_HINT_RANGE, "0.001,100.0,0.001"),
    "set_voxel_size", "get_voxel_size");
```

### 4.4 GPU SDF Generation Integration

```cpp
// File: generators/voxel_generator_script.h — add virtuals

GDVIRTUAL0RC(String, _get_sdf_compute_shader)
GDVIRTUAL0RC(Dictionary, _get_sdf_compute_params)
GDVIRTUAL3(_generate_materials, Ref<godot::VoxelBuffer>, Vector3i, int)

bool supports_gpu_sdf() const {
    return GDVIRTUAL_IS_OVERRIDDEN(_get_sdf_compute_shader);
}

// File: generators/voxel_generator_script.cpp

bool VoxelGeneratorScript::supports_gpu_sdf() const {
    String shader_source;
    if (GDVIRTUAL_CALL(_get_sdf_compute_shader, shader_source)) {
        return !shader_source.is_empty();
    }
    return false;
}

void VoxelGeneratorScript::generate_materials(
    std::shared_ptr<VoxelBuffer> voxels, Vector3i origin, int lod) {
    Ref<godot::VoxelBuffer> buffer_wrapper;
    buffer_wrapper.instantiate();
    buffer_wrapper->get_buffer() = *voxels;
    GDVIRTUAL_CALL(_generate_materials, buffer_wrapper, origin, lod);
}
```

---

## 5. Implementation Roadmap

### Phase 1: Foundation (Weeks 1–2)
**Lowest risk, highest immediate value**

- [x] **5.1** Add `voxel_size` property to `VoxelLodTerrain`
  - Added `float _voxel_size = 1.0f` with getter/setter
  - Bound to GDScript with PROPERTY_HINT_RANGE in "Advanced" group
  - Test: verify planet renders identically at `voxel_size = 1.0`

- [x] **5.2** Add `lod_distances` PackedFloat64Array property
  - Modified `VoxelLodTerrainUpdateData::Settings` with `custom_lod_distances` + `use_custom_lod_distances`
  - Modified `get_relative_lod_distance_in_chunks()` in clipbox streaming (both call sites)
  - Updated `get_lod_distances(Span<float>)` for VoxelInstancer to use custom table
  - Bound to GDScript with static_cast overload disambiguation
  - Test: verify custom distance table matches formula output for equivalent values

- [x] **5.3** Expose thread count to GDScript via `VoxelEngine`
  - `set_thread_count()` / `get_thread_count()` confirmed already bound
  - Added `get_hardware_thread_count()` via `zylann::Thread::get_hardware_concurrency()`

### Phase 2: Streaming Optimization (Weeks 3–4)
**Moderate risk, large performance gain for disk I/O**

- [x] **5.4** Configurable `VoxelStreamCache` size
  - Added `set_cache_size()` / `get_cache_size()` to `VoxelStreamSQLite`
  - Changed `CACHE_SIZE` constant to `DEFAULT_CACHE_SIZE` with runtime `_cache_size` member
  - Bound to GDScript with PROPERTY_HINT_RANGE (1–4096)

- [x] **5.5** Zstd compression option
  - Already implemented in `compressed_data.h/.cpp` (COMPRESSION_ZSTD = 3)
  - Already bound via `set_compression_mode()` on `VoxelStream` base class
  - No additional work needed

- [x] **5.6** LOD-filtered bulk load
  - Added `BulkLoadParams` struct and `load_blocks_bulk()` virtual to `VoxelStream`
  - Added `supports_bulk_load()` virtual
  - Default: filters `load_all_blocks` results by LOD range + region
  - `VoxelStreamSQLite` override with `supports_bulk_load() = true`

### Phase 3: Threading Improvements (Weeks 5–6)
**Moderate risk, improves multi-core utilization**

- [ ] **5.7** Dynamic thread scaling
  - Deferred: manual `set_thread_count()` + `get_hardware_thread_count()` covers the main use case
  - Auto-scaling can be implemented as GDScript logic using `VoxelEngine.get_stats()`
  - Test: monitor thread utilization during camera flyover

- [x] **5.8** Separate generation and meshing pools (optional)
  - Added second `ThreadedTaskRunner _generation_thread_pool` to `VoxelEngine`
  - Added `push_generation_task()` / `push_generation_tasks()` routing methods
  - Added `set_generation_thread_count()` / `get_generation_thread_count()` — setting count > 0 activates the pool
  - Updated `BufferedTaskScheduler` with `push_generation_task()` and `_generation_tasks` vector
  - Routed `GenerateBlockTask` via `push_generation_task` in update_task.cpp and load_block_data_task.cpp
  - Updated `process()`, `wait_and_clear_all_tasks()`, destructor, Stats struct
  - Exposed to GDScript via `VoxelEngine` singleton

### Phase 4: GPU Pipeline (Weeks 7–10)
**Higher risk, highest potential performance gain**

- [x] **5.9** GPU SDF for VoxelGeneratorScript
  - Added `_get_sdf_compute_shader` / `_get_sdf_compute_params` / `_generate_materials` GDVIRTUAL methods
  - Added `has_sdf_compute_shader()`, `get_sdf_compute_shader_source()`, `get_sdf_compute_params()` C++ methods
  - `_generate_materials` falls back to `_generate_block` if not implemented
  - Overrode `supports_shaders()` and `get_shader_source()` in `VoxelGeneratorScript` for GPU pipeline integration
  - Added `run_gpu_material_pass()` to `GenerateBlockTask` — after GPU SDF, calls `generate_materials()` on script generators
  - Full pipeline: GPU dispatches SDF via existing `GenerateBlockGPUTask`, then CPU runs material assignment
  - Test: planet SDF on GPU, materials on CPU

- [x] **5.10** Planet-specific SDF compute shader
  - **Engine extension** — Added `_get_sdf_shader_textures()` GDVIRTUAL to `VoxelGeneratorScript`:
    - Returns `Array` of `[String name, Resource/PackedByteArray]` pairs
    - `PackedByteArray` entries → `create_storage_buffer()` with `#define <name>_BINDING <N>`
    - `get_shader_source()` auto-prepends FastNoiseLite GLSL library (~2330 lines) from `g_fast_noise_lite_shader[]`
  - **Storage buffer pipeline** (replaced texture-based approach in session 25):
    - ~38KB buffer: header (32B) + 1200 Voronoi points × 32B + 16 plates × 4B (padded to 16B)
    - Per-voxel brute-force Voronoi lookup + domain warp — sub-metre accurate, no quantization
    - Domain warp seeds packed as raw int32 via `encode_s32()` / `floatBitsToInt()` (session 27 fix)
  - **GLSL compute shader** — `Scripts/planet/planet_sdf_compute.glsl.txt` (~520 lines):
    - Full port of `_compute_sdf_and_td()` to GPU — all 8 noise layers
    - Uses FNL GLSL: `fnlCreateState()`, `fnlGetNoise3D()` with exact same seeds/params as GDScript
    - **Continuous 0-1 feature masks** (session 28 redesign, session 31 fix):
      boundary-type-dependent layers (mountain, trench, rift) use **per-side evaluation
      + lerp** — each side's full feature contribution (using its own bnd_type's noise)
      is computed independently, then cross-faded via `blend_t` and scaled once by
      `border_w`.  This eliminates dips at boundary-type transitions (e.g. MOUNTAIN→RIFT)
      where the old per-mask multiplication caused both masks to go through zero.
    - Early-out for blocks fully inside/outside terrain budget (PLANET_RADIUS ± skip_margin)
    - All boundary logic: mountain floor fix, convergence bonus, continental floor
  - **GDScript integration** — `planet_generator.gd` GPU overrides:
    - `use_gpu_sdf: bool` flag (default false, opt-in)
    - `initialize()` packs tectonic storage buffer + loads GLSL when `use_gpu_sdf = true`
    - `_get_sdf_compute_shader()` → returns GLSL source
    - `_get_sdf_shader_textures()` → returns `[["u_tectonic_data", _gpu_tectonic_buffer]]`
    - `_generate_materials()` → CPU material pass after GPU SDF (skips at LOD≥3, sdf-proximity culled)
  - Pipeline: GPU dispatches SDF (stage 0→1), CPU assigns materials (stage 2 via `_generate_materials`)
  - CPU GDScript updated with identical continuous-mask logic for consistency

- [ ] **5.11** GPU Transvoxel meshing (future / experimental)
  - Port marching cubes tables to storage buffers
  - Implement prefix-sum vertex counting
  - Implement vertex/index generation compute pass
  - Handle mesh data transfer back to `RenderingServer`

### Phase 5: Advanced Streaming (Weeks 11–12)
**Higher risk, benefits large persistent worlds**

- [x] **5.12** Memory-mapped region files
  - Implemented `RegionFileMMap` utility class (cross-platform read-only mmap)
  - Windows: `CreateFileA` → `CreateFileMappingA` → `MapViewOfFile` (opaque `void*` handles to avoid leaking `<windows.h>`)
  - POSIX: `open()` → `fstat()` → `mmap(PROT_READ, MAP_PRIVATE)`
  - Integrated directly into existing `RegionFile` — no new stream class needed
  - `RegionFile::open()`: opens mmap alongside FileAccess (graceful fallback if mmap fails)
  - `RegionFile::load_block()`: mmap fast path reads size prefix + block data from mapped memory, calls `BlockSerializer::decompress_and_deserialize(Span<const uint8_t>, ...)` — zero FileAccess seeks/copies
  - `RegionFile::save_block()`: flushes FileAccess then remaps to pick up grown file
  - `RegionFile::close()`: closes mmap + FileAccess
  - Header uses opaque `void*` handles — no `<windows.h>` leak to other TUs
  - Test: compare throughput vs SQLite for bulk load

- [x] **5.13** LOD data downsampling cache
  - Implemented `try_downsample_block()` in `VoxelData` — reads 8 child blocks at LOD-1, downsamples into target block
  - Integrated into `GenerateBlockTask::run_cpu_generation()` — tries downsampling before invoking generator for LOD > 0
  - Uses existing `VoxelBuffer::downscale_to()` for correct SDF/material averaging
  - Thread-safe: acquires spatial lock + map read lock on source LOD
  - Test: verify correct SDF at LOD transitions

### Phase 6: Advanced Noise and SIMD (Weeks 13–14)
**High value for planet generation performance**

- [x] **5.14** VoxelAdvancedNoise resource
  - Ported UE5 VoxelPlugin's AdvancedNoise3D node as a Godot Resource class
  - 12 noise types: Smooth/Billowy/Ridged × Perlin/Cellular/Simplex/Value
  - Multi-octave mixing with per-octave type and strength control
  - Matches UE5 algorithm: per-octave seed LCG, NaN guard, amplitude normalization
  - Self-contained noise implementations (Perlin 2D/3D, Simplex 2D/3D, Cellular 2D/3D, Value 2D/3D)
  - `fill_buffer_3d()` for efficient batch operations (returns PackedFloat32Array)
  - Full ClassDB bindings with properties, enum, and static `is_simd_available()` method
  - Registered in `register_types.cpp`, added to build in `common.py`

- [x] **5.15** ISPC SIMD acceleration for noise
  - Added `voxel_ispc` build option (default off, x86-only)
  - Created `ispc_builder.py` — SCons tool that auto-detects ISPC compiler, compiles `.ispc` → `.obj`
  - Created `voxel_advanced_noise_ispc.ispc` — SIMD kernel with 3 exported functions:
    - `VoxelAdvancedNoise3D_FillBuffer` — grid-based 3D noise batch
    - `VoxelAdvancedNoise2D_FillBuffer` — grid-based 2D noise batch
    - `VoxelAdvancedNoise3D_Batch` — arbitrary position array batch
  - Created `voxel_noise_impl.isph` — ISPC noise functions (Perlin, Simplex, Cellular, Value 2D/3D)
  - Created `voxel_ispc_minimal.isph` — self-contained math types (float2/3, int2/3, interpolation)
  - Created `voxel_advanced_noise_simd.h/.cpp` — C++ dispatch wrapper
  - `fill_buffer_3d()` auto-dispatches to ISPC when compiled with `voxel_ispc=yes`
  - Falls back to scalar C++ when ISPC is not available
  - ISPC targets: SSE4 (4-wide) + AVX2 (8-wide) — expected 3-6× speedup for batch noise
  - Install: download ISPC from https://github.com/ispc/ispc/releases, set `ISPC_PATH` env var or add to PATH
  - Build: `scons ... voxel_ispc=yes`
  - Test: `VoxelAdvancedNoise.is_simd_available()` in GDScript to verify ISPC is active

- [x] **5.16** ISPC Noise Expansion — All noise types + Domain Warp
  - Completely rewrote `voxel_noise_impl.isph` (~700 lines) with SIMD-optimized implementations:
    - Branchless Simplex noise (2D/3D) — no divergent `if` chains, uses permutation table
    - TrueDistanceCellular noise (2D/3D) — Euclidean distance (not squared), matches VoxelPlugin 2.0
    - Domain Warp functions — warp input coordinates using gradient-based displacement
    - Optimized gradient tables with `MakeRegister` for uniform access
  - Updated `voxel_ispc_minimal.isph` with `lerp`, `InvSqrt`, `InterpHermite`, `SmoothStep`, operator overloads
  - Updated `voxel_advanced_noise_ispc.ispc` — added 3 TrueDistanceCellular octave types (Smooth/Billowy/Ridged)
  - Created `voxel_domain_warp_ispc.ispc` (~346 lines):
    - `VoxelDomainWarp3D_Grid` — grid-based 3D domain warp with configurable amplitude/frequency
    - `VoxelDomainWarp2D_Grid` — grid-based 2D domain warp
    - `VoxelDomainWarp3D_Batch` — arbitrary position array domain warp
    - 10 standalone noise batch exports: `VoxelNoise_{Perlin,Simplex,Cellular,Value,TrueDistanceCellular}_{2D,3D}_Batch`
  - Updated `voxel_advanced_noise.h/.cpp` — 3 new enum values (TYPE_TRUE_DISTANCE_CELLULAR_*)

- [x] **5.17** VoxelNoiseBatch — GDScript/C++ wrapper for ISPC noise
  - Created `voxel_noise_batch.h/.cpp` — `VoxelNoiseBatch` utility class (Object-derived)
  - 10 static noise batch methods callable from GDScript and C++:
    - `perlin_3d_batch(px, py, pz, seed)` / `perlin_2d_batch(px, py, seed)`
    - `simplex_3d_batch(px, py, pz, seed)` / `simplex_2d_batch(px, py, seed)`
    - `cellular_3d_batch(px, py, pz, seed)` / `cellular_2d_batch(px, py, seed)`
    - `value_3d_batch(px, py, pz, seed)` / `value_2d_batch(px, py, seed)`
    - `true_distance_cellular_3d_batch(...)` / `true_distance_cellular_2d_batch(...)`
  - 3 domain warp methods:
    - `domain_warp_3d_grid(origin, size, spacing, seed, amp, freq)` → `[warped_x, warped_y, warped_z]`
    - `domain_warp_2d_grid(origin, size, spacing, seed, amp, freq)` → `[warped_x, warped_y]`
    - `domain_warp_3d_batch(px, py, pz, seed, amp, freq)` → `[warped_x, warped_y, warped_z]`
  - Position helper methods for VoxelGeneratorScript integration:
    - `prepare_positions_3d(origin, size, lod)` → `[PackedFloat32Array x, y, z]`
    - `prepare_positions_2d(origin_xz, size_xz, spacing)` → `[PackedFloat32Array x, y]`
  - `is_simd_available()` — returns true when compiled with `voxel_ispc=yes`
  - All methods auto-dispatch to ISPC (SSE4/AVX2) or fall back to scalar C++
  - Updated `voxel_advanced_noise_simd.h/.cpp` with 13 new C++ ↔ ISPC dispatch functions
  - Registered in `register_types.cpp`, added to build in `common.py`
  - **VoxelGeneratorScript integration**: Call from `_generate_block()`:
    ```gdscript
    var positions = VoxelNoiseBatch.prepare_positions_3d(origin, size, lod)
    var base = VoxelNoiseBatch.simplex_3d_batch(positions[0], positions[1], positions[2], 42)
    # base is PackedFloat32Array with one value per voxel — use to fill SDF buffer
    ```

---

## 6. Conclusion and Considerations

### Performance Budget Estimates

| Operation | Current (CPU, GDScript) | After Phase 4 (GPU SDF + CPU mat) | Projected Speedup |
|---|---|---|---|
| Block generation (32³, LOD0) | ~15–30 ms | ~0.5–2 ms (GPU SDF) + ~5 ms (CPU mat) | 3–6× |
| Block meshing (32³ Transvoxel) | ~2–8 ms | ~2–8 ms (still CPU) | 1× (future: 10× with GPU) |
| Block save (SQLite, LZ4) | ~0.5–2 ms | ~0.3–1.5 ms (Zstd, larger cache) | 1.3× |
| Block load (SQLite) | ~0.3–1 ms | ~0.1–0.3 ms (mmap region) | 3× |
| LOD transition recompute | Full regeneration | Downsample from children | 5–10× |
| Batch noise (32³ C++ scalar) | ~2–5 ms | ~0.4–1.2 ms (ISPC AVX2) | 3–6× |

### Risk Mitigation

1. **Backward compatibility:** All new properties have defaults matching current behavior. `voxel_size = 1.0`, empty `lod_distances`, thread scaling disabled by default.

2. **Incremental deployment:** Each phase is independently testable. The planet-voxels project can adopt features one at a time.

3. **GDScript generator compatibility:** The GPU SDF path is opt-in via `_get_sdf_compute_shader`. Generators that don't override it continue to work identically.

4. **Thread safety:** All new features respect the existing threading contract: `_generate_block` on workers, `_plates` read-only, scene tree access only on main thread.

5. **Build impact:** Phase 1–3 changes are header/cpp only in `modules/voxel/`. No new third-party dependencies. Phase 4 adds GLSL compute shaders (already supported infrastructure). Phase 5 adds platform-specific mmap (conditional compilation). Phase 6 adds optional ISPC dependency — `voxel_ispc=yes` flag, auto-detected, scalar fallback when not available.

### UE5 VoxelPlugin Concepts NOT Adapted (and why)

| UE5 Feature | Reason for Exclusion |
|---|---|
| Hierarchical instanced static mesh (HISM) for voxels | Godot uses `MultiMeshInstance3D` via `VoxelInstancer` — different API |
| Nanite integration | Godot has no Nanite equivalent |
| World Partition streaming | Godot uses `VoxelViewer`-based streaming which is more flexible for non-flat worlds |
| Async collision cooking | Godot 4 + Jolt doesn't expose async collision mesh building |
| Voxel painting with UE5 material instances | Godot uses MIXEL4 channel packing — already more efficient |

### Key Files to Modify (Summary)

| File | Changes |
|---|---|
| `terrain/variable_lod/voxel_lod_terrain.h/.cpp` | `voxel_size`, `lod_distances`, property bindings |
| `terrain/variable_lod/voxel_lod_terrain_update_data.h` | `Settings` struct: custom LOD distances |
| `terrain/variable_lod/voxel_lod_terrain_update_clipbox_streaming.cpp` | Modified distance computation |
| `engine/voxel_engine.h/.cpp` | Thread scaling, pool splitting, GDScript bindings |
| `generators/voxel_generator_script.h/.cpp` | GPU SDF virtuals |
| `generators/generate_block_task.cpp` | Hybrid GPU/CPU path, downsampling optimization |
| `streams/compressed_data.h/.cpp` | Zstd compression |
| `streams/voxel_stream_cache.h/.cpp` | Configurable cache size, priority eviction |
| `storage/voxel_data.h/.cpp` | `try_downsample_block()` |
| `constants/voxel_constants.h` | SDF scale adjustments for variable voxel size |
| `util/noise/voxel_advanced_noise.h/.cpp` | VoxelAdvancedNoise resource, ISPC dispatch in fill_buffer_3d |
| `util/noise/voxel_advanced_noise_simd.h/.cpp` | C++ ↔ ISPC dispatch wrapper (16 functions total) |
| `util/noise/voxel_noise_batch.h/.cpp` | GDScript/C++ batch noise API — 10 noise + 3 domain warp + position helpers |
| `util/noise/voxel_advanced_noise_ispc.ispc` | ISPC SIMD noise kernel (15 octave types) |
| `util/noise/voxel_domain_warp_ispc.ispc` | ISPC domain warp kernel + 10 standalone noise batch exports |
| `util/noise/voxel_noise_impl.isph` | ISPC noise function implementations (Perlin, Simplex, Cellular, Value, TrueDistanceCellular, DomainWarp) |
| `util/noise/voxel_ispc_minimal.isph` | ISPC math types and utilities |
| `ispc_builder.py` | SCons ISPC compiler integration |
| `common.py` | `voxel_ispc` build option |
| `SCsub` | ISPC build rules |
| `shaders/fast_noise_lite_shader.h` | FNL GLSL library (~2330 lines), auto-prepended to user compute shaders |
| `Scripts/planet/planet_sdf_compute.glsl` | GPU SDF compute shader — full planet terrain (354 lines, 8 noise layers + plate textures) |
| `Scripts/planet/bake_gpu_tectonic_textures.gd` | Bakes 3 RGBAH equirect textures from PlateTectonics for GPU shader |
| `streams/region/region_file_mmap.h` | Cross-platform read-only mmap utility (opaque handles, no `<windows.h>` leak) |
| `streams/region/region_file_mmap.cpp` | Windows `CreateFileMapping`/`MapViewOfFile` + POSIX `mmap`/`munmap` implementations |
| `streams/region/region_file.h/.cpp` | Added `RegionFileMMap` member; mmap fast path in `load_block()`; remap after `save_block()` |
