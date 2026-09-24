#pragma once

// Uses engine-internal renderer headers, so module builds only.
#if defined(VOXEL_ENABLE_GPU) && defined(ZN_GODOT)
#define VOXEL_ENABLE_GPU_DRIVEN_RENDERING

#include "core/math/transform_3d.h"
#include "core/object/object.h"
#include "core/os/mutex.h"
#include "core/templates/hash_map.h"
#include "core/templates/rid.h"
#include "core/variant/dictionary.h"
#include "core/variant/typed_array.h"
#include "scene/resources/compositor.h"

#include "core/gpu_range_allocator.h"
#include "core/gpu_vertex_pack.h"

#include <atomic>
#include <cstring>
#include <vector>

class RenderData;
class World3D;

namespace zylann::voxel {

// GPU-driven renderer used by VoxelLodTerrain's RENDER_MODE_GPU_DRIVEN.
// Chunk geometry lives in RenderingDevice storage "mega-buffers". Every frame a compute shader frustum-culls chunks
// into an indirect argument buffer, then all chunks are drawn with one indirect multi-draw, vertex-pulling packed
// vertices from the storage buffers. The draw is injected with a CompositorEffect (POST_OPAQUE) attached to the
// terrain's scenario, which editor viewports share, so it also renders in the editor with the editor camera.
//
// Threading: public methods are main-thread only. They queue commands consumed on the render thread, which owns every
// RenderingDevice resource.
//
// Shading is a built-in port of planet_v4.gdshader (parameters taken from the terrain material), lit like Godot's
// scene shader: the render's first directional light, the environment's ambient (sky radiance included), sky
// reflections and fog. Limitations vs the traditional path: no custom ShaderMaterial code, shadows, GI, volumetric
// fog or fog aerial perspective, single view (no XR), Camera3D-level compositors override the scenario one.
class VoxelGpuDrivenRenderer : public Object {
public:
	static constexpr uint32_t INVALID_CHUNK = 0xffffffffu;

	// Material look, same parameters (and defaults) as demo_eden/shaders/planet_v4.gdshader
	struct Style {
		float planet_radius = 0.f; // 0: flat terrain, up is +Y
		float vibrance = 1.18f;
		float slope_rock_start = 0.6f;
		float slope_rock_end = 0.38f;
		float snow_temperature = 0.22f;
		float snow_blend = 0.06f;
		float gully_darkening = 0.3f;
		float ridge_highlight = 0.25f;
		float dryness_tint = 0.6f;
		float detail_normal_strength = 0.f;
		float detail_normal_scale = 6.f;
		// Low-poly styling. All zero is the smooth look, so existing materials are unaffected.
		float facet_shading = 0.f; // 0: smooth normals, 1: one normal per triangle
		float light_bands = 0.f; // 0: continuous, >= 2: quantize diffuse into that many steps
		float palette_snap = 0.f; // 0: blended materials, 1: flat per-triangle palette color
		float facet_tint = 0.f; // random per-triangle brightness, keeps flat facets readable
		// Material palette: albedo rgb + roughness, one row per material index. The terrain material can
		// override it through `material_colors` / `material_roughness`, like far_terrain.gdshader does.
		float palette[8][4] = {
			{ 0.075f, 0.20f, 0.045f, 0.88f }, // grass
			{ 0.27f, 0.245f, 0.215f, 0.72f }, // rock
			{ 0.82f, 0.86f, 0.92f, 0.42f }, // snow
			{ 0.56f, 0.44f, 0.24f, 0.78f }, // sand
			{ 0.21f, 0.13f, 0.065f, 0.88f }, // dirt
			{ 0.05f, 0.14f, 0.065f, 0.88f }, // moss
			{ 0.20f, 0.30f, 0.36f, 0.62f }, // ocean floor, bright until water exists
			{ 1.f, 0.f, 1.f, 0.88f }, // unused
		};

		bool operator==(const Style &o) const {
			return memcmp(this, &o, sizeof(Style)) == 0;
		}
	};

	VoxelGpuDrivenRenderer();
	// Don't memdelete: RenderingDevice resources must be released on the render thread. This defers deletion there.
	void destroy();

	// Call every frame. Keeps the effect registered in whichever compositor the scenario currently uses.
	void update(World3D &world, const Transform3D &terrain_transform, bool visible, const Style &style);
	// Unregisters the effect (e.g. terrain left the tree), and removes our fallback compositor from `world` if we
	// installed it there. `update` re-registers it.
	void detach(World3D *world);

	uint32_t create_chunk();
	// `origin` in terrain-local units, packed positions are relative to it
	void set_chunk_mesh(uint32_t id, gpu_driven::PackedMesh &&mesh, Vector3 origin);
	void set_chunk_visible(uint32_t id, bool visible);
	// Far-field sectors shade from baked AO and a single material, without the Transvoxel seam and climate logic
	void set_chunk_far(uint32_t id, bool far);
	// Mask in shader order (-x +x -y +y -z +z), same as the `u_transition_mask` uniform
	void set_chunk_transition_mask(uint32_t id, uint8_t mask);
	void remove_chunk(uint32_t id);

	Dictionary get_stats() const;

	// Only for the deferred deletion path
	~VoxelGpuDrivenRenderer();

private:
	struct Op {
		enum Type : uint8_t { SET_MESH, SET_VISIBLE, SET_TRANSITION_MASK, SET_FAR, REMOVE };
		Type type;
		bool visible = false;
		bool far = false;
		uint8_t transition_mask = 0;
		uint32_t id;
		Vector3 origin;
		gpu_driven::PackedMesh mesh;
	};

	// Mirrors `Chunk` in the shaders, std430, 64 bytes
	struct ChunkGPU {
		float origin[4]; // w unused
		float aabb_min[4];
		float aabb_max[4];
		// x = vertex offset, y = first index (draw's first_vertex), z = index count,
		// w = bit 0 visible | bit 1 wide (32-bit) indices | transition mask << 8
		uint32_t info[4];
	};
	static_assert(sizeof(ChunkGPU) == 64, "Must match shader");

	struct ChunkRecord {
		uint32_t vertex_offset = 0;
		uint32_t vertex_count = 0;
		uint32_t index_offset = 0;
		uint32_t index_count = 0;
	};

	void push_op(Op &&op);

	// Render thread
	void _render_callback(int p_callback_type, RenderData *p_render_data);
	void _render_thread_free();
	bool _ensure_initialized();
	void _apply_ops();
	void _apply_set_mesh(Op &op);
	void _free_chunk_geometry(uint32_t id);
	void _ensure_chunk_capacity(uint32_t id);
	bool _grow_buffer(RID &buffer, uint32_t old_size, uint32_t new_size, bool indirect);
	bool _grow_index_buffer(uint32_t old_count, uint32_t new_count);
	void _mark_chunk_dirty(uint32_t id);
	void _upload_dirty_chunks();
	bool _ensure_uniform_sets();
	RID _get_render_pipeline(int64_t framebuffer_format, int samples);
	RID _get_radiance_uniform_set(RID radiance);
	void _on_visible_count_read(const Vector<uint8_t> &data);

	// Main thread state
	RID _effect;
	// Installed in the World3D when it has no compositor
	Ref<Compositor> _own_compositor;
	// Compositor we last injected our effect into, and the resource's own effect list we saw there
	Ref<Compositor> _injected_compositor;
	TypedArray<RID> _injected_user_effects;
	std::vector<uint32_t> _free_ids;
	uint32_t _next_id = 0;

	// Shared, guarded by _mutex
	mutable Mutex _mutex;
	std::vector<Op> _pending_ops;
	Transform3D _terrain_transform;
	bool _visible = true;
	Style _style;
	uint32_t _style_version = 1;

	// Stats written on the render thread, read on the main thread (guarded by _mutex)
	struct Stats {
		uint32_t chunk_count = 0;
		uint32_t chunk_slots = 0;
		uint64_t vertex_count = 0;
		uint64_t index_count = 0;
		uint64_t vertex_capacity = 0;
		uint64_t index_capacity = 0;
		uint64_t uploaded_bytes_total = 0;
		uint32_t indirect_draws_total = 0;
		// Chunks that passed frustum culling in the last render that was asked for stats
		uint32_t visible_chunks = 0;
		uint32_t visible_triangles = 0;
		bool initialized = false;
		bool failed = false;
	};
	Stats _stats;

	// Render thread state
	bool _initialized = false;
	bool _init_failed = false;
	RID _render_shader;
	RID _cull_shader;
	RID _cull_pipeline;
	HashMap<int64_t, RID> _render_pipelines;
	RID _vertex_buffer;
	RID _index_buffer;
	RID _index_array;
	RID _chunk_buffer;
	RID _indirect_buffer;
	// Chunk slots near to far, see the cull shader
	RID _order_buffer;
	std::vector<std::pair<float, uint32_t>> _order_keys;
	std::vector<uint32_t> _order;
	// Chunks listed in _order this frame: the cull dispatch and the indirect draw cover only these
	uint32_t _draw_count = 0;
	RID _style_buffer;
	RID _scene_buffer;
	RID _visible_counter_buffer;
	mutable std::atomic_bool _visible_count_requested = false;
	// Render thread: one async readback at a time, so polling stats every frame queues nothing extra
	bool _visible_readback_in_flight = false;
	RID _radiance_sampler;
	RID _radiance_uniform_set;
	RID _radiance_uniform_set_texture;
	uint32_t _uploaded_style_version = 0;
	RID _render_uniform_set;
	RID _cull_uniform_set;
	gpu_driven::RangeAllocator _vertex_allocator;
	gpu_driven::RangeAllocator _index_allocator;
	uint32_t _chunk_capacity = 0;
	uint32_t _chunk_high_water = 0;
	std::vector<ChunkRecord> _chunks;
	std::vector<ChunkGPU> _chunks_gpu;
	uint32_t _dirty_min = 0xffffffffu;
	uint32_t _dirty_max = 0;
};

} // namespace zylann::voxel

#endif // VOXEL_ENABLE_GPU && ZN_GODOT
