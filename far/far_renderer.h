#ifndef VOXEL_FAR_RENDERER_H
#define VOXEL_FAR_RENDERER_H

#include "../gpu_driven/voxel_gpu_driven_renderer.h"
#include "../util/godot/classes/array_mesh.h"
#include "../util/godot/classes/material.h"
#include "../util/godot/direct_mesh_instance.h"
#include "core/far_lod_tree.h"
#include "tasks/far_tasks.h"

#include <memory>
#include <unordered_map>

namespace zylann::voxel::far {

// ---------------------------------------------------------------------------
// FarRenderer
//
// Main-thread half of the far field: turns the mesh arrays produced by
// FarBuildSectorTask into mesh instances, and keeps them in step with the
// residency decided by the update task.
//
// It owns no threading and makes no LOD decisions. Those live in
// `voxel_lod_terrain_update_far_streaming.cpp`, which runs on the update
// thread. This split is the same one VoxelLodTerrain already uses for near
// blocks, and it is why this class only ever touches RenderingServer.
// ---------------------------------------------------------------------------

class FarRenderer {
public:
	~FarRenderer();

	void set_world(World3D *world);
	void set_material(Ref<Material> material);

#ifdef VOXEL_ENABLE_GPU_DRIVEN_RENDERING
	// When set, sectors go into the terrain's GPU-driven buffers (one indirect draw for near and far together)
	// instead of getting a mesh instance each. Switching requires a far restart so sectors are rebuilt.
	void set_gpu_driven_renderer(VoxelGpuDrivenRenderer *renderer) {
		_gpu_renderer = renderer;
	}
#endif
	void set_cast_shadows(bool enabled);
	void set_render_layers_mask(int mask);
	void set_visible(bool visible);

	// Builds meshes for up to `max_uploads` results. Building a Mesh must happen
	// on the main thread, so this is the knob that trades streaming speed
	// against frame time.
	void drain_results(FarSharedState &shared, const FarLodTree &lod_tree, const Transform3D &volume_transform,
			unsigned int max_uploads);

	// Destroys meshes for sectors the LOD tree no longer holds resident.
	//
	// Deliberately a reconcile against current residency rather than a queue of
	// unload events: a sector can be dropped and re-required between two main
	// thread frames, and replaying the drop would destroy a mesh that is
	// resident again -- which nothing would ever rebuild, because residency is
	// what suppresses a reload request.
	void reconcile(const FarLodTree &lod_tree);
	void destroy_all();

	// Re-evaluates which sectors are hidden under the near terrain. Cheap, and
	// it means moving does not require rebuilding anything.
	void refresh_visibility(const FarLodTree &lod_tree, const Vec3f &viewer_position);

	inline size_t get_sector_count() const {
		return _sectors.size();
	}

	uint64_t get_sectors_built() const {
		return _sectors_built;
	}

	uint64_t get_stale_results_discarded() const {
		return _stale_results_discarded;
	}

	// Highest material index seen in any built sector. Stays 0 when the
	// generator exposes no material channel the far field understands, which is
	// the quickest way to tell a working palette from a silently blank one.
	unsigned int get_max_material() const {
		return _max_material;
	}

	void get_statistics(int64_t &out_vertices, int64_t &out_triangles, int64_t &out_empty, int64_t &out_clipped,
			AABB &out_bounds) const;

private:
	struct SectorRender {
		zylann::godot::DirectMeshInstance mesh_instance;
		Ref<ArrayMesh> mesh;
		// Kept so statistics can report world bounds without recomputing where
		// the sector is; on a planet that is not derivable from its grid coords.
		Vector3 local_origin;
		bool visible = true;
		// Counted here rather than from the Mesh, which the GPU-driven path doesn't create
		uint32_t vertex_count = 0;
		uint32_t triangle_count = 0;
#ifdef VOXEL_ENABLE_GPU_DRIVEN_RENDERING
		uint32_t gpu_chunk = VoxelGpuDrivenRenderer::INVALID_CHUNK;
#endif
	};

	Ref<ArrayMesh> build_mesh_resource(const FarMeshArrays &arrays);
	void release_sector(SectorRender &sector);

	std::unordered_map<SectorId, SectorRender, SectorIdHash> _sectors;

	World3D *_world = nullptr;
#ifdef VOXEL_ENABLE_GPU_DRIVEN_RENDERING
	// Not owned. The terrain rebuilds the far field when this changes or the renderer goes away.
	VoxelGpuDrivenRenderer *_gpu_renderer = nullptr;
#endif
	Ref<Material> _material;
	bool _cast_shadows = false;
	int _render_layers_mask = 1;
	bool _visible = true;

	uint64_t _sectors_built = 0;
	uint64_t _stale_results_discarded = 0;
	unsigned int _max_material = 0;
};

} // namespace zylann::voxel::far

#endif // VOXEL_FAR_RENDERER_H
