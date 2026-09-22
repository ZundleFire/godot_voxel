#ifndef VOXEL_FAR_TASKS_H
#define VOXEL_FAR_TASKS_H

#include "../core/far_extract.h"
#include "../core/far_geometry.h"
#include "../core/far_lod_tree.h"
#include "../core/far_mesher.h"
#include "far_cache.h"

#include <atomic>
#include <memory>
#include <vector>

#include "../../generators/voxel_generator.h"
#include "../../util/tasks/threaded_task.h"

namespace zylann::voxel::far {

// ---------------------------------------------------------------------------
// The generate -> mesh -> upload pipeline.
//
// Stages, and why they are split this way:
//
//   1. Load (IO thread pool). Try the cache. A hit skips straight to meshing,
//      which is the whole point of having a cache.
//   2. Generate (compute pool). Sample the generator into a voxel buffer,
//      extract column data, write it back to the cache.
//   3. Mesh (compute pool). Column data to triangles.
//   4. Upload (main thread). Build the mesh resource and hand it to the
//      renderer.
//
// Only stage 4 touches the main thread, and it is deliberately a separate step
// rather than part of stage 3: creating a Mesh resource has to happen on the
// main thread in Godot, so doing it inside the mesh task would mean either
// blocking a worker on the main thread or building the resource on a thread
// that is not allowed to. Splitting it lets the main thread take a bounded
// number of uploads per frame and defer the rest, which is what keeps sector
// streaming from producing frame spikes.
//
// Stages 1-3 are chained on the worker side rather than bounced back through
// the main thread between each, so a cache miss costs one scheduling round
// trip rather than three.
// ---------------------------------------------------------------------------

class VoxelFarTerrain;

// Parameters shared by every task, snapshotted when the pipeline starts so that
// tasks never read mutable node state from a worker thread.
struct FarTaskParams {
	Ref<VoxelGenerator> generator;

	// Vertical span the far field represents, in world units.
	float vertical_min = -256.f;
	float vertical_max = 512.f;

	// Sample the generator one level finer and reduce, instead of sampling
	// directly at the sector's own level.
	//
	// Direct sampling is exact but aliased: level N's sample points are a
	// strict subset of level N-1's, so the levels agree perfectly where they
	// share points, but a feature narrower than a cell is either caught at full
	// strength or missed entirely. Moving between levels then makes ridges and
	// spires blink in and out.
	//
	// Reducing from one level finer costs four times the generator evaluations
	// and antialiases that away. Worth it for the levels close enough to look
	// at; not worth it for the horizon.
	bool antialias = true;
	// Levels at or above this are sampled directly, regardless of `antialias`.
	uint8_t antialias_max_lod = 8;

	// -- Planet mode --------------------------------------------------------
	// When set, columns run radially from `planet_centre` instead of along Y,
	// and the sector grid is a cube face rather than the XZ plane. See
	// far_geometry.h and far_sphere.h.
	bool spherical = false;
	Vec3f planet_centre;
	// Radii, not heights: the terrain lives in a shell between these.
	float radial_min = 0.f;
	float radial_max = 0.f;
	// LOD at which one sector covers a whole cube face.
	uint8_t sphere_root_lod = 1;

	ExtractSettings extract;
	MesherSettings mesher;

	std::shared_ptr<FarCache> cache;
};

// Result handed back to the main thread.
struct FarSectorResult {
	SectorId id;
	// Point the mesh's vertices are relative to. The renderer puts this in the
	// mesh instance's transform -- on a flat world it is the sector's corner,
	// on a planet the centre of its surface patch.
	Vec3f local_origin;
	std::shared_ptr<ColumnField> field;
	std::shared_ptr<FarMeshArrays> mesh;
	bool valid = false;
};

// Thread-safe queue for finished sectors. The node drains it each frame.
class FarResultQueue {
public:
	void push(FarSectorResult &&result);
	// Moves up to `max_count` results out. Returns how many were taken.
	unsigned int pop_batch(std::vector<FarSectorResult> &out, unsigned int max_count);
	void clear();
	size_t size() const;

private:
	mutable Mutex _mutex;
	std::vector<FarSectorResult> _results;
};

// Shared state that outlives any individual task, so that a task finishing
// after the node has been destroyed has somewhere safe to land.
//
// Tasks hold a shared_ptr to this rather than a raw pointer to the node. A far
// sector can be in flight for hundreds of milliseconds; if the node is freed or
// the terrain reconfigured in that window, writing through a node pointer is a
// use-after-free. The cancellation generation lets in-flight work from a
// previous configuration be discarded rather than applied to the new one.
struct FarSharedState {
	FarResultQueue results;
	std::atomic<uint32_t> generation{ 0 };
	std::atomic<bool> shutting_down{ false };
	std::atomic<int32_t> tasks_in_flight{ 0 };

	inline bool is_stale(uint32_t task_generation) const {
		return shutting_down.load(std::memory_order_relaxed) ||
				generation.load(std::memory_order_relaxed) != task_generation;
	}
};

// Builds one sector end to end: cache lookup, generation on a miss, extraction,
// meshing, then enqueue for upload.
//
// One task rather than a chain of three because the stages are short and
// strictly sequential. Splitting them across separate scheduled tasks would add
// queue latency between each without gaining any parallelism -- the parallelism
// here comes from running many sectors at once, not from pipelining one.
class FarBuildSectorTask : public IThreadedTask {
public:
	FarBuildSectorTask(
			SectorId id,
			float priority_distance,
			std::shared_ptr<FarTaskParams> params,
			std::shared_ptr<FarSharedState> shared,
			uint32_t generation
	);
	~FarBuildSectorTask() override;

	void run(ThreadedTaskContext &ctx) override;
	TaskPriority get_priority() override;
	bool is_cancelled() override;

	const char *get_debug_name() const override {
		return "FarBuildSector";
	}

private:
	// Samples the generator over the sector's footprint and extracts columns.
	bool generate_field(uint8_t sample_lod, float world_x, float world_z, ColumnField &out_field) const;

	// Planet counterpart: samples along radial columns over a cube-face patch.
	bool generate_field_spherical(uint8_t sample_lod, const SectorGeometry &geom, ColumnField &out_field) const;

	// Builds the geometry for this sector, flat or spherical.
	void build_geometry(const SectorId &id, SectorGeometry &out) const;

	SectorId _id;
	float _priority_distance = 0.f;
	std::shared_ptr<FarTaskParams> _params;
	std::shared_ptr<FarSharedState> _shared;
	uint32_t _generation = 0;
};

} // namespace zylann::voxel::far

#endif // VOXEL_FAR_TASKS_H
