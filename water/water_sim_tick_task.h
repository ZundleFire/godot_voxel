#ifndef VOXEL_WATER_SIM_TICK_TASK_H
#define VOXEL_WATER_SIM_TICK_TASK_H

#include "../storage/voxel_data.h"
#include "../util/containers/std_unordered_map.h"
#include "../util/containers/std_vector.h"
#include "../util/godot/classes/object.h"
#include "../util/math/box3i.h"
#include "../util/tasks/threaded_task.h"
#include "voxel_water_simulator.h"

namespace zylann::voxel {

// One water-simulation tick, computed entirely off the main thread. This is the fix for
// EdenWaterSimulator's original design flaw: that prototype scanned every active block
// synchronously inside `_process()`, so a large active set (a real coastline) could stall the
// main thread for whole seconds. Here, `run()` (worker thread) only ever *reads* voxel data
// and computes mass deltas into local maps; `apply_result()` (main thread, called
// automatically by VoxelEngine::process() once the task completes) is the only place voxel
// data is actually written, kept deliberately cheap (a handful of set_voxel_f calls plus one
// post_edit_area) so it can't reintroduce the same stall.
class WaterSimTickTask : public IThreadedTask {
public:
	WaterSimTickTask(
			ObjectID p_simulator_id,
			std::shared_ptr<VoxelData> p_data,
			unsigned int p_block_size,
			unsigned int p_lod_index,
			StdVector<Vector3i> p_blocks_to_process,
			VoxelWaterSimulator::GravityMode p_gravity_mode,
			Vector3 p_planet_center,
			float p_absorption_rate,
			float p_max_absorption_per_voxel,
			StdVector<uint64_t> p_absorbing_type_ids
	);

	const char *get_debug_name() const override {
		return "WaterSimTick";
	}

	// Worker thread. Reads voxel data under a read spatial-lock covering every processed
	// block plus its 6 face-neighbors (flow can cross block boundaries), and never writes.
	void run(ThreadedTaskContext &ctx) override;

	// Main thread, called by VoxelEngine::process() once this task is dequeued as complete.
	// Writes the deltas computed in run() and triggers a remesh via post_edit_area().
	void apply_result() override;

private:
	ObjectID _simulator_id;
	std::shared_ptr<VoxelData> _data;
	unsigned int _block_size;
	unsigned int _lod_index;
	StdVector<Vector3i> _blocks_to_process;
	VoxelWaterSimulator::GravityMode _gravity_mode;
	Vector3 _planet_center;
	float _absorption_rate;
	float _max_absorption_per_voxel;
	StdVector<uint64_t> _absorbing_type_ids;

	// Outputs of run(), consumed by apply_result(). Keyed by global voxel position.
	StdUnorderedMap<Vector3i, float> _mass_deltas;
	StdUnorderedMap<Vector3i, float> _absorbed_deltas;
	StdVector<WaterSimBlockResult> _block_results;
	// Union of every touched block's voxel-space box, used for a single post_edit_area() call.
	Box3i _touched_voxel_box;
	bool _any_touched = false;
};

} // namespace zylann::voxel

#endif // VOXEL_WATER_SIM_TICK_TASK_H
