#include "water_sim_tick_task.h"

#include "../storage/voxel_buffer.h"
#include "../terrain/variable_lod/voxel_lod_terrain.h"
#include "../util/math/funcs.h"
#include "../util/math/vector3i.h"

namespace zylann::voxel {

namespace {

const Vector3i FACE_OFFSETS[6] = { Vector3i(1, 0, 0), Vector3i(-1, 0, 0), Vector3i(0, 1, 0), Vector3i(0, -1, 0),
	Vector3i(0, 0, 1), Vector3i(0, 0, -1) };

// Caches fetched block buffers for the duration of one tick's scan, so a voxel near a block
// boundary doesn't re-fetch its neighbor's buffer for each of its up-to-4 sideways checks.
// Every block a tick can touch was already established to be within the read-locked region
// by WaterSimTickTask::run() before this is used (see the +1-block margin around
// `_blocks_to_process` there).
class TickDataView {
public:
	TickDataView(VoxelData &data, unsigned int block_size, unsigned int lod_index) :
			_data(data), _block_size(int(block_size)), _lod_index(lod_index) {}

	inline Vector3i block_of(Vector3i global_pos) const {
		return math::floordiv(global_pos, _block_size);
	}

	std::shared_ptr<VoxelBuffer> get_block(Vector3i block_pos) {
		auto it = _cache.find(block_pos);
		if (it != _cache.end()) {
			return it->second;
		}
		std::shared_ptr<VoxelBuffer> buf = _data.try_get_block_voxels(block_pos, _lod_index);
		_cache[block_pos] = buf;
		return buf;
	}

	bool is_solid(Vector3i global_pos) {
		const Vector3i block_pos = block_of(global_pos);
		std::shared_ptr<VoxelBuffer> buf = get_block(block_pos);
		if (buf == nullptr) {
			// Unloaded/ungenerated region: treated as solid, matching EdenWaterSimulator --
			// water never leaks into or creates data outside what's actually resident.
			return true;
		}
		const Vector3i local = global_pos - block_pos * _block_size;
		return buf->get_voxel_f(local.x, local.y, local.z, VoxelBuffer::CHANNEL_SDF) <= 0.0f;
	}

	float get_mass(Vector3i global_pos) {
		const Vector3i block_pos = block_of(global_pos);
		std::shared_ptr<VoxelBuffer> buf = get_block(block_pos);
		if (buf == nullptr) {
			return 0.0f;
		}
		const Vector3i local = global_pos - block_pos * _block_size;
		return buf->get_voxel_f(local.x, local.y, local.z, VoxelWaterSimulator::WATER_CHANNEL);
	}

	uint64_t get_type_id(Vector3i global_pos) {
		const Vector3i block_pos = block_of(global_pos);
		std::shared_ptr<VoxelBuffer> buf = get_block(block_pos);
		if (buf == nullptr) {
			return 0;
		}
		const Vector3i local = global_pos - block_pos * _block_size;
		return buf->get_voxel(local.x, local.y, local.z, VoxelBuffer::CHANNEL_TYPE);
	}

private:
	VoxelData &_data;
	int _block_size;
	unsigned int _lod_index;
	StdUnorderedMap<Vector3i, std::shared_ptr<VoxelBuffer>> _cache;
};

// Ranks the 6 face-neighbor offsets of `global_pos` (expressed in `lod_index`'s own coordinate
// space) by alignment with "down": index 0 is the single most-down neighbor, index 5 the single
// most-up, 1-4 the remaining sideways targets.
void classify_neighbors(
		VoxelWaterSimulator::GravityMode mode, Vector3 planet_center, Vector3i global_pos,
		unsigned int lod_index, Vector3i r_ranked[6]
) {
	// GRAVITY_RADIAL needs a real-world-space position to compute a direction relative to
	// planet_center (also real-world-space) -- global_pos here is in lod_index's own grid,
	// where each unit is 2^lod_index world units, so it must be scaled up first. Neighbor
	// ranking below stays in global_pos's own (unscaled) space, since r_ranked indexes voxels
	// in that same LOD's buffer.
	const Vector3i world_pos = global_pos << int(lod_index);
	const Vector3 down = VoxelWaterSimulator::compute_down_dir(mode, planet_center, world_pos);
	float dots[6];
	for (int i = 0; i < 6; ++i) {
		r_ranked[i] = global_pos + FACE_OFFSETS[i];
		const Vector3 dir(float(FACE_OFFSETS[i].x), float(FACE_OFFSETS[i].y), float(FACE_OFFSETS[i].z));
		dots[i] = dir.dot(down);
	}
	// Insertion sort, 6 elements, descending by dot (most-aligned-with-down first).
	for (int i = 1; i < 6; ++i) {
		const float d = dots[i];
		const Vector3i v = r_ranked[i];
		int j = i - 1;
		while (j >= 0 && dots[j] < d) {
			dots[j + 1] = dots[j];
			r_ranked[j + 1] = r_ranked[j];
			--j;
		}
		dots[j + 1] = d;
		r_ranked[j + 1] = v;
	}
}

// Target mass a "lower" cell should settle at given the combined mass of it and its "upper"
// neighbor, modeling water as a slightly compressible liquid. Same formula/constants as
// EdenWaterSimulator (itself matching the standard reference "falling sand" water CA).
float stable_state_b(float total_mass) {
	if (total_mass <= VoxelWaterSimulator::MAX_MASS) {
		return VoxelWaterSimulator::MAX_MASS;
	} else if (total_mass < (2.0f * VoxelWaterSimulator::MAX_MASS + VoxelWaterSimulator::MAX_COMPRESS)) {
		return (VoxelWaterSimulator::MAX_MASS * VoxelWaterSimulator::MAX_MASS +
					   total_mass * VoxelWaterSimulator::MAX_COMPRESS) /
				(VoxelWaterSimulator::MAX_MASS + VoxelWaterSimulator::MAX_COMPRESS);
	} else {
		return (total_mass + VoxelWaterSimulator::MAX_COMPRESS) / 2.0f;
	}
}

bool is_absorbing(uint64_t type_id, const StdVector<uint64_t> &absorbing_ids) {
	for (uint64_t id : absorbing_ids) {
		if (id == type_id) {
			return true;
		}
	}
	return false;
}

Box3i blocks_to_box(const StdUnorderedSet<Vector3i> &blocks) {
	if (blocks.empty()) {
		return Box3i();
	}
	auto it = blocks.begin();
	Vector3i mn = *it;
	Vector3i mx = *it;
	for (const Vector3i &b : blocks) {
		mn = math::min(mn, b);
		mx = math::max(mx, b);
	}
	return Box3i::from_min_max(mn, mx + Vector3i(1, 1, 1));
}

} // namespace

WaterSimTickTask::WaterSimTickTask(
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
) :
		_simulator_id(p_simulator_id),
		_data(p_data),
		_block_size(p_block_size),
		_lod_index(p_lod_index),
		_blocks_to_process(p_blocks_to_process),
		_gravity_mode(p_gravity_mode),
		_planet_center(p_planet_center),
		_absorption_rate(p_absorption_rate),
		_max_absorption_per_voxel(p_max_absorption_per_voxel),
		_absorbing_type_ids(p_absorbing_type_ids) {}

void WaterSimTickTask::run(ThreadedTaskContext &ctx) {
	if (_data == nullptr || _blocks_to_process.size() == 0) {
		return;
	}

	// Read-lock a box covering every block we might process plus their 6 face-neighbors --
	// flow can read/write across a block boundary -- for the whole duration of the scan. This
	// is the only synchronization needed: run() never mutates VoxelBuffer, only local delta
	// maps, so it can't race with streaming/meshing tasks touching the same blocks.
	Vector3i lock_min = _blocks_to_process[0];
	Vector3i lock_max = _blocks_to_process[0];
	for (const Vector3i &b : _blocks_to_process) {
		lock_min = math::min(lock_min, b);
		lock_max = math::max(lock_max, b);
	}
	lock_min -= Vector3i(1, 1, 1);
	lock_max += Vector3i(2, 2, 2); // -1/+1 neighbor margin, plus 1 because max is exclusive

	SpatialLock3D &lock = _data->get_spatial_lock(_lod_index);
	SpatialLock3D::Read rlock(lock, BoxBounds3i(lock_min, lock_max));

	TickDataView view(*_data, _block_size, _lod_index);
	StdUnorderedSet<Vector3i> touched_blocks;
	StdUnorderedSet<Vector3i> missing_blocks;

	const int block_size = int(_block_size);

	for (const Vector3i &block_pos : _blocks_to_process) {
		std::shared_ptr<VoxelBuffer> buf = view.get_block(block_pos);
		if (buf == nullptr) {
			missing_blocks.insert(block_pos);
			continue;
		}

		for (int z = 0; z < block_size; ++z) {
			for (int y = 0; y < block_size; ++y) {
				for (int x = 0; x < block_size; ++x) {
					const float mass = buf->get_voxel_f(x, y, z, VoxelWaterSimulator::WATER_CHANNEL);
					if (mass <= VoxelWaterSimulator::MIN_MASS) {
						continue;
					}
					if (buf->get_voxel_f(x, y, z, VoxelBuffer::CHANNEL_SDF) <= 0.0f) {
						// Solid voxel holding stale mass (shouldn't normally happen); skip.
						continue;
					}

					const Vector3i global = block_pos * block_size + Vector3i(x, y, z);

					Vector3i ranked[6];
					classify_neighbors(_gravity_mode, _planet_center, global, _lod_index, ranked);
					const Vector3i down_n = ranked[0];
					const Vector3i up_n = ranked[5];

					float remaining = mass;

					// 0. Absorption into the ground: only from the single "down" neighbor, and
					// only when it's solid, an absorbing type, and hasn't hit capacity yet.
					if (_absorption_rate > 0.0f && remaining > VoxelWaterSimulator::MIN_MASS &&
							view.is_solid(down_n) && is_absorbing(view.get_type_id(down_n), _absorbing_type_ids)) {
						// CHANNEL_DATA5 reused as the absorbed-amount tracker on solid voxels,
						// same convention as EdenWaterSimulator's CHANNEL_WATER reuse.
						const float absorbed_so_far = view.get_mass(down_n);
						const float capacity = _max_absorption_per_voxel - absorbed_so_far;
						if (capacity > 0.0f) {
							const float amount = math::min(_absorption_rate, math::min(remaining, capacity));
							if (amount > 0.0f) {
								_mass_deltas[global] -= amount;
								_absorbed_deltas[down_n] += amount;
								remaining -= amount;
								touched_blocks.insert(block_pos);
								// down_n's block may differ from the source block (a voxel at the
								// bottom of a block, absorbing into the block below) -- must be
								// included so apply_result()'s write-lock actually covers it.
								touched_blocks.insert(view.block_of(down_n));
							}
						}
					}

					// 1. Flow toward "down".
					if (!view.is_solid(down_n)) {
						const float neighbor_mass = view.get_mass(down_n);
						const float stable = stable_state_b(remaining + neighbor_mass);
						const float flow = math::clamp(
								(stable - neighbor_mass) * 0.5f, 0.0f, math::min(VoxelWaterSimulator::MAX_SPEED, remaining)
						);
						if (flow > 0.0f) {
							_mass_deltas[global] -= flow;
							_mass_deltas[down_n] += flow;
							remaining -= flow;
							touched_blocks.insert(block_pos);
							touched_blocks.insert(view.block_of(down_n));
						}
					}

					// 2. Flow sideways, equalizing with the 4 non-vertical neighbors. Compared
					// against the voxel's original `mass` (not `remaining`), matching Eden.
					if (remaining > VoxelWaterSimulator::MIN_MASS) {
						for (int i = 1; i <= 4 && remaining > VoxelWaterSimulator::MIN_MASS; ++i) {
							const Vector3i side_n = ranked[i];
							if (view.is_solid(side_n)) {
								continue;
							}
							const float side_mass = view.get_mass(side_n);
							const float flow = math::clamp((mass - side_mass) * 0.125f, 0.0f, remaining);
							if (flow > 0.0f) {
								_mass_deltas[global] -= flow;
								_mass_deltas[side_n] += flow;
								remaining -= flow;
								touched_blocks.insert(block_pos);
								touched_blocks.insert(view.block_of(side_n));
							}
						}
					}

					// 3. Flow "up" -- only the compressed excess above stable equilibrium.
					if (remaining > VoxelWaterSimulator::MIN_MASS && !view.is_solid(up_n)) {
						const float neighbor_mass = view.get_mass(up_n);
						const float stable = stable_state_b(remaining + neighbor_mass);
						const float flow = math::clamp(
								(remaining - stable) * 0.5f, 0.0f, math::min(VoxelWaterSimulator::MAX_SPEED, remaining)
						);
						if (flow > 0.0f) {
							_mass_deltas[global] -= flow;
							_mass_deltas[up_n] += flow;
							touched_blocks.insert(block_pos);
							touched_blocks.insert(view.block_of(up_n));
						}
					}
				}
			}
		}
	}

	// Build results: every processed block reports whether it changed (or is missing), and
	// every touched block outside the processed set (a neighbor that received flow) reports
	// in too, so the simulator can activate it even though it wasn't already active.
	for (const Vector3i &block_pos : _blocks_to_process) {
		WaterSimBlockResult r;
		r.block_pos = block_pos;
		if (missing_blocks.find(block_pos) != missing_blocks.end()) {
			r.missing = true;
		} else {
			r.had_flow = touched_blocks.find(block_pos) != touched_blocks.end();
		}
		_block_results.push_back(r);
	}
	for (const Vector3i &block_pos : touched_blocks) {
		bool already_reported = false;
		for (const Vector3i &b : _blocks_to_process) {
			if (b == block_pos) {
				already_reported = true;
				break;
			}
		}
		if (!already_reported) {
			WaterSimBlockResult r;
			r.block_pos = block_pos;
			r.had_flow = true;
			_block_results.push_back(r);
		}
	}

	_any_touched = !touched_blocks.empty();
	if (_any_touched) {
		const Box3i block_box = blocks_to_box(touched_blocks);
		_touched_voxel_box =
				Box3i(block_box.position * block_size, block_box.size * block_size);
	}
}

void WaterSimTickTask::apply_result() {
	Object *obj = ObjectDB::get_instance(_simulator_id);
	VoxelWaterSimulator *sim = Object::cast_to<VoxelWaterSimulator>(obj);
	if (sim == nullptr) {
		// The simulator node was freed while this tick was in flight. Nothing left to update.
		return;
	}

	if (_data != nullptr && _any_touched) {
		SpatialLock3D &lock = _data->get_spatial_lock(_lod_index);
		const Box3i block_box = Box3i(
				math::floordiv(_touched_voxel_box.position, int(_block_size)),
				_touched_voxel_box.size / int(_block_size)
		);
		SpatialLock3D::Write wlock(lock, BoxBounds3i(block_box));

		// Deliberately NOT skipping near-zero individual deltas here (e.g. via
		// Math::is_zero_approx): each map entry is the NET SUM of every flow event touching
		// that voxel this tick, and that sum is asymmetric between a "spreads to several
		// neighbors" source (several small contributions summing to something applied) and
		// each of those neighbors (a single small credit that could individually look
		// negligible). Skipping small-looking individual entries broke mass conservation --
		// a source with several tiny outgoing flows would clear the threshold and get its
		// debit applied, while each receiving neighbor's single tiny credit stayed under it
		// and got silently dropped, leaking mass with no matching debit anywhere.
		for (const auto &kv : _mass_deltas) {
			const Vector3i block_pos = math::floordiv(kv.first, int(_block_size));
			std::shared_ptr<VoxelBuffer> buf = _data->try_get_block_voxels(block_pos, _lod_index);
			if (buf == nullptr) {
				continue;
			}
			const Vector3i local = kv.first - block_pos * int(_block_size);
			const float cur = buf->get_voxel_f(local.x, local.y, local.z, VoxelWaterSimulator::WATER_CHANNEL);
			const float next = math::max(0.0f, cur + kv.second);
			buf->set_voxel_f(next, local.x, local.y, local.z, VoxelWaterSimulator::WATER_CHANNEL);
		}
		for (const auto &kv : _absorbed_deltas) {
			const Vector3i block_pos = math::floordiv(kv.first, int(_block_size));
			std::shared_ptr<VoxelBuffer> buf = _data->try_get_block_voxels(block_pos, _lod_index);
			if (buf == nullptr) {
				continue;
			}
			const Vector3i local = kv.first - block_pos * int(_block_size);
			const float cur = buf->get_voxel_f(local.x, local.y, local.z, VoxelWaterSimulator::WATER_CHANNEL);
			const float next = math::clamp(cur + kv.second, 0.0f, _max_absorption_per_voxel);
			buf->set_voxel_f(next, local.x, local.y, local.z, VoxelWaterSimulator::WATER_CHANNEL);
		}
	}

	// post_edit_area()'s update_mesh=true path exists to cascade an SDF-affecting edit up to
	// coarser LODs so their downsampled mesh stays correct (see VoxelData::mark_area_modified()
	// -- update_mesh gates block->set_needs_lodding(true), which is what later triggers
	// VoxelData::update_lods()'s cascade). Water edits only ever touch CHANNEL_DATA5, never
	// CHANNEL_SDF, and our own water rendering re-scans+regenerates independently every refresh
	// (see eden_stress_test.gd/voxel_water_demo.gd's _refresh_water_meshes()) rather than
	// depending on this notification system at all -- so update_mesh=false here is correct, not
	// just an optimization: real terrain meshing genuinely has nothing to redo. Passing true
	// was benign at the old, much lower LOD0 tick rate, but at the default 0.2s cadence it was
	// cascading to coarser LODs constantly, and printing a real (not just verbose) error every
	// time the destination block wasn't resident -- itself a measurable per-tick main-thread
	// cost once ticks got frequent enough (confirmed: eliminated a full-second-plus stall).
	// Still LOD0-only: coarser-LOD ticks never called this at all (their water is rendered the
	// same script-side-rescan way, so there was never anything for it to invalidate there either).
	VoxelLodTerrain *terrain = sim->_get_terrain();
	if (terrain != nullptr && _any_touched && _lod_index == 0) {
		terrain->post_edit_area(_touched_voxel_box, false);
	}

	sim->_on_tick_completed(_block_results, _lod_index);
}

} // namespace zylann::voxel
