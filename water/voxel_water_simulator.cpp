#include "voxel_water_simulator.h"
#include "water_sim_tick_task.h"

#include "../engine/voxel_engine.h"
#include "../storage/voxel_data.h"
#include "../storage/voxel_format_gd.h"
#include "../terrain/variable_lod/voxel_lod_terrain.h"
#include "../util/godot/core/class_db.h"
#include "../util/io/log.h"
#include "../util/math/box3i.h"
#include "../util/math/funcs.h"
#include "../util/math/vector3i.h"

namespace zylann::voxel {

namespace {
const Vector3i FACE_OFFSETS[6] = { Vector3i(1, 0, 0), Vector3i(-1, 0, 0), Vector3i(0, 1, 0), Vector3i(0, -1, 0),
	Vector3i(0, 0, 1), Vector3i(0, 0, -1) };
} // namespace

VoxelWaterSimulator::VoxelWaterSimulator() {}

VoxelWaterSimulator::~VoxelWaterSimulator() {
	// Any in-flight WaterSimTickTask only holds this node's ObjectID, and checks
	// ObjectDB::get_instance() before touching `this` in apply_result() -- safe to destroy
	// while a tick is running.
}

void VoxelWaterSimulator::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY:
			set_process(true);
			break;
		case NOTIFICATION_PROCESS:
			_process_time(get_process_delta_time());
			break;
	}
}

void VoxelWaterSimulator::set_terrain(Node *p_terrain) {
	if (p_terrain == nullptr) {
		_terrain_id = ObjectID();
		return;
	}
	VoxelLodTerrain *terrain = Object::cast_to<VoxelLodTerrain>(p_terrain);
	if (terrain == nullptr) {
		ZN_PRINT_ERROR("VoxelWaterSimulator.terrain must be a VoxelLodTerrain");
		return;
	}
	_terrain_id = terrain->get_instance_id();

	// Bump the water channel's format depth from the 8-bit default to 16-bit for finer mass
	// resolution. Only matters before this terrain's blocks are generated/loaded.
	Ref<zylann::voxel::godot::VoxelFormat> fmt = terrain->get_format();
	if (fmt.is_null()) {
		fmt.instantiate();
		terrain->set_format(fmt);
	}
	// `godot::VoxelFormat` takes the GDScript-facing `godot::VoxelBuffer::ChannelId`/`Depth`
	// enums, distinct C++ types from the internal `VoxelBuffer` ones this file otherwise uses,
	// even though their values are defined to match 1:1 -- same cast VoxelFormat::
	// set_channel_depth() does internally at the module/wrapper boundary.
	using GdChannelId = zylann::voxel::godot::VoxelBuffer::ChannelId;
	using GdDepth = zylann::voxel::godot::VoxelBuffer::Depth;
	const GdChannelId gd_water_channel = static_cast<GdChannelId>(WATER_CHANNEL);
	const GdDepth gd_depth_16_bit = static_cast<GdDepth>(VoxelBuffer::DEPTH_16_BIT);

	const unsigned int current_bytes =
			VoxelBuffer::get_depth_byte_count(static_cast<VoxelBuffer::Depth>(fmt->get_channel_depth(gd_water_channel))
			);
	const unsigned int wanted_bytes = VoxelBuffer::get_depth_byte_count(VoxelBuffer::DEPTH_16_BIT);
	if (current_bytes < wanted_bytes) {
		fmt->set_channel_depth(gd_water_channel, gd_depth_16_bit);
	}
}

Node *VoxelWaterSimulator::get_terrain() const {
	return _get_terrain();
}

void VoxelWaterSimulator::set_gravity_mode(GravityMode p_mode) {
	_gravity_mode = p_mode;
}

VoxelWaterSimulator::GravityMode VoxelWaterSimulator::get_gravity_mode() const {
	return _gravity_mode;
}

void VoxelWaterSimulator::set_planet_center(Vector3 p_center) {
	_planet_center = p_center;
}

Vector3 VoxelWaterSimulator::get_planet_center() const {
	return _planet_center;
}

void VoxelWaterSimulator::set_update_interval(float p_seconds) {
	_update_interval = math::max(p_seconds, 0.01f);
}

float VoxelWaterSimulator::get_update_interval() const {
	return _update_interval;
}

void VoxelWaterSimulator::set_max_blocks_per_tick(int p_count) {
	_max_blocks_per_tick = math::max(p_count, 1);
}

int VoxelWaterSimulator::get_max_blocks_per_tick() const {
	return _max_blocks_per_tick;
}

void VoxelWaterSimulator::set_settle_ticks_threshold(int p_ticks) {
	_settle_ticks_threshold = math::max(p_ticks, 1);
}

int VoxelWaterSimulator::get_settle_ticks_threshold() const {
	return _settle_ticks_threshold;
}

void VoxelWaterSimulator::set_absorption_rate(float p_rate) {
	_absorption_rate = math::max(p_rate, 0.0f);
}

float VoxelWaterSimulator::get_absorption_rate() const {
	return _absorption_rate;
}

void VoxelWaterSimulator::set_max_absorption_per_voxel(float p_max) {
	_max_absorption_per_voxel = math::max(p_max, 0.0f);
}

float VoxelWaterSimulator::get_max_absorption_per_voxel() const {
	return _max_absorption_per_voxel;
}

void VoxelWaterSimulator::set_absorbing_type_ids(PackedInt32Array p_ids) {
	_absorbing_type_ids = p_ids;
}

PackedInt32Array VoxelWaterSimulator::get_absorbing_type_ids() const {
	return _absorbing_type_ids;
}

VoxelLodTerrain *VoxelWaterSimulator::_get_terrain() const {
	Object *obj = ObjectDB::get_instance(_terrain_id);
	return Object::cast_to<VoxelLodTerrain>(obj);
}

std::shared_ptr<VoxelData> VoxelWaterSimulator::_get_data() const {
	VoxelLodTerrain *terrain = _get_terrain();
	if (terrain == nullptr) {
		return nullptr;
	}
	return terrain->get_storage_shared();
}

void VoxelWaterSimulator::add_water(Vector3i p_global_voxel, float p_amount) {
	if (Math::is_zero_approx(p_amount)) {
		return;
	}
	VoxelLodTerrain *terrain = _get_terrain();
	std::shared_ptr<VoxelData> voxel_data = _get_data();
	if (terrain == nullptr || voxel_data == nullptr) {
		return;
	}

	const int block_size = int(voxel_data->get_block_size());
	const Vector3i block_pos = math::floordiv(p_global_voxel, block_size);
	const Vector3i local = p_global_voxel - block_pos * block_size;

	bool applied = false;
	StdVector<Vector3i> woken_neighbors;
	{
		// Lock the target block plus its 6 face-neighbors: a source placed right at a block
		// boundary should be able to wake the neighbor on this same call (see below), matching
		// EdenWaterSimulator's add_water().
		SpatialLock3D::Write wlock(
				voxel_data->get_spatial_lock(0), BoxBounds3i(block_pos - Vector3i(1, 1, 1), block_pos + Vector3i(2, 2, 2))
		);

		std::shared_ptr<VoxelBuffer> buf = voxel_data->try_get_block_voxels(block_pos);
		if (buf == nullptr) {
			return;
		}
		if (buf->get_voxel_f(local.x, local.y, local.z, VoxelBuffer::CHANNEL_SDF) <= 0.0f) {
			// Solid voxel: no-op, same as Eden.
			return;
		}
		const float cur = buf->get_voxel_f(local.x, local.y, local.z, WATER_CHANNEL);
		const float next = math::max(0.0f, cur + p_amount);
		buf->set_voxel_f(next, local.x, local.y, local.z, WATER_CHANNEL);
		applied = true;

		for (int i = 0; i < 6; ++i) {
			const Vector3i npos = math::floordiv(p_global_voxel + FACE_OFFSETS[i], block_size);
			if (npos != block_pos && voxel_data->try_get_block_voxels(npos) != nullptr) {
				woken_neighbors.push_back(npos);
			}
		}
	}
	if (!applied) {
		return;
	}

	_active_blocks.insert(block_pos);
	_settle_counters[block_pos] = 0;
	for (const Vector3i &npos : woken_neighbors) {
		_active_blocks.insert(npos);
		_settle_counters[npos] = 0;
	}

	terrain->post_edit_area(
			Box3i(block_pos * block_size, Vector3i(block_size, block_size, block_size)), true
	);
}

float VoxelWaterSimulator::get_water_mass(Vector3i p_global_voxel) const {
	std::shared_ptr<VoxelData> voxel_data = _get_data();
	if (voxel_data == nullptr) {
		return 0.0f;
	}
	const int block_size = int(voxel_data->get_block_size());
	const Vector3i block_pos = math::floordiv(p_global_voxel, block_size);
	const Vector3i local = p_global_voxel - block_pos * block_size;
	SpatialLock3D::Read rlock(voxel_data->get_spatial_lock(0), BoxBounds3i::from_position(block_pos));
	std::shared_ptr<VoxelBuffer> buf = voxel_data->try_get_block_voxels(block_pos);
	if (buf == nullptr) {
		return 0.0f;
	}
	return buf->get_voxel_f(local.x, local.y, local.z, WATER_CHANNEL);
}

bool VoxelWaterSimulator::is_solid_voxel(Vector3i p_global_voxel) const {
	std::shared_ptr<VoxelData> voxel_data = _get_data();
	if (voxel_data == nullptr) {
		return true;
	}
	const int block_size = int(voxel_data->get_block_size());
	const Vector3i block_pos = math::floordiv(p_global_voxel, block_size);
	const Vector3i local = p_global_voxel - block_pos * block_size;
	SpatialLock3D::Read rlock(voxel_data->get_spatial_lock(0), BoxBounds3i::from_position(block_pos));
	std::shared_ptr<VoxelBuffer> buf = voxel_data->try_get_block_voxels(block_pos);
	if (buf == nullptr) {
		return true;
	}
	return buf->get_voxel_f(local.x, local.y, local.z, VoxelBuffer::CHANNEL_SDF) <= 0.0f;
}

void VoxelWaterSimulator::activate_block(Vector3i p_block_pos) {
	std::shared_ptr<VoxelData> voxel_data = _get_data();
	if (voxel_data == nullptr) {
		return;
	}
	SpatialLock3D::Read rlock(voxel_data->get_spatial_lock(0), BoxBounds3i::from_position(p_block_pos));
	if (voxel_data->try_get_block_voxels(p_block_pos) == nullptr) {
		return;
	}
	_active_blocks.insert(p_block_pos);
	_settle_counters[p_block_pos] = 0;
}

void VoxelWaterSimulator::deactivate_block(Vector3i p_block_pos) {
	_active_blocks.erase(p_block_pos);
	_settle_counters.erase(p_block_pos);
}

int VoxelWaterSimulator::get_active_block_count() const {
	return int(_active_blocks.size());
}

bool VoxelWaterSimulator::is_block_active(Vector3i p_block_pos) const {
	return _active_blocks.find(p_block_pos) != _active_blocks.end();
}

TypedArray<Vector3i> VoxelWaterSimulator::get_active_block_positions() const {
	TypedArray<Vector3i> arr;
	arr.resize(_active_blocks.size());
	int i = 0;
	for (const Vector3i &b : _active_blocks) {
		arr[i++] = b;
	}
	return arr;
}

bool VoxelWaterSimulator::is_tick_in_flight() const {
	return _tick_in_flight.load(std::memory_order_acquire);
}

void VoxelWaterSimulator::_process_time(double p_delta) {
	_time_accumulator += p_delta;
	if (_time_accumulator < double(_update_interval)) {
		return;
	}
	if (_tick_in_flight.load(std::memory_order_acquire)) {
		// Previous tick is still running on the worker pool -- wait for it instead of queuing
		// more work. Not resetting the accumulator means the very next _process() call will
		// retry immediately once the in-flight tick clears, rather than waiting a full extra
		// `update_interval`.
		return;
	}
	_time_accumulator = 0.0;
	_dispatch_tick();
}

void VoxelWaterSimulator::_dispatch_tick() {
	VoxelLodTerrain *terrain = _get_terrain();
	if (terrain == nullptr || _active_blocks.empty()) {
		return;
	}
	std::shared_ptr<VoxelData> voxel_data = terrain->get_storage_shared();
	if (voxel_data == nullptr) {
		return;
	}

	StdVector<Vector3i> all_active;
	all_active.reserve(_active_blocks.size());
	for (const Vector3i &b : _active_blocks) {
		all_active.push_back(b);
	}

	const int total = int(all_active.size());
	const int process_count = math::min(total, _max_blocks_per_tick);
	StdVector<Vector3i> snapshot;
	snapshot.reserve(process_count);
	for (int i = 0; i < process_count; ++i) {
		snapshot.push_back(all_active[(_round_robin_cursor + i) % total]);
	}
	_round_robin_cursor = (total > 0) ? (_round_robin_cursor + process_count) % total : 0;

	StdVector<uint64_t> absorbing_ids;
	absorbing_ids.reserve(_absorbing_type_ids.size());
	for (int i = 0; i < _absorbing_type_ids.size(); ++i) {
		absorbing_ids.push_back(uint64_t(_absorbing_type_ids[i]));
	}

	_tick_in_flight.store(true, std::memory_order_release);

	WaterSimTickTask *task = ZN_NEW(WaterSimTickTask(
			get_instance_id(),
			voxel_data,
			voxel_data->get_block_size(),
			snapshot,
			_gravity_mode,
			_planet_center,
			_absorption_rate,
			_max_absorption_per_voxel,
			absorbing_ids
	));

	VoxelEngine::get_singleton().push_async_task(task);
}

void VoxelWaterSimulator::_on_tick_completed(const StdVector<WaterSimBlockResult> &p_results) {
	for (const WaterSimBlockResult &r : p_results) {
		if (r.missing) {
			_active_blocks.erase(r.block_pos);
			_settle_counters.erase(r.block_pos);
			continue;
		}
		if (r.had_flow) {
			_active_blocks.insert(r.block_pos);
			_settle_counters[r.block_pos] = 0;
		} else {
			auto it = _settle_counters.find(r.block_pos);
			const int count = (it != _settle_counters.end()) ? it->second + 1 : 1;
			_settle_counters[r.block_pos] = count;
			if (count >= _settle_ticks_threshold) {
				_active_blocks.erase(r.block_pos);
				_settle_counters.erase(r.block_pos);
			}
		}
	}
	_tick_in_flight.store(false, std::memory_order_release);
}

Vector3 VoxelWaterSimulator::compute_down_dir(GravityMode p_mode, Vector3 p_planet_center, Vector3i p_global_pos) {
	if (p_mode == GRAVITY_Y_DOWN) {
		return Vector3(0.0f, -1.0f, 0.0f);
	}
	const Vector3 voxel_center(
			float(p_global_pos.x) + 0.5f, float(p_global_pos.y) + 0.5f, float(p_global_pos.z) + 0.5f
	);
	const Vector3 dir = p_planet_center - voxel_center;
	if (dir.length_squared() < 0.0001f) {
		return Vector3(0.0f, -1.0f, 0.0f);
	}
	return dir.normalized();
}

PackedVector3Array VoxelWaterSimulator::generate_water_mesh_faces(Vector3i p_block_pos, float p_min_render_mass)
		const {
	PackedVector3Array faces;
	std::shared_ptr<VoxelData> voxel_data = _get_data();
	if (voxel_data == nullptr) {
		return faces;
	}
	const int bs = int(voxel_data->get_block_size());

	SpatialLock3D::Read rlock(
			voxel_data->get_spatial_lock(0),
			BoxBounds3i(p_block_pos - Vector3i(1, 1, 1), p_block_pos + Vector3i(2, 2, 2))
	);

	std::shared_ptr<VoxelBuffer> buf = voxel_data->try_get_block_voxels(p_block_pos);
	if (buf == nullptr) {
		return faces;
	}

	auto get_mass_at = [&](Vector3i g) -> float {
		const Vector3i bp = math::floordiv(g, bs);
		std::shared_ptr<VoxelBuffer> b = (bp == p_block_pos) ? buf : voxel_data->try_get_block_voxels(bp);
		if (b == nullptr) {
			return 0.0f;
		}
		const Vector3i local = g - bp * bs;
		return b->get_voxel_f(local.x, local.y, local.z, WATER_CHANNEL);
	};
	auto is_solid_at = [&](Vector3i g) -> bool {
		const Vector3i bp = math::floordiv(g, bs);
		std::shared_ptr<VoxelBuffer> b = (bp == p_block_pos) ? buf : voxel_data->try_get_block_voxels(bp);
		if (b == nullptr) {
			return true;
		}
		const Vector3i local = g - bp * bs;
		return b->get_voxel_f(local.x, local.y, local.z, VoxelBuffer::CHANNEL_SDF) <= 0.0f;
	};

	// Absorption (see set_absorption_rate) reuses a SOLID voxel's own WATER_CHANNEL to track
	// how much it has absorbed, so mass > threshold alone isn't "wet, render a face" -- it must
	// also not be solid, or an absorbing ground voxel near saturation would grow a phantom
	// water face fused into the terrain.
	auto is_wet = [&](int x, int y, int z) -> bool {
		if (x < 0 || y < 0 || z < 0 || x >= bs || y >= bs || z >= bs) {
			const Vector3i g = p_block_pos * bs + Vector3i(x, y, z);
			return get_mass_at(g) > p_min_render_mass && !is_solid_at(g);
		}
		return buf->get_voxel_f(x, y, z, WATER_CHANNEL) > p_min_render_mass &&
				buf->get_voxel_f(x, y, z, VoxelBuffer::CHANNEL_SDF) > 0.0f;
	};
	auto is_blocking = [&](int x, int y, int z) -> bool {
		if (x < 0 || y < 0 || z < 0 || x >= bs || y >= bs || z >= bs) {
			return is_solid_at(p_block_pos * bs + Vector3i(x, y, z));
		}
		return buf->get_voxel_f(x, y, z, VoxelBuffer::CHANNEL_SDF) <= 0.0f;
	};

	// Per-axis greedy sweep: for each of the 3 axes and both directions along it, build a
	// boolean "visible wet face" mask on the 2D slice perpendicular to that axis, then merge it
	// into maximal rectangles (standard greedy meshing).
	for (int axis = 0; axis < 3; ++axis) {
		const int u_axis = (axis + 1) % 3;
		const int v_axis = (axis + 2) % 3;

		for (int dir = -1; dir <= 1; dir += 2) {
			StdVector<bool> mask;
			mask.resize(bs * bs);

			for (int d = -1; d < bs; ++d) {
				for (int vv = 0; vv < bs; ++vv) {
					for (int uu = 0; uu < bs; ++uu) {
						int pos[3];
						pos[axis] = d;
						pos[u_axis] = uu;
						pos[v_axis] = vv;
						int npos[3] = { pos[0], pos[1], pos[2] };
						npos[axis] = d + 1;

						const bool cur_wet = is_wet(pos[0], pos[1], pos[2]);
						const bool next_wet = is_wet(npos[0], npos[1], npos[2]);
						const bool cur_block = is_blocking(pos[0], pos[1], pos[2]);
						const bool next_block = is_blocking(npos[0], npos[1], npos[2]);

						bool face_here;
						if (dir < 0) {
							// Face on the negative side of the boundary, owned by the "next" cell.
							face_here = next_wet && !cur_wet && !cur_block;
						} else {
							// Face on the positive side, owned by the "current" cell.
							face_here = cur_wet && !next_wet && !next_block;
						}
						mask[vv * bs + uu] = face_here;
					}
				}

				int n = 0;
				for (int vv = 0; vv < bs; ++vv) {
					for (int uu = 0; uu < bs;) {
						if (!mask[n + uu]) {
							++uu;
							continue;
						}
						int w = 1;
						while (uu + w < bs && mask[n + uu + w]) {
							++w;
						}
						int h = 1;
						bool done = false;
						while (vv + h < bs && !done) {
							for (int k = 0; k < w; ++k) {
								if (!mask[(vv + h) * bs + uu + k]) {
									done = true;
									break;
								}
							}
							if (!done) {
								++h;
							}
						}

						float pos[3];
						pos[axis] = float(d + (dir < 0 ? 0 : 1));
						pos[u_axis] = float(uu);
						pos[v_axis] = float(vv);
						float size_u[3] = { 0, 0, 0 };
						size_u[u_axis] = float(w);
						float size_v[3] = { 0, 0, 0 };
						size_v[v_axis] = float(h);

						const Vector3 origin(pos[0], pos[1], pos[2]);
						const Vector3 du(size_u[0], size_u[1], size_u[2]);
						const Vector3 dv(size_v[0], size_v[1], size_v[2]);
						const Vector3 p00 = origin;
						const Vector3 p10 = origin + du;
						const Vector3 p01 = origin + dv;
						const Vector3 p11 = origin + du + dv;

						// Winding depends on face direction so the quad faces outward.
						if (dir < 0) {
							faces.push_back(p00);
							faces.push_back(p11);
							faces.push_back(p10);
							faces.push_back(p00);
							faces.push_back(p01);
							faces.push_back(p11);
						} else {
							faces.push_back(p00);
							faces.push_back(p10);
							faces.push_back(p11);
							faces.push_back(p00);
							faces.push_back(p11);
							faces.push_back(p01);
						}

						for (int hh = 0; hh < h; ++hh) {
							for (int ww = 0; ww < w; ++ww) {
								mask[(vv + hh) * bs + uu + ww] = false;
							}
						}
						uu += w;
					}
					n += bs;
				}
			}
		}
	}

	return faces;
}

PackedVector3Array VoxelWaterSimulator::generate_water_mesh_smooth_faces(
		Vector3i p_block_pos, float p_min_render_mass
) const {
	PackedVector3Array faces;
	std::shared_ptr<VoxelData> voxel_data = _get_data();
	if (voxel_data == nullptr) {
		return faces;
	}
	const int bs = int(voxel_data->get_block_size());

	SpatialLock3D::Read rlock(voxel_data->get_spatial_lock(0), BoxBounds3i::from_position(p_block_pos));

	if (voxel_data->try_get_block_voxels(p_block_pos) == nullptr) {
		return faces;
	}

	auto get_mass_at = [&](Vector3i g) -> float {
		const Vector3i bp = math::floordiv(g, bs);
		std::shared_ptr<VoxelBuffer> b = voxel_data->try_get_block_voxels(bp);
		if (b == nullptr) {
			return 0.0f;
		}
		const Vector3i local = g - bp * bs;
		return b->get_voxel_f(local.x, local.y, local.z, WATER_CHANNEL);
	};
	auto is_solid_at = [&](Vector3i g) -> bool {
		const Vector3i bp = math::floordiv(g, bs);
		std::shared_ptr<VoxelBuffer> b = voxel_data->try_get_block_voxels(bp);
		if (b == nullptr) {
			return true;
		}
		const Vector3i local = g - bp * bs;
		return b->get_voxel_f(local.x, local.y, local.z, VoxelBuffer::CHANNEL_SDF) <= 0.0f;
	};

	// This block's own locally-approximated "up": a single block spans a few dozen world
	// units, negligible next to a planet's radius, so treating "up" as constant across one
	// block instead of a true per-voxel radial direction is accurate enough at this scale.
	const Vector3 up_dir =
			-compute_down_dir(_gravity_mode, _planet_center, p_block_pos * bs + Vector3i(bs / 2, bs / 2, bs / 2));
	int up_axis = 1;
	if (Math::abs(up_dir.x) >= Math::abs(up_dir.y) && Math::abs(up_dir.x) >= Math::abs(up_dir.z)) {
		up_axis = 0;
	} else if (Math::abs(up_dir.z) >= Math::abs(up_dir.y)) {
		up_axis = 2;
	}
	const float up_sign = (up_dir[up_axis] >= 0.0f) ? 1.0f : -1.0f;
	static const int OTHER_AXES[3][2] = { { 1, 2 }, { 0, 2 }, { 0, 1 } };
	const int u_axis = OTHER_AXES[up_axis][0];
	const int v_axis = OTHER_AXES[up_axis][1];

	// Column height search range along up_axis, in GLOBAL voxel-index units: exactly this
	// block's own up-axis span, no padding into a neighbor -- peeking into an unloaded
	// neighbor would make the scan bail on a phantom "roof" (is_solid_at treats an absent
	// block as solid) instead of ever reaching this block's real water.
	const int block_up_origin = p_block_pos[up_axis] * bs;
	const int scan_lo = block_up_origin;
	const int scan_hi = block_up_origin + bs - 1;

	// Per-column surface height, in GLOBAL up_axis voxel-index units (fractional -- see the
	// sub-voxel interpolation below), or NAN if this column has no water. Cached over the
	// (bs+1)x(bs+1) grid of corner lines spanning this block (matching the shared-vertex
	// convention any heightfield quad grid needs).
	const int grid = bs + 1;
	StdVector<float> heights;
	heights.resize(size_t(grid) * size_t(grid));

	auto column_height = [&](int p_u, int p_v) -> float {
		Vector3i g;
		g[u_axis] = p_u;
		g[v_axis] = p_v;
		if (up_sign > 0.0f) {
			for (int idx = scan_hi; idx >= scan_lo; --idx) {
				g[up_axis] = idx;
				if (is_solid_at(g)) {
					return NAN; // Solid breaks the surface here (no cave/overhang water via this path).
				}
				const float mass = get_mass_at(g);
				if (mass > p_min_render_mass) {
					return float(idx) + math::clamp(mass / MAX_MASS, 0.0f, 1.0f);
				}
			}
		} else {
			for (int idx = scan_lo; idx <= scan_hi; ++idx) {
				g[up_axis] = idx;
				if (is_solid_at(g)) {
					return NAN;
				}
				const float mass = get_mass_at(g);
				if (mass > p_min_render_mass) {
					return float(idx) + (1.0f - math::clamp(mass / MAX_MASS, 0.0f, 1.0f));
				}
			}
		}
		return NAN; // No water anywhere in range -- fully dry column.
	};

	for (int lv = 0; lv < grid; ++lv) {
		for (int lu = 0; lu < grid; ++lu) {
			const int global_u = p_block_pos[u_axis] * bs + lu;
			const int global_v = p_block_pos[v_axis] * bs + lv;
			heights[size_t(lv) * size_t(grid) + size_t(lu)] = column_height(global_u, global_v);
		}
	}

	Vector3 up_local; // Unit vector, local coordinate space, used only for outward winding.
	up_local[up_axis] = up_sign;

	for (int lv = 0; lv < bs; ++lv) {
		for (int lu = 0; lu < bs; ++lu) {
			const float h00 = heights[size_t(lv) * size_t(grid) + size_t(lu)];
			const float h10 = heights[size_t(lv) * size_t(grid) + size_t(lu) + 1];
			const float h01 = heights[size_t(lv + 1) * size_t(grid) + size_t(lu)];
			const float h11 = heights[size_t(lv + 1) * size_t(grid) + size_t(lu) + 1];
			if (Math::is_nan(h00) || Math::is_nan(h10) || Math::is_nan(h01) || Math::is_nan(h11)) {
				continue; // At least one corner is dry -- skip this cell entirely (no partial/torn quads).
			}

			Vector3 p00, p10, p01, p11;
			p00[u_axis] = float(lu);
			p00[v_axis] = float(lv);
			p00[up_axis] = h00 - float(block_up_origin);
			p10[u_axis] = float(lu + 1);
			p10[v_axis] = float(lv);
			p10[up_axis] = h10 - float(block_up_origin);
			p01[u_axis] = float(lu);
			p01[v_axis] = float(lv + 1);
			p01[up_axis] = h01 - float(block_up_origin);
			p11[u_axis] = float(lu + 1);
			p11[v_axis] = float(lv + 1);
			p11[up_axis] = h11 - float(block_up_origin);

			const bool flip = (p10 - p00).cross(p01 - p00).dot(up_local) < 0.0f;
			if (!flip) {
				faces.push_back(p00);
				faces.push_back(p10);
				faces.push_back(p11);
				faces.push_back(p00);
				faces.push_back(p11);
				faces.push_back(p01);
			} else {
				faces.push_back(p00);
				faces.push_back(p11);
				faces.push_back(p10);
				faces.push_back(p00);
				faces.push_back(p01);
				faces.push_back(p11);
			}
		}
	}

	return faces;
}

void VoxelWaterSimulator::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_terrain", "terrain"), &VoxelWaterSimulator::set_terrain);
	ClassDB::bind_method(D_METHOD("get_terrain"), &VoxelWaterSimulator::get_terrain);
	ADD_PROPERTY(
			PropertyInfo(Variant::OBJECT, "terrain", PROPERTY_HINT_NODE_TYPE, "VoxelLodTerrain"),
			"set_terrain",
			"get_terrain"
	);

	ClassDB::bind_method(D_METHOD("set_gravity_mode", "mode"), &VoxelWaterSimulator::set_gravity_mode);
	ClassDB::bind_method(D_METHOD("get_gravity_mode"), &VoxelWaterSimulator::get_gravity_mode);
	ADD_PROPERTY(
			PropertyInfo(Variant::INT, "gravity_mode", PROPERTY_HINT_ENUM, "YDown,Radial"),
			"set_gravity_mode",
			"get_gravity_mode"
	);

	ClassDB::bind_method(D_METHOD("set_planet_center", "center"), &VoxelWaterSimulator::set_planet_center);
	ClassDB::bind_method(D_METHOD("get_planet_center"), &VoxelWaterSimulator::get_planet_center);
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "planet_center"), "set_planet_center", "get_planet_center");

	ClassDB::bind_method(D_METHOD("set_update_interval", "seconds"), &VoxelWaterSimulator::set_update_interval);
	ClassDB::bind_method(D_METHOD("get_update_interval"), &VoxelWaterSimulator::get_update_interval);
	ADD_PROPERTY(
			PropertyInfo(Variant::FLOAT, "update_interval"), "set_update_interval", "get_update_interval"
	);

	ClassDB::bind_method(
			D_METHOD("set_max_blocks_per_tick", "count"), &VoxelWaterSimulator::set_max_blocks_per_tick
	);
	ClassDB::bind_method(D_METHOD("get_max_blocks_per_tick"), &VoxelWaterSimulator::get_max_blocks_per_tick);
	ADD_PROPERTY(
			PropertyInfo(Variant::INT, "max_blocks_per_tick"),
			"set_max_blocks_per_tick",
			"get_max_blocks_per_tick"
	);

	ClassDB::bind_method(
			D_METHOD("set_settle_ticks_threshold", "ticks"), &VoxelWaterSimulator::set_settle_ticks_threshold
	);
	ClassDB::bind_method(
			D_METHOD("get_settle_ticks_threshold"), &VoxelWaterSimulator::get_settle_ticks_threshold
	);
	ADD_PROPERTY(
			PropertyInfo(Variant::INT, "settle_ticks_threshold"),
			"set_settle_ticks_threshold",
			"get_settle_ticks_threshold"
	);

	ClassDB::bind_method(D_METHOD("set_absorption_rate", "rate"), &VoxelWaterSimulator::set_absorption_rate);
	ClassDB::bind_method(D_METHOD("get_absorption_rate"), &VoxelWaterSimulator::get_absorption_rate);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "absorption_rate"), "set_absorption_rate", "get_absorption_rate");

	ClassDB::bind_method(
			D_METHOD("set_max_absorption_per_voxel", "max_amount"),
			&VoxelWaterSimulator::set_max_absorption_per_voxel
	);
	ClassDB::bind_method(
			D_METHOD("get_max_absorption_per_voxel"), &VoxelWaterSimulator::get_max_absorption_per_voxel
	);
	ADD_PROPERTY(
			PropertyInfo(Variant::FLOAT, "max_absorption_per_voxel"),
			"set_max_absorption_per_voxel",
			"get_max_absorption_per_voxel"
	);

	ClassDB::bind_method(
			D_METHOD("set_absorbing_type_ids", "ids"), &VoxelWaterSimulator::set_absorbing_type_ids
	);
	ClassDB::bind_method(D_METHOD("get_absorbing_type_ids"), &VoxelWaterSimulator::get_absorbing_type_ids);
	ADD_PROPERTY(
			PropertyInfo(Variant::PACKED_INT32_ARRAY, "absorbing_type_ids"),
			"set_absorbing_type_ids",
			"get_absorbing_type_ids"
	);

	ClassDB::bind_method(D_METHOD("add_water", "global_voxel", "amount"), &VoxelWaterSimulator::add_water);
	ClassDB::bind_method(D_METHOD("get_water_mass", "global_voxel"), &VoxelWaterSimulator::get_water_mass);
	ClassDB::bind_method(D_METHOD("is_solid_voxel", "global_voxel"), &VoxelWaterSimulator::is_solid_voxel);
	ClassDB::bind_method(D_METHOD("activate_block", "block_pos"), &VoxelWaterSimulator::activate_block);
	ClassDB::bind_method(D_METHOD("deactivate_block", "block_pos"), &VoxelWaterSimulator::deactivate_block);
	ClassDB::bind_method(D_METHOD("get_active_block_count"), &VoxelWaterSimulator::get_active_block_count);
	ClassDB::bind_method(D_METHOD("is_block_active", "block_pos"), &VoxelWaterSimulator::is_block_active);
	ClassDB::bind_method(
			D_METHOD("get_active_block_positions"), &VoxelWaterSimulator::get_active_block_positions
	);
	ClassDB::bind_method(D_METHOD("is_tick_in_flight"), &VoxelWaterSimulator::is_tick_in_flight);

	ClassDB::bind_method(
			D_METHOD("generate_water_mesh_faces", "block_pos", "min_render_mass"),
			&VoxelWaterSimulator::generate_water_mesh_faces,
			DEFVAL(0.05f)
	);
	ClassDB::bind_method(
			D_METHOD("generate_water_mesh_smooth_faces", "block_pos", "min_render_mass"),
			&VoxelWaterSimulator::generate_water_mesh_smooth_faces,
			DEFVAL(0.05f)
	);

	BIND_ENUM_CONSTANT(GRAVITY_Y_DOWN);
	BIND_ENUM_CONSTANT(GRAVITY_RADIAL);
}

} // namespace zylann::voxel
