#include "voxel_water_simulator.h"
#include "water_sim_tick_task.h"

#include "../engine/voxel_engine.h"
#include "../storage/voxel_data.h"
#include "../storage/voxel_data_block.h"
#include "../storage/voxel_format_gd.h"
#include "../terrain/variable_lod/voxel_lod_terrain.h"
#include "../util/godot/classes/engine.h"
#include "../util/godot/classes/surface_tool.h"
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
			set_process(!Engine::get_singleton()->is_editor_hint() || _enabled_in_editor);
			break;
		case NOTIFICATION_PROCESS:
			_process_time(get_process_delta_time());
			break;
		case NOTIFICATION_PREDELETE:
			// Stop processing immediately rather than leaving it to whatever happens to already
			// be queued -- during editor scene teardown, node destruction order relative to a
			// still-pending _process() call isn't guaranteed, and _process_time() walks _terrain
			// (via a NodePath lookup) and _mesh_nodes (raw pointers into children that may
			// already be mid-free in the same teardown cascade). This closes that window.
			set_process(false);
			break;
	}
}

void VoxelWaterSimulator::set_enabled_in_editor(bool p_enabled) {
	if (_enabled_in_editor == p_enabled) {
		return;
	}
	_enabled_in_editor = p_enabled;
	// Note: `is_editor_hint()` doesn't change at runtime, so this only ever matters in the
	// editor -- in game, processing is already unconditionally on regardless of this flag.
	if (is_inside_tree() && Engine::get_singleton()->is_editor_hint()) {
		set_process(_enabled_in_editor);
	}
}

bool VoxelWaterSimulator::is_enabled_in_editor() const {
	return _enabled_in_editor;
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

	// VoxelLodTerrain defaults to NOT persisting generated voxel data into its VoxelData map
	// (voxels are generated transiently for meshing and discarded) -- without this, every query
	// this simulator makes against the terrain's VoxelData (is_solid_voxel(), get_water_mass(),
	// scan_and_activate_wet_blocks(), get_wet_block_positions()) silently sees an empty map
	// forever, no matter how much terrain has actually streamed in or how long you wait.
	// Confirmed by direct A/B probing: identical global voxel position, generator confirmed
	// deterministic and baking real water there, yet every simulator query returned the
	// not-resident fallback until this was turned on. This is the actual reason a
	// VoxelWaterSimulator attached to a fresh VoxelLodTerrain can appear to not render any
	// water at all with no error anywhere -- silent by design (cache_generated_blocks exists
	// for a real memory/perf tradeoff), but a simulator that depends on real data existing
	// needs it on, so default it on here rather than leaving it as a trap every caller has to
	// discover independently.
	terrain->set_cache_generated_blocks(true);

	// VoxelLodTerrain defaults to the legacy octree streaming system, whose LOD0 data-block
	// loading is not the one this comment already relied on: with cache_generated_blocks alone
	// (still under legacy octree), the terrain's `data_blocks` count grows into the thousands
	// (coarser LODs load fine) but LOD0 data specifically never becomes resident anywhere near
	// the viewer -- confirmed by instrumenting both systems directly: the legacy octree path
	// never even calls into the data-box-diff logic that would request LOD0 blocks for a plain
	// VoxelViewer, while the clipbox system (traced end to end: data_box_per_lod derived from
	// the mesh box, diffed each frame, request -> generate -> apply_data_block_response ->
	// try_set_block) does. The simulator's LOD0-only queries (is_solid_voxel, get_water_mass,
	// get_wet_block_positions) need real LOD0 residency to ever see anything but the
	// not-resident fallback, so force the modern system on here too.
	terrain->set_streaming_system(VoxelLodTerrain::STREAMING_SYSTEM_CLIPBOX);

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

	// Default planet_center to wherever the terrain actually is, rather than requiring the
	// caller to also set it manually -- see set_planet_radius()'s doc comment. Every planet
	// generator in this repo places the planet at its terrain node's own origin, so this is
	// right for the overwhelmingly common case; set_planet_center() afterward still overrides
	// it for the rare case where that's not true. get_global_position() requires being inside
	// the tree to resolve parent transforms correctly (this can run during scene deserialization
	// before that's true) -- local get_position() is a fine approximation for that case, since a
	// terrain nested under another transformed node is already an unusual setup.
	_planet_center = terrain->is_inside_tree() ? terrain->get_global_position() : terrain->get_position();
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

void VoxelWaterSimulator::set_planet_radius(float p_radius) {
	_planet_radius = math::max(p_radius, 0.0f);
	if (_planet_radius > 0.0f) {
		_gravity_mode = GRAVITY_RADIAL;
	}
}

float VoxelWaterSimulator::get_planet_radius() const {
	return _planet_radius;
}

void VoxelWaterSimulator::set_material(Ref<Material> p_material) {
	if (_material == p_material) {
		return;
	}
	_material = p_material;

	// Previously this just stored the new material and waited for a future _refresh_meshes()
	// pass to naturally rebuild nodes with it -- in practice that meant an already-built mesh
	// node kept showing whatever material it was created with until something else caused it to
	// be rebuilt from scratch (e.g. new water data streaming in), which could be indefinitely,
	// so changing the material in the editor visually did nothing. Worse, if the new material is
	// null, _refresh_meshes() early-returns before even reaching its own prune-stale-nodes step
	// (see its own doc comment), so previously-built nodes were never cleaned up either -- they
	// just sat there forever showing stale geometry with their last material. Both are fixed by
	// applying the change to already-tracked nodes immediately, right here.
	for (unsigned int lod = 0; lod < LOD_TIER_COUNT; ++lod) {
		if (_material.is_null()) {
			for (auto &kv : _mesh_nodes[lod]) {
				if (kv.second != nullptr) {
					kv.second->queue_free();
				}
			}
			_mesh_nodes[lod].clear();
			_coarse_mesh_keys[lod].clear();
		} else {
			for (auto &kv : _mesh_nodes[lod]) {
				if (kv.second != nullptr) {
					kv.second->set_material_override(_material);
				}
			}
		}
	}
}

Ref<Material> VoxelWaterSimulator::get_material() const {
	return _material;
}

void VoxelWaterSimulator::set_min_render_mass(float p_mass) {
	_min_render_mass = math::max(p_mass, 0.0f);
}

float VoxelWaterSimulator::get_min_render_mass() const {
	return _min_render_mass;
}

void VoxelWaterSimulator::set_mesh_refresh_interval(float p_seconds) {
	_mesh_refresh_interval = math::max(p_seconds, 0.01f);
}

float VoxelWaterSimulator::get_mesh_refresh_interval() const {
	return _mesh_refresh_interval;
}

void VoxelWaterSimulator::set_coarse_mesh_refresh_interval(float p_seconds) {
	_coarse_mesh_refresh_interval = math::max(p_seconds, 0.01f);
}

float VoxelWaterSimulator::get_coarse_mesh_refresh_interval() const {
	return _coarse_mesh_refresh_interval;
}

void VoxelWaterSimulator::set_max_query_lod(int p_lod) {
	_max_query_lod = math::max(p_lod, 0);
}

int VoxelWaterSimulator::get_max_query_lod() const {
	return _max_query_lod;
}

void VoxelWaterSimulator::set_update_interval(float p_seconds) {
	_update_interval = math::max(p_seconds, 0.01f);
}

float VoxelWaterSimulator::get_update_interval() const {
	return _update_interval;
}

void VoxelWaterSimulator::set_lod_update_interval_step(float p_seconds) {
	_lod_update_interval_step = math::max(p_seconds, 0.0f);
}

float VoxelWaterSimulator::get_lod_update_interval_step() const {
	return _lod_update_interval_step;
}

float VoxelWaterSimulator::get_lod_tick_interval(int p_lod) const {
	return _update_interval + float(math::max(p_lod, 0)) * _lod_update_interval_step;
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

	_active_blocks[0].insert(block_pos);
	_settle_counters[0][block_pos] = 0;
	for (const Vector3i &npos : woken_neighbors) {
		_active_blocks[0].insert(npos);
		_settle_counters[0][npos] = 0;
	}

	// update_mesh=false: see WaterSimTickTask::apply_result()'s comment on its own
	// post_edit_area() call -- a water-only edit never needs real terrain mesh/LOD cascading.
	terrain->post_edit_area(
			Box3i(block_pos * block_size, Vector3i(block_size, block_size, block_size)), false
	);
}

bool VoxelWaterSimulator::_find_resident_lod(
		const std::shared_ptr<VoxelData> &p_data,
		Vector3i p_global_voxel_lod0,
		unsigned int &out_lod,
		std::shared_ptr<VoxelBuffer> &out_buf,
		Vector3i &out_local
) const {
	const int bs = int(p_data->get_block_size());
	const unsigned int max_lod = math::min(
			static_cast<unsigned int>(math::max(_max_query_lod, 0)), p_data->get_lod_count() - 1
	);
	for (unsigned int lod = 0; lod <= max_lod; ++lod) {
		const Vector3i pos_at_lod = p_global_voxel_lod0 >> int(lod);
		const Vector3i bpos = math::floordiv(pos_at_lod, bs);
		SpatialLock3D::Read rlock(p_data->get_spatial_lock(lod), BoxBounds3i::from_position(bpos));
		std::shared_ptr<VoxelBuffer> buf = p_data->try_get_block_voxels(bpos, lod);
		if (buf != nullptr) {
			out_lod = lod;
			out_buf = buf;
			out_local = pos_at_lod - bpos * bs;
			return true;
		}
	}
	return false;
}

float VoxelWaterSimulator::get_water_mass(Vector3i p_global_voxel) const {
	std::shared_ptr<VoxelData> voxel_data = _get_data();
	if (voxel_data == nullptr) {
		return 0.0f;
	}
	unsigned int lod;
	std::shared_ptr<VoxelBuffer> buf;
	Vector3i local;
	if (!_find_resident_lod(voxel_data, p_global_voxel, lod, buf, local)) {
		return 0.0f;
	}
	// At lod > 0, WATER_CHANNEL holds whatever the generator baked, or that level's own
	// (slower-cadence, see get_lod_tick_interval()) simulation last wrote there -- an
	// approximation of the real LOD0 value, not a downsample of it. Good enough for "is there
	// roughly water here" until LOD0 arrives.
	return buf->get_voxel_f(local.x, local.y, local.z, WATER_CHANNEL);
}

bool VoxelWaterSimulator::is_solid_voxel(Vector3i p_global_voxel) const {
	std::shared_ptr<VoxelData> voxel_data = _get_data();
	if (voxel_data == nullptr) {
		return true;
	}
	unsigned int lod;
	std::shared_ptr<VoxelBuffer> buf;
	Vector3i local;
	if (!_find_resident_lod(voxel_data, p_global_voxel, lod, buf, local)) {
		return true;
	}
	return buf->get_voxel_f(local.x, local.y, local.z, VoxelBuffer::CHANNEL_SDF) <= 0.0f;
}

void VoxelWaterSimulator::activate_block(Vector3i p_block_pos) {
	activate_block_at_lod(p_block_pos, 0);
}

void VoxelWaterSimulator::deactivate_block(Vector3i p_block_pos) {
	deactivate_block_at_lod(p_block_pos, 0);
}

void VoxelWaterSimulator::activate_block_at_lod(Vector3i p_block_pos, int p_lod) {
	if (p_lod < 0 || p_lod > int(WATER_SIM_MAX_SIM_LOD)) {
		return;
	}
	std::shared_ptr<VoxelData> voxel_data = _get_data();
	if (voxel_data == nullptr) {
		return;
	}
	const unsigned int lod = static_cast<unsigned int>(p_lod);
	SpatialLock3D::Read rlock(voxel_data->get_spatial_lock(lod), BoxBounds3i::from_position(p_block_pos));
	if (voxel_data->try_get_block_voxels(p_block_pos, lod) == nullptr) {
		return;
	}
	_active_blocks[lod].insert(p_block_pos);
	_settle_counters[lod][p_block_pos] = 0;
}

void VoxelWaterSimulator::deactivate_block_at_lod(Vector3i p_block_pos, int p_lod) {
	if (p_lod < 0 || p_lod > int(WATER_SIM_MAX_SIM_LOD)) {
		return;
	}
	const unsigned int lod = static_cast<unsigned int>(p_lod);
	_active_blocks[lod].erase(p_block_pos);
	_settle_counters[lod].erase(p_block_pos);
}

void VoxelWaterSimulator::set_auto_scan_wet_blocks(bool p_enabled) {
	_auto_scan_wet_blocks = p_enabled;
}

bool VoxelWaterSimulator::get_auto_scan_wet_blocks() const {
	return _auto_scan_wet_blocks;
}

void VoxelWaterSimulator::set_auto_scan_interval(float p_seconds) {
	_auto_scan_interval = math::max(p_seconds, 0.1f);
}

float VoxelWaterSimulator::get_auto_scan_interval() const {
	return _auto_scan_interval;
}

void VoxelWaterSimulator::scan_and_activate_wet_blocks() {
	scan_and_activate_wet_blocks_at_lod(0);
}

void VoxelWaterSimulator::scan_and_activate_wet_blocks_at_lod(int p_lod) {
	if (p_lod < 0 || p_lod > int(WATER_SIM_MAX_SIM_LOD)) {
		return;
	}
	std::shared_ptr<VoxelData> voxel_data = _get_data();
	if (voxel_data == nullptr) {
		return;
	}
	const unsigned int lod = static_cast<unsigned int>(p_lod);
	// Collect first (activate_block_at_lod() below takes its own spatial read-lock per block,
	// which would deadlock against the global read-lock for_each_block_at_lod_r already holds
	// for the whole scan), then activate outside the scan.
	StdVector<Vector3i> to_activate;
	voxel_data->for_each_block_at_lod_r(
			[&to_activate](Vector3i p_bpos, const VoxelDataBlock &p_block) {
				if (!p_block.has_voxels()) {
					return;
				}
				std::shared_ptr<VoxelBuffer> buf = p_block.get_voxels_shared();
				if (buf != nullptr && !buf->is_uniform(WATER_CHANNEL)) {
					to_activate.push_back(p_bpos);
				}
			},
			lod
	);
	for (const Vector3i &bpos : to_activate) {
		activate_block_at_lod(bpos, p_lod);
	}
}

namespace {
void collect_wet_block_positions(
		const std::shared_ptr<VoxelData> &p_data, unsigned int p_lod, float p_min_render_mass,
		StdVector<Vector3i> &out_found
) {
	p_data->for_each_block_at_lod_r(
			[&out_found, p_min_render_mass](Vector3i p_bpos, const VoxelDataBlock &p_block) {
				if (!p_block.has_voxels()) {
					return;
				}
				std::shared_ptr<VoxelBuffer> buf = p_block.get_voxels_shared();
				if (buf == nullptr) {
					return;
				}
				if (buf->is_uniform(VoxelWaterSimulator::WATER_CHANNEL)) {
					// Uniform blocks are either entirely dry or entirely wet (e.g. open ocean
					// far from any shore/floor boundary). get_voxel_f at any position decodes
					// the same uniform value regardless of compression state.
					if (buf->get_voxel_f(0, 0, 0, VoxelWaterSimulator::WATER_CHANNEL) > p_min_render_mass) {
						out_found.push_back(p_bpos);
					}
				} else {
					// Has variation, so it contains water somewhere (a fully-dry block with no
					// variation would be uniform).
					out_found.push_back(p_bpos);
				}
			},
			p_lod
	);
}
} // namespace

TypedArray<Vector3i> VoxelWaterSimulator::get_wet_block_positions(float p_min_render_mass) const {
	TypedArray<Vector3i> arr;
	std::shared_ptr<VoxelData> voxel_data = _get_data();
	if (voxel_data == nullptr) {
		return arr;
	}
	StdVector<Vector3i> found;
	collect_wet_block_positions(voxel_data, 0, p_min_render_mass, found);
	arr.resize(found.size());
	for (unsigned int i = 0; i < found.size(); ++i) {
		arr[i] = found[i];
	}
	return arr;
}

TypedArray<Vector3i> VoxelWaterSimulator::get_wet_block_positions_at_lod(int p_lod, float p_min_render_mass) const {
	TypedArray<Vector3i> arr;
	std::shared_ptr<VoxelData> voxel_data = _get_data();
	if (voxel_data == nullptr || p_lod < 0) {
		return arr;
	}
	const unsigned int lod = math::min(static_cast<unsigned int>(p_lod), voxel_data->get_lod_count() - 1);
	StdVector<Vector3i> found;
	collect_wet_block_positions(voxel_data, lod, p_min_render_mass, found);
	arr.resize(found.size());
	for (unsigned int i = 0; i < found.size(); ++i) {
		arr[i] = found[i];
	}
	return arr;
}

int VoxelWaterSimulator::get_active_block_count() const {
	return get_active_block_count_at_lod(0);
}

bool VoxelWaterSimulator::is_block_active(Vector3i p_block_pos) const {
	return is_block_active_at_lod(p_block_pos, 0);
}

TypedArray<Vector3i> VoxelWaterSimulator::get_active_block_positions() const {
	return get_active_block_positions_at_lod(0);
}

int VoxelWaterSimulator::get_active_block_count_at_lod(int p_lod) const {
	if (p_lod < 0 || p_lod > int(WATER_SIM_MAX_SIM_LOD)) {
		return 0;
	}
	return int(_active_blocks[p_lod].size());
}

bool VoxelWaterSimulator::is_block_active_at_lod(Vector3i p_block_pos, int p_lod) const {
	if (p_lod < 0 || p_lod > int(WATER_SIM_MAX_SIM_LOD)) {
		return false;
	}
	const auto &blocks = _active_blocks[p_lod];
	return blocks.find(p_block_pos) != blocks.end();
}

TypedArray<Vector3i> VoxelWaterSimulator::get_active_block_positions_at_lod(int p_lod) const {
	TypedArray<Vector3i> arr;
	if (p_lod < 0 || p_lod > int(WATER_SIM_MAX_SIM_LOD)) {
		return arr;
	}
	const auto &blocks = _active_blocks[p_lod];
	arr.resize(blocks.size());
	int i = 0;
	for (const Vector3i &b : blocks) {
		arr[i++] = b;
	}
	return arr;
}

bool VoxelWaterSimulator::is_tick_in_flight() const {
	// True if ANY simulated LOD tier currently has a tick in flight.
	for (unsigned int lod = 0; lod < LOD_TIER_COUNT; ++lod) {
		if (_tick_in_flight[lod].load(std::memory_order_acquire)) {
			return true;
		}
	}
	return false;
}

void VoxelWaterSimulator::_process_time(double p_delta) {
	if (_auto_scan_wet_blocks) {
		// Runs synchronously on the main thread (unlike ticking) and, for lod > 0, walks every
		// resident block at that level regardless of how many are actually wet -- coarser LODs
		// can have block counts comparable to LOD0's (see get_lod_tick_interval()'s doc comment
		// on the same issue for ticking), so scanning all of them at a flat auto_scan_interval
		// measurably stalled the main thread once enough streamed in. Same per-LOD cadence
		// tiering as ticking fixes it the same way.
		for (unsigned int lod = 0; lod < LOD_TIER_COUNT; ++lod) {
			_scan_time_accumulator[lod] += p_delta;
			if (_scan_time_accumulator[lod] < double(get_lod_tick_interval(int(lod))) + double(_auto_scan_interval)) {
				continue;
			}
			_scan_time_accumulator[lod] = 0.0;
			scan_and_activate_wet_blocks_at_lod(int(lod));
		}
	}

	// Each simulated LOD tier ticks independently on its own cadence (see
	// get_lod_tick_interval()'s doc comment) and has its own in-flight guard, so a slow coarse
	// tick never blocks LOD0's fast one or vice versa.
	for (unsigned int lod = 0; lod < LOD_TIER_COUNT; ++lod) {
		_time_accumulator[lod] += p_delta;
		if (_time_accumulator[lod] < double(get_lod_tick_interval(int(lod)))) {
			continue;
		}
		if (_tick_in_flight[lod].load(std::memory_order_acquire)) {
			// Previous tick for this tier is still running -- wait for it instead of queuing
			// more work. Not resetting the accumulator means the very next _process() call
			// will retry immediately once it clears, rather than waiting a full extra interval.
			continue;
		}
		_time_accumulator[lod] = 0.0;
		_dispatch_tick(lod);
	}

	_refresh_meshes(p_delta);
}

void VoxelWaterSimulator::_dispatch_tick(unsigned int p_lod) {
	VoxelLodTerrain *terrain = _get_terrain();
	StdUnorderedSet<Vector3i> &active_blocks = _active_blocks[p_lod];
	if (terrain == nullptr || active_blocks.empty()) {
		return;
	}
	std::shared_ptr<VoxelData> voxel_data = terrain->get_storage_shared();
	if (voxel_data == nullptr) {
		return;
	}

	StdVector<Vector3i> all_active;
	all_active.reserve(active_blocks.size());
	for (const Vector3i &b : active_blocks) {
		all_active.push_back(b);
	}

	const int total = int(all_active.size());
	const int process_count = math::min(total, _max_blocks_per_tick);
	StdVector<Vector3i> snapshot;
	snapshot.reserve(process_count);
	uint32_t &cursor = _round_robin_cursor[p_lod];
	for (int i = 0; i < process_count; ++i) {
		snapshot.push_back(all_active[(cursor + i) % total]);
	}
	cursor = (total > 0) ? (cursor + process_count) % total : 0;

	StdVector<uint64_t> absorbing_ids;
	absorbing_ids.reserve(_absorbing_type_ids.size());
	for (int i = 0; i < _absorbing_type_ids.size(); ++i) {
		absorbing_ids.push_back(uint64_t(_absorbing_type_ids[i]));
	}

	_tick_in_flight[p_lod].store(true, std::memory_order_release);

	WaterSimTickTask *task = ZN_NEW(WaterSimTickTask(
			get_instance_id(),
			voxel_data,
			voxel_data->get_block_size(),
			p_lod,
			snapshot,
			_gravity_mode,
			_planet_center,
			_absorption_rate,
			_max_absorption_per_voxel,
			absorbing_ids
	));

	VoxelEngine::get_singleton().push_async_task(task);
}

void VoxelWaterSimulator::_on_tick_completed(const StdVector<WaterSimBlockResult> &p_results, unsigned int p_lod) {
	StdUnorderedSet<Vector3i> &active_blocks = _active_blocks[p_lod];
	StdUnorderedMap<Vector3i, int> &settle_counters = _settle_counters[p_lod];
	for (const WaterSimBlockResult &r : p_results) {
		if (r.missing) {
			active_blocks.erase(r.block_pos);
			settle_counters.erase(r.block_pos);
			continue;
		}
		if (r.had_flow) {
			active_blocks.insert(r.block_pos);
			settle_counters[r.block_pos] = 0;
		} else {
			auto it = settle_counters.find(r.block_pos);
			const int count = (it != settle_counters.end()) ? it->second + 1 : 1;
			settle_counters[r.block_pos] = count;
			if (count >= _settle_ticks_threshold) {
				active_blocks.erase(r.block_pos);
				settle_counters.erase(r.block_pos);
			}
		}
	}
	_tick_in_flight[p_lod].store(false, std::memory_order_release);
}

void VoxelWaterSimulator::_lod0_corners(Vector3i block_pos, unsigned int lod, Vector3i (&out_corners)[8]) {
	const Vector3i lo = block_pos * (1 << lod);
	const Vector3i hi = lo + Vector3i(1, 1, 1) * ((1 << lod) - 1);
	int i = 0;
	for (int x = 0; x < 2; ++x) {
		for (int y = 0; y < 2; ++y) {
			for (int z = 0; z < 2; ++z) {
				out_corners[i++] = Vector3i(x == 0 ? lo.x : hi.x, y == 0 ? lo.y : hi.y, z == 0 ? lo.z : hi.z);
			}
		}
	}
}

void VoxelWaterSimulator::_update_mesh_node(unsigned int lod, Vector3i block_pos, int block_size) {
	PackedVector3Array faces = generate_water_mesh_smooth_faces(block_pos, _min_render_mass, int(lod));
	if (faces.size() == 0) {
		_remove_mesh_node(lod, block_pos);
		return;
	}

	MeshInstance3D *node = nullptr;
	auto it = _mesh_nodes[lod].find(block_pos);
	if (it == _mesh_nodes[lod].end()) {
		node = memnew(MeshInstance3D);
		node->set_material_override(_material);
		node->set_cast_shadows_setting(GeometryInstance3D::SHADOW_CASTING_SETTING_OFF);
		add_child(node);
		const float scale_factor = float(1 << lod);
		node->set_position(Vector3(block_pos) * float(block_size) * scale_factor);
		node->set_scale(Vector3(1.0f, 1.0f, 1.0f) * scale_factor);
		_mesh_nodes[lod][block_pos] = node;
	} else {
		node = it->second;
	}

	Ref<SurfaceTool> st;
	st.instantiate();
	st->begin(Mesh::PRIMITIVE_TRIANGLES);
	for (int i = 0; i < faces.size(); ++i) {
		st->add_vertex(faces[i]);
	}
	st->generate_normals();
	node->set_mesh(st->commit());
}

void VoxelWaterSimulator::_remove_mesh_node(unsigned int lod, Vector3i block_pos) {
	auto it = _mesh_nodes[lod].find(block_pos);
	if (it != _mesh_nodes[lod].end()) {
		it->second->queue_free();
		_mesh_nodes[lod].erase(it);
	}
	// No-op for lod 0 (never inserted there), harmless.
	_coarse_mesh_keys[lod].erase(block_pos);
}

void VoxelWaterSimulator::_refresh_meshes(double p_delta) {
	if (_material.is_null()) {
		// Opt-out: no material assigned means this node builds no meshes at all, matching its
		// behavior before this existed -- see set_material()'s doc comment.
		return;
	}
	VoxelLodTerrain *terrain = _get_terrain();
	if (terrain == nullptr) {
		return;
	}

	_mesh_refresh_accum += p_delta;
	if (_mesh_refresh_accum < double(_mesh_refresh_interval)) {
		return;
	}
	_mesh_refresh_accum = 0.0;

	const int block_size = int(terrain->get_data_block_size());
	StdUnorderedSet<Vector3i> seen[LOD_TIER_COUNT];
	StdUnorderedSet<Vector3i> claimed;

	{
		TypedArray<Vector3i> wet0 = get_wet_block_positions_at_lod(0, _min_render_mass);
		for (int i = 0; i < wet0.size(); ++i) {
			const Vector3i bp = wet0[i];
			seen[0].insert(bp);
			Vector3i corners[8];
			_lod0_corners(bp, 0, corners);
			for (int c = 0; c < 8; ++c) {
				claimed.insert(corners[c]);
			}
			_update_mesh_node(0, bp, block_size);
		}
	}

	// Coarser LODs (1..max render level): their data boxes can be comparable in block count to
	// LOD0's, so they're throttled to their own slower cadence and skip regenerating geometry
	// for blocks that already have a mesh (a lower-cadence fallback isn't real-time-critical --
	// see mesh_refresh_interval's doc comment).
	const unsigned int max_render_lod =
			math::min(static_cast<unsigned int>(math::max(_max_query_lod, 0)), WATER_SIM_MAX_SIM_LOD);
	_coarse_mesh_refresh_accum += _mesh_refresh_interval;
	if (_coarse_mesh_refresh_accum >= double(_coarse_mesh_refresh_interval)) {
		_coarse_mesh_refresh_accum = 0.0;
		for (unsigned int lod = 1; lod <= max_render_lod; ++lod) {
			TypedArray<Vector3i> wet = get_wet_block_positions_at_lod(int(lod), _min_render_mass);
			for (int i = 0; i < wet.size(); ++i) {
				const Vector3i bp = wet[i];
				Vector3i corners[8];
				_lod0_corners(bp, lod, corners);
				bool covered = false;
				for (int c = 0; c < 8; ++c) {
					if (claimed.find(corners[c]) != claimed.end()) {
						covered = true;
						break;
					}
				}
				if (covered) {
					_remove_mesh_node(lod, bp);
					continue;
				}
				for (int c = 0; c < 8; ++c) {
					claimed.insert(corners[c]);
				}
				_coarse_mesh_keys[lod].insert(bp);
				if (_mesh_nodes[lod].find(bp) != _mesh_nodes[lod].end()) {
					continue;
				}
				_update_mesh_node(lod, bp, block_size);
			}
		}
	}
	// Coarse meshes persist across passes that didn't touch them (not re-scanned every pass),
	// so keep them out of the prune step below unless a scan actually dropped them.
	for (unsigned int lod = 1; lod < LOD_TIER_COUNT; ++lod) {
		for (const Vector3i &bp : _coarse_mesh_keys[lod]) {
			seen[lod].insert(bp);
		}
	}

	for (unsigned int lod = 0; lod < LOD_TIER_COUNT; ++lod) {
		StdVector<Vector3i> to_remove;
		for (const auto &kv : _mesh_nodes[lod]) {
			if (seen[lod].find(kv.first) == seen[lod].end()) {
				to_remove.push_back(kv.first);
			}
		}
		for (const Vector3i &bp : to_remove) {
			_remove_mesh_node(lod, bp);
		}
	}
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

PackedVector3Array VoxelWaterSimulator::generate_water_mesh_faces(
		Vector3i p_block_pos, float p_min_render_mass, int p_lod
) const {
	PackedVector3Array faces;
	std::shared_ptr<VoxelData> voxel_data = _get_data();
	if (voxel_data == nullptr || p_lod < 0) {
		return faces;
	}
	const unsigned int lod = math::min(static_cast<unsigned int>(p_lod), voxel_data->get_lod_count() - 1);
	const int bs = int(voxel_data->get_block_size());

	SpatialLock3D::Read rlock(
			voxel_data->get_spatial_lock(lod),
			BoxBounds3i(p_block_pos - Vector3i(1, 1, 1), p_block_pos + Vector3i(2, 2, 2))
	);

	std::shared_ptr<VoxelBuffer> buf = voxel_data->try_get_block_voxels(p_block_pos, lod);
	if (buf == nullptr) {
		return faces;
	}

	auto get_mass_at = [&](Vector3i g) -> float {
		const Vector3i bp = math::floordiv(g, bs);
		std::shared_ptr<VoxelBuffer> b = (bp == p_block_pos) ? buf : voxel_data->try_get_block_voxels(bp, lod);
		if (b == nullptr) {
			return 0.0f;
		}
		const Vector3i local = g - bp * bs;
		return b->get_voxel_f(local.x, local.y, local.z, WATER_CHANNEL);
	};
	auto is_solid_at = [&](Vector3i g) -> bool {
		const Vector3i bp = math::floordiv(g, bs);
		std::shared_ptr<VoxelBuffer> b = (bp == p_block_pos) ? buf : voxel_data->try_get_block_voxels(bp, lod);
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
		Vector3i p_block_pos, float p_min_render_mass, int p_lod
) const {
	PackedVector3Array faces;
	std::shared_ptr<VoxelData> voxel_data = _get_data();
	if (voxel_data == nullptr || p_lod < 0) {
		return faces;
	}
	const unsigned int lod = math::min(static_cast<unsigned int>(p_lod), voxel_data->get_lod_count() - 1);
	const int bs = int(voxel_data->get_block_size());

	SpatialLock3D::Read rlock(voxel_data->get_spatial_lock(lod), BoxBounds3i::from_position(p_block_pos));

	if (voxel_data->try_get_block_voxels(p_block_pos, lod) == nullptr) {
		return faces;
	}

	auto get_mass_at = [&](Vector3i g) -> float {
		const Vector3i bp = math::floordiv(g, bs);
		std::shared_ptr<VoxelBuffer> b = voxel_data->try_get_block_voxels(bp, lod);
		if (b == nullptr) {
			return 0.0f;
		}
		const Vector3i local = g - bp * bs;
		return b->get_voxel_f(local.x, local.y, local.z, WATER_CHANNEL);
	};
	auto is_solid_at = [&](Vector3i g) -> bool {
		const Vector3i bp = math::floordiv(g, bs);
		std::shared_ptr<VoxelBuffer> b = voxel_data->try_get_block_voxels(bp, lod);
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

			// Branches intentionally swapped from the "expected" (!flip -> p00,p10,p11) order --
			// verified empirically this session that this winding was backwards for Godot's
			// front-face convention (the water mesh was only visible from underwater looking
			// outward, i.e. from the back side of every triangle).
			const bool flip = (p10 - p00).cross(p01 - p00).dot(up_local) < 0.0f;
			if (!flip) {
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

	ClassDB::bind_method(
			D_METHOD("set_enabled_in_editor", "enabled"), &VoxelWaterSimulator::set_enabled_in_editor
	);
	ClassDB::bind_method(D_METHOD("is_enabled_in_editor"), &VoxelWaterSimulator::is_enabled_in_editor);
	ADD_PROPERTY(
			PropertyInfo(Variant::BOOL, "enabled_in_editor"), "set_enabled_in_editor", "is_enabled_in_editor"
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

	ClassDB::bind_method(D_METHOD("set_planet_radius", "radius"), &VoxelWaterSimulator::set_planet_radius);
	ClassDB::bind_method(D_METHOD("get_planet_radius"), &VoxelWaterSimulator::get_planet_radius);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "planet_radius"), "set_planet_radius", "get_planet_radius");

	ClassDB::bind_method(D_METHOD("set_material", "material"), &VoxelWaterSimulator::set_material);
	ClassDB::bind_method(D_METHOD("get_material"), &VoxelWaterSimulator::get_material);
	ADD_PROPERTY(
			PropertyInfo(Variant::OBJECT, "material", PROPERTY_HINT_RESOURCE_TYPE, "Material"),
			"set_material",
			"get_material"
	);

	ClassDB::bind_method(D_METHOD("set_min_render_mass", "mass"), &VoxelWaterSimulator::set_min_render_mass);
	ClassDB::bind_method(D_METHOD("get_min_render_mass"), &VoxelWaterSimulator::get_min_render_mass);
	ADD_PROPERTY(
			PropertyInfo(Variant::FLOAT, "min_render_mass"), "set_min_render_mass", "get_min_render_mass"
	);

	ClassDB::bind_method(
			D_METHOD("set_mesh_refresh_interval", "seconds"), &VoxelWaterSimulator::set_mesh_refresh_interval
	);
	ClassDB::bind_method(D_METHOD("get_mesh_refresh_interval"), &VoxelWaterSimulator::get_mesh_refresh_interval);
	ADD_PROPERTY(
			PropertyInfo(Variant::FLOAT, "mesh_refresh_interval"),
			"set_mesh_refresh_interval",
			"get_mesh_refresh_interval"
	);

	ClassDB::bind_method(
			D_METHOD("set_coarse_mesh_refresh_interval", "seconds"),
			&VoxelWaterSimulator::set_coarse_mesh_refresh_interval
	);
	ClassDB::bind_method(
			D_METHOD("get_coarse_mesh_refresh_interval"), &VoxelWaterSimulator::get_coarse_mesh_refresh_interval
	);
	ADD_PROPERTY(
			PropertyInfo(Variant::FLOAT, "coarse_mesh_refresh_interval"),
			"set_coarse_mesh_refresh_interval",
			"get_coarse_mesh_refresh_interval"
	);

	ClassDB::bind_method(D_METHOD("set_max_query_lod", "lod"), &VoxelWaterSimulator::set_max_query_lod);
	ClassDB::bind_method(D_METHOD("get_max_query_lod"), &VoxelWaterSimulator::get_max_query_lod);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_query_lod"), "set_max_query_lod", "get_max_query_lod");

	ClassDB::bind_method(D_METHOD("set_update_interval", "seconds"), &VoxelWaterSimulator::set_update_interval);
	ClassDB::bind_method(D_METHOD("get_update_interval"), &VoxelWaterSimulator::get_update_interval);
	ADD_PROPERTY(
			PropertyInfo(Variant::FLOAT, "update_interval"), "set_update_interval", "get_update_interval"
	);

	ClassDB::bind_method(
			D_METHOD("set_lod_update_interval_step", "seconds"),
			&VoxelWaterSimulator::set_lod_update_interval_step
	);
	ClassDB::bind_method(
			D_METHOD("get_lod_update_interval_step"), &VoxelWaterSimulator::get_lod_update_interval_step
	);
	ADD_PROPERTY(
			PropertyInfo(Variant::FLOAT, "lod_update_interval_step"),
			"set_lod_update_interval_step",
			"get_lod_update_interval_step"
	);
	ClassDB::bind_method(D_METHOD("get_lod_tick_interval", "lod"), &VoxelWaterSimulator::get_lod_tick_interval);

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

	ClassDB::bind_method(
			D_METHOD("set_auto_scan_wet_blocks", "enabled"), &VoxelWaterSimulator::set_auto_scan_wet_blocks
	);
	ClassDB::bind_method(D_METHOD("get_auto_scan_wet_blocks"), &VoxelWaterSimulator::get_auto_scan_wet_blocks);
	ADD_PROPERTY(
			PropertyInfo(Variant::BOOL, "auto_scan_wet_blocks"),
			"set_auto_scan_wet_blocks",
			"get_auto_scan_wet_blocks"
	);

	ClassDB::bind_method(
			D_METHOD("set_auto_scan_interval", "seconds"), &VoxelWaterSimulator::set_auto_scan_interval
	);
	ClassDB::bind_method(D_METHOD("get_auto_scan_interval"), &VoxelWaterSimulator::get_auto_scan_interval);
	ADD_PROPERTY(
			PropertyInfo(Variant::FLOAT, "auto_scan_interval"),
			"set_auto_scan_interval",
			"get_auto_scan_interval"
	);

	ClassDB::bind_method(
			D_METHOD("scan_and_activate_wet_blocks"), &VoxelWaterSimulator::scan_and_activate_wet_blocks
	);
	ClassDB::bind_method(
			D_METHOD("scan_and_activate_wet_blocks_at_lod", "lod"),
			&VoxelWaterSimulator::scan_and_activate_wet_blocks_at_lod
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
	ClassDB::bind_method(
			D_METHOD("activate_block_at_lod", "block_pos", "lod"), &VoxelWaterSimulator::activate_block_at_lod
	);
	ClassDB::bind_method(
			D_METHOD("deactivate_block_at_lod", "block_pos", "lod"),
			&VoxelWaterSimulator::deactivate_block_at_lod
	);
	ClassDB::bind_method(
			D_METHOD("get_active_block_count_at_lod", "lod"),
			&VoxelWaterSimulator::get_active_block_count_at_lod
	);
	ClassDB::bind_method(
			D_METHOD("is_block_active_at_lod", "block_pos", "lod"),
			&VoxelWaterSimulator::is_block_active_at_lod
	);
	ClassDB::bind_method(
			D_METHOD("get_active_block_positions_at_lod", "lod"),
			&VoxelWaterSimulator::get_active_block_positions_at_lod
	);
	ClassDB::bind_method(
			D_METHOD("get_wet_block_positions", "min_render_mass"), &VoxelWaterSimulator::get_wet_block_positions,
			DEFVAL(0.05f)
	);
	ClassDB::bind_method(
			D_METHOD("get_wet_block_positions_at_lod", "lod", "min_render_mass"),
			&VoxelWaterSimulator::get_wet_block_positions_at_lod,
			DEFVAL(0.05f)
	);
	ClassDB::bind_method(D_METHOD("is_tick_in_flight"), &VoxelWaterSimulator::is_tick_in_flight);

	ClassDB::bind_method(
			D_METHOD("generate_water_mesh_faces", "block_pos", "min_render_mass", "lod"),
			&VoxelWaterSimulator::generate_water_mesh_faces,
			DEFVAL(0.05f), DEFVAL(0)
	);
	ClassDB::bind_method(
			D_METHOD("generate_water_mesh_smooth_faces", "block_pos", "min_render_mass", "lod"),
			&VoxelWaterSimulator::generate_water_mesh_smooth_faces,
			DEFVAL(0.05f), DEFVAL(0)
	);

	BIND_ENUM_CONSTANT(GRAVITY_Y_DOWN);
	BIND_ENUM_CONSTANT(GRAVITY_RADIAL);
}

} // namespace zylann::voxel
