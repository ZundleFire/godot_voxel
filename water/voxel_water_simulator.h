#ifndef VOXEL_WATER_SIMULATOR_H
#define VOXEL_WATER_SIMULATOR_H

#include "../storage/voxel_buffer.h"
#include "../util/containers/std_unordered_map.h"
#include "../util/containers/std_unordered_set.h"
#include "../util/containers/std_vector.h"
#include "../util/godot/classes/node.h"
#include "../util/godot/classes/object.h"

#include <atomic>
#include <memory>

namespace zylann::voxel {

class VoxelLodTerrain;
class VoxelData;
class WaterSimTickTask;

// Outcome of processing one block during a tick, reported from WaterSimTickTask (worker
// thread, computed in run()) back to VoxelWaterSimulator (main thread, consumed in
// apply_result()/_on_tick_completed()). Free struct (not nested in either class) so both
// headers can reference it without a circular include.
struct WaterSimBlockResult {
	Vector3i block_pos;
	// True if this block's mass changed by more than MIN_MASS this tick. Drives the
	// simulator's settle/dormancy bookkeeping.
	bool had_flow = false;
	// True if the block no longer exists in VoxelData (e.g. unloaded while the tick was in
	// flight) and should be dropped from the active set unconditionally.
	bool missing = false;
};

// Sparse 3D cellular-automaton fluid simulation for a VoxelLodTerrain.
//
// Ported from GodotEden's `EdenWaterSimulator` prototype (mass-conserving, gravity-ranked
// 6-neighbor flow with a slightly-compressible-liquid stable-state model, radial or flat
// gravity, per-block settle/dormancy tracking so simulation cost tracks "how much water is
// currently in motion", not "how many chunks are loaded"). The core per-voxel math is the
// same. What's different, and the entire reason for this port: that prototype ran its
// per-block voxel scan synchronously on the main thread, once per fixed_timestep tick
// (capped, but still measured at 15-30+ real seconds for a single tick over a few hundred
// active coastal blocks). This version:
//   1. Dispatches a tick at most once per `update_interval` (default 0.5s) of wall-clock
//      time, not once per frame/physics-tick.
//   2. Runs the actual scan on VoxelEngine's general worker thread pool (see
//      WaterSimTickTask), never on the main thread. The main thread only ever applies a
//      small delta map and calls VoxelLodTerrain::post_edit_area() to queue a remesh.
//   3. Keeps a single-flight guard (_tick_in_flight) so a slow tick is simply awaited by the
//      next `_process()` check rather than queuing up backlogged work.
// `max_blocks_per_tick` (round-robin, ported as-is from Eden) remains as defense-in-depth:
// even at a 0.5s cadence, a real coastline can activate hundreds of blocks in one dispatch.
//
// Water mass is stored in VoxelBuffer::CHANNEL_DATA5 (one of 3 channels with no built-in
// meaning, reserved for exactly this kind of per-game-data use) as a normalized float,
// independent of CHANNEL_SDF/terrain occupancy — a voxel can be simultaneously "air" to the
// terrain mesher and "full of water" to this simulator. attach_to_terrain() bumps that
// channel's format depth to 16-bit (from the 8-bit default) for finer mass resolution; call
// it before any blocks of the terrain are generated for the depth change to take effect.
class VoxelWaterSimulator : public Node {
	GDCLASS(VoxelWaterSimulator, Node)
public:
	enum GravityMode {
		// Flat-world gravity: -Y is always "down". Cheapest, correct for non-spherical worlds.
		GRAVITY_Y_DOWN = 0,
		// Planet gravity: "down" at a voxel is toward planet_center, computed from that
		// voxel's own world position - no stored per-voxel direction data needed.
		GRAVITY_RADIAL = 1,
	};

	// Slightly-compressible-liquid tuning, identical meaning/defaults to EdenWaterSimulator.
	static constexpr float MAX_MASS = 1.0f;
	static constexpr float MAX_COMPRESS = 0.02f;
	static constexpr float MIN_MASS = 0.0001f;
	static constexpr float MAX_SPEED = 0.8f;

	// Water mass channel. One of VoxelBuffer's 3 free "no builtin meaning" channels.
	static constexpr VoxelBuffer::ChannelId WATER_CHANNEL = VoxelBuffer::CHANNEL_DATA5;

	VoxelWaterSimulator();
	~VoxelWaterSimulator();

	// Terrain this simulator reads/writes. Also bumps WATER_CHANNEL's format depth to
	// DEPTH_16_BIT on the terrain if it's currently lower (only matters before blocks exist).
	void set_terrain(Node *p_terrain);
	Node *get_terrain() const;

	void set_gravity_mode(GravityMode p_mode);
	GravityMode get_gravity_mode() const;

	void set_planet_center(Vector3 p_center);
	Vector3 get_planet_center() const;

	// Minimum wall-clock seconds between dispatched ticks. Default 0.5 - the explicit
	// performance requirement this port exists to satisfy (Eden's version scanned every
	// tick, i.e. every frame).
	void set_update_interval(float p_seconds);
	float get_update_interval() const;

	// Round-robin cap on how many of the currently-active blocks get processed in any one
	// dispatched tick. Defense-in-depth alongside update_interval: bounds worker-thread scan
	// cost even if the active set is very large.
	void set_max_blocks_per_tick(int p_count);
	int get_max_blocks_per_tick() const;

	// Consecutive quiet ticks (no change > MIN_MASS) before a block is dropped from the
	// active set and stops being simulated.
	void set_settle_ticks_threshold(int p_ticks);
	int get_settle_ticks_threshold() const;

	void set_absorption_rate(float p_rate);
	float get_absorption_rate() const;
	void set_max_absorption_per_voxel(float p_max);
	float get_max_absorption_per_voxel() const;
	// VoxelBuffer::CHANNEL_TYPE ids of materials that absorb water (default: empty, i.e.
	// disabled - matches Eden's own default of absorption_rate=0).
	void set_absorbing_type_ids(PackedInt32Array p_ids);
	PackedInt32Array get_absorbing_type_ids() const;

	// Adds (or removes, with a negative amount) water mass at a global voxel coordinate.
	// Applied immediately on the main thread (not batched into the async tick) since this is
	// a rare, caller-driven op, not the per-tick hot path. No-op on a solid voxel or a voxel
	// in an unloaded block. Marks the voxel's block (and any resident face-neighbor block)
	// active.
	void add_water(Vector3i p_global_voxel, float p_amount);
	float get_water_mass(Vector3i p_global_voxel) const;
	bool is_solid_voxel(Vector3i p_global_voxel) const;

	// Marks a block active without touching its mass (e.g. for externally-baked water) /
	// pauses simulation for a block without touching its stored mass (e.g. camera-distance
	// culling by caller code). No-op if the block doesn't exist / isn't active respectively.
	void activate_block(Vector3i p_block_pos);
	void deactivate_block(Vector3i p_block_pos);

	int get_active_block_count() const;
	bool is_block_active(Vector3i p_block_pos) const;
	TypedArray<Vector3i> get_active_block_positions() const;

	// True if a WaterSimTickTask is currently running on the worker pool. Mainly for tests
	// (verifying the single-flight guard) and introspection.
	bool is_tick_in_flight() const;

	// "Down" direction at a voxel: -Y for GRAVITY_Y_DOWN, or normalized (planet_center - voxel
	// world position) for GRAVITY_RADIAL. Shared by WaterSimTickTask's per-voxel flow ranking
	// and generate_water_mesh_smooth_faces() below (both need the same gravity model).
	static Vector3 compute_down_dir(GravityMode p_mode, Vector3 p_planet_center, Vector3i p_global_pos);

	// Greedy-meshed volumetric water surface for one block: one quad per maximal same-face run
	// of "wet" voxels (mass > p_min_render_mass) whose neighbor in that face direction is not
	// itself wet and not solid terrain -- i.e. only visible water surface, never faces buried
	// under terrain or sandwiched between two wet voxels. Blocky (not smooth-contoured), but
	// topologically exact through caves/overhangs, since the CA's mass field has no gradient to
	// contour against. Cross-block boundary faces resample the real neighbor block. Vertices
	// are in this block's local voxel-index space (0..block_size).
	PackedVector3Array generate_water_mesh_faces(Vector3i p_block_pos, float p_min_render_mass = 0.05f) const;

	// Heightfield alternative to generate_water_mesh_faces(): for every (u, v) lateral position
	// (the two axes perpendicular to this block's own locally-approximated "up"), scans along
	// "up" for the first non-solid wet voxel and sub-voxel-interpolates its exact surface
	// height from that voxel's own fill fraction -- one smooth surface per water body instead
	// of a per-voxel ripple traced from the raw CA mass field. A column is left dry if a solid
	// voxel is hit before any wet one scanning up -- this is a heightfield, not the full
	// volumetric CA render, so it cannot represent cave/overhang water (use
	// generate_water_mesh_faces() for that case). Cross-block continuity is automatic and
	// bit-exact: a shared grid line's height is recomputed independently by each neighboring
	// block from the same global voxel data. Vertices are in this block's local voxel-index
	// space.
	PackedVector3Array generate_water_mesh_smooth_faces(Vector3i p_block_pos, float p_min_render_mass = 0.05f) const;

protected:
	void _notification(int p_what);

private:
	friend class WaterSimTickTask;

	static void _bind_methods();

	void _process_time(double p_delta);
	void _dispatch_tick();
	// Called on the main thread by WaterSimTickTask::apply_result(). Not exposed to
	// scripting.
	void _on_tick_completed(const StdVector<WaterSimBlockResult> &p_results);

	VoxelLodTerrain *_get_terrain() const;
	std::shared_ptr<VoxelData> _get_data() const;

	ObjectID _terrain_id;

	GravityMode _gravity_mode = GRAVITY_RADIAL;
	Vector3 _planet_center;
	float _update_interval = 0.5f;
	int _max_blocks_per_tick = 64;
	int _settle_ticks_threshold = 4;
	float _absorption_rate = 0.0f;
	float _max_absorption_per_voxel = 1.0f;
	PackedInt32Array _absorbing_type_ids;

	double _time_accumulator = 0.0;
	std::atomic_bool _tick_in_flight = { false };
	uint32_t _round_robin_cursor = 0;

	StdUnorderedSet<Vector3i> _active_blocks;
	StdUnorderedMap<Vector3i, int> _settle_counters;
};

} // namespace zylann::voxel

VARIANT_ENUM_CAST(zylann::voxel::VoxelWaterSimulator::GravityMode);

#endif // VOXEL_WATER_SIMULATOR_H
