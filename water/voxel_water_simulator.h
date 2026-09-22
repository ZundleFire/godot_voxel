#ifndef VOXEL_WATER_SIMULATOR_H
#define VOXEL_WATER_SIMULATOR_H

#include "../storage/voxel_buffer.h"
#include "../util/containers/std_unordered_map.h"
#include "../util/containers/std_unordered_set.h"
#include "../util/containers/std_vector.h"
#include "../util/godot/classes/material.h"
#include "../util/godot/classes/mesh_instance_3d.h"
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

// Simulation runs on LOD0 (real detail) plus this many coarser levels above it (LOD 1..N),
// each on its own, progressively slower cadence -- see VoxelWaterSimulator::get_lod_tick_interval().
// A level above this never ticks: its water is whatever the generator baked or a finer level's
// simulation left behind last time that area was at a finer LOD, permanently static otherwise.
// Matches max_query_lod's read-side fallback depth by default expectation, though the two are
// independent knobs.
constexpr unsigned int WATER_SIM_MAX_SIM_LOD = 3;

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
//
// Also builds and manages its own visible water meshes: set_terrain(), a material (see
// set_material()), and optionally set_planet_radius(), are the only setup a caller needs --
// no script has to be attached to make water actually appear on screen. This was originally an
// external responsibility (a script calling generate_water_mesh_*_faces() and managing its own
// MeshInstance3D children every frame, e.g. GodotEden's demo_eden/voxel_water_demo.gd), moved
// in here so the node is usable standalone. Leaving material null keeps the old opt-out
// behavior for a caller that wants to build meshes its own way instead.
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

	// Off by default, matching VoxelViewer's own default: simulating water in the editor (not
	// just Play mode) is opt-in since it's extra background work while editing.
	void set_enabled_in_editor(bool p_enabled);
	bool is_enabled_in_editor() const;

	void set_gravity_mode(GravityMode p_mode);
	GravityMode get_gravity_mode() const;

	void set_planet_center(Vector3 p_center);
	Vector3 get_planet_center() const;

	// Convenience for the common case (this node's job end to end: point it at a terrain, tell
	// it the world is a sphere of this radius, assign a material -- everything else automatic).
	// Setting a radius > 0 also forces gravity_mode to GRAVITY_RADIAL. planet_center still
	// defaults to wherever set_terrain() found the terrain node (its global position at the
	// time set_terrain() was called), not necessarily world origin -- call set_planet_center()
	// afterward to override that if the planet isn't centered on the terrain node itself. 0
	// (the default) leaves gravity_mode/planet_center exactly as already configured -- this is
	// purely a convenience setter, not a mode switch on its own.
	void set_planet_radius(float p_radius);
	float get_planet_radius() const;

	// Material for the water surface meshes this node builds and manages itself (see class
	// comment) -- one MeshInstance3D child per rendered wet block, kept in sync automatically
	// every mesh_refresh_interval seconds. Left null (the default): this node builds no meshes
	// at all, matching its old behavior before this existed (a caller can still do its own
	// mesh-building via generate_water_mesh_*_faces() same as always -- assigning a material
	// here and rolling your own are mutually exclusive, not additive, so leave this null if
	// you're doing the latter).
	void set_material(Ref<Material> p_material);
	Ref<Material> get_material() const;

	// Minimum mass (0..MAX_MASS) for a voxel to be considered "wet" for the built-in mesh
	// rendering above -- same semantics/default as generate_water_mesh_*_faces()'s own
	// p_min_render_mass parameter, which this uses internally.
	void set_min_render_mass(float p_mass);
	float get_min_render_mass() const;

	// Wall-clock seconds between rebuilding LOD0's water meshes (real detail, the most
	// dynamic/responsive tier). Coarser tiers (1..min(max_query_lod, WATER_SIM_MAX_SIM_LOD),
	// see max_query_lod's own doc comment) are a lower-detail progressive-refinement fallback
	// rendered on their own, much slower cadence (mesh_refresh_interval +
	// coarse_mesh_refresh_interval) -- their data boxes can be comparable in block count to
	// LOD0's, so refreshing them every pass like LOD0 is unaffordable (measured: real stalls).
	void set_mesh_refresh_interval(float p_seconds);
	float get_mesh_refresh_interval() const;
	void set_coarse_mesh_refresh_interval(float p_seconds);
	float get_coarse_mesh_refresh_interval() const;

	// How many LOD levels above 0 the read-side queries (is_solid_voxel, get_water_mass,
	// get_wet_block_positions_at_lod, generate_water_mesh_*_faces) are allowed to fall back to
	// when a finer level isn't resident yet. LOD0 data can take a long time to stream in for a
	// fast-moving viewer over a large world (each level is ~8x cheaper to generate and become
	// resident than the one below it, since a block covers 2x the world extent per axis for the
	// same voxel count) -- querying a coarser level that IS already resident gives an
	// approximate-but-immediate answer instead of "nothing here yet", and callers naturally get
	// upgraded to real LOD0 detail the moment it streams in, since every query re-resolves the
	// best available level fresh each time it's called. 0 disables the fallback (LOD0-only,
	// the original behavior).
	void set_max_query_lod(int p_lod);
	int get_max_query_lod() const;

	// Minimum wall-clock seconds between dispatched LOD0 ticks. Default 0.2 - the explicit
	// performance requirement this port exists to satisfy (Eden's version scanned every
	// tick, i.e. every frame) while staying responsive for the level a player actually stands
	// in/on. Coarser levels (1..WATER_SIM_MAX_SIM_LOD) tick on their own slower cadence, see
	// get_lod_tick_interval().
	void set_update_interval(float p_seconds);
	float get_update_interval() const;

	// How many additional wall-clock seconds each LOD level above 0 adds to its own tick
	// interval, on top of update_interval: interval(lod) = update_interval + lod * this. LOD0
	// is real detail a player can be standing in, so it gets the fast update_interval cadence;
	// coarser levels are an approximate fallback for area that hasn't streamed in real detail
	// yet (see max_query_lod), so they don't need to look nearly as responsive, and ticking
	// them less often matters since their data boxes are comparable in block count to LOD0's
	// (see eden_stress_test.gd's own perf notes on this) -- default 1.0 gives interval(0..3) =
	// 0.2, 1.2, 2.2, 3.2s with the default update_interval. Levels above WATER_SIM_MAX_SIM_LOD
	// never tick at all.
	void set_lod_update_interval_step(float p_seconds);
	float get_lod_update_interval_step() const;

	// Wall-clock seconds between dispatched ticks for `p_lod` (0..WATER_SIM_MAX_SIM_LOD):
	// update_interval + p_lod * lod_update_interval_step. Exposed mainly for
	// introspection/tests; the actual per-LOD cadence in set_lod_update_interval_step()'s
	// formula is what _process_time() uses.
	float get_lod_tick_interval(int p_lod) const;

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

	// If true (default), periodically scans every currently resident block at LOD 0..
	// WATER_SIM_MAX_SIM_LOD and activates any whose water channel is non-uniform (i.e. has an
	// actual wet/dry boundary: a shoreline, a partially-filled basin, a spring). Uniform blocks
	// -- no water anywhere, or fully submerged and already at rest -- have provably zero flow
	// work to do and are skipped (an O(1) check per block, see VoxelBuffer::is_uniform()). This
	// is what makes generator-baked water (e.g. EdenPlanetGeneratorV1/V3's baked ocean, see
	// their own generate_block() comments) actually start flowing/rendering with zero
	// scripting: baked water is deliberately inert until something wakes it, and this is that
	// "something", running entirely off Inspector properties instead of requiring a scan loop
	// in a script.
	void set_auto_scan_wet_blocks(bool p_enabled);
	bool get_auto_scan_wet_blocks() const;
	// Wall-clock seconds between auto-scans. Independent of update_interval (the tick
	// dispatch cadence) since scanning and simulating are different costs.
	void set_auto_scan_interval(float p_seconds);
	float get_auto_scan_interval() const;

	// Scans every currently resident LOD0 block of the attached terrain and calls
	// activate_block() on any with a non-uniform water channel. Safe to call manually (e.g.
	// once from a script after loading a save) even with auto_scan_wet_blocks disabled.
	void scan_and_activate_wet_blocks();
	// Same as scan_and_activate_wet_blocks(), but for a specific coarser LOD (1..
	// WATER_SIM_MAX_SIM_LOD) -- called automatically for every simulated level when
	// auto_scan_wet_blocks is on.
	void scan_and_activate_wet_blocks_at_lod(int p_lod);

	// Unlike scan_and_activate_wet_blocks() / get_active_block_positions(), this also reports
	// blocks that are wet but uniform (a whole block of open, at-rest water with no shore or
	// flow boundary inside it) -- those are deliberately never added to the simulator's active
	// set (see scan_and_activate_wet_blocks()'s own comment: they have no flow work to do), but
	// they still need to be *rendered*. Intended for a visualization script to call instead of
	// get_active_block_positions() when it wants to draw the whole ocean, not just the parts
	// currently being simulated.
	TypedArray<Vector3i> get_wet_block_positions(float p_min_render_mass = 0.05f) const;

	// Same as get_wet_block_positions(), but scans a specific LOD level's resident blocks
	// instead of always LOD0. Positions are in that LOD's own block grid, not LOD0's. Intended
	// for progressive-refinement rendering: a caller scans LOD0 first, then increasingly coarse
	// levels up to max_query_lod, using coarser results only to fill in areas LOD0 hasn't
	// streamed in yet (see set_max_query_lod()'s comment).
	TypedArray<Vector3i> get_wet_block_positions_at_lod(int p_lod, float p_min_render_mass = 0.05f) const;

	// Adds (or removes, with a negative amount) water mass at a global voxel coordinate.
	// Applied immediately on the main thread (not batched into the async tick) since this is
	// a rare, caller-driven op, not the per-tick hot path. No-op on a solid voxel or a voxel
	// in an unloaded block. Marks the voxel's block (and any resident face-neighbor block)
	// active.
	void add_water(Vector3i p_global_voxel, float p_amount);
	// get_water_mass()/is_solid_voxel() fall back to progressively coarser LODs (up to
	// max_query_lod) when the LOD0 block covering p_global_voxel isn't resident yet -- see
	// set_max_query_lod(). add_water() does not: writing at a coarser LOD would silently mutate
	// a whole 2^lod-wide region of terrain the caller didn't intend to touch, and edits should
	// wait for real LOD0 data rather than land on an approximation.
	float get_water_mass(Vector3i p_global_voxel) const;
	bool is_solid_voxel(Vector3i p_global_voxel) const;

	// Marks a block active without touching its mass (e.g. for externally-baked water) /
	// pauses simulation for a block without touching its stored mass (e.g. camera-distance
	// culling by caller code). No-op if the block doesn't exist / isn't active respectively.
	// LOD0 shorthand for activate_block_at_lod(p_block_pos, 0) / deactivate_block_at_lod(...).
	void activate_block(Vector3i p_block_pos);
	void deactivate_block(Vector3i p_block_pos);
	// p_lod must be in 0..WATER_SIM_MAX_SIM_LOD; a no-op outside that range (there is nothing
	// to activate -- coarser levels are never simulated).
	void activate_block_at_lod(Vector3i p_block_pos, int p_lod);
	void deactivate_block_at_lod(Vector3i p_block_pos, int p_lod);

	int get_active_block_count() const;
	bool is_block_active(Vector3i p_block_pos) const;
	TypedArray<Vector3i> get_active_block_positions() const;
	int get_active_block_count_at_lod(int p_lod) const;
	bool is_block_active_at_lod(Vector3i p_block_pos, int p_lod) const;
	TypedArray<Vector3i> get_active_block_positions_at_lod(int p_lod) const;

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
	// are in this block's local voxel-index space (0..block_size). `p_block_pos` and the
	// returned vertices are both in `p_lod`'s own grid/space -- the caller is responsible for
	// scaling by (1 << p_lod) and offsetting by p_block_pos * block_size * (1 << p_lod) when
	// placing the resulting mesh in world space, same as any other LOD content in this engine.
	PackedVector3Array generate_water_mesh_faces(
			Vector3i p_block_pos, float p_min_render_mass = 0.05f, int p_lod = 0
	) const;

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
	// space. See generate_water_mesh_faces() above for what `p_lod` means for placement.
	PackedVector3Array generate_water_mesh_smooth_faces(
			Vector3i p_block_pos, float p_min_render_mass = 0.05f, int p_lod = 0
	) const;

protected:
	void _notification(int p_what);

private:
	friend class WaterSimTickTask;

	static void _bind_methods();

	void _process_time(double p_delta);
	void _dispatch_tick(unsigned int p_lod);
	// Called on the main thread by WaterSimTickTask::apply_result(). Not exposed to
	// scripting.
	void _on_tick_completed(const StdVector<WaterSimBlockResult> &p_results, unsigned int p_lod);

	// Builds/updates/prunes this node's own water MeshInstance3D children. No-op if _material
	// is null. See set_material()'s doc comment for the overall design.
	void _refresh_meshes(double p_delta);
	// (Re)generates one block's mesh and creates its MeshInstance3D child on first use. Removes
	// the node instead if the block turned out to have no visible water surface after all.
	void _update_mesh_node(unsigned int lod, Vector3i block_pos, int block_size);
	void _remove_mesh_node(unsigned int lod, Vector3i block_pos);
	// The 8 corners of `block_pos` (at `lod`)'s footprint, in LOD0's block-grid units -- see
	// generate_water_mesh_smooth_faces()'s own doc comment for why block grids nest this way (a
	// block at lod L covers exactly the LOD0-block-grid range [pos << L, (pos << L) + 2^L)).
	// Used to detect when a coarser fallback mesh is already covered by finer rendered data.
	static void _lod0_corners(Vector3i block_pos, unsigned int lod, Vector3i (&out_corners)[8]);

	VoxelLodTerrain *_get_terrain() const;
	std::shared_ptr<VoxelData> _get_data() const;

	// Walks LOD 0..max_query_lod (inclusive) looking for the finest resident block covering
	// `p_global_voxel_lod0` (expressed in LOD0 global voxel coordinates, i.e. the same space
	// every other public method here uses). On success returns true and fills out_lod (the
	// level it found data at), out_buf (that block's voxels) and out_local (p_global_voxel_lod0
	// converted into out_lod's own local block-voxel coordinates). Takes and releases each
	// level's spatial lock itself -- callers must not already hold one for the same block.
	bool _find_resident_lod(
			const std::shared_ptr<VoxelData> &p_data,
			Vector3i p_global_voxel_lod0,
			unsigned int &out_lod,
			std::shared_ptr<VoxelBuffer> &out_buf,
			Vector3i &out_local
	) const;

	ObjectID _terrain_id;
	bool _enabled_in_editor = false;

	GravityMode _gravity_mode = GRAVITY_RADIAL;
	Vector3 _planet_center;
	int _max_query_lod = 0;
	float _update_interval = 0.2f;
	float _lod_update_interval_step = 1.0f;
	int _max_blocks_per_tick = 64;
	int _settle_ticks_threshold = 4;
	float _absorption_rate = 0.0f;
	float _max_absorption_per_voxel = 1.0f;
	PackedInt32Array _absorbing_type_ids;
	bool _auto_scan_wet_blocks = true;
	float _auto_scan_interval = 2.0f;
	float _planet_radius = 0.0f;

	Ref<Material> _material;
	float _min_render_mass = 0.05f;
	float _mesh_refresh_interval = 0.5f;
	float _coarse_mesh_refresh_interval = 2.0f;
	double _mesh_refresh_accum = 0.0;
	double _coarse_mesh_refresh_accum = 0.0;

	static constexpr unsigned int LOD_TIER_COUNT = WATER_SIM_MAX_SIM_LOD + 1;

	// Per-simulated-LOD state (index 0..WATER_SIM_MAX_SIM_LOD). std::atomic_bool has no
	// copy/move constructor, hence a plain array here instead of e.g. StdVector.
	double _scan_time_accumulator[LOD_TIER_COUNT] = {};
	double _time_accumulator[LOD_TIER_COUNT] = {};
	// Explicit per-element {false} initializers (count must match LOD_TIER_COUNT): std::atomic's
	// default constructor doesn't guarantee zero-init before C++20, unlike a plain bool.
	std::atomic_bool _tick_in_flight[LOD_TIER_COUNT] = { false, false, false, false };
	static_assert(LOD_TIER_COUNT == 4, "Update _tick_in_flight's initializer list if this changes");
	uint32_t _round_robin_cursor[LOD_TIER_COUNT] = {};
	StdUnorderedSet<Vector3i> _active_blocks[LOD_TIER_COUNT];
	StdUnorderedMap<Vector3i, int> _settle_counters[LOD_TIER_COUNT];

	// Mesh-rendering state (see set_material()). _coarse_mesh_keys (lod >= 1 only) persists
	// between coarse refresh passes so already-built coarse meshes aren't pruned just because a
	// given pass's throttled scan didn't happen to touch them this time.
	StdUnorderedMap<Vector3i, MeshInstance3D *> _mesh_nodes[LOD_TIER_COUNT];
	StdUnorderedSet<Vector3i> _coarse_mesh_keys[LOD_TIER_COUNT];
};

} // namespace zylann::voxel

VARIANT_ENUM_CAST(zylann::voxel::VoxelWaterSimulator::GravityMode);

#endif // VOXEL_WATER_SIMULATOR_H
