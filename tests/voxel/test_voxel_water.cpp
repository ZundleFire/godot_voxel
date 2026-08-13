#include "test_voxel_water.h"

#include "../../storage/voxel_data.h"
#include "../../terrain/variable_lod/voxel_lod_terrain.h"
#include "../../util/math/funcs.h"
#include "../../util/memory/memory.h"
#include "../../util/tasks/threaded_task.h"
#include "../../util/testing/test_macros.h"
#include "../../water/voxel_water_simulator.h"
#include "../../water/water_sim_tick_task.h"

namespace zylann::voxel::tests {

namespace {

// Populates `p_data` (assumed empty) with a solid floor at global y <= 0 and open air above,
// over every block position in `p_block_box`. CHANNEL_SDF and the water channel are both set
// to 32-bit float depth so test assertions aren't muddied by quantization noise (production
// terrains default to 16-bit for the water channel, coarser but still fine for gameplay).
// Populates in place (rather than building and returning a fresh VoxelData) since VoxelData
// holds per-LOD RWLock/SpatialLock3D members and isn't copyable or movable.
void populate_floor_test_data(VoxelData &p_data, Box3i p_block_box) {
	const int bs = int(p_data.get_block_size());
	p_block_box.for_each_cell_zxy([&p_data, bs](Vector3i block_pos) {
		std::shared_ptr<VoxelBuffer> buffer = make_shared_instance<VoxelBuffer>(VoxelBuffer::ALLOCATOR_DEFAULT);
		buffer->create(Vector3iUtil::create(bs));
		buffer->set_channel_depth(VoxelBuffer::CHANNEL_SDF, VoxelBuffer::DEPTH_32_BIT);
		buffer->set_channel_depth(VoxelWaterSimulator::WATER_CHANNEL, VoxelBuffer::DEPTH_32_BIT);
		for (int z = 0; z < bs; ++z) {
			for (int y = 0; y < bs; ++y) {
				for (int x = 0; x < bs; ++x) {
					const int global_y = block_pos.y * bs + y;
					buffer->set_voxel_f(global_y <= 0 ? -1.0f : 1.0f, x, y, z, VoxelBuffer::CHANNEL_SDF);
				}
			}
		}
		VoxelDataBlock block(buffer, 0);
		block.set_edited(true);
		ZN_TEST_ASSERT(p_data.try_set_block(block_pos, block));
	});
}

float get_mass(VoxelData &p_data, Vector3i p_global_pos) {
	const int bs = int(p_data.get_block_size());
	const Vector3i bp = math::floordiv(p_global_pos, bs);
	std::shared_ptr<VoxelBuffer> buf = p_data.try_get_block_voxels(bp);
	if (buf == nullptr) {
		return 0.0f;
	}
	const Vector3i local = p_global_pos - bp * bs;
	return buf->get_voxel_f(local.x, local.y, local.z, VoxelWaterSimulator::WATER_CHANNEL);
}

void set_mass(VoxelData &p_data, Vector3i p_global_pos, float p_mass) {
	const int bs = int(p_data.get_block_size());
	const Vector3i bp = math::floordiv(p_global_pos, bs);
	std::shared_ptr<VoxelBuffer> buf = p_data.try_get_block_voxels(bp);
	ZN_TEST_ASSERT(buf != nullptr);
	const Vector3i local = p_global_pos - bp * bs;
	buf->set_voxel_f(p_mass, local.x, local.y, local.z, VoxelWaterSimulator::WATER_CHANNEL);
}

float total_mass(VoxelData &p_data, Box3i p_block_box) {
	float total = 0.0f;
	const int bs = int(p_data.get_block_size());
	p_block_box.for_each_cell_zxy([&](Vector3i bp) {
		std::shared_ptr<VoxelBuffer> buf = p_data.try_get_block_voxels(bp);
		if (buf == nullptr) {
			return;
		}
		for (int z = 0; z < bs; ++z) {
			for (int y = 0; y < bs; ++y) {
				for (int x = 0; x < bs; ++x) {
					total += buf->get_voxel_f(x, y, z, VoxelWaterSimulator::WATER_CHANNEL);
				}
			}
		}
	});
	return total;
}

// Runs one WaterSimTickTask::run()+apply_result() pass synchronously, exercising the exact
// production split (worker-thread compute, main-thread commit) just invoked inline for
// deterministic tests -- mirrors VoxelLodTerrain's own synchronous fallback pattern for
// IThreadedTask (see voxel_lod_terrain.cpp's `_threaded_update_enabled == false` branch).
void run_one_tick(
		std::shared_ptr<VoxelData> p_data,
		VoxelWaterSimulator *p_sim,
		const StdVector<Vector3i> &p_blocks,
		VoxelWaterSimulator::GravityMode p_gravity_mode = VoxelWaterSimulator::GRAVITY_Y_DOWN,
		Vector3 p_planet_center = Vector3()
) {
	WaterSimTickTask task(
			p_sim->get_instance_id(),
			p_data,
			p_data->get_block_size(),
			p_blocks,
			p_gravity_mode,
			p_planet_center,
			0.0f,
			1.0f,
			StdVector<uint64_t>()
	);
	ThreadedTaskContext ctx(0, TaskPriority());
	task.run(ctx);
	task.apply_result();
}

} // namespace

void test_voxel_water_mass_conservation() {
	const Box3i block_box(Vector3i(-1, -1, -1), Vector3i(3, 3, 3));
	std::shared_ptr<VoxelData> data = make_shared_instance<VoxelData>();
	populate_floor_test_data(*data, block_box);

	VoxelWaterSimulator *sim = memnew(VoxelWaterSimulator);

	// A column of water two voxels above an open floor.
	const Vector3i source(0, 2, 0);
	set_mass(*data, source, 1.0f);

	// Adding water on a solid voxel must never be possible via direct write in this test
	// helper's own invariant: the floor stays dry throughout.
	ZN_TEST_ASSERT(get_mass(*data, Vector3i(0, 0, 0)) == 0.0f);

	const float initial_total = total_mass(*data, block_box);
	ZN_TEST_ASSERT(initial_total > 0.0f);

	StdVector<Vector3i> active_blocks;
	block_box.for_each_cell_zxy([&active_blocks](Vector3i bp) { active_blocks.push_back(bp); });

	// 15 ticks: enough for the water to fall the 1 voxel onto the floor (MAX_SPEED=0.8/tick),
	// not so many that lateral spreading across the whole open floor thins the source spot
	// below any threshold worth asserting (that dispersal is real, expected behavior -- an
	// open floor has no walls to pool against -- and is exercised at a contained, walled scale
	// by the settle-dormancy test below instead).
	for (int i = 0; i < 15; ++i) {
		run_one_tick(data, sim, active_blocks);
	}

	const float final_total = total_mass(*data, block_box);
	// Only float rounding is allowed to move the total over many ticks -- see
	// WaterSimTickTask::apply_result()'s own comment on why individual deltas are never
	// skipped for looking small (that used to be a real, if tiny-looking-per-tick, leak).
	ZN_TEST_ASSERT(Math::abs(final_total - initial_total) < 0.01f);

	// Water must have actually moved off its exact source voxel and reached the floor.
	ZN_TEST_ASSERT(get_mass(*data, source) < 0.9f);
	ZN_TEST_ASSERT(get_mass(*data, Vector3i(0, 1, 0)) > 0.01f);

	memdelete(sim);
}

void test_voxel_water_flows_through_cave() {
	// A one-cell-tall solid-roofed tunnel connecting two open columns, several voxels apart --
	// this is the architecturally load-bearing test: it proves the sparse 3D CA can carry water
	// through a cave/overhang, which a heightfield-only water model fundamentally cannot do
	// (see VoxelWaterSimulator's own class doc comment on why a 3D CA was chosen at all).
	const Box3i block_box(Vector3i(-1, -1, -1), Vector3i(3, 3, 3));
	std::shared_ptr<VoxelData> data = make_shared_instance<VoxelData>();
	populate_floor_test_data(*data, block_box);

	const int tunnel_y = 3;
	const int tunnel_len = 6;
	// Carve the tunnel: open at tunnel_y across x in [0, tunnel_len], with its OWN solid floor
	// at tunnel_y-1 (without this, the tunnel is just open air sitting above the test's real
	// floor at y=0, and water would simply fall straight through to y=1 instead of ever
	// traveling horizontally through the tunnel) and solid roof at tunnel_y+1.
	auto set_sdf = [&](Vector3i g, float sdf) {
		const int bs = int(data->get_block_size());
		const Vector3i bp = math::floordiv(g, bs);
		std::shared_ptr<VoxelBuffer> buf = data->try_get_block_voxels(bp);
		ZN_TEST_ASSERT(buf != nullptr);
		const Vector3i local = g - bp * bs;
		buf->set_voxel_f(sdf, local.x, local.y, local.z, VoxelBuffer::CHANNEL_SDF);
	};
	for (int x = 0; x <= tunnel_len; ++x) {
		set_sdf(Vector3i(x, tunnel_y, 0), 1.0f); // air (open tunnel)
		set_sdf(Vector3i(x, tunnel_y - 1, 0), -1.0f); // solid floor under the tunnel
		set_sdf(Vector3i(x, tunnel_y + 1, 0), -1.0f); // solid roof over the tunnel
		// Seal both side walls too (z=+-1, all 3 rows) so this is a fully enclosed tube open
		// only at its two x-ends -- otherwise mass continuously leaks sideways into the vast
		// open floor beyond and never returns, masking whether it actually traveled the
		// tunnel's length (it does -- confirmed via debug trace -- the leak just also drains
		// the tunnel itself almost completely over hundreds of ticks).
		for (int dz = -1; dz <= 1; dz += 2) {
			for (int y = tunnel_y - 1; y <= tunnel_y + 1; ++y) {
				set_sdf(Vector3i(x, y, dz), -1.0f);
			}
		}
	}
	// Cap both x-ends solid too: otherwise mass draining past either end falls into the vast
	// open floor beyond and disperses across it, never returning -- after hundreds of ticks
	// that leak alone drains the tunnel almost completely, masking whether the CA actually
	// carried water through it. Capping both ends makes this a closed system: any mass this
	// tunnel-shaped path can't hold must have failed to traverse it, which is exactly the
	// property this test needs to check.
	for (int x : { -1, tunnel_len + 1 }) {
		for (int z = -1; z <= 1; ++z) {
			for (int y = tunnel_y - 1; y <= tunnel_y + 1; ++y) {
				set_sdf(Vector3i(x, y, z), -1.0f);
			}
		}
	}

	VoxelWaterSimulator *sim = memnew(VoxelWaterSimulator);

	set_mass(*data, Vector3i(0, tunnel_y, 0), 1.0f);
	const float initial_total = total_mass(*data, block_box);

	StdVector<Vector3i> active_blocks;
	block_box.for_each_cell_zxy([&active_blocks](Vector3i bp) { active_blocks.push_back(bp); });

	for (int i = 0; i < 600; ++i) {
		run_one_tick(data, sim, active_blocks);
	}

	// Mass reached the far end of the tunnel, under a solid roof the whole way.
	ZN_TEST_ASSERT(get_mass(*data, Vector3i(tunnel_len, tunnel_y, 0)) > 0.05f);
	// Conservation still holds despite the longer run.
	const float final_total = total_mass(*data, block_box);
	ZN_TEST_ASSERT(Math::abs(final_total - initial_total) < 0.2f);
	// The tunnel's roof voxels never accumulate mass (solid voxels are always excluded).
	for (int x = 0; x <= tunnel_len; ++x) {
		ZN_TEST_ASSERT(get_mass(*data, Vector3i(x, tunnel_y + 1, 0)) == 0.0f);
	}

	memdelete(sim);
}

void test_voxel_water_gravity_radial() {
	// planet_center placed far along -X: water seeded in a corridor running along X should
	// migrate toward low X (toward planet_center), not spread symmetrically the way
	// GRAVITY_Y_DOWN would (there, x=-1 and x=+1 are equivalent since gravity has no X
	// component at all) -- proving gravity is derived per-voxel from world position rather
	// than hardcoded to world "down".
	const Box3i block_box(Vector3i(-1, -1, -1), Vector3i(3, 3, 3));
	std::shared_ptr<VoxelData> data = make_shared_instance<VoxelData>();
	populate_floor_test_data(*data, block_box);

	// A corridor running along X at y=2, z=0, walled on every other side (y=1/y=3, z=-1/z=1,
	// and both x-ends) so the only direction water can move is along X -- otherwise, with
	// "down" here being -X rather than -Y, the ordinary Y/Z directions all count as
	// "sideways" flow targets and (being wide open by default, see populate_floor_test_data)
	// let mass disperse through the whole 3D region before this test's small tick budget can
	// show any X-direction asymmetry (see test_voxel_water_flows_through_cave's identical
	// lesson, hit first).
	auto set_sdf = [&](Vector3i g, float sdf) {
		const int bs = int(data->get_block_size());
		const Vector3i bp = math::floordiv(g, bs);
		std::shared_ptr<VoxelBuffer> buf = data->try_get_block_voxels(bp);
		ZN_TEST_ASSERT(buf != nullptr);
		const Vector3i local = g - bp * bs;
		buf->set_voxel_f(sdf, local.x, local.y, local.z, VoxelBuffer::CHANNEL_SDF);
	};
	const int corridor_half = 4;
	for (int x = -corridor_half; x <= corridor_half; ++x) {
		set_sdf(Vector3i(x, 2, 0), 1.0f); // air (open corridor)
		for (int dz = -1; dz <= 1; dz += 2) {
			set_sdf(Vector3i(x, 2, dz), -1.0f);
		}
		for (int dy = 1; dy <= 3; dy += 2) {
			set_sdf(Vector3i(x, dy, 0), -1.0f);
		}
	}
	for (int x : { -corridor_half - 1, corridor_half + 1 }) {
		for (int z = -1; z <= 1; ++z) {
			for (int y = 1; y <= 3; ++y) {
				set_sdf(Vector3i(x, y, z), -1.0f);
			}
		}
	}

	VoxelWaterSimulator *sim = memnew(VoxelWaterSimulator);

	const Vector3i source(0, 2, 0);
	set_mass(*data, source, 1.0f);

	const Vector3 planet_center(-1000.0f, 2.5f, 0.0f);

	StdVector<Vector3i> active_blocks;
	block_box.for_each_cell_zxy([&active_blocks](Vector3i bp) { active_blocks.push_back(bp); });

	for (int i = 0; i < 100; ++i) {
		run_one_tick(data, sim, active_blocks, VoxelWaterSimulator::GRAVITY_RADIAL, planet_center);
	}

	// Mass should have migrated toward -X (toward planet_center, where this corridor's wall
	// now acts as "the floor"), not stayed at the source or spread symmetrically the way
	// GRAVITY_Y_DOWN would (there, +-X are equivalent since gravity has no X component at
	// all). planet_center is placed extremely far away, so the pull is effectively constant
	// and strong across this whole corridor -- mass free-falls to the -X wall almost
	// immediately (well within 100 ticks) rather than settling gradually, so the correct
	// check is "piled up against the -X wall", not "still mid-transit at x=-+1".
	ZN_TEST_ASSERT(get_mass(*data, Vector3i(-corridor_half, 2, 0)) > get_mass(*data, Vector3i(corridor_half, 2, 0)));
	ZN_TEST_ASSERT(get_mass(*data, Vector3i(-corridor_half, 2, 0)) > 0.5f);

	memdelete(sim);
}

void test_voxel_water_settle_dormancy() {
	const Box3i block_box(Vector3i(-1, -1, -1), Vector3i(3, 3, 3));
	std::shared_ptr<VoxelData> data = make_shared_instance<VoxelData>();
	populate_floor_test_data(*data, block_box);

	VoxelWaterSimulator *sim = memnew(VoxelWaterSimulator);
	sim->set_settle_ticks_threshold(2);

	// A tiny puddle in a single fully walled-in pit (floor below already solid; wall every
	// side too): with nowhere to go, it reaches true equilibrium (zero flow) on the very
	// first tick. Without side walls, the exact same puddle on an unconfined open floor
	// keeps diffusing sideways indefinitely instead -- always producing *some* nonzero flow
	// each tick and never actually settling, which tests the wrong thing (this class's own
	// settle/dormancy bookkeeping, not "does water ever stop moving on an infinite plane").
	{
		const int bs = int(data->get_block_size());
		auto set_sdf = [&](Vector3i g, float sdf) {
			const Vector3i bp = math::floordiv(g, bs);
			std::shared_ptr<VoxelBuffer> buf = data->try_get_block_voxels(bp);
			ZN_TEST_ASSERT(buf != nullptr);
			const Vector3i local = g - bp * bs;
			buf->set_voxel_f(sdf, local.x, local.y, local.z, VoxelBuffer::CHANNEL_SDF);
		};
		set_sdf(Vector3i(1, 1, 0), -1.0f);
		set_sdf(Vector3i(-1, 1, 0), -1.0f);
		set_sdf(Vector3i(0, 1, 1), -1.0f);
		set_sdf(Vector3i(0, 1, -1), -1.0f);
	}
	set_mass(*data, Vector3i(0, 1, 0), 0.01f);

	StdVector<Vector3i> active_blocks;
	active_blocks.push_back(Vector3i(0, 0, 0));

	// Drive enough ticks for it to settle and drop out of the active set via
	// VoxelWaterSimulator::_on_tick_completed (exercised through apply_result() each tick).
	for (int i = 0; i < 10; ++i) {
		run_one_tick(data, sim, active_blocks);
	}

	ZN_TEST_ASSERT(sim->get_active_block_count() == 0);
	ZN_TEST_ASSERT(!sim->is_block_active(Vector3i(0, 0, 0)));

	memdelete(sim);
}

namespace {

// generate_water_mesh_faces()/generate_water_mesh_smooth_faces() both read through
// _get_data(), which requires a live terrain attached (see VoxelWaterSimulator::set_terrain())
// -- unlike the tick-driving tests above (which construct a WaterSimTickTask directly and so
// never need one), these two do need a real VoxelLodTerrain. It's never added to a SceneTree
// or given a generator/mesher -- its constructor is explicitly documented as safe to
// instantiate standalone (see its own "don't do anything heavy in the constructor" comment,
// and test_voxel_graph.cpp's identical memnew/memdelete pattern) -- only its VoxelData storage
// is used directly, bypassing streaming/generation entirely.
struct MeshTestFixture {
	VoxelLodTerrain *terrain;
	VoxelWaterSimulator *sim;

	MeshTestFixture() {
		terrain = memnew(VoxelLodTerrain);
		sim = memnew(VoxelWaterSimulator);
		sim->set_terrain(terrain);
		// VoxelWaterSimulator defaults to GRAVITY_RADIAL with planet_center=(0,0,0) -- fine
		// for a real planet, but generate_water_mesh_smooth_faces() picks its "up" axis from
		// this, and these mesh tests build simple flat-Y-up test geometry, so pin it to
		// GRAVITY_Y_DOWN explicitly rather than relying on defaults meant for a different use.
		sim->set_gravity_mode(VoxelWaterSimulator::GRAVITY_Y_DOWN);
	}

	~MeshTestFixture() {
		memdelete(sim);
		memdelete(terrain);
	}
};

} // namespace

void test_voxel_water_mesh_faces_flat_pool() {
	MeshTestFixture fx;

	std::shared_ptr<VoxelData> data = fx.terrain->get_storage_shared();
	populate_floor_test_data(*data, Box3i(Vector3i(0, 0, 0), Vector3i(1, 1, 1)));

	const int bs = int(data->get_block_size());
	std::shared_ptr<VoxelBuffer> buf = data->try_get_block_voxels(Vector3i(0, 0, 0));
	ZN_TEST_ASSERT(buf != nullptr);
	// Flat, fully-settled pool one voxel deep, everywhere in this block, resting on the floor.
	for (int z = 0; z < bs; ++z) {
		for (int x = 0; x < bs; ++x) {
			buf->set_voxel_f(1.0f, x, 1, z, VoxelWaterSimulator::WATER_CHANNEL);
		}
	}

	const PackedVector3Array faces = fx.sim->generate_water_mesh_faces(Vector3i(0, 0, 0));

	// A full-block flat pool with no loaded neighbor blocks: the bottom face is suppressed by
	// the solid floor, the 4 side faces are suppressed because the (unloaded) neighbor blocks
	// are treated as solid -- only the top face remains, and it must greedy-merge into exactly
	// one quad (2 triangles = 6 vertices), not one quad per voxel.
	ZN_TEST_ASSERT(faces.size() == 6);
	for (int i = 0; i < faces.size(); ++i) {
		ZN_TEST_ASSERT(Math::is_equal_approx(faces[i].y, 2.0f));
	}
}

void test_voxel_water_mesh_smooth_faces_flat_pool() {
	MeshTestFixture fx;

	std::shared_ptr<VoxelData> data = fx.terrain->get_storage_shared();
	populate_floor_test_data(*data, Box3i(Vector3i(0, 0, 0), Vector3i(1, 1, 1)));

	const int bs = int(data->get_block_size());
	std::shared_ptr<VoxelBuffer> buf = data->try_get_block_voxels(Vector3i(0, 0, 0));
	ZN_TEST_ASSERT(buf != nullptr);
	for (int z = 0; z < bs; ++z) {
		for (int x = 0; x < bs; ++x) {
			buf->set_voxel_f(1.0f, x, 1, z, VoxelWaterSimulator::WATER_CHANNEL);
		}
	}

	const PackedVector3Array faces = fx.sim->generate_water_mesh_smooth_faces(Vector3i(0, 0, 0));

	// (bs-1) x (bs-1) cells, 6 vertices (2 triangles) each -- not the full bs x bs: the last
	// row/column of corner columns (grid index bs) samples one step into the +u/+v neighbor
	// block, which doesn't exist in this single-block test, so those columns come back NaN
	// ("no data" is correctly NOT the same as "dry") and every cell touching them is skipped.
	// A real multi-block terrain has no such gap (see this method's own doc comment on
	// cross-block continuity).
	ZN_TEST_ASSERT(faces.size() == 6 * (bs - 1) * (bs - 1));
	// Every vertex lands at exactly the same height -- the whole point of the heightfield
	// rewrite (no per-voxel ripple traced from the raw CA mass field, unlike the blocky
	// variant, which is deliberately exact-but-blocky instead).
	for (int i = 0; i < faces.size(); ++i) {
		ZN_TEST_ASSERT(Math::is_equal_approx(faces[i].y, 2.0f));
	}
}

} // namespace zylann::voxel::tests
