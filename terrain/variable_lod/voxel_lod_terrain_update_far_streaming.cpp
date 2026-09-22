#include "voxel_lod_terrain_update_far_streaming.h"
#include "../../engine/voxel_engine.h"
#include "../../util/math/funcs.h"
#include "../../util/profiling.h"

namespace zylann::voxel {

using namespace zylann::voxel::far;

namespace {

// The clip radius is what stops the far field drawing ground the near terrain
// already owns. Deriving it from the near system's own view distance is the
// whole point of living in here: as a separate node it was a number you had to
// keep in step by hand, and getting it wrong gave either a ring of missing
// world or a band of double-drawn ground.
//
// The margin pulls it slightly inside the near edge so the two overlap rather
// than leaving a gap; the shader's dithered fade covers the overlap.
float resolve_near_clip_radius(const VoxelLodTerrainUpdateData::Settings &settings) {
	if (settings.far_near_clip_radius != 0.f) {
		// Positive is an explicit radius; negative turns clipping off, which is
		// only useful when the far field is being looked at on its own.
		return math::max(settings.far_near_clip_radius, 0.f);
	}
	constexpr float NEAR_EDGE_MARGIN = 0.94f;
	return static_cast<float>(settings.view_distance_voxels) * NEAR_EDGE_MARGIN;
}

void configure_lod_tree(VoxelLodTerrainUpdateData::FarStreamingState &far, //
		const VoxelLodTerrainUpdateData::Settings &settings) {
	FarLodSettings lod_settings;
	lod_settings.first_lod = settings.far_first_lod;
	lod_settings.lod_count = settings.far_lod_count;
	lod_settings.ring_radius_sectors = settings.far_ring_radius;
	lod_settings.near_clip_radius = resolve_near_clip_radius(settings);

	if (settings.far_planet_radius > 0.f) {
		lod_settings.spherical = true;
		// The planet is centred on the terrain's own origin, and this all runs
		// in the terrain's local space, so the centre is simply zero.
		lod_settings.planet_centre = Vec3f(0.f, 0.f, 0.f);
		lod_settings.sphere_reference_radius =
				settings.far_planet_radius + (settings.far_vertical_min + settings.far_vertical_max) * 0.5f;
		// Keeps `first_lod` meaning what it means on a flat world: cells about
		// 2^first_lod units across.
		lod_settings.sphere_root_lod = choose_sphere_root_lod(settings.far_planet_radius, 1.f);
	}

	if (lod_settings.first_lod == far.configured.first_lod && lod_settings.lod_count == far.configured.lod_count &&
		lod_settings.ring_radius_sectors == far.configured.ring_radius_sectors &&
		lod_settings.near_clip_radius == far.configured.near_clip_radius &&
		lod_settings.spherical == far.configured.spherical &&
		lod_settings.sphere_reference_radius == far.configured.sphere_reference_radius) {
		return;
	}

	far.configured = lod_settings;
	far.lod_tree.configure(lod_settings);
}

std::shared_ptr<FarTaskParams> build_task_params(const VoxelLodTerrainUpdateData::FarStreamingState &far,
		const VoxelLodTerrainUpdateData::Settings &settings, Ref<VoxelGenerator> generator) {
	auto params = std::make_shared<FarTaskParams>();
	params->generator = generator;
	params->vertical_min = settings.far_vertical_min;
	params->vertical_max = settings.far_vertical_max;
	params->antialias = settings.far_antialias;
	params->cache = far.cache;

	params->mesher.generate_bottoms = settings.far_generate_bottoms;
	params->mesher.skirt_depth_cells = settings.far_skirt_depth_cells;

	if (settings.far_planet_radius > 0.f) {
		params->spherical = true;
		params->planet_centre = Vec3f(0.f, 0.f, 0.f);
		// vertical_min/max are heights above the planet radius here, so the
		// terrain lives in a shell rather than a slab.
		params->radial_min = settings.far_planet_radius + settings.far_vertical_min;
		params->radial_max = settings.far_planet_radius + settings.far_vertical_max;
		params->sphere_root_lod = far.configured.sphere_root_lod;
	}

	return params;
}

} // namespace

void process_far_streaming(
		VoxelLodTerrainUpdateData::State &state,
		const VoxelLodTerrainUpdateData::Settings &settings,
		Vector3 viewer_position_in_voxels,
		Ref<VoxelGenerator> generator
) {
	ZN_PROFILE_SCOPE();

	VoxelLodTerrainUpdateData::FarStreamingState &far = state.far_streaming;

	if (!settings.far_enabled || generator.is_null()) {
		return;
	}
	if (far.shared == nullptr) {
		return;
	}

	configure_lod_tree(far, settings);

	const Vec3f viewer(
			static_cast<float>(viewer_position_in_voxels.x),
			static_cast<float>(viewer_position_in_voxels.y),
			static_cast<float>(viewer_position_in_voxels.z)
	);
	far.viewer_position = viewer;

	far.to_load.clear();
	far.to_unload.clear();
	far.lod_tree.update(viewer, far.to_load, far.to_unload);

	// Nothing is done with  here: the main thread reconciles its mesh
	// instances against residency instead, which cannot lose a sector that is
	// dropped and re-required between two of its frames.

	if (far.to_load.empty()) {
		return;
	}

	std::shared_ptr<FarTaskParams> params = build_task_params(far, settings, generator);
	const uint32_t generation = far.shared->generation.load(std::memory_order_relaxed);

	for (const SectorRequest &request : far.to_load) {
		// Sectors hidden under the near terrain are withheld by FarLodTree, not
		// here: skipping them at this point left them marked resident but
		// unbuilt, so they were never offered again once they came back into
		// view.
		FarBuildSectorTask *task =
				ZN_NEW(FarBuildSectorTask(request.id, request.distance, params, far.shared, generation));
		VoxelEngine::get_singleton().push_async_task(task);
	}
}

} // namespace zylann::voxel
