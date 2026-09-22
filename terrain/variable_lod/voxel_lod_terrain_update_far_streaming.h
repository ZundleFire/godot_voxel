#ifndef VOXEL_LOD_TERRAIN_UPDATE_FAR_STREAMING_H
#define VOXEL_LOD_TERRAIN_UPDATE_FAR_STREAMING_H

#include "voxel_lod_terrain_update_data.h"

namespace zylann::voxel {

// Third streaming system, and the only one that does not stream voxels.
//
// Octree and clipbox streaming decide which *blocks* a viewer needs, out to
// `view_distance_voxels`. This one takes over past that edge and decides which
// far-field *sectors* a viewer needs, out to the horizon. It runs in addition
// to whichever near system is selected, not instead of it -- so the two never
// compete, and `far_enabled` can be toggled without touching near streaming.
//
// It shares the update task, the viewer positions and the generator with the
// near systems. Notably it takes its clip radius from `view_distance_voxels`
// rather than a hand-tuned number, which is the one setting that was easy to
// get wrong when the far field lived in its own node.
//
// Runs on the update thread. The main-thread half is `far::FarRenderer`.
void process_far_streaming(
		VoxelLodTerrainUpdateData::State &state,
		const VoxelLodTerrainUpdateData::Settings &settings,
		Vector3 viewer_position_in_voxels,
		Ref<VoxelGenerator> generator
);

} // namespace zylann::voxel

#endif // VOXEL_LOD_TERRAIN_UPDATE_FAR_STREAMING_H
