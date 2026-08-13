#ifndef VOXEL_WATER_RECLAIM_H
#define VOXEL_WATER_RECLAIM_H

#include "../storage/voxel_data.h"
#include "../util/math/box3i.h"

namespace zylann::voxel {

// Clears VoxelWaterSimulator::WATER_CHANNEL wherever a terrain edit just turned a voxel solid
// (CHANNEL_SDF <= 0) inside `p_voxel_box_lod0`. Without this, digging into water-filled air or
// building over a wet voxel leaves stale water mass sitting in what's now rock -- a gap
// EdenWaterSimulator's own dig/build tools never closed (see this port's plan doc, step 5).
// Cheap: only iterates the edited box's blocks, and only writes WATER_CHANNEL for voxels that
// are ALREADY solid after the edit (the common edit-boundary case), never a full rescan.
// Caller (VoxelToolLodTerrain::do_sphere[_async]) must call this AFTER the SDF edit has been
// applied, with the same voxel-space box it just edited.
void reclaim_water_in_edited_box(VoxelData &p_data, Box3i p_voxel_box_lod0);

} // namespace zylann::voxel

#endif // VOXEL_WATER_RECLAIM_H
