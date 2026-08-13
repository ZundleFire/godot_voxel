#include "voxel_water_reclaim.h"
#include "voxel_water_simulator.h"

#include "../storage/voxel_buffer.h"
#include "../util/math/vector3i.h"

namespace zylann::voxel {

void reclaim_water_in_edited_box(VoxelData &p_data, Box3i p_voxel_box_lod0) {
	if (p_voxel_box_lod0.size.x <= 0 || p_voxel_box_lod0.size.y <= 0 || p_voxel_box_lod0.size.z <= 0) {
		return;
	}

	const int block_size = int(p_data.get_block_size());
	const Box3i block_box = Box3i::from_min_max(
			math::floordiv(p_voxel_box_lod0.position, block_size),
			math::floordiv(p_voxel_box_lod0.position + p_voxel_box_lod0.size - Vector3i(1, 1, 1), block_size) +
					Vector3i(1, 1, 1)
	);

	SpatialLock3D::Write wlock(p_data.get_spatial_lock(0), BoxBounds3i(block_box));

	block_box.for_each_cell([&p_data, &p_voxel_box_lod0, block_size](Vector3i block_pos) {
		std::shared_ptr<VoxelBuffer> buf = p_data.try_get_block_voxels(block_pos);
		if (buf == nullptr) {
			return;
		}
		const Vector3i buf_size = buf->get_size();
		for (int z = 0; z < buf_size.z; ++z) {
			for (int y = 0; y < buf_size.y; ++y) {
				for (int x = 0; x < buf_size.x; ++x) {
					const Vector3i global = block_pos * block_size + Vector3i(x, y, z);
					if (!p_voxel_box_lod0.contains(global)) {
						continue;
					}
					if (buf->get_voxel_f(x, y, z, VoxelBuffer::CHANNEL_SDF) > 0.0f) {
						// Still air (or newly-dug-open) -- water there is legitimate, leave it.
						continue;
					}
					if (buf->get_voxel_f(x, y, z, VoxelWaterSimulator::WATER_CHANNEL) >
							VoxelWaterSimulator::MIN_MASS) {
						buf->set_voxel_f(0.0f, x, y, z, VoxelWaterSimulator::WATER_CHANNEL);
					}
				}
			}
		}
	});
}

} // namespace zylann::voxel
