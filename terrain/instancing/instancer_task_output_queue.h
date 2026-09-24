#ifndef VOXEL_INSTANCER_TASK_OUTPUT_QUEUE_H
#define VOXEL_INSTANCER_TASK_OUTPUT_QUEUE_H

#include "../../util/containers/std_vector.h"
#include "../../util/math/transform3f.h"
#include "../../util/math/vector3i.h"
#include "../../util/thread/mutex.h"
#include <cstdint>

namespace zylann::voxel {

struct InstanceLoadingTaskOutput {
	Vector3i render_block_position;
	// uint16_t like every other layer id (library ids go up to 0xffff): as uint8_t, results for ids above 255 were
	// truncated onto the wrong or a missing layer and silently dropped
	uint16_t layer_id;
	// Tells which parts of the block contain edited data (non-generated).
	// When data chunks are half the size of render chunks, this is 8 bits in XYZ order.
	uint8_t edited_mask;
	StdVector<Transform3f> transforms;
};

struct InstancerTaskOutputQueue {
	StdVector<InstanceLoadingTaskOutput> results;
	Mutex mutex;
};

} // namespace zylann::voxel

#endif // VOXEL_INSTANCER_TASK_OUTPUT_QUEUE_H
