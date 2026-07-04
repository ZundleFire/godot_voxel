#ifndef VOXEL_GRAPH_MATH_OPS_SIMD_H
#define VOXEL_GRAPH_MATH_OPS_SIMD_H

#include "voxel_graph_runtime.h"

namespace zylann::voxel::pg {

void process_add_buffer(Runtime::ProcessBufferContext &ctx);
void process_subtract_buffer(Runtime::ProcessBufferContext &ctx);
void process_multiply_buffer(Runtime::ProcessBufferContext &ctx);
void process_divide_buffer(Runtime::ProcessBufferContext &ctx);

} // namespace zylann::voxel::pg

#endif // VOXEL_GRAPH_MATH_OPS_SIMD_H
