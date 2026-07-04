// ISPC backend for core voxel graph math nodes.
// This is the first native SIMD step for `VoxelGeneratorGraph` buffer execution.

export void VoxelGraphMath_BinaryOp(
		uniform const float * uniform a_data,
		uniform int32 a_is_constant,
		uniform float a_constant,
		uniform const float * uniform b_data,
		uniform int32 b_is_constant,
		uniform float b_constant,
		uniform float * uniform out_data,
		uniform int32 op,
		uniform int32 count) {
	foreach (i = 0 ... count) {
		float a = (a_is_constant != 0) ? a_constant : a_data[i];
		float b = (b_is_constant != 0) ? b_constant : b_data[i];
		float out_v = 0.0f;

		if (op == 0) {
			out_v = a + b;
		} else if (op == 1) {
			out_v = a - b;
		} else if (op == 2) {
			out_v = a * b;
		} else {
			out_v = (b == 0.0f) ? 0.0f : a / b;
		}

		out_data[i] = out_v;
	}
}
