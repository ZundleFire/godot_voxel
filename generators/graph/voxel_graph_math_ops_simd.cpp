#include "voxel_graph_math_ops_simd.h"

#ifdef VOXEL_ISPC_ENABLED
#include "voxel_graph_math_ops_ispc_ispc.h"
#endif

namespace zylann::voxel::pg {
namespace {

enum MathOp {
	MATH_OP_ADD = 0,
	MATH_OP_SUBTRACT = 1,
	MATH_OP_MULTIPLY = 2,
	MATH_OP_DIVIDE = 3
};

template <typename F>
inline void process_scalar_binop(const Runtime::Buffer &a, const Runtime::Buffer &b, Runtime::Buffer &out, F op) {
	const uint32_t buffer_size = out.size;

	if (a.is_constant || b.is_constant) {
		if (!b.is_constant) {
			const float c = a.constant_value;
			const float *v = b.data;
			for (uint32_t i = 0; i < buffer_size; ++i) {
				out.data[i] = op(c, v[i]);
			}

		} else if (!a.is_constant) {
			const float c = b.constant_value;
			const float *v = a.data;
			for (uint32_t i = 0; i < buffer_size; ++i) {
				out.data[i] = op(v[i], c);
			}

		} else {
			const float c = op(a.constant_value, b.constant_value);
			for (uint32_t i = 0; i < buffer_size; ++i) {
				out.data[i] = c;
			}
		}

	} else {
		for (uint32_t i = 0; i < buffer_size; ++i) {
			out.data[i] = op(a.data[i], b.data[i]);
		}
	}
}

inline void process_scalar_division(const Runtime::Buffer &a, const Runtime::Buffer &b, Runtime::Buffer &out) {
	const uint32_t buffer_size = out.size;

	if (a.is_constant || b.is_constant) {
		if (!b.is_constant) {
			const float c = a.constant_value;
			const float *v = b.data;
			for (uint32_t i = 0; i < buffer_size; ++i) {
				out.data[i] = v[i] == 0.f ? 0.f : c / v[i];
			}

		} else if (!a.is_constant) {
			if (b.constant_value == 0.f) {
				for (uint32_t i = 0; i < buffer_size; ++i) {
					out.data[i] = 0.f;
				}
			} else {
				const float c = 1.f / b.constant_value;
				const float *v = a.data;
				for (uint32_t i = 0; i < buffer_size; ++i) {
					out.data[i] = v[i] * c;
				}
			}

		} else {
			const float v = b.constant_value == 0.f ? 0.f : a.constant_value / b.constant_value;
			for (uint32_t i = 0; i < buffer_size; ++i) {
				out.data[i] = v;
			}
		}

	} else {
		for (uint32_t i = 0; i < buffer_size; ++i) {
			out.data[i] = b.data[i] == 0.f ? 0.f : a.data[i] / b.data[i];
		}
	}
}

inline void process_ispc_binop(const Runtime::Buffer &a, const Runtime::Buffer &b, Runtime::Buffer &out, int32_t op) {
#ifdef VOXEL_ISPC_ENABLED
	ispc::VoxelGraphMath_BinaryOp(
			a.data,
			a.is_constant ? 1 : 0,
			a.constant_value,
			b.data,
			b.is_constant ? 1 : 0,
			b.constant_value,
			out.data,
			op,
			static_cast<int32_t>(out.size));
#else
		(void)a;
		(void)b;
		(void)out;
		(void)op;
#endif
}

} // namespace

void process_add_buffer(Runtime::ProcessBufferContext &ctx) {
	const Runtime::Buffer &a = ctx.get_input(0);
	const Runtime::Buffer &b = ctx.get_input(1);
	Runtime::Buffer &out = ctx.get_output(0);
#ifdef VOXEL_ISPC_ENABLED
	process_ispc_binop(a, b, out, MATH_OP_ADD);
#else
	process_scalar_binop(a, b, out, [](float x, float y) { return x + y; });
#endif
}

void process_subtract_buffer(Runtime::ProcessBufferContext &ctx) {
	const Runtime::Buffer &a = ctx.get_input(0);
	const Runtime::Buffer &b = ctx.get_input(1);
	Runtime::Buffer &out = ctx.get_output(0);
#ifdef VOXEL_ISPC_ENABLED
	process_ispc_binop(a, b, out, MATH_OP_SUBTRACT);
#else
	process_scalar_binop(a, b, out, [](float x, float y) { return x - y; });
#endif
}

void process_multiply_buffer(Runtime::ProcessBufferContext &ctx) {
	const Runtime::Buffer &a = ctx.get_input(0);
	const Runtime::Buffer &b = ctx.get_input(1);
	Runtime::Buffer &out = ctx.get_output(0);
#ifdef VOXEL_ISPC_ENABLED
	process_ispc_binop(a, b, out, MATH_OP_MULTIPLY);
#else
	process_scalar_binop(a, b, out, [](float x, float y) { return x * y; });
#endif
}

void process_divide_buffer(Runtime::ProcessBufferContext &ctx) {
	const Runtime::Buffer &a = ctx.get_input(0);
	const Runtime::Buffer &b = ctx.get_input(1);
	Runtime::Buffer &out = ctx.get_output(0);
#ifdef VOXEL_ISPC_ENABLED
	process_ispc_binop(a, b, out, MATH_OP_DIVIDE);
#else
	process_scalar_division(a, b, out);
#endif
}

} // namespace zylann::voxel::pg
