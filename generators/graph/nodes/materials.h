#ifndef VOXEL_GRAPH_NODES_MATERIALS_H
#define VOXEL_GRAPH_NODES_MATERIALS_H

#include "../../../util/godot/classes/standard_material_3d.h"
#include "../node_type_db.h"

namespace zylann::voxel::pg {

namespace {

template <typename T>
Variant create_material_ref_to_variant() {
	Ref<T> res;
	res.instantiate();
	return Variant(res);
}

inline float get_safe_falloff(float falloff) {
	return MAX(Math::abs(falloff), 0.0001f);
}

inline void sort_min_max(float a, float b, float &out_min, float &out_max) {
	if (a <= b) {
		out_min = a;
		out_max = b;
	} else {
		out_min = b;
		out_max = a;
	}
}

inline void compute_height_split(
		float value,
		float height,
		float falloff,
		float reference,
		float &out_above,
		float &out_below
) {
	const float threshold = reference + height;
	const float distance = value - threshold;
	const float f = get_safe_falloff(falloff);
	out_above = math::smoothstep(0.f, f, distance);
	out_below = 1.f - out_above;
}

} // namespace

void register_material_nodes(Span<NodeType> types) {
	using namespace math;

	{
		NodeType &t = types[VoxelGraphFunction::NODE_HEIGHT_SPLITTER];
		t.name = "HeightSplitter";
		t.category = CATEGORY_MASKS;
		t.inputs.push_back(NodeType::Port("value"));
		t.inputs.push_back(NodeType::Port("height_offset", 0.f));
		t.inputs.push_back(NodeType::Port("falloff", 1.f));
		t.inputs.push_back(NodeType::Port("reference", 0.f));
		t.outputs.push_back(NodeType::Port("above"));
		t.outputs.push_back(NodeType::Port("below"));
		t.process_buffer_func = [](Runtime::ProcessBufferContext &ctx) {
			const Runtime::Buffer &value = ctx.get_input(0);
			const Runtime::Buffer &height = ctx.get_input(1);
			const Runtime::Buffer &falloff = ctx.get_input(2);
			const Runtime::Buffer &reference = ctx.get_input(3);
			Runtime::Buffer &above = ctx.get_output(0);
			Runtime::Buffer &below = ctx.get_output(1);

			for (uint32_t i = 0; i < above.size; ++i) {
				compute_height_split(
						value.data[i],
						height.data[i],
						falloff.data[i],
						reference.data[i],
						above.data[i],
						below.data[i]);
			}
		};
		t.range_analysis_func = [](Runtime::RangeAnalysisContext &ctx) {
			ctx.set_output(0, Interval(0.f, 1.f));
			ctx.set_output(1, Interval(0.f, 1.f));
		};
#ifdef VOXEL_ENABLE_GPU
		t.shader_gen_func = [](ShaderGenContext &ctx) {
			ctx.add_format(
					"float {}_f = max(abs({}), 0.0001);\n"
					"float {}_threshold = {} + {};\n"
					"float {}_distance = {} - {}_threshold;\n"
					"{} = smoothstep(0.0, {}_f, {}_distance);\n"
					"{} = 1.0 - {};\n",
					ctx.get_output_name(0),
					ctx.get_input_name(2),
					ctx.get_output_name(0),
					ctx.get_input_name(3),
					ctx.get_input_name(1),
					ctx.get_output_name(0),
					ctx.get_input_name(0),
					ctx.get_output_name(0),
					ctx.get_output_name(0),
					ctx.get_output_name(0),
					ctx.get_output_name(0),
					ctx.get_output_name(1),
					ctx.get_output_name(0));
		};
#endif
	}

	{
		NodeType &t = types[VoxelGraphFunction::NODE_SLOPE_MASK];
		t.name = "SlopeMask";
		t.category = CATEGORY_MASKS;
		t.inputs.push_back(NodeType::Port("up_dot", 1.f));
		t.inputs.push_back(NodeType::Port("min_degrees", 20.f));
		t.inputs.push_back(NodeType::Port("max_degrees", 50.f));
		t.outputs.push_back(NodeType::Port("mask"));
		t.outputs.push_back(NodeType::Port("flat"));
		t.outputs.push_back(NodeType::Port("slope"));
		t.process_buffer_func = [](Runtime::ProcessBufferContext &ctx) {
			const Runtime::Buffer &up_dot = ctx.get_input(0);
			const Runtime::Buffer &min_degrees = ctx.get_input(1);
			const Runtime::Buffer &max_degrees = ctx.get_input(2);
			Runtime::Buffer &mask = ctx.get_output(0);
			Runtime::Buffer &flat = ctx.get_output(1);
			Runtime::Buffer &slope = ctx.get_output(2);

			for (uint32_t i = 0; i < mask.size; ++i) {
				float min_deg;
				float max_deg;
				sort_min_max(min_degrees.data[i], max_degrees.data[i], min_deg, max_deg);
				const float clamped_up = clamp(up_dot.data[i], -1.f, 1.f);
				const float degrees = Math::rad_to_deg(Math::acos(clamped_up));
				const float steep = smoothstep(min_deg, max_deg, degrees);
				mask.data[i] = steep;
				flat.data[i] = 1.f - steep;
				slope.data[i] = clamp((degrees - min_deg) / MAX(max_deg - min_deg, 0.0001f), 0.f, 1.f);
			}
		};
		t.range_analysis_func = [](Runtime::RangeAnalysisContext &ctx) {
			ctx.set_output(0, Interval(0.f, 1.f));
			ctx.set_output(1, Interval(0.f, 1.f));
			ctx.set_output(2, Interval(0.f, 1.f));
		};
#ifdef VOXEL_ENABLE_GPU
		t.shader_gen_func = [](ShaderGenContext &ctx) {
			ctx.add_format(
					"float {}_min = min({}, {});\n"
					"float {}_max = max({}, {});\n"
					"float {}_degrees = degrees(acos(clamp({}, -1.0, 1.0)));\n"
					"{} = smoothstep({}_min, {}_max, {}_degrees);\n"
					"{} = 1.0 - {};\n"
					"{} = clamp(({}_degrees - {}_min) / max({}_max - {}_min, 0.0001), 0.0, 1.0);\n",
					ctx.get_output_name(0),
					ctx.get_input_name(1),
					ctx.get_input_name(2),
					ctx.get_output_name(1),
					ctx.get_input_name(1),
					ctx.get_input_name(2),
					ctx.get_output_name(2),
					ctx.get_input_name(0),
					ctx.get_output_name(0),
					ctx.get_output_name(0),
					ctx.get_output_name(1),
					ctx.get_output_name(2),
					ctx.get_output_name(0),
					ctx.get_output_name(1),
					ctx.get_output_name(1));
		};
#endif
	}

	{
		NodeType &t = types[VoxelGraphFunction::NODE_ALTITUDE_MASK];
		t.name = "AltitudeMask";
		t.category = CATEGORY_MASKS;
		t.inputs.push_back(NodeType::Port("value"));
		t.inputs.push_back(NodeType::Port("min", 0.f));
		t.inputs.push_back(NodeType::Port("max", 100.f));
		t.inputs.push_back(NodeType::Port("falloff", 10.f));
		t.outputs.push_back(NodeType::Port("mask"));
		t.outputs.push_back(NodeType::Port("below"));
		t.outputs.push_back(NodeType::Port("above"));
		t.outputs.push_back(NodeType::Port("normalized"));
		t.process_buffer_func = [](Runtime::ProcessBufferContext &ctx) {
			const Runtime::Buffer &value = ctx.get_input(0);
			const Runtime::Buffer &min_value = ctx.get_input(1);
			const Runtime::Buffer &max_value = ctx.get_input(2);
			const Runtime::Buffer &falloff = ctx.get_input(3);
			Runtime::Buffer &mask = ctx.get_output(0);
			Runtime::Buffer &below = ctx.get_output(1);
			Runtime::Buffer &above = ctx.get_output(2);
			Runtime::Buffer &normalized = ctx.get_output(3);

			for (uint32_t i = 0; i < mask.size; ++i) {
				float min_v;
				float max_v;
				sort_min_max(min_value.data[i], max_value.data[i], min_v, max_v);
				const float f = get_safe_falloff(falloff.data[i]);
				const float lower = smoothstep(min_v - f, min_v + f, value.data[i]);
				const float upper = 1.f - smoothstep(max_v - f, max_v + f, value.data[i]);
				mask.data[i] = lower * upper;
				below.data[i] = 1.f - lower;
				above.data[i] = 1.f - upper;
				normalized.data[i] = clamp((value.data[i] - min_v) / MAX(max_v - min_v, 0.0001f), 0.f, 1.f);
			}
		};
		t.range_analysis_func = [](Runtime::RangeAnalysisContext &ctx) {
			ctx.set_output(0, Interval(0.f, 1.f));
			ctx.set_output(1, Interval(0.f, 1.f));
			ctx.set_output(2, Interval(0.f, 1.f));
			ctx.set_output(3, Interval(0.f, 1.f));
		};
#ifdef VOXEL_ENABLE_GPU
		t.shader_gen_func = [](ShaderGenContext &ctx) {
			ctx.add_format(
					"float {}_min = min({}, {});\n"
					"float {}_max = max({}, {});\n"
					"float {}_f = max(abs({}), 0.0001);\n"
					"float {}_lower = smoothstep({}_min - {}_f, {}_min + {}_f, {});\n"
					"float {}_upper = 1.0 - smoothstep({}_max - {}_f, {}_max + {}_f, {});\n"
					"{} = {}_lower * {}_upper;\n"
					"{} = 1.0 - {}_lower;\n"
					"{} = 1.0 - {}_upper;\n"
					"{} = clamp(({} - {}_min) / max({}_max - {}_min, 0.0001), 0.0, 1.0);\n",
					ctx.get_output_name(0),
					ctx.get_input_name(1),
					ctx.get_input_name(2),
					ctx.get_output_name(0),
					ctx.get_input_name(1),
					ctx.get_input_name(2),
					ctx.get_output_name(0),
					ctx.get_input_name(3),
					ctx.get_output_name(0),
					ctx.get_output_name(0),
					ctx.get_output_name(0),
					ctx.get_input_name(0),
					ctx.get_output_name(0),
					ctx.get_output_name(0),
					ctx.get_output_name(0),
					ctx.get_output_name(0),
					ctx.get_input_name(0),
					ctx.get_output_name(0),
					ctx.get_output_name(0),
					ctx.get_output_name(0),
					ctx.get_output_name(1),
					ctx.get_output_name(0),
					ctx.get_output_name(2),
					ctx.get_output_name(0),
					ctx.get_output_name(3),
					ctx.get_input_name(0),
					ctx.get_output_name(0),
					ctx.get_output_name(0),
					ctx.get_output_name(0));
		};
#endif
	}

	{
		NodeType &t = types[VoxelGraphFunction::NODE_NOISE_MASK];
		t.name = "NoiseMask";
		t.category = CATEGORY_MASKS;
		t.inputs.push_back(NodeType::Port("value"));
		t.inputs.push_back(NodeType::Port("threshold", 0.5f));
		t.inputs.push_back(NodeType::Port("falloff", 0.1f));
		t.outputs.push_back(NodeType::Port("mask"));
		t.outputs.push_back(NodeType::Port("inverted"));
		t.outputs.push_back(NodeType::Port("distance"));
		t.process_buffer_func = [](Runtime::ProcessBufferContext &ctx) {
			const Runtime::Buffer &value = ctx.get_input(0);
			const Runtime::Buffer &threshold = ctx.get_input(1);
			const Runtime::Buffer &falloff = ctx.get_input(2);
			Runtime::Buffer &mask = ctx.get_output(0);
			Runtime::Buffer &inverted = ctx.get_output(1);
			Runtime::Buffer &distance = ctx.get_output(2);

			for (uint32_t i = 0; i < mask.size; ++i) {
				const float f = get_safe_falloff(falloff.data[i]);
				const float m = smoothstep(threshold.data[i] - f, threshold.data[i] + f, value.data[i]);
				mask.data[i] = m;
				inverted.data[i] = 1.f - m;
				distance.data[i] = value.data[i] - threshold.data[i];
			}
		};
		t.range_analysis_func = [](Runtime::RangeAnalysisContext &ctx) {
			ctx.set_output(0, Interval(0.f, 1.f));
			ctx.set_output(1, Interval(0.f, 1.f));
			ctx.set_output(2, ctx.get_input(0) - ctx.get_input(1));
		};
#ifdef VOXEL_ENABLE_GPU
		t.shader_gen_func = [](ShaderGenContext &ctx) {
			ctx.add_format(
					"float {}_f = max(abs({}), 0.0001);\n"
					"{} = {} - {};\n"
					"{} = smoothstep({} - {}_f, {} + {}_f, {});\n"
					"{} = 1.0 - {};\n",
					ctx.get_output_name(0),
					ctx.get_input_name(2),
					ctx.get_output_name(2),
					ctx.get_input_name(0),
					ctx.get_input_name(1),
					ctx.get_output_name(0),
					ctx.get_input_name(1),
					ctx.get_output_name(0),
					ctx.get_input_name(1),
					ctx.get_output_name(0),
					ctx.get_input_name(0),
					ctx.get_output_name(1),
					ctx.get_output_name(0));
		};
#endif
	}

	{
		NodeType &t = types[VoxelGraphFunction::NODE_MATERIAL];
		t.name = "Material";
		t.category = CATEGORY_MATERIAL;
		NodeType::Param material_param(
				"material", "StandardMaterial3D", create_material_ref_to_variant<StandardMaterial3D>);
		t.params.push_back(material_param);
		NodeType::Param layer_param("layer", Variant::INT, -1);
		layer_param.has_range = true;
		layer_param.min_value = -1;
		layer_param.max_value = 15;
		t.params.push_back(layer_param);
		t.outputs.push_back(NodeType::Port("material"));
		t.process_buffer_func = [](Runtime::ProcessBufferContext &ctx) {
			Runtime::Buffer &out = ctx.get_output(0);
			for (uint32_t i = 0; i < out.size; ++i) {
				out.data[i] = 0.f;
			}
		};
		t.range_analysis_func = [](Runtime::RangeAnalysisContext &ctx) {
			ctx.set_output(0, Interval::from_single_value(0.f));
		};
#ifdef VOXEL_ENABLE_GPU
		t.shader_gen_func = [](ShaderGenContext &ctx) {
			ctx.add_format("{} = 0.0;\n", ctx.get_output_name(0));
		};
#endif
	}

	{
		NodeType &t = types[VoxelGraphFunction::NODE_BLEND_MATERIAL];
		t.name = "BlendMaterial";
		t.category = CATEGORY_MATERIAL;
		t.inputs.push_back(NodeType::Port("a"));
		t.inputs.push_back(NodeType::Port("b"));
		t.inputs.push_back(NodeType::Port("alpha", 0.5f));
		t.outputs.push_back(NodeType::Port("material"));
		t.process_buffer_func = [](Runtime::ProcessBufferContext &ctx) {
			const Runtime::Buffer &a = ctx.get_input(0);
			const Runtime::Buffer &b = ctx.get_input(1);
			const Runtime::Buffer &alpha = ctx.get_input(2);
			Runtime::Buffer &out = ctx.get_output(0);
			for (uint32_t i = 0; i < out.size; ++i) {
				const float w = clamp(alpha.data[i], 0.f, 1.f);
				out.data[i] = Math::lerp(a.data[i], b.data[i], w);
			}
		};
		t.range_analysis_func = [](Runtime::RangeAnalysisContext &ctx) {
			ctx.set_output(0, Interval::from_union(ctx.get_input(0), ctx.get_input(1)));
		};
#ifdef VOXEL_ENABLE_GPU
		t.shader_gen_func = [](ShaderGenContext &ctx) {
			ctx.add_format(
					"{} = mix({}, {}, clamp({}, 0.0, 1.0));\n",
					ctx.get_output_name(0),
					ctx.get_input_name(0),
					ctx.get_input_name(1),
					ctx.get_input_name(2));
		};
#endif
	}

	{
		NodeType &t = types[VoxelGraphFunction::NODE_MATERIAL_SWITCH];
		t.name = "MaterialSwitch";
		t.category = CATEGORY_MATERIAL;
		t.inputs.push_back(NodeType::Port("a"));
		t.inputs.push_back(NodeType::Port("b"));
		t.inputs.push_back(NodeType::Port("selector", 0.f));
		t.inputs.push_back(NodeType::Port("threshold", 0.5f));
		t.outputs.push_back(NodeType::Port("material"));
		t.process_buffer_func = [](Runtime::ProcessBufferContext &ctx) {
			const Runtime::Buffer &a = ctx.get_input(0);
			const Runtime::Buffer &b = ctx.get_input(1);
			const Runtime::Buffer &selector = ctx.get_input(2);
			const Runtime::Buffer &threshold = ctx.get_input(3);
			Runtime::Buffer &out = ctx.get_output(0);
			for (uint32_t i = 0; i < out.size; ++i) {
				out.data[i] = selector.data[i] < threshold.data[i] ? a.data[i] : b.data[i];
			}
		};
		t.range_analysis_func = [](Runtime::RangeAnalysisContext &ctx) {
			ctx.set_output(0, Interval::from_union(ctx.get_input(0), ctx.get_input(1)));
		};
#ifdef VOXEL_ENABLE_GPU
		t.shader_gen_func = [](ShaderGenContext &ctx) {
			ctx.add_format(
					"{} = mix({}, {}, step({}, {}));\n",
					ctx.get_output_name(0),
					ctx.get_input_name(0),
					ctx.get_input_name(1),
					ctx.get_input_name(3),
					ctx.get_input_name(2));
		};
#endif
	}

	{
		NodeType &t = types[VoxelGraphFunction::NODE_MATERIAL_PROPERTY_OVERRIDE];
		t.name = "MaterialPropertyOverride";
		t.category = CATEGORY_MATERIAL;
		t.inputs.push_back(NodeType::Port("material"));
		t.inputs.push_back(NodeType::Port("albedo_r", 1.f));
		t.inputs.push_back(NodeType::Port("albedo_g", 1.f));
		t.inputs.push_back(NodeType::Port("albedo_b", 1.f));
		t.inputs.push_back(NodeType::Port("roughness", 1.f));
		t.inputs.push_back(NodeType::Port("metallic", 0.f));
		t.inputs.push_back(NodeType::Port("specular", 0.5f));
		t.inputs.push_back(NodeType::Port("emission_r", 0.f));
		t.inputs.push_back(NodeType::Port("emission_g", 0.f));
		t.inputs.push_back(NodeType::Port("emission_b", 0.f));
		t.inputs.push_back(NodeType::Port("emission_energy", 1.f));
		t.inputs.push_back(NodeType::Port("normal_scale", 1.f));
		t.outputs.push_back(NodeType::Port("material"));
		t.process_buffer_func = [](Runtime::ProcessBufferContext &ctx) {
			const Runtime::Buffer &input = ctx.get_input(0);
			Runtime::Buffer &out = ctx.get_output(0);
			memcpy(out.data, input.data, input.size * sizeof(float));
		};
		t.range_analysis_func = [](Runtime::RangeAnalysisContext &ctx) {
			ctx.set_output(0, ctx.get_input(0));
		};
#ifdef VOXEL_ENABLE_GPU
		t.shader_gen_func = [](ShaderGenContext &ctx) {
			ctx.add_format("{} = {};\n", ctx.get_output_name(0), ctx.get_input_name(0));
		};
#endif
	}

	{
		NodeType &t = types[VoxelGraphFunction::NODE_MATERIAL_STACK];
		t.name = "MaterialStack";
		t.category = CATEGORY_MATERIAL;
		t.inputs.push_back(NodeType::Port("base"));
		t.inputs.push_back(NodeType::Port("middle"));
		t.inputs.push_back(NodeType::Port("top"));
		t.inputs.push_back(NodeType::Port("middle_alpha", 0.5f));
		t.inputs.push_back(NodeType::Port("top_alpha", 0.5f));
		t.outputs.push_back(NodeType::Port("material"));
		t.process_buffer_func = [](Runtime::ProcessBufferContext &ctx) {
			const Runtime::Buffer &base = ctx.get_input(0);
			const Runtime::Buffer &middle = ctx.get_input(1);
			const Runtime::Buffer &top = ctx.get_input(2);
			const Runtime::Buffer &middle_alpha = ctx.get_input(3);
			const Runtime::Buffer &top_alpha = ctx.get_input(4);
			Runtime::Buffer &out = ctx.get_output(0);
			for (uint32_t i = 0; i < out.size; ++i) {
				const float blended = Math::lerp(base.data[i], middle.data[i], clamp(middle_alpha.data[i], 0.f, 1.f));
				out.data[i] = Math::lerp(blended, top.data[i], clamp(top_alpha.data[i], 0.f, 1.f));
			}
		};
		t.range_analysis_func = [](Runtime::RangeAnalysisContext &ctx) {
			ctx.set_output(0, Interval::from_union(ctx.get_input(0), Interval::from_union(ctx.get_input(1), ctx.get_input(2))));
		};
#ifdef VOXEL_ENABLE_GPU
		t.shader_gen_func = [](ShaderGenContext &ctx) {
			ctx.add_format(
					"{} = mix(mix({}, {}, clamp({}, 0.0, 1.0)), {}, clamp({}, 0.0, 1.0));\n",
					ctx.get_output_name(0),
					ctx.get_input_name(0),
					ctx.get_input_name(1),
					ctx.get_input_name(3),
					ctx.get_input_name(2),
					ctx.get_input_name(4));
		};
#endif
	}

	{
		NodeType &t = types[VoxelGraphFunction::NODE_MATERIAL_MIXER];
		t.name = "MaterialMixer";
		t.category = CATEGORY_MATERIAL;
		// Connect a Material to each `material_i` and its coverage mask (from Climate2D, SlopeMask,
		// AltitudeMask, NoiseMask...) to the matching `weight_i`. The weights don't have to sum to 1:
		// the node normalizes them, so masks that overlap or leave gaps still give a usable blend.
		// Unconnected material slots are skipped entirely. If every weight is 0 the first connected
		// material wins, so a base material with no mask is a valid "fallback" slot.
		for (unsigned int i = 0; i < VoxelGraphFunction::MATERIAL_MIXER_SLOT_COUNT; ++i) {
			t.inputs.push_back(NodeType::Port(String("material_{0}").format(varray(i))));
			t.inputs.push_back(NodeType::Port(String("weight_{0}").format(varray(i)), 0.f));
		}
		t.outputs.push_back(NodeType::Port("material"));
		t.process_buffer_func = [](Runtime::ProcessBufferContext &ctx) {
			Runtime::Buffer &out = ctx.get_output(0);
			// Materials only carry a placeholder scalar at runtime (the real routing happens at
			// compile time, see build_material_layer_weights in voxel_generator_graph.cpp). Mirror the
			// blend anyway so previews and BlendMaterial chains stay consistent.
			const unsigned int slot_count = VoxelGraphFunction::MATERIAL_MIXER_SLOT_COUNT;
			for (uint32_t i = 0; i < out.size; ++i) {
				float acc = 0.f;
				float total = 0.f;
				for (unsigned int slot = 0; slot < slot_count; ++slot) {
					const float w = MAX(ctx.get_input(slot * 2 + 1).data[i], 0.f);
					acc += ctx.get_input(slot * 2).data[i] * w;
					total += w;
				}
				out.data[i] = total > 0.f ? acc / total : ctx.get_input(0).data[i];
			}
		};
		t.range_analysis_func = [](Runtime::RangeAnalysisContext &ctx) {
			Interval r = ctx.get_input(0);
			for (unsigned int slot = 1; slot < VoxelGraphFunction::MATERIAL_MIXER_SLOT_COUNT; ++slot) {
				r = Interval::from_union(r, ctx.get_input(slot * 2));
			}
			ctx.set_output(0, r);
		};
#ifdef VOXEL_ENABLE_GPU
		t.shader_gen_func = [](ShaderGenContext &ctx) {
			ctx.add_format(
					"float {}_acc = 0.0;\n"
					"float {}_total = 0.0;\n"
					"float {}_w = 0.0;\n",
					ctx.get_output_name(0),
					ctx.get_output_name(0),
					ctx.get_output_name(0));
			for (unsigned int slot = 0; slot < VoxelGraphFunction::MATERIAL_MIXER_SLOT_COUNT; ++slot) {
				ctx.add_format(
						"{}_w = max({}, 0.0);\n"
						"{}_acc += {} * {}_w;\n"
						"{}_total += {}_w;\n",
						ctx.get_output_name(0),
						ctx.get_input_name(slot * 2 + 1),
						ctx.get_output_name(0),
						ctx.get_input_name(slot * 2),
						ctx.get_output_name(0),
						ctx.get_output_name(0),
						ctx.get_output_name(0));
			}
			ctx.add_format(
					"{} = {}_total > 0.0 ? {}_acc / {}_total : {};\n",
					ctx.get_output_name(0),
					ctx.get_output_name(0),
					ctx.get_output_name(0),
					ctx.get_output_name(0),
					ctx.get_input_name(0));
		};
#endif
	}

	{
		NodeType &t = types[VoxelGraphFunction::NODE_OUTPUT_MATERIAL];
		t.name = "MaterialOutput";
		t.category = CATEGORY_OUTPUT;
		t.inputs.push_back(NodeType::Port("material"));
		t.outputs.push_back(NodeType::Port("_out"));
		t.process_buffer_func = [](Runtime::ProcessBufferContext &ctx) {
			const Runtime::Buffer &input = ctx.get_input(0);
			Runtime::Buffer &out = ctx.get_output(0);
			memcpy(out.data, input.data, input.size * sizeof(float));
		};
		t.range_analysis_func = [](Runtime::RangeAnalysisContext &ctx) {
			ctx.set_output(0, ctx.get_input(0));
		};
	}
}

} // namespace zylann::voxel::pg

#endif // VOXEL_GRAPH_NODES_MATERIALS_H
