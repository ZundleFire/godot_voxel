#ifndef VOXEL_GRAPH_NODES_PLANET_H
#define VOXEL_GRAPH_NODES_PLANET_H

#include "../../../../eden_planet_gen/planet_tectonics.h"
#include "../../../util/math/funcs.h"
#include "../../../util/profiling.h"
#include "../node_type_db.h"

namespace zylann::voxel::pg {

namespace {

template <typename T>
Variant create_ref_object_to_variant() {
	Ref<T> res;
	res.instantiate();
	return Variant(res);
}

struct PlanetTectonicsResourceHolder {
	Ref<PlanetTectonics> tectonics;
};

struct PlanetCellNoiseResourceHolder {
	Ref<PlanetTectonics> tectonics;
	VoronoiSphere fallback_voronoi;
};

inline float safe_length3(float x, float y, float z) {
	return MAX(Math::sqrt(x * x + y * y + z * z), 0.00001f);
}

inline uint32_t hash_u32(uint32_t x) {
	x ^= x >> 16;
	x *= 0x7feb352dU;
	x ^= x >> 15;
	x *= 0x846ca68bU;
	x ^= x >> 16;
	return x;
}

inline float hash_to_01(uint32_t x) {
	return float(hash_u32(x) & 0x00ffffffU) / float(0x00ffffffU);
}

inline float get_cell_noise_value(int cell_index, int seed) {
	return hash_to_01(uint32_t(cell_index + 1) * 747796405U ^ uint32_t(seed + 1) * 2891336453U);
}

} // namespace

void register_planet_nodes(Span<NodeType> types) {
	using namespace math;

	{
		struct Params {
			float radius;
		};
		NodeType &t = types[VoxelGraphFunction::NODE_PLANET_ALTITUDE];
		t.name = "PlanetAltitude";
		t.category = CATEGORY_GENERATE;
		t.inputs.push_back(NodeType::Port("x", 0.f, VoxelGraphFunction::AUTO_CONNECT_X));
		t.inputs.push_back(NodeType::Port("y", 0.f, VoxelGraphFunction::AUTO_CONNECT_Y));
		t.inputs.push_back(NodeType::Port("z", 0.f, VoxelGraphFunction::AUTO_CONNECT_Z));
		t.outputs.push_back(NodeType::Port("altitude"));
		t.params.push_back(NodeType::Param("radius", Variant::FLOAT, 40000.f));
		t.compile_func = [](CompileContext &ctx) {
			Params p;
			p.radius = ctx.get_param(0);
			ctx.set_params(p);
		};
		t.process_buffer_func = [](Runtime::ProcessBufferContext &ctx) {
			ZN_PROFILE_SCOPE_NAMED("NODE_PLANET_ALTITUDE");
			const Runtime::Buffer &x = ctx.get_input(0);
			const Runtime::Buffer &y = ctx.get_input(1);
			const Runtime::Buffer &z = ctx.get_input(2);
			Runtime::Buffer &out = ctx.get_output(0);
			const Params p = ctx.get_params<Params>();
			for (uint32_t i = 0; i < out.size; ++i) {
				out.data[i] = safe_length3(x.data[i], y.data[i], z.data[i]) - p.radius;
			}
		};
		t.range_analysis_func = [](Runtime::RangeAnalysisContext &ctx) {
			const Interval x = ctx.get_input(0);
			const Interval y = ctx.get_input(1);
			const Interval z = ctx.get_input(2);
			const Params p = ctx.get_params<Params>();
			ctx.set_output(0, get_length(x, y, z) - p.radius);
		};
#ifdef VOXEL_ENABLE_GPU
		t.shader_gen_func = [](ShaderGenContext &ctx) {
			const float radius = ctx.get_param(0);
			ctx.add_format(
					"{} = length(vec3({}, {}, {})) - {}f;\n",
					ctx.get_output_name(0),
					ctx.get_input_name(0),
					ctx.get_input_name(1),
					ctx.get_input_name(2),
					radius);
		};
#endif
	}

	{
		NodeType &t = types[VoxelGraphFunction::NODE_PLANET_LATITUDE];
		t.name = "PlanetLatitude";
		t.category = CATEGORY_CONVERT;
		t.inputs.push_back(NodeType::Port("x", 0.f, VoxelGraphFunction::AUTO_CONNECT_X));
		t.inputs.push_back(NodeType::Port("y", 0.f, VoxelGraphFunction::AUTO_CONNECT_Y));
		t.inputs.push_back(NodeType::Port("z", 0.f, VoxelGraphFunction::AUTO_CONNECT_Z));
		t.outputs.push_back(NodeType::Port("signed"));
		t.outputs.push_back(NodeType::Port("abs"));
		t.process_buffer_func = [](Runtime::ProcessBufferContext &ctx) {
			ZN_PROFILE_SCOPE_NAMED("NODE_PLANET_LATITUDE");
			const Runtime::Buffer &x = ctx.get_input(0);
			const Runtime::Buffer &y = ctx.get_input(1);
			const Runtime::Buffer &z = ctx.get_input(2);
			Runtime::Buffer &signed_out = ctx.get_output(0);
			Runtime::Buffer &abs_out = ctx.get_output(1);
			for (uint32_t i = 0; i < signed_out.size; ++i) {
				const float lat = y.data[i] / safe_length3(x.data[i], y.data[i], z.data[i]);
				signed_out.data[i] = lat;
				abs_out.data[i] = Math::abs(lat);
			}
		};
		t.range_analysis_func = [](Runtime::RangeAnalysisContext &ctx) {
			ctx.set_output(0, Interval(-1.f, 1.f));
			ctx.set_output(1, Interval(0.f, 1.f));
		};
#ifdef VOXEL_ENABLE_GPU
		t.shader_gen_func = [](ShaderGenContext &ctx) {
			ctx.add_format(
					"vec3 {}_p = vec3({}, {}, {});\n"
					"float {}_lat = {}_p.y / max(length({}_p), 0.00001);\n"
					"{} = {}_lat;\n"
					"{} = abs({}_lat);\n",
					ctx.get_output_name(0),
					ctx.get_input_name(0),
					ctx.get_input_name(1),
					ctx.get_input_name(2),
					ctx.get_output_name(0),
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
		NodeType &t = types[VoxelGraphFunction::NODE_PLANET_LONGITUDE];
		t.name = "PlanetLongitude";
		t.category = CATEGORY_CONVERT;
		t.inputs.push_back(NodeType::Port("x", 0.f, VoxelGraphFunction::AUTO_CONNECT_X));
		t.inputs.push_back(NodeType::Port("z", 0.f, VoxelGraphFunction::AUTO_CONNECT_Z));
		t.outputs.push_back(NodeType::Port("radians"));
		t.outputs.push_back(NodeType::Port("normalized"));
		t.process_buffer_func = [](Runtime::ProcessBufferContext &ctx) {
			ZN_PROFILE_SCOPE_NAMED("NODE_PLANET_LONGITUDE");
			const Runtime::Buffer &x = ctx.get_input(0);
			const Runtime::Buffer &z = ctx.get_input(1);
			Runtime::Buffer &rad_out = ctx.get_output(0);
			Runtime::Buffer &norm_out = ctx.get_output(1);
			const float inv_tau = 1.0f / float(Math::TAU);
			for (uint32_t i = 0; i < rad_out.size; ++i) {
				const float lon = Math::atan2(z.data[i], x.data[i]);
				rad_out.data[i] = lon;
				norm_out.data[i] = 0.5f - lon * inv_tau;
			}
		};
		t.range_analysis_func = [](Runtime::RangeAnalysisContext &ctx) {
			ctx.set_output(0, Interval(-float(Math::PI), float(Math::PI)));
			ctx.set_output(1, Interval(0.f, 1.f));
		};
#ifdef VOXEL_ENABLE_GPU
		t.shader_gen_func = [](ShaderGenContext &ctx) {
			ctx.add_format(
					"{} = atan({}, {});\n"
					"{} = 0.5 - {} * {};\n",
					ctx.get_output_name(0),
					ctx.get_input_name(1),
					ctx.get_input_name(0),
					ctx.get_output_name(1),
					ctx.get_output_name(0),
					(1.0 / double(Math::TAU)));
		};
#endif
	}

	{
		struct Params {
			float equator_temp_c;
			float polar_temp_c;
			float ocean_humidity;
			float interior_humidity;
			float lapse_rate_c_per_m;
		};
		NodeType &t = types[VoxelGraphFunction::NODE_PLANET_CLIMATE];
		t.name = "PlanetClimate";
		t.category = CATEGORY_GENERATE;
		t.inputs.push_back(NodeType::Port("x", 0.f, VoxelGraphFunction::AUTO_CONNECT_X));
		t.inputs.push_back(NodeType::Port("y", 0.f, VoxelGraphFunction::AUTO_CONNECT_Y));
		t.inputs.push_back(NodeType::Port("z", 0.f, VoxelGraphFunction::AUTO_CONNECT_Z));
		t.inputs.push_back(NodeType::Port("altitude", 0.f));
		t.inputs.push_back(NodeType::Port("coast_proximity", 0.5f));
		t.inputs.push_back(NodeType::Port("moisture_bias", 0.f));
		t.inputs.push_back(NodeType::Port("temp_bias", 0.f));
		t.outputs.push_back(NodeType::Port("temp_c"));
		t.outputs.push_back(NodeType::Port("humidity"));
		t.params.push_back(NodeType::Param("equator_temp_c", Variant::FLOAT, 30.f));
		t.params.push_back(NodeType::Param("polar_temp_c", Variant::FLOAT, -10.f));
		t.params.push_back(NodeType::Param("ocean_humidity", Variant::FLOAT, 0.85f));
		t.params.push_back(NodeType::Param("interior_humidity", Variant::FLOAT, 0.28f));
		t.params.push_back(NodeType::Param("lapse_rate_c_per_m", Variant::FLOAT, 0.0048f));
		t.compile_func = [](CompileContext &ctx) {
			Params p;
			p.equator_temp_c = ctx.get_param(0);
			p.polar_temp_c = ctx.get_param(1);
			p.ocean_humidity = ctx.get_param(2);
			p.interior_humidity = ctx.get_param(3);
			p.lapse_rate_c_per_m = ctx.get_param(4);
			ctx.set_params(p);
		};
		t.process_buffer_func = [](Runtime::ProcessBufferContext &ctx) {
			ZN_PROFILE_SCOPE_NAMED("NODE_PLANET_CLIMATE");
			const Runtime::Buffer &x = ctx.get_input(0);
			const Runtime::Buffer &y = ctx.get_input(1);
			const Runtime::Buffer &z = ctx.get_input(2);
			const Runtime::Buffer &altitude = ctx.get_input(3);
			const Runtime::Buffer &coast = ctx.get_input(4);
			const Runtime::Buffer &moisture_bias = ctx.get_input(5);
			const Runtime::Buffer &temp_bias = ctx.get_input(6);
			Runtime::Buffer &temp_out = ctx.get_output(0);
			Runtime::Buffer &humidity_out = ctx.get_output(1);
			const Params p = ctx.get_params<Params>();
			for (uint32_t i = 0; i < temp_out.size; ++i) {
				const float len = safe_length3(x.data[i], y.data[i], z.data[i]);
				const float lat01 = Math::abs(y.data[i] / len);
				const float lat_curve = Math::pow(lat01, 1.35f);
				const float terrain_alt = MAX(altitude.data[i], 0.f);
				const float coast_p = CLAMP(coast.data[i], 0.f, 1.f);
				const float temp_c = Math::lerp(p.equator_temp_c, p.polar_temp_c, lat_curve)
						- terrain_alt * p.lapse_rate_c_per_m + temp_bias.data[i];
				float humidity = Math::lerp(p.interior_humidity, p.ocean_humidity, coast_p)
						+ moisture_bias.data[i]
						- terrain_alt * 0.0000105f;
				temp_out.data[i] = temp_c;
				humidity_out.data[i] = CLAMP(humidity, 0.f, 1.f);
			}
		};
		t.range_analysis_func = [](Runtime::RangeAnalysisContext &ctx) {
			const Params p = ctx.get_params<Params>();
			const Interval altitude = ctx.get_input(3);
			const Interval moisture_bias = ctx.get_input(5);
			const Interval temp_bias = ctx.get_input(6);
			const float min_base = MIN(p.equator_temp_c, p.polar_temp_c);
			const float max_base = MAX(p.equator_temp_c, p.polar_temp_c);
			const float min_alt = MAX(altitude.min, 0.f);
			const float max_alt = MAX(altitude.max, 0.f);
			ctx.set_output(0, Interval(min_base - max_alt * p.lapse_rate_c_per_m, max_base - min_alt * p.lapse_rate_c_per_m) + temp_bias);
			ctx.set_output(1, Interval(0.f, 1.f) + moisture_bias);
		};
#ifdef VOXEL_ENABLE_GPU
		t.shader_gen_func = [](ShaderGenContext &ctx) {
			const float eq_temp = ctx.get_param(0);
			const float polar_temp = ctx.get_param(1);
			const float ocean_humidity = ctx.get_param(2);
			const float interior_humidity = ctx.get_param(3);
			const float lapse_rate = ctx.get_param(4);
			ctx.add_format(
					"vec3 {}_p = vec3({}, {}, {});\n"
					"float {}_lat01 = abs({}_p.y) / max(length({}_p), 0.00001);\n"
					"float {}_lat_curve = pow({}_lat01, 1.35);\n"
					"float {}_terrain_alt = max({}, 0.0);\n"
					"float {}_coast = clamp({}, 0.0, 1.0);\n"
					"{} = mix({}, {}, {}_lat_curve) - {}_terrain_alt * {} + {};\n"
					"{} = clamp(mix({}, {}, {}_coast) + {} - {}_terrain_alt * 0.0000105, 0.0, 1.0);\n",
					ctx.get_output_name(0),
					ctx.get_input_name(0),
					ctx.get_input_name(1),
					ctx.get_input_name(2),
					ctx.get_output_name(0),
					ctx.get_output_name(0),
					ctx.get_output_name(0),
					ctx.get_output_name(0),
					ctx.get_output_name(0),
					ctx.get_output_name(0),
					ctx.get_input_name(3),
					ctx.get_output_name(0),
					ctx.get_input_name(4),
					ctx.get_output_name(0),
					eq_temp,
					polar_temp,
					ctx.get_output_name(0),
					ctx.get_output_name(0),
					lapse_rate,
					ctx.get_input_name(6),
					ctx.get_output_name(1),
					interior_humidity,
					ocean_humidity,
					ctx.get_output_name(0),
					ctx.get_input_name(5),
					ctx.get_output_name(0));
		};
#endif
	}

	{
		NodeType &t = types[VoxelGraphFunction::NODE_PLANET_BIOME];
		t.name = "PlanetBiome";
		t.category = CATEGORY_GENERATE;
		t.inputs.push_back(NodeType::Port("temp_c", 15.f));
		t.inputs.push_back(NodeType::Port("humidity", 0.5f));
		t.inputs.push_back(NodeType::Port("altitude", 0.f));
		t.inputs.push_back(NodeType::Port("oceanic", 0.f));
		t.inputs.push_back(NodeType::Port("shore_proximity", 0.5f));
		t.outputs.push_back(NodeType::Port("desert"));
		t.outputs.push_back(NodeType::Port("forest"));
		t.outputs.push_back(NodeType::Port("tropical"));
		t.outputs.push_back(NodeType::Port("tundra"));
		t.outputs.push_back(NodeType::Port("grassland"));
		t.outputs.push_back(NodeType::Port("alpine"));
		t.outputs.push_back(NodeType::Port("water"));
		t.outputs.push_back(NodeType::Port("dominant"));
		t.process_buffer_func = [](Runtime::ProcessBufferContext &ctx) {
			ZN_PROFILE_SCOPE_NAMED("NODE_PLANET_BIOME");
			const Runtime::Buffer &temp_in = ctx.get_input(0);
			const Runtime::Buffer &humidity_in = ctx.get_input(1);
			const Runtime::Buffer &altitude_in = ctx.get_input(2);
			const Runtime::Buffer &oceanic_in = ctx.get_input(3);
			const Runtime::Buffer &shore_in = ctx.get_input(4);
			Runtime::Buffer &desert_out = ctx.get_output(0);
			Runtime::Buffer &forest_out = ctx.get_output(1);
			Runtime::Buffer &tropical_out = ctx.get_output(2);
			Runtime::Buffer &tundra_out = ctx.get_output(3);
			Runtime::Buffer &grassland_out = ctx.get_output(4);
			Runtime::Buffer &alpine_out = ctx.get_output(5);
			Runtime::Buffer &water_out = ctx.get_output(6);
			Runtime::Buffer &dominant_out = ctx.get_output(7);
			for (uint32_t i = 0; i < desert_out.size; ++i) {
				const float temp_c = temp_in.data[i];
				const float humidity = CLAMP(humidity_in.data[i], 0.f, 1.f);
				const float altitude = altitude_in.data[i];
				const float oceanic = CLAMP(oceanic_in.data[i], 0.f, 1.f);
				const float shore = CLAMP(shore_in.data[i], 0.f, 1.f);
				const float lowland = 1.0f - Math::smoothstep(300.0f, 1800.0f, altitude);
				const float upland = Math::smoothstep(350.0f, 2200.0f, altitude);
				const float highland = Math::smoothstep(1200.0f, 3400.0f, altitude);
				const float warm = Math::smoothstep(12.0f, 30.0f, temp_c);
				const float hot = Math::smoothstep(22.0f, 34.0f, temp_c);
				const float cold = 1.0f - Math::smoothstep(-2.0f, 18.0f, temp_c);
				const float temperate = 1.0f - CLAMP(Math::abs(temp_c - 14.0f) / 18.0f, 0.0f, 1.0f);
				const float lush = Math::smoothstep(0.58f, 0.90f, humidity);
				const float wet = Math::smoothstep(0.45f, 0.78f, humidity);
				const float arid = 1.0f - Math::smoothstep(0.22f, 0.56f, humidity);
				const float inland = 1.0f - shore;
				const float water = CLAMP(
						MAX(
								1.0f - Math::smoothstep(-120.0f, 80.0f, altitude),
								oceanic * (0.35f + 0.65f * (1.0f - Math::smoothstep(-40.0f, 160.0f, altitude)))
						),
						0.0f,
						1.0f
				);
				const float land = 1.0f - water;
				float scores[6];
				scores[0] = land * arid * (0.30f + 0.70f * warm) * (0.55f + 0.45f * lowland) * (0.65f + 0.35f * inland);
				scores[1] = land * wet * (0.35f + 0.65f * (1.0f - cold)) * (0.45f + 0.55f * (1.0f - arid)) * (0.55f + 0.45f * lowland);
				scores[2] = land * lush * hot * (0.50f + 0.50f * lowland) * (0.75f + 0.25f * shore);
				scores[3] = land * cold * (0.35f + 0.65f * (1.0f - hot)) * (0.45f + 0.55f * (0.4f + 0.6f * highland));
				scores[4] = land * temperate * (0.55f + 0.45f * lowland) * (0.45f + 0.55f * (1.0f - Math::abs(humidity - 0.50f) * 1.8f));
				scores[5] = land * highland * (0.20f + 0.80f * (0.55f * cold + 0.45f * upland));
				float score_total = 0.0f;
				int dominant = 6;
				float dominant_score = water;
				for (int j = 0; j < 6; ++j) {
					scores[j] = MAX(scores[j], 0.0f);
					score_total += scores[j];
					if (scores[j] > dominant_score) {
						dominant_score = scores[j];
						dominant = j;
					}
				}
				const float inv_total = 1.0f / MAX(score_total, 0.0001f);
				desert_out.data[i] = scores[0] * inv_total;
				forest_out.data[i] = scores[1] * inv_total;
				tropical_out.data[i] = scores[2] * inv_total;
				tundra_out.data[i] = scores[3] * inv_total;
				grassland_out.data[i] = scores[4] * inv_total;
				alpine_out.data[i] = scores[5] * inv_total;
				water_out.data[i] = water;
				dominant_out.data[i] = float(dominant);
			}
		};
		t.range_analysis_func = [](Runtime::RangeAnalysisContext &ctx) {
			for (unsigned int i = 0; i < 7; ++i) {
				ctx.set_output(i, Interval(0.f, 1.f));
			}
			ctx.set_output(7, Interval(0.f, 6.f));
		};
	}

	{
		struct Params {
			const PlanetCellNoiseResourceHolder *holder;
			int seed;
			float edge_width_rad;
		};
		NodeType &t = types[VoxelGraphFunction::NODE_PLANET_CELL_NOISE];
		t.name = "PlanetCellNoise";
		t.category = CATEGORY_GENERATE;
		t.inputs.push_back(NodeType::Port("x", 0.f, VoxelGraphFunction::AUTO_CONNECT_X));
		t.inputs.push_back(NodeType::Port("y", 0.f, VoxelGraphFunction::AUTO_CONNECT_Y));
		t.inputs.push_back(NodeType::Port("z", 0.f, VoxelGraphFunction::AUTO_CONNECT_Z));
		t.outputs.push_back(NodeType::Port("noise"));
		t.outputs.push_back(NodeType::Port("signed"));
		t.outputs.push_back(NodeType::Port("cell_id"));
		t.outputs.push_back(NodeType::Port("edge"));
		t.params.push_back(NodeType::Param("tectonics", PlanetTectonics::get_class_static(), nullptr));
		t.params.push_back(NodeType::Param("num_points", Variant::INT, 600));
		t.params.push_back(NodeType::Param("seed", Variant::INT, 0));
		t.params.push_back(NodeType::Param("edge_width_rad", Variant::FLOAT, 0.055f));
		t.compile_func = [](CompileContext &ctx) {
			Ref<PlanetTectonics> tectonics = ctx.get_param(0);
			PlanetCellNoiseResourceHolder *holder = ZN_NEW(PlanetCellNoiseResourceHolder);
			if (tectonics.is_valid()) {
				holder->tectonics = tectonics;
			} else {
				const int num_points = MAX(int(ctx.get_param(1)), 12);
				const int seed = int(ctx.get_param(2));
				holder->fallback_voronoi.generate(num_points, seed);
			}
			Params p;
			p.holder = holder;
			p.seed = int(ctx.get_param(2));
			p.edge_width_rad = MAX(float(ctx.get_param(3)), 0.0001f);
			ctx.add_delete_cleanup(holder);
			ctx.set_params(p);
		};
		t.process_buffer_func = [](Runtime::ProcessBufferContext &ctx) {
			ZN_PROFILE_SCOPE_NAMED("NODE_PLANET_CELL_NOISE");
			const Runtime::Buffer &x = ctx.get_input(0);
			const Runtime::Buffer &y = ctx.get_input(1);
			const Runtime::Buffer &z = ctx.get_input(2);
			Runtime::Buffer &noise_out = ctx.get_output(0);
			Runtime::Buffer &signed_out = ctx.get_output(1);
			Runtime::Buffer &cell_id_out = ctx.get_output(2);
			Runtime::Buffer &edge_out = ctx.get_output(3);
			const Params p = ctx.get_params<Params>();
			PlanetTectonics *tectonics = nullptr;
			const VoronoiSphere *voronoi = nullptr;
			if (p.holder != nullptr) {
				if (p.holder->tectonics.is_valid() && p.holder->tectonics->get_voronoi().get_num_points() > 1) {
					tectonics = *p.holder->tectonics;
					voronoi = &tectonics->get_voronoi();
				} else if (p.holder->fallback_voronoi.get_num_points() > 1) {
					voronoi = &p.holder->fallback_voronoi;
				}
			}
			const int point_count = voronoi != nullptr ? voronoi->get_num_points() : 0;
			for (uint32_t i = 0; i < noise_out.size; ++i) {
				if (point_count < 2) {
					noise_out.data[i] = 0.f;
					signed_out.data[i] = 0.f;
					cell_id_out.data[i] = 0.f;
					edge_out.data[i] = 0.f;
					continue;
				}
				const float inv_len = 1.0f / safe_length3(x.data[i], y.data[i], z.data[i]);
				const Vector3 dir(x.data[i] * inv_len, y.data[i] * inv_len, z.data[i] * inv_len);
				const Vector3 sample_dir = tectonics != nullptr ? tectonics->warp_direction(dir) : dir;
				int best_idx = 0;
				int second_idx = 0;
				float best_dot = -2.0f;
				float second_dot = -2.0f;
				voronoi->get_two_nearest(sample_dir, best_idx, second_idx, best_dot, second_dot);
				const float ang_a = Math::acos(CLAMP(best_dot, -1.0f, 1.0f));
				const float ang_b = Math::acos(CLAMP(second_dot, -1.0f, 1.0f));
				const float blend_denom = MAX(ang_a + ang_b, 0.0001f);
				const float blend_to_a = CLAMP(0.5f + 0.5f * (ang_b - ang_a) / blend_denom, 0.0f, 1.0f);
				const float edge = Math::pow(1.0f - CLAMP((ang_b - ang_a) / p.edge_width_rad, 0.0f, 1.0f), 0.72f);
				const float va = get_cell_noise_value(best_idx, p.seed);
				const float vb = get_cell_noise_value(second_idx, p.seed);
				const float noise = Math::lerp(vb, va, blend_to_a);
				noise_out.data[i] = noise;
				signed_out.data[i] = noise * 2.0f - 1.0f;
				cell_id_out.data[i] = float(best_idx) / MAX(float(point_count - 1), 1.0f);
				edge_out.data[i] = CLAMP(edge, 0.0f, 1.0f);
			}
		};
		t.range_analysis_func = [](Runtime::RangeAnalysisContext &ctx) {
			ctx.set_output(0, Interval(0.f, 1.f));
			ctx.set_output(1, Interval(-1.f, 1.f));
			ctx.set_output(2, Interval(0.f, 1.f));
			ctx.set_output(3, Interval(0.f, 1.f));
		};
	}

	{
		struct Params {
			const PlanetTectonicsResourceHolder *holder;
			float planet_radius;
		};
		NodeType &t = types[VoxelGraphFunction::NODE_PLANET_TECTONICS];
		t.name = "PlanetTectonics";
		t.category = CATEGORY_GENERATE;
		t.inputs.push_back(NodeType::Port("x", 0.f, VoxelGraphFunction::AUTO_CONNECT_X));
		t.inputs.push_back(NodeType::Port("y", 0.f, VoxelGraphFunction::AUTO_CONNECT_Y));
		t.inputs.push_back(NodeType::Port("z", 0.f, VoxelGraphFunction::AUTO_CONNECT_Z));
		t.outputs.push_back(NodeType::Port("falloff"));
		t.outputs.push_back(NodeType::Port("oceanic"));
		t.outputs.push_back(NodeType::Port("bias"));
		t.outputs.push_back(NodeType::Port("border_dist_m"));
		t.outputs.push_back(NodeType::Port("boundary_type"));
		t.params.push_back(NodeType::Param("tectonics", PlanetTectonics::get_class_static(), nullptr));
		t.params.push_back(NodeType::Param("planet_radius", Variant::FLOAT, 40000.f));
		t.compile_func = [](CompileContext &ctx) {
			Ref<PlanetTectonics> tectonics = ctx.get_param(0);
			Params p;
			p.holder = nullptr;
			if (tectonics.is_valid()) {
				PlanetTectonicsResourceHolder *holder = ZN_NEW(PlanetTectonicsResourceHolder);
				holder->tectonics = tectonics;
				p.holder = holder;
				ctx.add_delete_cleanup(holder);
			}
			p.planet_radius = ctx.get_param(1);
			ctx.set_params(p);
		};
		t.process_buffer_func = [](Runtime::ProcessBufferContext &ctx) {
			ZN_PROFILE_SCOPE_NAMED("NODE_PLANET_TECTONICS");
			const Runtime::Buffer &x = ctx.get_input(0);
			const Runtime::Buffer &y = ctx.get_input(1);
			const Runtime::Buffer &z = ctx.get_input(2);
			Runtime::Buffer &falloff_out = ctx.get_output(0);
			Runtime::Buffer &oceanic_out = ctx.get_output(1);
			Runtime::Buffer &bias_out = ctx.get_output(2);
			Runtime::Buffer &border_out = ctx.get_output(3);
			Runtime::Buffer &boundary_out = ctx.get_output(4);
			const Params p = ctx.get_params<Params>();
			PlanetTectonics *tectonics = nullptr;
			if (p.holder != nullptr && p.holder->tectonics.is_valid()) {
				tectonics = *p.holder->tectonics;
			}
			const bool tectonics_ready = tectonics != nullptr && tectonics->get_plate_ids().size() > 1;
			for (uint32_t i = 0; i < falloff_out.size; ++i) {
				if (!tectonics_ready) {
					falloff_out.data[i] = 0.f;
					oceanic_out.data[i] = 0.f;
					bias_out.data[i] = 0.f;
					border_out.data[i] = p.planet_radius;
					boundary_out.data[i] = float(PlanetTectonics::BND_INTERIOR);
					continue;
				}
				const float inv_len = 1.0f / safe_length3(x.data[i], y.data[i], z.data[i]);
				const Vector3 dir(x.data[i] * inv_len, y.data[i] * inv_len, z.data[i] * inv_len);
				const PlanetTectonics::TerrainData td = tectonics->get_terrain_data(dir);
				falloff_out.data[i] = td.falloff_s;
				oceanic_out.data[i] = td.oceanic_s;
				bias_out.data[i] = td.bias_s;
				border_out.data[i] = td.border_dist_rad * p.planet_radius;
				boundary_out.data[i] = float(td.bnd_type_a);
			}
		};
		t.range_analysis_func = [](Runtime::RangeAnalysisContext &ctx) {
			ctx.set_output(0, Interval(0.f, 1.f));
			ctx.set_output(1, Interval(0.f, 1.f));
			ctx.set_output(2, Interval(-4000.f, 4000.f));
			ctx.set_output(3, Interval(0.f, 20000.f));
			ctx.set_output(4, Interval(0.f, 7.f));
		};
	}
}

} // namespace zylann::voxel::pg

#endif // VOXEL_GRAPH_NODES_PLANET_H
