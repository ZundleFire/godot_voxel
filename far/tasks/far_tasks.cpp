#include "far_tasks.h"
#include "../../storage/mixel4.h"

#include "../../storage/voxel_buffer.h"
#include "../../util/profiling.h"

#include <algorithm>
#include <cmath>

namespace zylann::voxel::far {

namespace {

// Which channel, if any, carries something the far field can use as a material.
//
// Two conventions are in play. `CHANNEL_TYPE` is a plain id per voxel, which is
// what the far field originally read. MIXEL4 instead packs four material indices
// and four blend weights into `CHANNEL_INDICES`/`CHANNEL_WEIGHTS`, and that is
// what this project's planet generators write -- so reading only TYPE gave every
// layer material 0 and the palette never worked at all.
//
// The far field stores one material per layer, so for MIXEL4 the heaviest of the
// four is taken. At far-field cell sizes a blend is not representable anyway.
enum MaterialSource { MATERIAL_NONE, MATERIAL_TYPE, MATERIAL_MIXEL4 };

MaterialSource choose_material_source(const VoxelGenerator &generator) {
	const int mask = generator.get_used_channels_mask();
	if ((mask & (1 << VoxelBuffer::CHANNEL_TYPE)) != 0) {
		return MATERIAL_TYPE;
	}
	if ((mask & (1 << VoxelBuffer::CHANNEL_INDICES)) != 0 && (mask & (1 << VoxelBuffer::CHANNEL_WEIGHTS)) != 0) {
		return MATERIAL_MIXEL4;
	}
	return MATERIAL_NONE;
}

uint16_t read_material(const VoxelBuffer &buffer, MaterialSource source, int x, int y, int z) {
	switch (source) {
		case MATERIAL_TYPE:
			return static_cast<uint16_t>(buffer.get_voxel(x, y, z, VoxelBuffer::CHANNEL_TYPE));

		case MATERIAL_MIXEL4: {
			const uint16_t packed_indices =
					static_cast<uint16_t>(buffer.get_voxel(x, y, z, VoxelBuffer::CHANNEL_INDICES));
			const uint16_t packed_weights =
					static_cast<uint16_t>(buffer.get_voxel(x, y, z, VoxelBuffer::CHANNEL_WEIGHTS));
			const FixedArray<uint8_t, 4> indices = mixel4::decode_indices_from_packed_u16(packed_indices);
			const FixedArray<uint8_t, 4> weights = mixel4::decode_weights_from_packed_u16(packed_weights);
			unsigned int heaviest = 0;
			for (unsigned int i = 1; i < 4; ++i) {
				if (weights[i] > weights[heaviest]) {
					heaviest = i;
				}
			}
			return indices[heaviest];
		}

		default:
			return 0;
	}
}

} // namespace

// ---------------------------------------------------------------------------
// FarResultQueue
// ---------------------------------------------------------------------------

void FarResultQueue::push(FarSectorResult &&result) {
	MutexLock lock(_mutex);
	_results.push_back(std::move(result));
}

unsigned int FarResultQueue::pop_batch(std::vector<FarSectorResult> &out, unsigned int max_count) {
	MutexLock lock(_mutex);
	const unsigned int count = std::min(static_cast<unsigned int>(_results.size()), max_count);
	if (count == 0) {
		return 0;
	}
	// Take from the back: the queue is unordered anyway (results arrive as
	// threads finish, not in request order), and popping the back avoids an
	// O(n) shift per item.
	for (unsigned int i = 0; i < count; ++i) {
		out.push_back(std::move(_results.back()));
		_results.pop_back();
	}
	return count;
}

void FarResultQueue::clear() {
	MutexLock lock(_mutex);
	_results.clear();
}

size_t FarResultQueue::size() const {
	MutexLock lock(_mutex);
	return _results.size();
}

// ---------------------------------------------------------------------------
// FarBuildSectorTask
// ---------------------------------------------------------------------------

FarBuildSectorTask::FarBuildSectorTask(
		SectorId id,
		float priority_distance,
		std::shared_ptr<FarTaskParams> params,
		std::shared_ptr<FarSharedState> shared,
		uint32_t generation
) :
		_id(id),
		_priority_distance(priority_distance),
		_params(std::move(params)),
		_shared(std::move(shared)),
		_generation(generation) {
	if (_shared != nullptr) {
		_shared->tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
	}
}

FarBuildSectorTask::~FarBuildSectorTask() {
	if (_shared != nullptr) {
		_shared->tasks_in_flight.fetch_sub(1, std::memory_order_relaxed);
	}
}

bool FarBuildSectorTask::is_cancelled() {
	return _shared == nullptr || _shared->is_stale(_generation);
}

TaskPriority FarBuildSectorTask::get_priority() {
	TaskPriority priority;

	// Finer levels first. They are nearer by construction, so they are what the
	// viewer is most likely to notice missing.
	priority.band3 = static_cast<uint8_t>(255 - std::min<unsigned int>(_id.lod, 255));

	// Then nearer first within a level. Bucketed into the remaining bands so
	// that comparisons stay a single integer compare.
	const float scale = std::max(_id.get_world_size(), 1.f);
	const unsigned int bucket =
			std::min<unsigned int>(static_cast<unsigned int>(_priority_distance / scale), 0xffffu);
	const unsigned int inverted = 0xffffu - bucket;
	priority.band2 = static_cast<uint8_t>((inverted >> 8) & 0xff);
	priority.band1 = static_cast<uint8_t>(inverted & 0xff);
	priority.band0 = 0;

	return priority;
}


// Samples the generator along radial columns instead of vertical ones.
//
// This is the one piece of the far field that genuinely differs on a planet.
// Everything else works in grid coordinates and a scalar along the column axis,
// and does not care what that axis is -- but the generator only speaks world
// space, and a radial column is a line of arbitrary 3D points rather than an
// axis-aligned run a `generate_block` can fill.
//
// Two ways to ask for those points, in order of preference:
//
//   generate_series()  arbitrary point batches, which is exactly this shape.
//                      Implemented by VoxelGeneratorGraph and VoxelGeneratorNoise2D.
//   generate_block()   on a reused 1x1x1 buffer, one call per sample.
//                      Correct for any generator and roughly two orders of
//                      magnitude slower. `VoxelGenerator::generate_single` does
//                      the same thing but allocates a VoxelBuffer *per sample*,
//                      which at ~270k samples per sector is not usable at all.
//
// Materials are not read here. The flat path pulls CHANNEL_TYPE alongside the
// SDF, but a batch series call returns one channel, and the MIXEL4 layout that
// this project's generators write into INDICES/WEIGHTS is not meaningful when
// sampled as floats. Every layer gets material 0, so the shader's height and
// slope colouring still works and the palette does not. See HANDOFF 9.4.
bool FarBuildSectorTask::generate_field_spherical(
		uint8_t sample_lod,
		const SectorGeometry &geom,
		ColumnField &out_field
) const {
	ZN_PROFILE_SCOPE();

	Ref<VoxelGenerator> generator = _params->generator;
	if (generator.is_null()) {
		return false;
	}

	const float step = static_cast<float>(1 << sample_lod);

	// Snap the radial range to the sample stride for the same reason the flat
	// path snaps the vertical range: a level whose samples sit off the grid
	// every other level uses will disagree with them about where the surface is.
	const float r_min = std::floor(_params->radial_min / step) * step;
	const float r_max = std::ceil(_params->radial_max / step) * step;

	int height_samples = static_cast<int>((r_max - r_min) / step) + 1;
	if (height_samples < 2) {
		height_samples = 2;
	}
	constexpr int MAX_HEIGHT_SAMPLES = 4096;
	if (height_samples > MAX_HEIGHT_SAMPLES) {
		height_samples = MAX_HEIGHT_SAMPLES;
	}
	const unsigned int height = static_cast<unsigned int>(height_samples);

	std::vector<float> sdf(static_cast<size_t>(SECTOR_RES) * SECTOR_RES * height);
	std::vector<uint16_t> material(sdf.size(), 0);

	SdfSlab slab;
	slab.sdf = sdf.data();
	slab.material = material.data();
	slab.res_xz = SECTOR_RES;
	slab.height = height;
	// The extractor reports crossings as `origin_y + i * step`, which on a
	// sphere reads as a radius from the planet centre rather than a world Y.
	slab.origin_y = r_min;
	slab.step = step;

	const bool use_series = generator->supports_series_generation();

	// One batch per row of columns. The slab is column-major, so a whole row of
	// constant z is contiguous in the output and can be written in place with no
	// second copy. Per-column batches would multiply the call overhead by 33 for
	// no benefit; the whole sector at once would need a several-megabyte scratch
	// per worker thread.
	const size_t row_samples = static_cast<size_t>(SECTOR_RES) * height;
	std::vector<float> px;
	std::vector<float> py;
	std::vector<float> pz;
	if (use_series) {
		px.resize(row_samples);
		py.resize(row_samples);
		pz.resize(row_samples);
	}

	VoxelBuffer point_buffer(VoxelBuffer::ALLOCATOR_POOL);
	if (!use_series) {
		point_buffer.create(1, 1, 1);
	}

	for (unsigned int z = 0; z < SECTOR_RES; ++z) {
		//  is non-const (it overrides IThreadedTask), and this
		// sampler is const like its flat counterpart.
		if (_shared == nullptr || _shared->is_stale(_generation)) {
			return false;
		}

		Vec3f row_min(1e30f, 1e30f, 1e30f);
		Vec3f row_max(-1e30f, -1e30f, -1e30f);
		size_t n = 0;

		for (unsigned int x = 0; x < SECTOR_RES; ++x) {
			const Vec3f dir = geom.dir[geom.column_index(x, z)];
			for (unsigned int y = 0; y < height; ++y) {
				const float r = r_min + static_cast<float>(y) * step;
				const Vec3f p = geom.planet_centre + dir * r;

				if (use_series) {
					px[n] = p.x;
					py[n] = p.y;
					pz[n] = p.z;
					row_min = Vec3f(std::min(row_min.x, p.x), std::min(row_min.y, p.y), std::min(row_min.z, p.z));
					row_max = Vec3f(std::max(row_max.x, p.x), std::max(row_max.y, p.y), std::max(row_max.z, p.z));
				} else {
					VoxelGenerator::VoxelQueryData q{
						point_buffer,
						Vector3i(static_cast<int32_t>(std::floor(p.x)), static_cast<int32_t>(std::floor(p.y)),
								static_cast<int32_t>(std::floor(p.z))),
						0,
					};
					generator->generate_block(q);
					sdf[slab.index(x, y, z)] = point_buffer.get_voxel_f(0, 0, 0, VoxelBuffer::CHANNEL_SDF);
				}
				++n;
			}
		}

		if (use_series) {
			// The row is contiguous from column 0's first sample.
			float *dst = sdf.data() + slab.index(0, 0, z);
			generator->generate_series(
					to_span(px),
					to_span(py),
					to_span(pz),
					VoxelBuffer::CHANNEL_SDF,
					Span<float>(dst, row_samples),
					Vector3f(row_min.x, row_min.y, row_min.z),
					Vector3f(row_max.x, row_max.y, row_max.z)
			);
		}
	}

	ExtractSettings extract = _params->extract;
	extract.min_gap = step * 0.25f;
	extract.min_thickness = step * 0.1f;

	std::vector<ColumnStack> stacks(SECTOR_COLUMN_COUNT);
	extract_columns_from_sdf(slab, extract, stacks.data());

	// Materials, sampled only at the layer boundaries that actually store one.
	//
	// The flat path reads a material alongside every SDF sample because it has
	// the whole block in hand anyway. Here a series call returns one channel,
	// and MIXEL4 -- which is what this project's planet generators write -- is a
	// packed index/weight pair that cannot be expressed as a float array at all.
	//
	// So materials come from `generate_block` on a reused 1x1x1 buffer instead,
	// at each layer's top and bottom. That sounds expensive and is not: a sector
	// has ~1100 columns with one or two layers each, against ~270k SDF samples,
	// so this is well under 1% of the work the sector already does.
	const MaterialSource material_source = choose_material_source(**generator);
	if (material_source != MATERIAL_NONE) {
		VoxelBuffer material_buffer(VoxelBuffer::ALLOCATOR_POOL);
		material_buffer.create(1, 1, 1);

		for (unsigned int z = 0; z < SECTOR_RES; ++z) {
			if (_shared == nullptr || _shared->is_stale(_generation)) {
				return false;
			}
			for (unsigned int x = 0; x < SECTOR_RES; ++x) {
				const Vec3f dir = geom.dir[geom.column_index(x, z)];
				ColumnStack &stack = stacks[geom.column_index(x, z)];

				for (unsigned int i = 0; i < stack.count; ++i) {
					// Just inside the surface, by half a *voxel* rather than half
					// a far-field cell. Generators commonly only compute real
					// materials in a narrow band around the surface and return a
					// default elsewhere -- EdenPlanetGeneratorV4's band is 28
					// units at LOD 0 -- so sampling half a cell down (32 units
					// at LOD 6) lands outside it and reads back the default for
					// every column.
					const float r = stack.layers[i].top - 0.5f;
					const Vec3f p = geom.planet_centre + dir * r;

					VoxelGenerator::VoxelQueryData q{
						material_buffer,
						Vector3i(
								static_cast<int32_t>(std::floor(p.x)),
								static_cast<int32_t>(std::floor(p.y)),
								static_cast<int32_t>(std::floor(p.z))
						),
						0,
					};
					generator->generate_block(q);
					stack.layers[i].material = read_material(material_buffer, material_source, 0, 0, 0);
				}
			}
		}
	}

	// Rotate the extracted normals out of slab space. The extractor takes its
	// gradient from index-space neighbours, which on a flat world are already
	// world axes; here they are the column's surface frame. Skipping this leaves
	// every normal pointing somewhere arbitrary.
	rotate_stack_normals_to_world(stacks.data(), geom);

	out_field.build_from_stacks(stacks.data());
	return true;
}

void FarBuildSectorTask::build_geometry(const SectorId &id, SectorGeometry &out) const {
	if (_params->spherical) {
		const float reference_radius = (_params->radial_min + _params->radial_max) * 0.5f;
		out.build_spherical(
				id,
				sectors_per_face_side(id.lod, _params->sphere_root_lod),
				_params->planet_centre,
				reference_radius
		);
		// Cell size on a sphere is the arc a cell spans, which is what every
		// tolerance in the extractor and mesher is expressed in.
		const float patch_arc = arc_length_of_sector(id, _params->sphere_root_lod, reference_radius);
		out.cell_size = patch_arc / static_cast<float>(SECTOR_CELLS);
	} else {
		out.build_flat(id, id.get_cell_size());
	}
}

bool FarBuildSectorTask::generate_field(uint8_t sample_lod, float world_x, float world_z, ColumnField &out_field)
		const {
	ZN_PROFILE_SCOPE();

	Ref<VoxelGenerator> generator = _params->generator;
	if (generator.is_null()) {
		return false;
	}

	const int step = 1 << sample_lod;

	// Snap the vertical range to the sample stride. `origin_in_voxels` is in
	// LOD-0 units, and an origin that is not a multiple of the stride would put
	// this level's samples off the grid that every other level uses -- which is
	// precisely the registration that keeps levels from disagreeing.
	const int32_t min_y = static_cast<int32_t>(std::floor(_params->vertical_min / step)) * step;
	const int32_t max_y = static_cast<int32_t>(std::ceil(_params->vertical_max / step)) * step;

	const int span = max_y - min_y;
	int height_samples = span / step + 1;
	if (height_samples < 2) {
		height_samples = 2;
	}
	// Guard against a pathological vertical range at a fine level allocating an
	// enormous buffer. Callers are expected to set `first_lod` so this does not
	// bite, but a mis-set range should degrade rather than exhaust memory.
	constexpr int MAX_HEIGHT_SAMPLES = 4096;
	if (height_samples > MAX_HEIGHT_SAMPLES) {
		height_samples = MAX_HEIGHT_SAMPLES;
	}

	VoxelBuffer buffer(VoxelBuffer::ALLOCATOR_POOL);
	buffer.create(SECTOR_RES, static_cast<unsigned int>(height_samples), SECTOR_RES);

	VoxelGenerator::VoxelQueryData query{
		buffer,
		Vector3i(static_cast<int32_t>(world_x), min_y, static_cast<int32_t>(world_z)),
		sample_lod,
	};
	generator->generate_block(query);

	// Transpose into the column-major layout the extractor wants. The extractor
	// walks a column top to bottom, so Y contiguous turns its inner loop into a
	// linear scan. Paying for the transpose once here is cheaper than a strided
	// walk over every column.
	std::vector<float> sdf(static_cast<size_t>(SECTOR_RES) * SECTOR_RES * height_samples);
	std::vector<uint16_t> material(sdf.size(), 0);

	SdfSlab slab;
	slab.sdf = sdf.data();
	slab.material = material.data();
	slab.res_xz = SECTOR_RES;
	slab.height = static_cast<unsigned int>(height_samples);
	slab.origin_y = static_cast<float>(min_y);
	slab.step = static_cast<float>(step);

	const MaterialSource material_source = choose_material_source(**generator);

	for (unsigned int z = 0; z < SECTOR_RES; ++z) {
		for (unsigned int x = 0; x < SECTOR_RES; ++x) {
			for (int y = 0; y < height_samples; ++y) {
				const size_t dst = slab.index(x, static_cast<unsigned int>(y), z);
				sdf[dst] = buffer.get_voxel_f(
						static_cast<int>(x), y, static_cast<int>(z), VoxelBuffer::CHANNEL_SDF
				);
				if (material_source != MATERIAL_NONE) {
					material[dst] =
							read_material(buffer, material_source, static_cast<int>(x), y, static_cast<int>(z));
				}
			}
		}
	}

	ExtractSettings extract = _params->extract;
	// Tolerances scale with the cell size: what counts as a negligible gap at
	// one unit per cell is not negligible at 64.
	extract.min_gap = static_cast<float>(step) * 0.25f;
	extract.min_thickness = static_cast<float>(step) * 0.1f;

	std::vector<ColumnStack> stacks(SECTOR_COLUMN_COUNT);
	extract_columns_from_sdf(slab, extract, stacks.data());
	out_field.build_from_stacks(stacks.data());

	return true;
}

void FarBuildSectorTask::run(ThreadedTaskContext &ctx) {
	ZN_PROFILE_SCOPE();

	if (is_cancelled()) {
		return;
	}

	// Built once and shared by generation, extraction and meshing: they all need
	// to agree on where this sector's columns are.
	auto geom = std::make_shared<SectorGeometry>();
	build_geometry(_id, *geom);

	auto field = std::make_shared<ColumnField>();
	bool have_field = false;

	// -- Stage 1: cache ----------------------------------------------------
	std::vector<uint8_t> bytes;
	if (_params->cache != nullptr && _params->cache->is_open()) {
		if (_params->cache->load(_id, bytes)) {
			have_field = field->deserialize(bytes.data(), bytes.size());
			// A failed deserialize is treated as a miss, not an error. The
			// format version may have moved on, or the file may be damaged;
			// either way regenerating is correct.
		}
	}

	// -- Stage 2: generate --------------------------------------------------
	if (!have_field) {
		if (is_cancelled()) {
			return;
		}

		const float world_x = _id.get_world_x();
		const float world_z = _id.get_world_z();

		// Antialiasing samples the four children one level finer and reduces.
		// On a sphere a child is a cube-face sub-patch rather than an XZ
		// quadrant, so it goes through the same geometry builder.
		const bool antialias = _params->antialias && _id.lod > 0 && _id.lod <= _params->antialias_max_lod;

		if (antialias && _params->spherical) {
			ColumnField children[4];
			const ColumnField *child_ptrs[4] = { nullptr, nullptr, nullptr, nullptr };

			bool all_ok = true;
			for (unsigned int q = 0; q < 4 && all_ok; ++q) {
				SectorGeometry child_geom;
				build_geometry(_id.get_child(q), child_geom);
				all_ok = generate_field_spherical(static_cast<uint8_t>(_id.lod - 1), child_geom, children[q]);
				child_ptrs[q] = &children[q];
				if (is_cancelled()) {
					return;
				}
			}

			if (all_ok) {
				downsample_field(child_ptrs, geom->cell_size, *field);
				have_field = true;
			}
		} else if (_params->spherical) {
			// no-op: handled below
		} else if (antialias) {
			// Sample the four quadrants one level finer, then reduce. Costs 4x
			// the generator evaluations and removes the level-to-level popping
			// that direct sampling produces on narrow features.
			const uint8_t child_lod = static_cast<uint8_t>(_id.lod - 1);
			const float child_size = static_cast<float>(SECTOR_CELLS) * static_cast<float>(1u << child_lod);

			ColumnField children[4];
			const ColumnField *child_ptrs[4] = { nullptr, nullptr, nullptr, nullptr };

			bool all_ok = true;
			for (unsigned int q = 0; q < 4 && all_ok; ++q) {
				const float cx = world_x + ((q & 1u) ? child_size : 0.f);
				const float cz = world_z + ((q >> 1) ? child_size : 0.f);
				all_ok = generate_field(child_lod, cx, cz, children[q]);
				child_ptrs[q] = &children[q];
				if (is_cancelled()) {
					return;
				}
			}

			if (all_ok) {
				downsample_field(child_ptrs, _id.get_cell_size(), *field);
				have_field = true;
			}
		}

		if (!have_field) {
			have_field = _params->spherical ? generate_field_spherical(_id.lod, *geom, *field)
											: generate_field(_id.lod, world_x, world_z, *field);
		}

		// -- Write back to the cache ---------------------------------------
		if (have_field && _params->cache != nullptr && _params->cache->is_open()) {
			bytes.clear();
			field->serialize(bytes);
			_params->cache->save(_id, bytes);
		}
	}

	if (!have_field || is_cancelled()) {
		return;
	}

	// -- Stage 3: mesh ------------------------------------------------------
	auto mesh = std::make_shared<FarMeshArrays>();
	if (!field->is_empty()) {
		MesherSettings mesher = _params->mesher;
		mesher.cell_size = geom->cell_size;
		mesher.geometry = geom.get();
		build_far_mesh(*field, mesher, *mesh);
	}

	if (is_cancelled()) {
		return;
	}

	// -- Hand off to the main thread ---------------------------------------
	FarSectorResult result;
	result.id = _id;
	result.field = std::move(field);
	result.mesh = std::move(mesh);
	result.local_origin = geom->local_origin;
	result.valid = true;
	_shared->results.push(std::move(result));

	ctx.status = ThreadedTaskContext::STATUS_COMPLETE;
}

} // namespace zylann::voxel::far
