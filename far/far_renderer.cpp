#include "far_renderer.h"

#include "../util/godot/classes/mesh.h"
#include "../util/godot/classes/world_3d.h"
#include "../util/godot/core/array.h"
#include "../util/godot/core/packed_arrays.h"
#include "../util/profiling.h"

#include <algorithm>

namespace zylann::voxel::far {

using namespace zylann::godot;

FarRenderer::~FarRenderer() {
	destroy_all();
}

void FarRenderer::set_world(World3D *world) {
	_world = world;
	for (auto &pair : _sectors) {
		// Sectors rendered through the GPU-driven path have no mesh instance
		if (pair.second.mesh_instance.is_valid()) {
			pair.second.mesh_instance.set_world(world);
		}
	}
}

void FarRenderer::set_material(Ref<Material> material) {
	_material = material;
	for (auto &pair : _sectors) {
		if (pair.second.mesh_instance.is_valid()) {
			pair.second.mesh_instance.set_material_override(_material);
		}
	}
}

void FarRenderer::set_cast_shadows(bool enabled) {
	if (_cast_shadows == enabled) {
		return;
	}
	_cast_shadows = enabled;
	const RenderingServerEnums::ShadowCastingSetting setting = enabled
			? RenderingServerEnums::SHADOW_CASTING_SETTING_ON
			: RenderingServerEnums::SHADOW_CASTING_SETTING_OFF;
	for (auto &pair : _sectors) {
		if (pair.second.mesh_instance.is_valid()) {
			pair.second.mesh_instance.set_cast_shadows_setting(setting);
		}
	}
}

void FarRenderer::set_render_layers_mask(int mask) {
	if (_render_layers_mask == mask) {
		return;
	}
	_render_layers_mask = mask;
	for (auto &pair : _sectors) {
		if (pair.second.mesh_instance.is_valid()) {
			pair.second.mesh_instance.set_render_layers_mask(mask);
		}
	}
}

void FarRenderer::set_visible(bool visible) {
	_visible = visible;
	for (auto &pair : _sectors) {
		if (pair.second.mesh_instance.is_valid()) {
			pair.second.mesh_instance.set_visible(visible && pair.second.visible);
		}
#ifdef VOXEL_ENABLE_GPU_DRIVEN_RENDERING
		if (pair.second.gpu_chunk != VoxelGpuDrivenRenderer::INVALID_CHUNK && _gpu_renderer != nullptr) {
			_gpu_renderer->set_chunk_visible(pair.second.gpu_chunk, visible && pair.second.visible);
		}
#endif
	}
}

#ifdef VOXEL_ENABLE_GPU_DRIVEN_RENDERING

namespace {

// Far vertices into the GPU-driven layout. They carry no Transvoxel seam data, so those fields are free: the
// baked AO goes where the secondary position would be, and the material index becomes a single MIXEL4 slot at
// full weight, so far sectors shade with the same palette as near terrain.
gpu_driven::PackedMesh pack_far_mesh(const FarMeshArrays &arrays, unsigned int &r_max_material) {
	ZN_PROFILE_SCOPE();
	gpu_driven::PackedMesh packed;
	packed.vertices.resize(arrays.vertices.size());

	for (size_t i = 0; i < arrays.vertices.size(); ++i) {
		const FarVertex &src = arrays.vertices[i];
		gpu_driven::PackedVertex &dst = packed.vertices[i];
		dst.position[0] = src.position.x;
		dst.position[1] = src.position.y;
		dst.position[2] = src.position.z;
		dst.normal_flags = gpu_driven::encode_oct14(src.normal.x, src.normal.y, src.normal.z) << 18;
		dst.secondary_xy = 0;
		const float ao = math::clamp(src.ao, 0.f, 1.f);
		dst.secondary_z = static_cast<uint32_t>(std::lround(ao * 65535.f)) << 16;
		const uint32_t material = static_cast<uint32_t>(math::clamp(src.material, 0.f, 255.f));
		r_max_material = math::max(r_max_material, material);
		dst.materials = math::min(material, gpu_driven::MAX_MATERIAL_INDEX) | (15u << 4);
		dst.surface = 0;
	}

	packed.indices.assign(arrays.indices.begin(), arrays.indices.end());
	if (arrays.has_bounds) {
		packed.aabb_min[0] = arrays.aabb_min.x;
		packed.aabb_min[1] = arrays.aabb_min.y;
		packed.aabb_min[2] = arrays.aabb_min.z;
		packed.aabb_max[0] = arrays.aabb_max.x;
		packed.aabb_max[1] = arrays.aabb_max.y;
		packed.aabb_max[2] = arrays.aabb_max.z;
	} else {
		for (unsigned int c = 0; c < 3; ++c) {
			packed.aabb_min[c] = std::numeric_limits<float>::max();
			packed.aabb_max[c] = std::numeric_limits<float>::lowest();
		}
		for (const gpu_driven::PackedVertex &v : packed.vertices) {
			for (unsigned int c = 0; c < 3; ++c) {
				packed.aabb_min[c] = math::min(packed.aabb_min[c], v.position[c]);
				packed.aabb_max[c] = math::max(packed.aabb_max[c], v.position[c]);
			}
		}
	}
	gpu_driven::finalize(packed);
	return packed;
}

} // namespace

#endif // VOXEL_ENABLE_GPU_DRIVEN_RENDERING

void FarRenderer::release_sector(SectorRender &sector) {
#ifdef VOXEL_ENABLE_GPU_DRIVEN_RENDERING
	if (sector.gpu_chunk != VoxelGpuDrivenRenderer::INVALID_CHUNK) {
		if (_gpu_renderer != nullptr) {
			_gpu_renderer->remove_chunk(sector.gpu_chunk);
		}
		sector.gpu_chunk = VoxelGpuDrivenRenderer::INVALID_CHUNK;
	}
#endif
	sector.mesh_instance.destroy();
}

Ref<ArrayMesh> FarRenderer::build_mesh_resource(const FarMeshArrays &arrays) {
	ZN_PROFILE_SCOPE();

	const size_t vertex_count = arrays.vertices.size();
	if (vertex_count == 0 || arrays.indices.empty()) {
		return Ref<ArrayMesh>();
	}

	PackedVector3Array positions;
	PackedVector3Array normals;
	PackedColorArray colors;
	PackedInt32Array indices;

	positions.resize(static_cast<int>(vertex_count));
	normals.resize(static_cast<int>(vertex_count));
	colors.resize(static_cast<int>(vertex_count));
	indices.resize(static_cast<int>(arrays.indices.size()));

	{
		Vector3 *positions_w = positions.ptrw();
		Vector3 *normals_w = normals.ptrw();
		Color *colors_w = colors.ptrw();

		for (size_t i = 0; i < vertex_count; ++i) {
			const FarVertex &v = arrays.vertices[i];
			positions_w[i] = Vector3(v.position.x, v.position.y, v.position.z);
			normals_w[i] = Vector3(v.normal.x, v.normal.y, v.normal.z);
			// Vertex colour carries the baked shading terms rather than a
			// colour: red is ambient occlusion, green is the material index
			// normalised to 0..1 for the shader to look up in a palette.
			colors_w[i] = Color(v.ao, v.material / 255.f, 0.f, 1.f);
			_max_material = std::max(_max_material, static_cast<unsigned int>(v.material));
		}
	}

	{
		int32_t *indices_w = indices.ptrw();
		for (size_t i = 0; i < arrays.indices.size(); ++i) {
			indices_w[i] = static_cast<int32_t>(arrays.indices[i]);
		}
	}

	Array surface;
	surface.resize(Mesh::ARRAY_MAX);
	surface[Mesh::ARRAY_VERTEX] = positions;
	surface[Mesh::ARRAY_NORMAL] = normals;
	surface[Mesh::ARRAY_COLOR] = colors;
	surface[Mesh::ARRAY_INDEX] = indices;

	Ref<ArrayMesh> mesh;
	mesh.instantiate();

	// Let Godot pack the vertex attributes. Normals drop to an octahedral pair
	// and colours to bytes, roughly halving the per-vertex footprint and the
	// bandwidth to match.
	//
	// Safe here specifically because vertices are sector-local: the quantisation
	// is relative to the mesh AABB, which spans one sector rather than the whole
	// world. The same flag on world-space far terrain would be destructive.
	const uint32_t surface_flags = Mesh::ARRAY_FLAG_COMPRESS_ATTRIBUTES;

	mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, surface, Array(), Dictionary(), surface_flags);

	// Supplying the AABB saves Godot a pass over every vertex, and the mesher
	// already tracked it for free while emitting them.
	if (arrays.has_bounds) {
		const Vector3 min(arrays.aabb_min.x, arrays.aabb_min.y, arrays.aabb_min.z);
		const Vector3 max(arrays.aabb_max.x, arrays.aabb_max.y, arrays.aabb_max.z);
		mesh->set_custom_aabb(AABB(min, max - min));
	}

	return mesh;
}

void FarRenderer::drain_results(FarSharedState &shared, const FarLodTree &lod_tree,
		const Transform3D &volume_transform, unsigned int max_uploads) {
	ZN_PROFILE_SCOPE();

	static thread_local std::vector<FarSectorResult> tls_batch;
	tls_batch.clear();
	shared.results.pop_batch(tls_batch, std::max(max_uploads, 1u));

	for (FarSectorResult &result : tls_batch) {
		if (!result.valid) {
			continue;
		}

		// The sector may have been unloaded while its task was running.
		// Residency is the authority: if the LOD tree no longer wants it, the
		// result is stale and building a mesh for it would leave orphaned
		// geometry.
		if (lod_tree.get_resident().find(result.id) == lod_tree.get_resident().end()) {
			++_stale_results_discarded;
			continue;
		}

		if (result.mesh == nullptr || result.mesh->is_empty()) {
			// A sector can legitimately be empty (all air, or all solid with no
			// surface in range). Record it as present so it is not requested
			// again every frame.
			SectorRender &render = _sectors[result.id];
			render.mesh.unref();
			if (render.mesh_instance.is_valid()) {
				render.mesh_instance.set_mesh(Ref<Mesh>());
			}
#ifdef VOXEL_ENABLE_GPU_DRIVEN_RENDERING
			if (render.gpu_chunk != VoxelGpuDrivenRenderer::INVALID_CHUNK && _gpu_renderer != nullptr) {
				_gpu_renderer->set_chunk_mesh(render.gpu_chunk, gpu_driven::PackedMesh(), Vector3());
			}
			render.vertex_count = 0;
			render.triangle_count = 0;
#endif
			++_sectors_built;
			continue;
		}

#ifdef VOXEL_ENABLE_GPU_DRIVEN_RENDERING
		if (_gpu_renderer != nullptr) {
			const Vector3 local_origin(result.local_origin.x, result.local_origin.y, result.local_origin.z);
			SectorRender &render = _sectors[result.id];
			if (render.gpu_chunk == VoxelGpuDrivenRenderer::INVALID_CHUNK) {
				render.gpu_chunk = _gpu_renderer->create_chunk();
			}
			render.vertex_count = static_cast<uint32_t>(result.mesh->vertices.size());
			render.triangle_count = static_cast<uint32_t>(result.mesh->indices.size() / 3);
			// Sector-local vertices with the origin passed separately, same as the mesh instance transform below
			_gpu_renderer->set_chunk_mesh(
					render.gpu_chunk, pack_far_mesh(*result.mesh, _max_material), local_origin
			);
			_gpu_renderer->set_chunk_far(render.gpu_chunk, true);
			render.local_origin = local_origin;
			render.visible = true;
			_gpu_renderer->set_chunk_visible(render.gpu_chunk, _visible);
			++_sectors_built;
			continue;
		}
#endif

		Ref<ArrayMesh> mesh = build_mesh_resource(*result.mesh);
		if (mesh.is_null()) {
			continue;
		}

		SectorRender &render = _sectors[result.id];
		if (!render.mesh_instance.is_valid()) {
			render.mesh_instance.create();
			render.mesh_instance.set_world(_world);
			render.mesh_instance.set_cast_shadows_setting(_cast_shadows
							? RenderingServerEnums::SHADOW_CASTING_SETTING_ON
							: RenderingServerEnums::SHADOW_CASTING_SETTING_OFF);
			render.mesh_instance.set_render_layers_mask(_render_layers_mask);
		}

		// Vertices are sector-local; the sector's world offset goes in the
		// transform. At 200 km out, absolute float coordinates have a
		// resolution of around 16 units, which would make the far terrain
		// visibly jitter and crawl as the camera moves.
		// Where the task said its vertices are relative to: the sector's corner
		// on a flat world, the centre of its surface patch on a planet.
		const Vector3 local_origin(result.local_origin.x, result.local_origin.y, result.local_origin.z);
		Transform3D transform = volume_transform;
		transform.origin += volume_transform.basis.xform(local_origin);
		render.mesh_instance.set_transform(transform);

		render.mesh = mesh;
		render.local_origin = local_origin;
		render.mesh_instance.set_mesh(mesh);
		if (_material.is_valid()) {
			render.mesh_instance.set_material_override(_material);
		}

		render.visible = true;
		render.mesh_instance.set_visible(_visible);

		++_sectors_built;
	}

	tls_batch.clear();
}

void FarRenderer::reconcile(const FarLodTree &lod_tree) {
	ZN_PROFILE_SCOPE();
	const auto &resident = lod_tree.get_resident();
	for (auto it = _sectors.begin(); it != _sectors.end();) {
		if (resident.find(it->first) != resident.end()) {
			++it;
			continue;
		}
		release_sector(it->second);
		it = _sectors.erase(it);
	}
}

void FarRenderer::destroy_all() {
	for (auto &pair : _sectors) {
		release_sector(pair.second);
	}
	_sectors.clear();
}

void FarRenderer::refresh_visibility(const FarLodTree &lod_tree, const Vec3f &viewer_position) {
	// The near clip radius follows the viewer, so which sectors are hidden
	// changes as they move -- without anything being reloaded.
	if (lod_tree.get_settings().near_clip_radius <= 0.f) {
		return;
	}

	for (auto &pair : _sectors) {
		const bool should_be_visible = !lod_tree.is_clipped(pair.first, viewer_position);
		if (should_be_visible != pair.second.visible) {
			pair.second.visible = should_be_visible;
			if (pair.second.mesh_instance.is_valid()) {
				pair.second.mesh_instance.set_visible(_visible && should_be_visible);
			}
#ifdef VOXEL_ENABLE_GPU_DRIVEN_RENDERING
			if (pair.second.gpu_chunk != VoxelGpuDrivenRenderer::INVALID_CHUNK && _gpu_renderer != nullptr) {
				_gpu_renderer->set_chunk_visible(pair.second.gpu_chunk, _visible && should_be_visible);
			}
#endif
		}
	}
}

void FarRenderer::get_statistics(int64_t &out_vertices, int64_t &out_triangles, int64_t &out_empty,
		int64_t &out_clipped, AABB &out_bounds) const {
	// Walked on demand rather than tracked incrementally: this is a debug call,
	// and a stale counter would be worse than a slow one.
	out_vertices = 0;
	out_triangles = 0;
	out_empty = 0;
	out_clipped = 0;
	bool has_bounds = false;

	for (const auto &pair : _sectors) {
		if (!pair.second.visible) {
			++out_clipped;
		}
		const Ref<ArrayMesh> &mesh = pair.second.mesh;
		if (mesh.is_null()) {
			if (pair.second.vertex_count == 0) {
				++out_empty;
				continue;
			}
			// GPU-driven path: no Mesh resource, and bounds are tracked by the renderer's chunks
			out_vertices += pair.second.vertex_count;
			out_triangles += pair.second.triangle_count;
			continue;
		}
		const int surface_count = mesh->get_surface_count();
		for (int i = 0; i < surface_count; ++i) {
			out_vertices += mesh->surface_get_array_len(i);
			const int index_count = mesh->surface_get_array_index_len(i);
			out_triangles += (index_count > 0 ? index_count : mesh->surface_get_array_len(i)) / 3;
		}
		AABB sector_bounds = mesh->get_aabb();
		sector_bounds.position += pair.second.local_origin;
		if (has_bounds) {
			out_bounds = out_bounds.merge(sector_bounds);
		} else {
			out_bounds = sector_bounds;
			has_bounds = true;
		}
	}
}

} // namespace zylann::voxel::far
