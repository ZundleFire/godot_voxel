#include "test_surface_nets.h"
#include "../../constants/cube_tables.h"
#include "../../meshers/surface_nets/surface_nets.h"
#include "../../meshers/surface_nets/voxel_mesher_surface_nets.h"
#include "../../storage/mixel4.h"
#include "../../util/godot/classes/rendering_server.h"
#include "../../util/testing/test_macros.h"

namespace zylann::voxel::tests {

namespace {

VoxelBuffer make_sphere_buffer(int size, float radius) {
	VoxelBuffer voxels(VoxelBuffer::ALLOCATOR_DEFAULT);
	voxels.create(Vector3iUtil::create(size));
	const Vector3 center(size * 0.5f, size * 0.5f, size * 0.5f);
	Vector3i pos;
	for (pos.z = 0; pos.z < size; ++pos.z) {
		for (pos.x = 0; pos.x < size; ++pos.x) {
			for (pos.y = 0; pos.y < size; ++pos.y) {
				const float d = Vector3(pos.x, pos.y, pos.z).distance_to(center) - radius;
				voxels.set_voxel_f(d, pos, VoxelBuffer::CHANNEL_SDF);
			}
		}
	}
	return voxels;
}

VoxelBuffer make_flat_plane_buffer(int size, float plane_height) {
	VoxelBuffer voxels(VoxelBuffer::ALLOCATOR_DEFAULT);
	voxels.create(Vector3iUtil::create(size));
	Vector3i pos;
	for (pos.z = 0; pos.z < size; ++pos.z) {
		for (pos.x = 0; pos.x < size; ++pos.x) {
			for (pos.y = 0; pos.y < size; ++pos.y) {
				voxels.set_voxel_f(float(pos.y) - plane_height, pos, VoxelBuffer::CHANNEL_SDF);
			}
		}
	}
	return voxels;
}

} // namespace

// A block must not emit geometry meaningfully outside its own volume. Dual contouring inherently
// places seam vertices up to one cell back (the neighbour block emits a coincident one, which is
// what makes the surface watertight), but anything beyond that means quads are being emitted for
// corners this block does not own -- which duplicates the neighbour's geometry and, at coarse LODs,
// hangs a large skirt into finer-LOD blocks.
void test_surface_nets_block_bounds() {
	const int buffer_size = 20;
	VoxelBuffer voxels = make_flat_plane_buffer(buffer_size, 9.5f);
	const int block_size = buffer_size - (surface_nets::MIN_PADDING + surface_nets::MAX_PADDING);

	surface_nets::MeshArrays mesh;
	surface_nets::build_regular_mesh(voxels, VoxelBuffer::CHANNEL_SDF, 0, mesh);

	ZN_TEST_ASSERT(mesh.indices.size() > 0);

	for (size_t i = 0; i < mesh.indices.size(); ++i) {
		const Vector3f v = mesh.vertices[mesh.indices[i]];
		for (unsigned int axis = 0; axis < 3; ++axis) {
			ZN_TEST_ASSERT(v[axis] >= -1.f);
			ZN_TEST_ASSERT(v[axis] <= float(block_size));
		}
	}
}

// VoxelLodTerrain positions mesh blocks with a translation-only transform, so the mesher must emit
// positions in LOD0 voxel units -- scaled by 2^lod_index. Without this, blocks at LOD n render at
// 1/2^n of their true size, bunched around the block origin: huge holes, and a flat surface landing
// at a different height in every LOD ring.
void test_surface_nets_lod_scaling() {
	const int buffer_size = 20;
	const float plane_height = 9.5f;
	VoxelBuffer voxels = make_flat_plane_buffer(buffer_size, plane_height);

	surface_nets::MeshArrays lod0;
	surface_nets::build_regular_mesh(voxels, VoxelBuffer::CHANNEL_SDF, 0, lod0);

	for (unsigned int lod_index = 1; lod_index < 4; ++lod_index) {
		surface_nets::MeshArrays lod_n;
		surface_nets::build_regular_mesh(voxels, VoxelBuffer::CHANNEL_SDF, lod_index, lod_n);

		// Same voxel data means the same topology; only the scale of the positions should differ.
		ZN_TEST_ASSERT(lod_n.vertices.size() == lod0.vertices.size());
		ZN_TEST_ASSERT(lod_n.indices.size() == lod0.indices.size());

		const float expected_scale = float(1 << lod_index);
		for (size_t i = 0; i < lod0.vertices.size(); ++i) {
			const Vector3f expected = lod0.vertices[i] * expected_scale;
			const Vector3f got = lod_n.vertices[i];
			for (unsigned int axis = 0; axis < 3; ++axis) {
				ZN_TEST_ASSERT(Math::abs(got[axis] - expected[axis]) < 0.001f);
			}
		}
	}
}

void test_surface_nets_regular_mesh() {
	VoxelBuffer voxels = make_sphere_buffer(16, 7.5f);

	surface_nets::MeshArrays mesh;
	surface_nets::build_regular_mesh(voxels, VoxelBuffer::CHANNEL_SDF, 0, mesh);

	ZN_TEST_ASSERT(mesh.vertices.size() > 0);
	ZN_TEST_ASSERT(mesh.indices.size() > 0);
	ZN_TEST_ASSERT(mesh.indices.size() % 3 == 0);
	ZN_TEST_ASSERT(mesh.normals.size() == mesh.vertices.size());
	for (size_t i = 0; i < mesh.indices.size(); ++i) {
		ZN_TEST_ASSERT(mesh.indices[i] >= 0 && mesh.indices[i] < int32_t(mesh.vertices.size()));
	}
}

void test_surface_nets_skirts() {
	// A sphere's mesh is an open shell inside the block, so it has a boundary rim for the skirt to
	// hang from.
	VoxelBuffer voxels = make_sphere_buffer(16, 7.5f);

	surface_nets::MeshArrays mesh;
	surface_nets::build_regular_mesh(voxels, VoxelBuffer::CHANNEL_SDF, 0, mesh);
	const size_t vertices_before = mesh.vertices.size();
	const size_t indices_before = mesh.indices.size();
	ZN_TEST_ASSERT(indices_before > 0);

	surface_nets::build_skirts(mesh, 0, 2.f);

	// Skirts only append; the surface itself must come through untouched so the collision submesh
	// range recorded before this call stays valid.
	ZN_TEST_ASSERT(mesh.vertices.size() > vertices_before);
	ZN_TEST_ASSERT(mesh.indices.size() > indices_before);
	ZN_TEST_ASSERT(mesh.indices.size() % 3 == 0);
	ZN_TEST_ASSERT(mesh.normals.size() == mesh.vertices.size());
	ZN_TEST_ASSERT(mesh.transition_tag.size() == mesh.vertices.size());
	for (size_t i = 0; i < mesh.indices.size(); ++i) {
		ZN_TEST_ASSERT(mesh.indices[i] >= 0 && mesh.indices[i] < int32_t(mesh.vertices.size()));
	}

	// Every added vertex is one of the originals pushed along -normal by exactly the requested depth,
	// which is what keeps the curtain inside the matter instead of poking through the surface.
	for (size_t i = vertices_before; i < mesh.vertices.size(); ++i) {
		const float len = math::length(mesh.normals[i]);
		ZN_TEST_ASSERT(Math::is_equal_approx(len, 1.f));
	}

	// Running it again finds no open edges left on the parts it already closed, so it must not keep
	// growing the mesh without bound.
	const size_t indices_after_first = mesh.indices.size();
	surface_nets::build_skirts(mesh, 0, 2.f);
	ZN_TEST_ASSERT(mesh.indices.size() < indices_after_first * 2);

	// A depth of zero is the documented way to turn skirts off.
	surface_nets::MeshArrays plain;
	surface_nets::build_regular_mesh(voxels, VoxelBuffer::CHANNEL_SDF, 0, plain);
	const size_t plain_indices = plain.indices.size();
	surface_nets::build_skirts(plain, 0, 0.f);
	ZN_TEST_ASSERT(plain.indices.size() == plain_indices);
}

void test_surface_nets_material_output() {
	// The voxel graph's MaterialOutput reaches the screen through CHANNEL_INDICES/CHANNEL_WEIGHTS,
	// which the mesher has to forward into ARRAY_CUSTOM1. When it doesn't, every vertex reports zero
	// weights and the auto material shader paints the whole terrain with layer 0.
	VoxelBuffer voxels = make_sphere_buffer(16, 7.5f);
	voxels.set_channel_depth(VoxelBuffer::CHANNEL_INDICES, VoxelBuffer::DEPTH_16_BIT);
	voxels.set_channel_depth(VoxelBuffer::CHANNEL_WEIGHTS, VoxelBuffer::DEPTH_16_BIT);

	// Two layers, split down the middle, so the result can't be mistaken for a uniform fallback.
	// Indices 0..3 specifically: the texture selection keeps the 4 highest-weighted layers and then
	// sorts them ascending, so with these the vertex slots line up with the layer numbers and the
	// assertions below can talk about slot 0 and slot 1 directly.
	const uint16_t encoded_indices = mixel4::encode_indices_to_packed_u16(0, 1, 2, 3);
	const uint16_t weights_a = mixel4::encode_weights_to_packed_u16_lossy(255, 0, 0, 0);
	const uint16_t weights_b = mixel4::encode_weights_to_packed_u16_lossy(0, 255, 0, 0);
	Vector3i pos;
	for (pos.z = 0; pos.z < 16; ++pos.z) {
		for (pos.x = 0; pos.x < 16; ++pos.x) {
			for (pos.y = 0; pos.y < 16; ++pos.y) {
				voxels.set_voxel(encoded_indices, pos, VoxelBuffer::CHANNEL_INDICES);
				voxels.set_voxel(pos.x < 8 ? weights_a : weights_b, pos, VoxelBuffer::CHANNEL_WEIGHTS);
			}
		}
	}

	Ref<VoxelMesherSurfaceNets> mesher;
	mesher.instantiate();
	ZN_TEST_ASSERT(mesher->get_texturing_mode() == VoxelMesherSurfaceNets::TEXTURES_MIXEL4_S4);
	// Without these channels in the mask the data layer never fills them in the first place.
	const int mask = mesher->get_used_channels_mask();
	ZN_TEST_ASSERT((mask & (1 << VoxelBuffer::CHANNEL_INDICES)) != 0);
	ZN_TEST_ASSERT((mask & (1 << VoxelBuffer::CHANNEL_WEIGHTS)) != 0);

	VoxelMesher::Output output;
	mesher->build(output, VoxelMesher::Input{ voxels, nullptr, Vector3i(), 0, false, false, false });
	ZN_TEST_ASSERT(output.surfaces.size() == 1);

	const Array arrays = output.surfaces[0].arrays;
	const PackedVector3Array vertices = arrays[Mesh::ARRAY_VERTEX];
	const PackedFloat32Array custom1 = arrays[Mesh::ARRAY_CUSTOM1];
	ZN_TEST_ASSERT(vertices.size() > 0);
	// 2 floats per vertex: packed indices, then packed weights.
	ZN_TEST_ASSERT(custom1.size() == vertices.size() * 2);
	ZN_TEST_ASSERT(
			(output.mesh_flags &
			 (::RenderingServerEnums::ARRAY_CUSTOM_RG_FLOAT << Mesh::ARRAY_FORMAT_CUSTOM1_SHIFT)) != 0
	);

	// Both halves of the sphere must show up, and no vertex may end up with all-zero weights (which
	// is what the shader reads as "no data" and renders as layer 0).
	bool saw_first_layer = false;
	bool saw_second_layer = false;
	for (int i = 0; i < vertices.size(); ++i) {
		uint32_t packed_indices;
		uint32_t packed_weights;
		const float fi = custom1[i * 2 + 0];
		const float w = custom1[i * 2 + 1];
		memcpy(&packed_indices, &fi, sizeof(uint32_t));
		memcpy(&packed_weights, &w, sizeof(uint32_t));
		ZN_TEST_ASSERT(packed_weights != 0);
		// Layers 0..3 in ascending slot order, as set up above.
		ZN_TEST_ASSERT(packed_indices == 0x03020100u);
		const uint32_t w0 = packed_weights & 0xff;
		const uint32_t w1 = (packed_weights >> 8) & 0xff;
		if (w0 > w1) {
			saw_first_layer = true;
		} else if (w1 > w0) {
			saw_second_layer = true;
		}
	}
	ZN_TEST_ASSERT(saw_first_layer);
	ZN_TEST_ASSERT(saw_second_layer);
}

} // namespace zylann::voxel::tests
