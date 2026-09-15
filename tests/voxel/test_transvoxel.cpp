#include "test_transvoxel.h"
#include "../../meshers/transvoxel/voxel_mesher_transvoxel.h"
#include "../../storage/mixel4.h"
#include "../../util/testing/test_macros.h"

namespace zylann::voxel::tests {

void test_transvoxel_issue772() {
	// There was a wrong assertion check on the values of component indices when texturing mode is SINGLE_S4

	VoxelBuffer voxels(VoxelBuffer::ALLOCATOR_DEFAULT);
	voxels.set_channel_depth(VoxelBuffer::CHANNEL_INDICES, VoxelBuffer::DEPTH_8_BIT);
	voxels.create(Vector3iUtil::create(8));
	{
		Vector3i pos;

		const float h = voxels.get_size().y / 2.f + 0.1f;

		for (pos.z = 0; pos.z < voxels.get_size().z; ++pos.z) {
			for (pos.x = 0; pos.x < voxels.get_size().x; ++pos.x) {
				for (pos.y = 0; pos.y < voxels.get_size().y; ++pos.y) {
					const float gy = pos.y;
					const float sd = gy - h;
					voxels.set_voxel_f(sd, pos, VoxelBuffer::CHANNEL_SDF);
					if (sd < 1.f) {
						const uint8_t material_index = (pos.x + pos.y + pos.z) & 0xff;
						voxels.set_voxel(material_index, pos, VoxelBuffer::CHANNEL_INDICES);
					}
				}
			}
		}
	}

	Ref<VoxelMesherTransvoxel> mesher;
	mesher.instantiate();
	mesher->set_texturing_mode(VoxelMesherTransvoxel::TEXTURES_SINGLE_S4);
	VoxelMesher::Output output;
	// Used to crash
	mesher->build(output, VoxelMesher::Input{ voxels, nullptr, Vector3i(), 0, false, false, false });

	ZN_TEST_ASSERT(!VoxelMesher::is_mesh_empty(output.surfaces));
}

void test_transvoxel_surface_data() {
	// CHANNEL_DATA6 packed RGBA8 must reach CUSTOM2, interpolated along vertex edges
	VoxelBuffer voxels(VoxelBuffer::ALLOCATOR_DEFAULT);
	voxels.set_channel_depth(VoxelBuffer::CHANNEL_DATA6, VoxelBuffer::DEPTH_32_BIT);
	voxels.create(Vector3iUtil::create(16));
	const uint16_t indices = mixel4::encode_indices_to_packed_u16(0, 1, 2, 3);
	const uint16_t weights = mixel4::encode_weights_to_packed_u16_lossy(255, 0, 0, 0);
	{
		Vector3i pos;
		for (pos.z = 0; pos.z < voxels.get_size().z; ++pos.z) {
			for (pos.x = 0; pos.x < voxels.get_size().x; ++pos.x) {
				for (pos.y = 0; pos.y < voxels.get_size().y; ++pos.y) {
					// Tilted plane so vertices land on edges along every axis
					const float sd = pos.y - 7.3f - 0.3f * pos.x + 0.2f * pos.z;
					voxels.set_voxel_f(sd, pos, VoxelBuffer::CHANNEL_SDF);
					voxels.set_voxel(indices, pos, VoxelBuffer::CHANNEL_INDICES);
					voxels.set_voxel(weights, pos, VoxelBuffer::CHANNEL_WEIGHTS);
					const uint32_t r = pos.x * 16;
					voxels.set_voxel(r | (0x55 << 8) | (0xAA << 16) | (0xFFu << 24), pos, VoxelBuffer::CHANNEL_DATA6);
				}
			}
		}
	}

	Ref<VoxelMesherTransvoxel> mesher;
	mesher.instantiate();
	mesher->set_texturing_mode(VoxelMesherTransvoxel::TEXTURES_MIXEL4_S4);
	mesher->set_surface_data_enabled(true);
	ZN_TEST_ASSERT((mesher->get_used_channels_mask() & (1 << VoxelBuffer::CHANNEL_DATA6)) != 0);

	for (const bool lod_hint : { false, true }) {
		VoxelMesher::Output output;
		mesher->build(output, VoxelMesher::Input{ voxels, nullptr, Vector3i(), 0, false, lod_hint, false });
		ZN_TEST_ASSERT(output.surfaces.size() == 1);
		ZN_TEST_ASSERT(((output.mesh_flags >> Mesh::ARRAY_FORMAT_CUSTOM2_SHIFT) & Mesh::ARRAY_FORMAT_CUSTOM_MASK) ==
				Mesh::ARRAY_CUSTOM_RGBA8_UNORM);

		const Array &arrays = output.surfaces[0].arrays;
		const PackedVector3Array vertices = arrays[Mesh::ARRAY_VERTEX];
		const PackedByteArray custom2 = arrays[Mesh::ARRAY_CUSTOM2];
		ZN_TEST_ASSERT(vertices.size() > 0);
		ZN_TEST_ASSERT(custom2.size() == vertices.size() * 4);

		float offset = 0.f;
		for (int i = 0; i < vertices.size(); ++i) {
			ZN_TEST_ASSERT(custom2[i * 4 + 1] == 0x55 && custom2[i * 4 + 2] == 0xAA && custom2[i * 4 + 3] == 0xFF);
			if (!lod_hint) {
				// R is linear in voxel x, so R/16 - vertex.x is the same (padding) offset everywhere
				const float o = custom2[i * 4] / 16.f - vertices[i].x;
				if (i == 0) {
					offset = o;
				}
				ZN_TEST_ASSERT(Math::abs(o - offset) < 0.1f);
			}
		}
	}
}

} // namespace zylann::voxel::tests
