#ifndef VOXEL_MESHER_SURFACE_NETS_H
#define VOXEL_MESHER_SURFACE_NETS_H

#include "../voxel_mesher.h"

ZN_GODOT_FORWARD_DECLARE(class ShaderMaterial)

namespace zylann::voxel {

// Smooth-terrain mesher using naive Surface Nets (dual contouring: one vertex per active cell)
// instead of Marching Cubes. See VoxelMesherTransvoxel for the other smooth mesher available.
class VoxelMesherSurfaceNets : public VoxelMesher {
	GDCLASS(VoxelMesherSurfaceNets, VoxelMesher)
public:
	enum TexturingMode {
		TEXTURES_NONE = 0,
		// 4 blend weights over 16 texture indices, packed into CHANNEL_INDICES and CHANNEL_WEIGHTS.
		// Same encoding as VoxelMesherTransvoxel's mode of the same name.
		TEXTURES_MIXEL4_S4 = 1,
		TEXTURING_MODE_COUNT = 2
	};

	static void load_static_resources();
	static void free_static_resources();

	VoxelMesherSurfaceNets();

	void build(VoxelMesher::Output &output, const VoxelMesher::Input &input) override;

	int get_used_channels_mask() const override;

	bool is_generating_collision_surface() const override;

	void set_skirts_enabled(bool enable);
	bool get_skirts_enabled() const;

	void set_skirt_depth_cells(float depth);
	float get_skirt_depth_cells() const;

	void set_texturing_mode(TexturingMode mode);
	TexturingMode get_texturing_mode() const;

	Ref<ShaderMaterial> get_default_lod_material() const override;

private:
	static void _bind_methods();

	bool _skirts_enabled = true;
	// In cells of the current LOD. Has to cover the coarse neighbour's decimation error, which for
	// ordinary terrain is a couple of cells; going deeper only risks poking out of thin geometry.
	float _skirt_depth_cells = 2.f;
	// Unlike VoxelMesherTransvoxel this defaults ON: this mesher exists to drive the voxel graph's
	// MaterialOutput pipeline, and that pipeline routes materials through exactly these two channels.
	// With texturing off, every vertex reports zero weights and the auto material shader falls back to
	// layer 0 everywhere -- a uniformly flat-coloured terrain with no indication of why.
	TexturingMode _texturing_mode = TEXTURES_MIXEL4_S4;
};

} // namespace zylann::voxel

VARIANT_ENUM_CAST(zylann::voxel::VoxelMesherSurfaceNets::TexturingMode);

#endif // VOXEL_MESHER_SURFACE_NETS_H
