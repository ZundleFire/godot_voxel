// Voxel Domain Warp — ISPC SIMD kernel
// Processes batches of 2D/3D positions through multi-octave domain warping using SIMD lanes.
// Adapted from UE5 VoxelPlugin's VoxelDomainWarpNodesImpl.ispc.
//
// Domain warp distorts input coordinates using hash-based gradient fields,
// producing organic, flowing deformations when composed with noise functions.

#include "voxel_noise_impl.isph"

// ============================================================================
// Domain Warp 3D — multi-octave batch processing on a grid
// ============================================================================

export void VoxelDomainWarp3D_FillBuffer(
    // Grid parameters
    uniform const float OriginX,
    uniform const float OriginY,
    uniform const float OriginZ,
    uniform const int32 SizeX,
    uniform const int32 SizeY,
    uniform const int32 SizeZ,
    uniform const float Spacing,
    // Warp parameters
    uniform const float InAmplitude,
    uniform const float FeatureScale,
    uniform const float Lacunarity,
    uniform const float Gain,
    uniform const float WeightedStrength,
    uniform const int32 NumOctaves,
    uniform const int32 InSeed,
    // Output buffer — warped positions (X, Y, Z interleaved) or just the warp magnitude
    uniform float ReturnValuesX[],
    uniform float ReturnValuesY[],
    uniform float ReturnValuesZ[],
    uniform const int32 TotalCount)
{
    // Precompute fractal bounding for amplitude normalization
    uniform float FractalBounding = 1.0f;
    {
        uniform float AmpFractal = 1.0f;
        uniform float Amp = Gain;
        for (uniform int32 OctaveIndex = 1; OctaveIndex < NumOctaves; OctaveIndex++) {
            AmpFractal += Amp;
            Amp *= Gain;
        }
        FractalBounding = 1.0f / AmpFractal;
    }

    foreach (Index = 0 ... TotalCount)
    {
        // Convert flat index to 3D grid position
        const int32 ix = Index % SizeX;
        const int32 iy = (Index / SizeX) % SizeY;
        const int32 iz = Index / (SizeX * SizeY);

        // World position
        const float wx = OriginX + (float)ix * Spacing;
        const float wy = OriginY + (float)iy * Spacing;
        const float wz = OriginZ + (float)iz * Spacing;

        // Warped position starts at original
        float3 WarpedPosition = MakeFloat3(wx, wy, wz);

        // Scaled position for noise sampling
        float3 ScaledPosition = MakeFloat3(wx, wy, wz) / FeatureScale;

        float Amplitude = FractalBounding * InAmplitude;
        int32 Seed = InSeed;

        float Strength = DomainWarp3D(Seed, Amplitude, ScaledPosition, &WarpedPosition);

        for (uniform int32 OctaveIndex = 1; OctaveIndex < NumOctaves; OctaveIndex++) {
            ScaledPosition = ScaledPosition * Lacunarity;
            Amplitude *= lerp(1.0f, 1.0f - Strength, WeightedStrength) * Gain;

            Seed = (Seed * 196314165) + 907633515;

            Strength = DomainWarp3D(Seed, Amplitude, ScaledPosition, &WarpedPosition);
        }

        ReturnValuesX[Index] = WarpedPosition.x;
        ReturnValuesY[Index] = WarpedPosition.y;
        ReturnValuesZ[Index] = WarpedPosition.z;
    }
}

// ============================================================================
// Domain Warp 2D — multi-octave batch processing on a grid
// ============================================================================

export void VoxelDomainWarp2D_FillBuffer(
    // Grid parameters
    uniform const float OriginX,
    uniform const float OriginY,
    uniform const int32 SizeX,
    uniform const int32 SizeY,
    uniform const float Spacing,
    // Warp parameters
    uniform const float InAmplitude,
    uniform const float FeatureScale,
    uniform const float Lacunarity,
    uniform const float Gain,
    uniform const float WeightedStrength,
    uniform const int32 NumOctaves,
    uniform const int32 InSeed,
    // Output buffer — warped positions
    uniform float ReturnValuesX[],
    uniform float ReturnValuesY[],
    uniform const int32 TotalCount)
{
    // Precompute fractal bounding
    uniform float FractalBounding = 1.0f;
    {
        uniform float AmpFractal = 1.0f;
        uniform float Amp = Gain;
        for (uniform int32 OctaveIndex = 1; OctaveIndex < NumOctaves; OctaveIndex++) {
            AmpFractal += Amp;
            Amp *= Gain;
        }
        FractalBounding = 1.0f / AmpFractal;
    }

    foreach (Index = 0 ... TotalCount)
    {
        const int32 ix = Index % SizeX;
        const int32 iy = Index / SizeX;

        const float wx = OriginX + (float)ix * Spacing;
        const float wy = OriginY + (float)iy * Spacing;

        float2 WarpedPosition = MakeFloat2(wx, wy);
        float2 ScaledPosition = MakeFloat2(wx, wy) / FeatureScale;

        float Amplitude = FractalBounding * InAmplitude;
        int32 Seed = InSeed;

        float Strength = DomainWarp2D(Seed, Amplitude, ScaledPosition, &WarpedPosition);

        for (uniform int32 OctaveIndex = 1; OctaveIndex < NumOctaves; OctaveIndex++) {
            ScaledPosition = ScaledPosition * Lacunarity;
            Amplitude *= lerp(1.0f, 1.0f - Strength, WeightedStrength) * Gain;

            Seed = (Seed * 196314165) + 907633515;

            Strength = DomainWarp2D(Seed, Amplitude, ScaledPosition, &WarpedPosition);
        }

        ReturnValuesX[Index] = WarpedPosition.x;
        ReturnValuesY[Index] = WarpedPosition.y;
    }
}

// ============================================================================
// Domain Warp 3D — arbitrary position batch (not on a grid)
// ============================================================================

export void VoxelDomainWarp3D_Batch(
    // Position arrays
    uniform const float PosX[],
    uniform const float PosY[],
    uniform const float PosZ[],
    // Warp parameters
    uniform const float InAmplitude,
    uniform const float FeatureScale,
    uniform const float Lacunarity,
    uniform const float Gain,
    uniform const float WeightedStrength,
    uniform const int32 NumOctaves,
    uniform const int32 InSeed,
    // Output buffer — warped positions
    uniform float ReturnValuesX[],
    uniform float ReturnValuesY[],
    uniform float ReturnValuesZ[],
    uniform const int32 Num)
{
    // Precompute fractal bounding
    uniform float FractalBounding = 1.0f;
    {
        uniform float AmpFractal = 1.0f;
        uniform float Amp = Gain;
        for (uniform int32 OctaveIndex = 1; OctaveIndex < NumOctaves; OctaveIndex++) {
            AmpFractal += Amp;
            Amp *= Gain;
        }
        FractalBounding = 1.0f / AmpFractal;
    }

    foreach (Index = 0 ... Num)
    {
        float3 WarpedPosition = MakeFloat3(PosX[Index], PosY[Index], PosZ[Index]);
        float3 ScaledPosition = WarpedPosition / FeatureScale;

        float Amplitude = FractalBounding * InAmplitude;
        int32 Seed = InSeed;

        float Strength = DomainWarp3D(Seed, Amplitude, ScaledPosition, &WarpedPosition);

        for (uniform int32 OctaveIndex = 1; OctaveIndex < NumOctaves; OctaveIndex++) {
            ScaledPosition = ScaledPosition * Lacunarity;
            Amplitude *= lerp(1.0f, 1.0f - Strength, WeightedStrength) * Gain;

            Seed = (Seed * 196314165) + 907633515;

            Strength = DomainWarp3D(Seed, Amplitude, ScaledPosition, &WarpedPosition);
        }

        ReturnValuesX[Index] = WarpedPosition.x;
        ReturnValuesY[Index] = WarpedPosition.y;
        ReturnValuesZ[Index] = WarpedPosition.z;
    }
}

// ============================================================================
// Standalone noise batch functions — for individual noise type evaluation
// These allow calling any single noise type on arbitrary position arrays.
// ============================================================================

export void VoxelNoise_Perlin3D_Batch(
    uniform const float PosX[],
    uniform const float PosY[],
    uniform const float PosZ[],
    uniform const int32 Seed,
    uniform float ReturnValues[],
    uniform const int32 Num)
{
    foreach (Index = 0 ... Num) {
        ReturnValues[Index] = GetPerlin3D(Seed, MakeFloat3(PosX[Index], PosY[Index], PosZ[Index]));
    }
}

export void VoxelNoise_Perlin2D_Batch(
    uniform const float PosX[],
    uniform const float PosY[],
    uniform const int32 Seed,
    uniform float ReturnValues[],
    uniform const int32 Num)
{
    foreach (Index = 0 ... Num) {
        ReturnValues[Index] = GetPerlin2D(Seed, MakeFloat2(PosX[Index], PosY[Index]));
    }
}

export void VoxelNoise_Simplex3D_Batch(
    uniform const float PosX[],
    uniform const float PosY[],
    uniform const float PosZ[],
    uniform const int32 Seed,
    uniform float ReturnValues[],
    uniform const int32 Num)
{
    foreach (Index = 0 ... Num) {
        ReturnValues[Index] = GetSimplex3D(Seed, MakeFloat3(PosX[Index], PosY[Index], PosZ[Index]));
    }
}

export void VoxelNoise_Simplex2D_Batch(
    uniform const float PosX[],
    uniform const float PosY[],
    uniform const int32 Seed,
    uniform float ReturnValues[],
    uniform const int32 Num)
{
    foreach (Index = 0 ... Num) {
        ReturnValues[Index] = GetSimplex2D(Seed, MakeFloat2(PosX[Index], PosY[Index]));
    }
}

export void VoxelNoise_Cellular3D_Batch(
    uniform const float PosX[],
    uniform const float PosY[],
    uniform const float PosZ[],
    uniform const int32 Seed,
    uniform const float Jitter,
    uniform float ReturnValues[],
    uniform const int32 Num)
{
    foreach (Index = 0 ... Num) {
        ReturnValues[Index] = GetCellularNoise3D(Seed, MakeFloat3(PosX[Index], PosY[Index], PosZ[Index]), Jitter);
    }
}

export void VoxelNoise_Cellular2D_Batch(
    uniform const float PosX[],
    uniform const float PosY[],
    uniform const int32 Seed,
    uniform const float Jitter,
    uniform float ReturnValues[],
    uniform const int32 Num)
{
    foreach (Index = 0 ... Num) {
        ReturnValues[Index] = GetCellularNoise2D(Seed, MakeFloat2(PosX[Index], PosY[Index]), Jitter);
    }
}

export void VoxelNoise_Value3D_Batch(
    uniform const float PosX[],
    uniform const float PosY[],
    uniform const float PosZ[],
    uniform const int32 Seed,
    uniform float ReturnValues[],
    uniform const int32 Num)
{
    foreach (Index = 0 ... Num) {
        ReturnValues[Index] = GetValue3D(Seed, MakeFloat3(PosX[Index], PosY[Index], PosZ[Index]));
    }
}

export void VoxelNoise_Value2D_Batch(
    uniform const float PosX[],
    uniform const float PosY[],
    uniform const int32 Seed,
    uniform float ReturnValues[],
    uniform const int32 Num)
{
    foreach (Index = 0 ... Num) {
        ReturnValues[Index] = GetValue2D(Seed, MakeFloat2(PosX[Index], PosY[Index]));
    }
}

export void VoxelNoise_TrueDistanceCellular3D_Batch(
    uniform const float PosX[],
    uniform const float PosY[],
    uniform const float PosZ[],
    uniform const int32 Seed,
    uniform const float Jitter,
    uniform float ReturnValues[],
    uniform const int32 Num)
{
    foreach (Index = 0 ... Num) {
        ReturnValues[Index] = GetTrueDistanceCellular3D(Seed, MakeFloat3(PosX[Index], PosY[Index], PosZ[Index]), Jitter);
    }
}

export void VoxelNoise_TrueDistanceCellular2D_Batch(
    uniform const float PosX[],
    uniform const float PosY[],
    uniform const int32 Seed,
    uniform const float Jitter,
    uniform float ReturnValues[],
    uniform const int32 Num)
{
    foreach (Index = 0 ... Num) {
        ReturnValues[Index] = GetTrueDistanceCellular2D(Seed, MakeFloat2(PosX[Index], PosY[Index]), Jitter);
    }
}
