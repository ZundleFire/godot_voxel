// Voxel Advanced Noise — ISPC SIMD kernel
// Processes batches of 3D/2D positions through multi-octave noise using SIMD lanes.
// This is the ISPC equivalent of VoxelAdvancedNoise::fill_buffer_3d() and get_noise_3d().
//
// The ISPC compiler auto-vectorizes this across SIMD lanes (SSE4: 4-wide, AVX2: 8-wide),
// giving 3-6x speedup over scalar C++ for noise computation.
//
// Matches UE5 VoxelPlugin's VoxelAdvancedNoiseNodesImpl.ispc algorithm exactly.

#include "voxel_noise_impl.isph"

// ============================================================================
// Octave types — must match VoxelAdvancedNoise::NoiseType enum in C++
// ============================================================================

enum OctaveType {
    OctaveType_SmoothPerlin = 0,
    OctaveType_BillowyPerlin = 1,
    OctaveType_RidgedPerlin = 2,

    OctaveType_SmoothCellular = 3,
    OctaveType_BillowyCellular = 4,
    OctaveType_RidgedCellular = 5,

    OctaveType_SmoothSimplex = 6,
    OctaveType_BillowySimplex = 7,
    OctaveType_RidgedSimplex = 8,

    OctaveType_SmoothValue = 9,
    OctaveType_BillowyValue = 10,
    OctaveType_RidgedValue = 11,

    OctaveType_SmoothTrueDistanceCellular = 12,
    OctaveType_BillowyTrueDistanceCellular = 13,
    OctaveType_RidgedTrueDistanceCellular = 14,
};

// Per-octave configuration, passed from C++ side
struct OctaveConfig {
    int32 type;       // OctaveType enum value
    float strength;   // Per-octave strength multiplier
};

// ============================================================================
// Advanced Noise 3D — batch processing
// ============================================================================

export void VoxelAdvancedNoise3D_FillBuffer(
    // Grid parameters
    uniform const float OriginX,
    uniform const float OriginY,
    uniform const float OriginZ,
    uniform const int32 SizeX,
    uniform const int32 SizeY,
    uniform const int32 SizeZ,
    uniform const float Spacing,
    // Noise parameters
    uniform const float InAmplitude,
    uniform const float FeatureScale,
    uniform const float Lacunarity,
    uniform const float Gain,
    uniform const float CellularJitter,
    uniform const int32 InSeed,
    // Octave configs
    uniform const OctaveConfig Octaves[],
    uniform const int32 NumOctaves,
    // Output buffer
    uniform float ReturnValues[],
    uniform const int32 TotalCount)
{
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

        // Scale by feature scale
        float px = wx / FeatureScale;
        float py = wy / FeatureScale;
        float pz = wz / FeatureScale;

        float Sum = 0.0f;
        float AmplitudeSum = 0.0f;
        float Amplitude = 1.0f;
        int32 Seed = InSeed;

        for (uniform int32 OctaveIndex = 0; OctaveIndex < NumOctaves; OctaveIndex++)
        {
            const uniform OctaveConfig Octave = Octaves[OctaveIndex];
            float3 Pos = MakeFloat3(px, py, pz);

            varying float Noise;
            switch (Octave.type)
            {
            case OctaveType_SmoothPerlin:
                Noise = GetPerlin3D(Seed, Pos);
                break;
            case OctaveType_BillowyPerlin:
                Noise = abs(GetPerlin3D(Seed, Pos)) * 2.0f - 1.0f;
                break;
            case OctaveType_RidgedPerlin:
                Noise = (1.0f - abs(GetPerlin3D(Seed, Pos))) * 2.0f - 1.0f;
                break;

            case OctaveType_SmoothCellular:
                Noise = GetCellularNoise3D(Seed, Pos, CellularJitter) - 0.5f;
                break;
            case OctaveType_BillowyCellular:
                Noise = abs(GetCellularNoise3D(Seed, Pos, CellularJitter)) * 2.0f - 1.0f;
                break;
            case OctaveType_RidgedCellular:
                Noise = (1.0f - abs(GetCellularNoise3D(Seed, Pos, CellularJitter))) * 2.0f - 1.0f;
                break;

            case OctaveType_SmoothSimplex:
                Noise = GetSimplex3D(Seed, Pos);
                break;
            case OctaveType_BillowySimplex:
                Noise = abs(GetSimplex3D(Seed, Pos)) * 2.0f - 1.0f;
                break;
            case OctaveType_RidgedSimplex:
                Noise = (1.0f - abs(GetSimplex3D(Seed, Pos))) * 2.0f - 1.0f;
                break;

            case OctaveType_SmoothValue:
                Noise = GetValue3D(Seed, Pos);
                break;
            case OctaveType_BillowyValue:
                Noise = abs(GetValue3D(Seed, Pos)) * 2.0f - 1.0f;
                break;
            case OctaveType_RidgedValue:
                Noise = (1.0f - abs(GetValue3D(Seed, Pos))) * 2.0f - 1.0f;
                break;

            case OctaveType_SmoothTrueDistanceCellular:
                Noise = GetTrueDistanceCellular3D(Seed, Pos, CellularJitter) - 0.5f;
                break;
            case OctaveType_BillowyTrueDistanceCellular:
                Noise = abs(GetTrueDistanceCellular3D(Seed, Pos, CellularJitter)) * 2.0f - 1.0f;
                break;
            case OctaveType_RidgedTrueDistanceCellular:
                Noise = (1.0f - abs(GetTrueDistanceCellular3D(Seed, Pos, CellularJitter))) * 2.0f - 1.0f;
                break;

            default:
                Noise = 0.0f;
                break;
            }

            const float Strength = Octave.strength;
            const float NewSum = Sum + Noise * Strength * Amplitude;

            // NaN guard — match UE5's select(IsFinite(...)) pattern
            Sum = IsFinite(NewSum) ? NewSum : Sum;

            AmplitudeSum += abs(Strength * Amplitude);
            Amplitude *= Gain;
            px *= Lacunarity;
            py *= Lacunarity;
            pz *= Lacunarity;
            Seed = (Seed * 196314165) + 907633515;
        }

        // Normalize and apply amplitude
        ReturnValues[Index] = Sum / (AmplitudeSum == 0.0f ? 1.0f : AmplitudeSum) * InAmplitude;
    }
}

// ============================================================================
// Advanced Noise 2D — batch processing
// ============================================================================

export void VoxelAdvancedNoise2D_FillBuffer(
    // Grid parameters
    uniform const float OriginX,
    uniform const float OriginY,
    uniform const int32 SizeX,
    uniform const int32 SizeY,
    uniform const float Spacing,
    // Noise parameters
    uniform const float InAmplitude,
    uniform const float FeatureScale,
    uniform const float Lacunarity,
    uniform const float Gain,
    uniform const float CellularJitter,
    uniform const int32 InSeed,
    // Octave configs
    uniform const OctaveConfig Octaves[],
    uniform const int32 NumOctaves,
    // Output buffer
    uniform float ReturnValues[],
    uniform const int32 TotalCount)
{
    foreach (Index = 0 ... TotalCount)
    {
        const int32 ix = Index % SizeX;
        const int32 iy = Index / SizeX;

        const float wx = OriginX + (float)ix * Spacing;
        const float wy = OriginY + (float)iy * Spacing;

        float px = wx / FeatureScale;
        float py = wy / FeatureScale;

        float Sum = 0.0f;
        float AmplitudeSum = 0.0f;
        float Amplitude = 1.0f;
        int32 Seed = InSeed;

        for (uniform int32 OctaveIndex = 0; OctaveIndex < NumOctaves; OctaveIndex++)
        {
            const uniform OctaveConfig Octave = Octaves[OctaveIndex];
            float2 Pos = MakeFloat2(px, py);

            varying float Noise;
            switch (Octave.type)
            {
            case OctaveType_SmoothPerlin:
                Noise = GetPerlin2D(Seed, Pos);
                break;
            case OctaveType_BillowyPerlin:
                Noise = abs(GetPerlin2D(Seed, Pos)) * 2.0f - 1.0f;
                break;
            case OctaveType_RidgedPerlin:
                Noise = (1.0f - abs(GetPerlin2D(Seed, Pos))) * 2.0f - 1.0f;
                break;

            case OctaveType_SmoothCellular:
                Noise = GetCellularNoise2D(Seed, Pos, CellularJitter);
                break;
            case OctaveType_BillowyCellular:
                Noise = abs(GetCellularNoise2D(Seed, Pos, CellularJitter)) * 2.0f - 1.0f;
                break;
            case OctaveType_RidgedCellular:
                Noise = (1.0f - abs(GetCellularNoise2D(Seed, Pos, CellularJitter))) * 2.0f - 1.0f;
                break;

            case OctaveType_SmoothSimplex:
                Noise = GetSimplex2D(Seed, Pos);
                break;
            case OctaveType_BillowySimplex:
                Noise = abs(GetSimplex2D(Seed, Pos)) * 2.0f - 1.0f;
                break;
            case OctaveType_RidgedSimplex:
                Noise = (1.0f - abs(GetSimplex2D(Seed, Pos))) * 2.0f - 1.0f;
                break;

            case OctaveType_SmoothValue:
                Noise = GetValue2D(Seed, Pos);
                break;
            case OctaveType_BillowyValue:
                Noise = abs(GetValue2D(Seed, Pos)) * 2.0f - 1.0f;
                break;
            case OctaveType_RidgedValue:
                Noise = (1.0f - abs(GetValue2D(Seed, Pos))) * 2.0f - 1.0f;
                break;

            case OctaveType_SmoothTrueDistanceCellular:
                Noise = GetTrueDistanceCellular2D(Seed, Pos, CellularJitter) - 0.5f;
                break;
            case OctaveType_BillowyTrueDistanceCellular:
                Noise = abs(GetTrueDistanceCellular2D(Seed, Pos, CellularJitter)) * 2.0f - 1.0f;
                break;
            case OctaveType_RidgedTrueDistanceCellular:
                Noise = (1.0f - abs(GetTrueDistanceCellular2D(Seed, Pos, CellularJitter))) * 2.0f - 1.0f;
                break;

            default:
                Noise = 0.0f;
                break;
            }

            const float Strength = Octave.strength;
            const float NewSum = Sum + Noise * Strength * Amplitude;
            Sum = IsFinite(NewSum) ? NewSum : Sum;

            AmplitudeSum += abs(Strength * Amplitude);
            Amplitude *= Gain;
            px *= Lacunarity;
            py *= Lacunarity;
            Seed = (Seed * 196314165) + 907633515;
        }

        ReturnValues[Index] = Sum / (AmplitudeSum == 0.0f ? 1.0f : AmplitudeSum) * InAmplitude;
    }
}

// ============================================================================
// Single-sample batch — for arbitrary position arrays (not on a grid)
// ============================================================================

export void VoxelAdvancedNoise3D_Batch(
    // Position arrays
    uniform const float PosX[],
    uniform const float PosY[],
    uniform const float PosZ[],
    // Noise parameters
    uniform const float InAmplitude,
    uniform const float FeatureScale,
    uniform const float Lacunarity,
    uniform const float Gain,
    uniform const float CellularJitter,
    uniform const int32 InSeed,
    // Octave configs
    uniform const OctaveConfig Octaves[],
    uniform const int32 NumOctaves,
    // Output buffer
    uniform float ReturnValues[],
    uniform const int32 Num)
{
    foreach (Index = 0 ... Num)
    {
        float px = PosX[Index] / FeatureScale;
        float py = PosY[Index] / FeatureScale;
        float pz = PosZ[Index] / FeatureScale;

        float Sum = 0.0f;
        float AmplitudeSum = 0.0f;
        float Amplitude = 1.0f;
        int32 Seed = InSeed;

        for (uniform int32 OctaveIndex = 0; OctaveIndex < NumOctaves; OctaveIndex++)
        {
            const uniform OctaveConfig Octave = Octaves[OctaveIndex];
            float3 Pos = MakeFloat3(px, py, pz);

            varying float Noise;
            switch (Octave.type)
            {
            case OctaveType_SmoothPerlin:
                Noise = GetPerlin3D(Seed, Pos);
                break;
            case OctaveType_BillowyPerlin:
                Noise = abs(GetPerlin3D(Seed, Pos)) * 2.0f - 1.0f;
                break;
            case OctaveType_RidgedPerlin:
                Noise = (1.0f - abs(GetPerlin3D(Seed, Pos))) * 2.0f - 1.0f;
                break;
            case OctaveType_SmoothCellular:
                Noise = GetCellularNoise3D(Seed, Pos, CellularJitter) - 0.5f;
                break;
            case OctaveType_BillowyCellular:
                Noise = abs(GetCellularNoise3D(Seed, Pos, CellularJitter)) * 2.0f - 1.0f;
                break;
            case OctaveType_RidgedCellular:
                Noise = (1.0f - abs(GetCellularNoise3D(Seed, Pos, CellularJitter))) * 2.0f - 1.0f;
                break;
            case OctaveType_SmoothSimplex:
                Noise = GetSimplex3D(Seed, Pos);
                break;
            case OctaveType_BillowySimplex:
                Noise = abs(GetSimplex3D(Seed, Pos)) * 2.0f - 1.0f;
                break;
            case OctaveType_RidgedSimplex:
                Noise = (1.0f - abs(GetSimplex3D(Seed, Pos))) * 2.0f - 1.0f;
                break;
            case OctaveType_SmoothValue:
                Noise = GetValue3D(Seed, Pos);
                break;
            case OctaveType_BillowyValue:
                Noise = abs(GetValue3D(Seed, Pos)) * 2.0f - 1.0f;
                break;
            case OctaveType_RidgedValue:
                Noise = (1.0f - abs(GetValue3D(Seed, Pos))) * 2.0f - 1.0f;
                break;
            case OctaveType_SmoothTrueDistanceCellular:
                Noise = GetTrueDistanceCellular3D(Seed, Pos, CellularJitter) - 0.5f;
                break;
            case OctaveType_BillowyTrueDistanceCellular:
                Noise = abs(GetTrueDistanceCellular3D(Seed, Pos, CellularJitter)) * 2.0f - 1.0f;
                break;
            case OctaveType_RidgedTrueDistanceCellular:
                Noise = (1.0f - abs(GetTrueDistanceCellular3D(Seed, Pos, CellularJitter))) * 2.0f - 1.0f;
                break;
            default:
                Noise = 0.0f;
                break;
            }

            const float Strength = Octave.strength;
            const float NewSum = Sum + Noise * Strength * Amplitude;
            Sum = IsFinite(NewSum) ? NewSum : Sum;

            AmplitudeSum += abs(Strength * Amplitude);
            Amplitude *= Gain;
            px *= Lacunarity;
            py *= Lacunarity;
            pz *= Lacunarity;
            Seed = (Seed * 196314165) + 907633515;
        }

        ReturnValues[Index] = Sum / (AmplitudeSum == 0.0f ? 1.0f : AmplitudeSum) * InAmplitude;
    }
}
