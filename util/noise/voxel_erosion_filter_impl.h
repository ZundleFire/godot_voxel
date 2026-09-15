// Advanced terrain erosion filter — port of "Better Terrain Noise"
// (https://godotshaders.com/shader/better-terrain-noise/), which wraps:
//
// Phacelle Noise and Advanced Terrain Erosion Filter
// Copyright (c) 2025 Rune Skovbo Johansen
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// Original: https://www.shadertoy.com/view/wXcfWn
// Technique: https://blog.runevision.com/2026/03/fast-and-gorgeous-erosion-filter.html
//
// Changes from the original GLSL:
// - Integer bit hash (FCHash22) instead of the fract-based hash: the original
//   loses float precision at planet-scale coordinates.
// - Seeded; base fBm octaves use decorrelated seeds.
// - FCPlanetErosion maps the 2D filter onto a sphere with triplanar blending.
//
// SINGLE-SOURCE FILE: included from voxel_procedural_noise_impl.h, so it compiles
// both as ISPC and as scalar C++ (inside namespace zylann::procedural_noise).
// GLSL twin for the graph GPU path: voxel_erosion_filter_glsl.h — keep in sync.

// Layout must match zylann::voxel::TerrainErosionParams (voxel_terrain_noise.h)
struct FCErosionParams {
	int32 Seed;
	float PlanetRadius;
	float TileSize; // world units per filter unit
	float TriplanarSharpness;
	// Base relief fBm (filter units)
	float HeightFrequency;
	float HeightAmp;
	int32 HeightOctaves;
	float HeightLacunarity;
	float HeightGain;
	// Erosion
	float Scale;
	float Strength;
	float GullyWeight;
	float Detail;
	float RoundingRidge;
	float RoundingCrease;
	float RoundingInputMult;
	float RoundingOctaveMult;
	float OnsetInitial;
	float OnsetOctave;
	float RidgeOnsetInitial;
	float RidgeOnsetOctave;
	float AssumedSlope;
	float AssumedSlopeBlend;
	float CellScale;
	float Normalization;
	int32 Octaves;
	float Lacunarity;
	float Gain;
	float HeightOffset;
	float HeightOffsetFadeBlend;
};

struct FCErodedTerrain {
	float Height; // Centered on 0. Filter units from FCErodedTerrain2D, world units from FCPlanetErosion
	float Ridge; // -1 creases .. 1 ridges
	float Erosion; // 0..1, 0.5 = untouched
};

FORCEINLINE float FCErClamp01(const float X) {
	return clamp(X, 0.f, 1.f);
}
FORCEINLINE float FCErSign(const float X) {
	return X > 0.f ? 1.f : (X < 0.f ? -1.f : 0.f);
}
FORCEINLINE float FCErPowInv(const float T, const float Power) {
	return 1.f - pow(1.f - FCErClamp01(T), Power);
}
FORCEINLINE float FCErEaseOut(const float T) {
	const float V = 1.f - FCErClamp01(T);
	return 1.f - V * V;
}
FORCEINLINE float FCErSmoothStart(const float T, const float Smoothing) {
	return T >= Smoothing ? T - 0.5f * Smoothing : 0.5f * T * T / max(Smoothing, 1e-10f);
}
FORCEINLINE float2 FCErSafeNormalize(const float2 N) {
	const float L = length(N);
	return L > 1e-10f ? N / L : N;
}

// Stripes aligned with NormDir. Returns normalized (cos, sin); OutSideDir gets the
// vector the sine multiplies to form the derivative of the cosine.
FORCEINLINE float2 FCPhacelleNoise(
		const uint32 Seed, const float2 P, const float2 NormDir, const float Freq, const float Offset,
		const float Normalization, varying float2 *OutSideDir) {
	const float2 SideDir = MakeFloat2(-NormDir.y, NormDir.x) * (Freq * 6.28318530718f);
	const float OffsetRad = Offset * 6.28318530718f;
	const float2 PInt = floor(P);
	const float2 PFrac = P - PInt;
	float2 PhaseDir = MakeFloat2(0.f, 0.f);
	float WeightSum = 0.f;
	for (uniform int32 I = -1; I <= 2; I++) {
		for (uniform int32 J = -1; J <= 2; J++) {
			const float2 GridOffset = MakeFloat2(I, J);
			const float2 V = PFrac - GridOffset - (FCHash22(Seed, PInt + GridOffset) - 0.5f);
			// Bell weight, exactly 0 at distance 1.5 to avoid grid line artifacts
			const float Weight = max(0.f, exp(-dot(V, V) * 2.f) - 0.01111f);
			WeightSum += Weight;
			const float WaveInput = dot(V, SideDir) + OffsetRad;
			PhaseDir = PhaseDir + MakeFloat2(cos(WaveInput), sin(WaveInput)) * Weight;
		}
	}
	const float2 Interpolated = PhaseDir / max(WeightSum, 1e-6f);
	const float Magnitude = max(max(1.f - Normalization, length(Interpolated)), 1e-10f);
	*OutSideDir = SideDir;
	return Interpolated / Magnitude;
}

struct FCErosionResult {
	float Delta;
	float Magnitude;
	float RidgeMap;
};

// InHeightAndSlope: height (x) and its derivative (y, z) at P
FORCEINLINE FCErosionResult FCErosionFilter2D(
		const uint32 Seed, const float2 P, const float3 InHeightAndSlope, const float InFadeTarget,
		const uniform FCErosionParams &Prm) {
	float Strength = Prm.Strength * Prm.Scale;
	float FadeTarget = clamp(InFadeTarget, -1.f, 1.f);
	float Height = InHeightAndSlope.x;
	float Freq = 1.f / (Prm.Scale * Prm.CellScale);
	float RoundingMult = 1.f;
	float Magnitude = 0.f;

	const float2 InSlope = MakeFloat2(InHeightAndSlope.y, InHeightAndSlope.z);
	const float SlopeLength = max(length(InSlope), 1e-10f);

	const float RoundingForInput =
			FCLerp(Prm.RoundingCrease, Prm.RoundingRidge, FCErClamp01(FadeTarget + 0.5f)) * Prm.RoundingInputMult;
	float CombiMask = FCErEaseOut(FCErSmoothStart(SlopeLength * Prm.OnsetInitial, RoundingForInput * Prm.OnsetInitial));

	float RidgeMapCombiMask = FCErEaseOut(SlopeLength * Prm.RidgeOnsetInitial);
	float RidgeMapFadeTarget = FadeTarget;

	float2 GullySlope = InSlope + (InSlope * (Prm.AssumedSlope / SlopeLength) - InSlope) * Prm.AssumedSlopeBlend;

	for (uniform int32 I = 0; I < Prm.Octaves; I++) {
		float2 SideDir;
		const float2 Phacelle = FCPhacelleNoise(
				Seed, P * Freq, FCErSafeNormalize(GullySlope), Prm.CellScale, 0.25f, Prm.Normalization, &SideDir);
		// P was multiplied by Freq; negated because slopes point downhill
		SideDir = SideDir * -Freq;
		const float Sloping = abs(Phacelle.y);

		GullySlope = GullySlope + SideDir * (FCErSign(Phacelle.y) * Strength * Prm.GullyWeight);

		const float GullyHeight = Phacelle.x;
		const float FadedHeight = FCLerp(FadeTarget, GullyHeight * Prm.GullyWeight, CombiMask);
		Height += FadedHeight * Strength;
		Magnitude += Strength;
		FadeTarget = FadedHeight;

		const float RoundingForOctave =
				FCLerp(Prm.RoundingCrease, Prm.RoundingRidge, FCErClamp01(Phacelle.x + 0.5f)) * RoundingMult;
		const float NewMask = FCErEaseOut(FCErSmoothStart(Sloping * Prm.OnsetOctave, RoundingForOctave * Prm.OnsetOctave));
		CombiMask = FCErPowInv(CombiMask, Prm.Detail) * NewMask;

		RidgeMapFadeTarget = FCLerp(RidgeMapFadeTarget, GullyHeight, RidgeMapCombiMask);
		RidgeMapCombiMask = RidgeMapCombiMask * FCErEaseOut(Sloping * Prm.RidgeOnsetOctave);

		Strength *= Prm.Gain;
		Freq *= Prm.Lacunarity;
		RoundingMult *= Prm.RoundingOctaveMult;
	}

	FCErosionResult Result;
	Result.Delta = Height - InHeightAndSlope.x;
	Result.Magnitude = Magnitude;
	Result.RidgeMap = RidgeMapFadeTarget * (1.f - RidgeMapCombiMask);
	return Result;
}

// Base derivative fBm + erosion, as in the shader's demo section. P in filter units.
FORCEINLINE FCErodedTerrain FCErodedTerrain2D(const float2 P, const uniform FCErosionParams &Prm) {
	float N = 0.f;
	float2 Slope = MakeFloat2(0.f, 0.f);
	uniform float Freq = Prm.HeightFrequency;
	uniform float Amp = 1.f;
	uniform int32 OctaveSeed = Prm.Seed;
	for (uniform int32 I = 0; I < Prm.HeightOctaves; I++) {
		float2 Gradient;
		const float Value = FCPerlinDeriv2D((uniform uint32)OctaveSeed, P * Freq, &Gradient);
		N += Value * Amp;
		Slope = Slope + Gradient * (Amp * Freq);
		Amp *= Prm.HeightGain;
		Freq *= Prm.HeightLacunarity;
		OctaveSeed = FCNextOctaveSeed(OctaveSeed);
	}
	N *= Prm.HeightAmp;
	Slope = Slope * Prm.HeightAmp;

	// Should be -1 in valleys and 1 on peaks, overshoot allowed
	const float FadeTarget = clamp(N / max(Prm.HeightAmp * 0.6f, 1e-6f), -1.f, 1.f);
	// The shader remaps height (and so slope) from [-1, 1] to [0, 1] before eroding
	const float3 HeightAndSlope = MakeFloat3(N, Slope.x, Slope.y) * 0.5f;

	const FCErosionResult E = FCErosionFilter2D(
			(uniform uint32)Prm.Seed ^ 0x2545F491u, P, HeightAndSlope, FadeTarget, Prm);

	const float Offset = FCLerp(Prm.HeightOffset, -FadeTarget, Prm.HeightOffsetFadeBlend) * E.Magnitude;

	FCErodedTerrain Result;
	Result.Height = HeightAndSlope.x + E.Delta + Offset;
	Result.Ridge = E.RidgeMap;
	Result.Erosion = FCErClamp01(E.Delta / max(E.Magnitude, 1e-10f) * 0.5f + 0.5f);
	return Result;
}

// Eroded relief on a sphere, sampled by direction only (altitude-independent).
// Height in world units. Triplanar: each axis plane runs its own 2D filter.
FORCEINLINE FCErodedTerrain FCPlanetErosion(const float3 Position, const uniform FCErosionParams &Prm) {
	const float3 Dir = Position / max(length(Position), 1e-6f);
	const float3 S = Dir * (Prm.PlanetRadius / Prm.TileSize);
	const float3 A = abs(Dir);

	float WX = pow(A.x, Prm.TriplanarSharpness);
	float WY = pow(A.y, Prm.TriplanarSharpness);
	float WZ = pow(A.z, Prm.TriplanarSharpness);
	float WSum = max(WX + WY + WZ, 1e-6f);
	// Fade out near-zero planes so they can be skipped entirely
	WX = max(WX / WSum - 0.02f, 0.f);
	WY = max(WY / WSum - 0.02f, 0.f);
	WZ = max(WZ / WSum - 0.02f, 0.f);
	WSum = max(WX + WY + WZ, 1e-6f);
	WX = WX / WSum;
	WY = WY / WSum;
	WZ = WZ / WSum;

	FCErodedTerrain Result;
	Result.Height = 0.f;
	Result.Ridge = 0.f;
	Result.Erosion = 0.f;

	// Opposite hemispheres get offset domains so they don't mirror each other
	if (WX > 0.f) {
		const FCErodedTerrain T = FCErodedTerrain2D(MakeFloat2(S.y + (Dir.x < 0.f ? 517.3f : 0.f), S.z), Prm);
		Result.Height += T.Height * WX;
		Result.Ridge += T.Ridge * WX;
		Result.Erosion += T.Erosion * WX;
	}
	if (WY > 0.f) {
		const FCErodedTerrain T = FCErodedTerrain2D(MakeFloat2(S.z + (Dir.y < 0.f ? 311.9f : 0.f), S.x), Prm);
		Result.Height += T.Height * WY;
		Result.Ridge += T.Ridge * WY;
		Result.Erosion += T.Erosion * WY;
	}
	if (WZ > 0.f) {
		const FCErodedTerrain T = FCErodedTerrain2D(MakeFloat2(S.x + (Dir.z < 0.f ? 743.1f : 0.f), S.y), Prm);
		Result.Height += T.Height * WZ;
		Result.Ridge += T.Ridge * WZ;
		Result.Erosion += T.Erosion * WZ;
	}

	Result.Height *= Prm.TileSize;
	return Result;
}
