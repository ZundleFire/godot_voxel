// Procedural noise collection — shared implementation.
// Ported from the VCET Unreal plugin (https://github.com/ZundleFire/VCET, MIT),
// itself ported from the Procedural Noise Collection by @lumiey (MIT):
//
// MIT License
//
// Copyright (c) 2026 @lumiey
// https://fragcoord.xyz/s/pxmcvnpc
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Changes from the original GLSL:
// - A seed is mixed into every hash so octaves can be decorrelated
// - Feature point offsets in Worley use a 2D/3D hash instead of a broadcasted scalar hash
// - Screen-space derivative based smoothing (fwidth) is replaced by explicit inputs
// - All noises are remapped to output roughly [-1, 1]
// - 3D variants are added for Voronoi, Blue, HilbertBlue, Crater, Gabor, Curl, Scratch,
//   Wavelet, Erosion, Paper, Stone and Wool
//
// SINGLE-SOURCE FILE: this header compiles both as ISPC (included from
// voxel_procedural_noise_ispc.ispc) and as scalar C++ (included from
// voxel_procedural_noise.cpp). Keep the body in the common subset of both
// languages — the preludes below map the differences.

#ifdef ISPC

#define FORCEINLINE inline
#define FC_GOLDEN_RATIO 0.6180339887498948482d

#else // C++ prelude

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

namespace zylann::procedural_noise {

#define FORCEINLINE inline
#define FC_GOLDEN_RATIO 0.6180339887498948482
#define uniform
#define varying

typedef int32_t int32;
typedef uint32_t uint32;

FORCEINLINE float floor(const float v) {
	return ::floorf(v);
}
FORCEINLINE double floor(const double v) {
	return ::floor(v);
}
FORCEINLINE float abs(const float v) {
	return ::fabsf(v);
}
FORCEINLINE float min(const float a, const float b) {
	return a < b ? a : b;
}
FORCEINLINE float max(const float a, const float b) {
	return a > b ? a : b;
}
FORCEINLINE float clamp(const float v, const float lo, const float hi) {
	return v < lo ? lo : (v > hi ? hi : v);
}
FORCEINLINE float sqrt(const float v) {
	return ::sqrtf(v);
}
FORCEINLINE float sin(const float v) {
	return ::sinf(v);
}
FORCEINLINE float cos(const float v) {
	return ::cosf(v);
}
FORCEINLINE float exp(const float v) {
	return ::expf(v);
}
FORCEINLINE float pow(const float a, const float b) {
	return ::powf(a, b);
}
FORCEINLINE uint32 intbits(const float v) {
	uint32 u;
	memcpy(&u, &v, 4);
	return u;
}

#endif // ISPC / C++ preludes

///////////////////////////////////////////////////////////////////////////////
// Noise type enum, shared by the ISPC kernels and the C++ resource
///////////////////////////////////////////////////////////////////////////////

enum FCProceduralNoiseType {
	FC_NOISE_PERLIN = 0,
	FC_NOISE_SIMPLEX = 1,
	FC_NOISE_VALUE = 2,
	FC_NOISE_WORLEY = 3,
	FC_NOISE_VORONOI = 4,
	FC_NOISE_BLUE = 5,
	FC_NOISE_HILBERT_BLUE = 6,
	FC_NOISE_CRATER = 7,
	FC_NOISE_GABOR = 8,
	FC_NOISE_CURL = 9,
	FC_NOISE_SCRATCH = 10,
	FC_NOISE_WAVELET = 11,
	FC_NOISE_EROSION = 12,
	FC_NOISE_PAPER = 13,
	FC_NOISE_STONE = 14,
	FC_NOISE_WOOL = 15,
	FC_NOISE_INTERLEAVED_GRADIENT = 16,
	FC_NOISE_TYPE_COUNT = 17
};

///////////////////////////////////////////////////////////////////////////////
// Minimal vector library
///////////////////////////////////////////////////////////////////////////////

struct float2 {
	float x;
	float y;
};

struct float3 {
	float x;
	float y;
	float z;
};

FORCEINLINE float2 MakeFloat2(const float X, const float Y) {
	float2 Result;
	Result.x = X;
	Result.y = Y;
	return Result;
}
FORCEINLINE float3 MakeFloat3(const float X, const float Y, const float Z) {
	float3 Result;
	Result.x = X;
	Result.y = Y;
	Result.z = Z;
	return Result;
}

FORCEINLINE float2 operator+(const float2 A, const float2 B) { return MakeFloat2(A.x + B.x, A.y + B.y); }
FORCEINLINE float2 operator-(const float2 A, const float2 B) { return MakeFloat2(A.x - B.x, A.y - B.y); }
FORCEINLINE float2 operator*(const float2 A, const float2 B) { return MakeFloat2(A.x * B.x, A.y * B.y); }
FORCEINLINE float2 operator+(const float2 A, const float B) { return MakeFloat2(A.x + B, A.y + B); }
FORCEINLINE float2 operator-(const float2 A, const float B) { return MakeFloat2(A.x - B, A.y - B); }
FORCEINLINE float2 operator*(const float2 A, const float B) { return MakeFloat2(A.x * B, A.y * B); }
FORCEINLINE float2 operator*(const float A, const float2 B) { return MakeFloat2(A * B.x, A * B.y); }
FORCEINLINE float2 operator+(const float A, const float2 B) { return MakeFloat2(A + B.x, A + B.y); }
FORCEINLINE float2 operator-(const float A, const float2 B) { return MakeFloat2(A - B.x, A - B.y); }
FORCEINLINE float2 operator/(const float2 A, const float B) { return MakeFloat2(A.x / B, A.y / B); }

FORCEINLINE float3 operator+(const float3 A, const float3 B) { return MakeFloat3(A.x + B.x, A.y + B.y, A.z + B.z); }
FORCEINLINE float3 operator-(const float3 A, const float3 B) { return MakeFloat3(A.x - B.x, A.y - B.y, A.z - B.z); }
FORCEINLINE float3 operator*(const float3 A, const float3 B) { return MakeFloat3(A.x * B.x, A.y * B.y, A.z * B.z); }
FORCEINLINE float3 operator+(const float3 A, const float B) { return MakeFloat3(A.x + B, A.y + B, A.z + B); }
FORCEINLINE float3 operator-(const float3 A, const float B) { return MakeFloat3(A.x - B, A.y - B, A.z - B); }
FORCEINLINE float3 operator*(const float3 A, const float B) { return MakeFloat3(A.x * B, A.y * B, A.z * B); }
FORCEINLINE float3 operator*(const float A, const float3 B) { return MakeFloat3(A * B.x, A * B.y, A * B.z); }
FORCEINLINE float3 operator+(const float A, const float3 B) { return MakeFloat3(A + B.x, A + B.y, A + B.z); }
FORCEINLINE float3 operator-(const float A, const float3 B) { return MakeFloat3(A - B.x, A - B.y, A - B.z); }
FORCEINLINE float3 operator/(const float3 A, const float B) { return MakeFloat3(A.x / B, A.y / B, A.z / B); }

FORCEINLINE float dot(const float2 A, const float2 B) {
	return A.x * B.x + A.y * B.y;
}
FORCEINLINE float dot(const float3 A, const float3 B) {
	return A.x * B.x + A.y * B.y + A.z * B.z;
}
FORCEINLINE float length(const float2 A) {
	return sqrt(dot(A, A));
}
FORCEINLINE float length(const float3 A) {
	return sqrt(dot(A, A));
}
FORCEINLINE float2 normalize(const float2 A) {
	return A / length(A);
}
FORCEINLINE float3 normalize(const float3 A) {
	return A / length(A);
}
FORCEINLINE float3 cross(const float3 A, const float3 B) {
	return MakeFloat3(
			A.y * B.z - A.z * B.y,
			A.z * B.x - A.x * B.z,
			A.x * B.y - A.y * B.x);
}
FORCEINLINE float2 floor(const float2 A) {
	return MakeFloat2(floor(A.x), floor(A.y));
}
FORCEINLINE float3 floor(const float3 A) {
	return MakeFloat3(floor(A.x), floor(A.y), floor(A.z));
}
FORCEINLINE float2 abs(const float2 A) {
	return MakeFloat2(abs(A.x), abs(A.y));
}
FORCEINLINE float3 abs(const float3 A) {
	return MakeFloat3(abs(A.x), abs(A.y), abs(A.z));
}

FORCEINLINE float FCLerp(const float A, const float B, const float Alpha) {
	return A + Alpha * (B - A);
}
FORCEINLINE float FCMax3(const float A, const float B, const float C) {
	return max(A, max(B, C));
}
FORCEINLINE bool FCIsFinite(const float Value) {
	// False for NaN and infinity
	return abs(Value) <= 3.402823e38f;
}
FORCEINLINE float FCFract(const float Value) {
	return Value - floor(Value);
}
FORCEINLINE float2 FCFract(const float2 Value) {
	return Value - floor(Value);
}
FORCEINLINE float3 FCFract(const float3 Value) {
	return Value - floor(Value);
}

// GLSL-style smoothstep, supports Edge0 > Edge1
FORCEINLINE float FCSmoothstep(const float Edge0, const float Edge1, const float Value) {
	const float Alpha = clamp((Value - Edge0) / (Edge1 - Edge0), 0.f, 1.f);
	return Alpha * Alpha * (3.f - 2.f * Alpha);
}

FORCEINLINE float2 FCClamp01(const float2 Value) {
	return MakeFloat2(clamp(Value.x, 0.f, 1.f), clamp(Value.y, 0.f, 1.f));
}
FORCEINLINE float3 FCClamp01(const float3 Value) {
	return MakeFloat3(clamp(Value.x, 0.f, 1.f), clamp(Value.y, 0.f, 1.f), clamp(Value.z, 0.f, 1.f));
}

///////////////////////////////////////////////////////////////////////////////
// Seeded ports of the fi hashes
///////////////////////////////////////////////////////////////////////////////

FORCEINLINE uint32 FCHashBase(const uint32 Seed, const float2 Position) {
	const uint32 X = intbits(Position.x * 141421356.f);
	const uint32 Y = intbits(Position.y * 2718281828.f);
	return X ^ Y ^ Seed;
}
FORCEINLINE uint32 FCHashBase(const uint32 Seed, const float3 Position) {
	const uint32 X = intbits(Position.x * 141421356.f);
	const uint32 Y = intbits(Position.y * 2718281828.f);
	const uint32 Z = intbits(Position.z * 1618033988.f);
	return X ^ Y ^ Z ^ Seed;
}

// Signed conversion is much faster than uint32 to float in ISPC
FORCEINLINE float FCUintToFloat01(const uint32 Value) {
	return (float)((int32)Value) * (1.f / 4294967296.f) + 0.5f;
}

FORCEINLINE float FCHash12(const uint32 Seed, const float2 Position) {
	return FCUintToFloat01(FCHashBase(Seed, Position) * 3141592653u);
}
FORCEINLINE float2 FCHash22(const uint32 Seed, const float2 Position) {
	const uint32 Hash = FCHashBase(Seed, Position);
	return MakeFloat2(
			FCUintToFloat01(Hash * 3141592653u),
			FCUintToFloat01(Hash * 1618033988u));
}
FORCEINLINE float3 FCHash32(const uint32 Seed, const float2 Position) {
	const uint32 Hash = FCHashBase(Seed, Position);
	return MakeFloat3(
			FCUintToFloat01(Hash * 1732050807u),
			FCUintToFloat01(Hash * 2645751311u),
			FCUintToFloat01(Hash * 3316624790u));
}
FORCEINLINE float FCHash13(const uint32 Seed, const float3 Position) {
	return FCUintToFloat01(FCHashBase(Seed, Position) * 3141592653u);
}
FORCEINLINE float3 FCHash33(const uint32 Seed, const float3 Position) {
	const uint32 Hash = FCHashBase(Seed, Position);
	return MakeFloat3(
			FCUintToFloat01(Hash * 1732050807u),
			FCUintToFloat01(Hash * 2645751311u),
			FCUintToFloat01(Hash * 3316624790u));
}

// Pseudo random offset used to seed noises whose formulation has no hash to mix a seed into
FORCEINLINE float2 FCSeedOffset2(const uint32 Seed) {
	const uint32 Hash = (Seed ^ 0x9E3779B9u) * 3141592653u;
	return MakeFloat2(
			(float)((int32)(Hash & 0xffffu)) * (1024.f / 65535.f),
			(float)((int32)((Hash >> 16) & 0xffffu)) * (1024.f / 65535.f));
}
FORCEINLINE float3 FCSeedOffset3(const uint32 Seed) {
	const uint32 Hash = (Seed ^ 0x9E3779B9u) * 3141592653u;
	const uint32 TenBits = (1u << 10) - 1;
	return MakeFloat3(
			(float)((int32)((Hash >> 0) & TenBits)),
			(float)((int32)((Hash >> 10) & TenBits)),
			(float)((int32)((Hash >> 20) & TenBits)));
}

///////////////////////////////////////////////////////////////////////////////
// 2D noises
///////////////////////////////////////////////////////////////////////////////

FORCEINLINE float FCValue2D(const uint32 Seed, const float2 Position) {
	const float2 Cell = floor(Position);
	float2 Alpha = Position - Cell;
	Alpha = Alpha * Alpha * (3.f - 2.f * Alpha);

	const float Result = FCLerp(
			FCLerp(FCHash12(Seed, Cell), FCHash12(Seed, Cell + MakeFloat2(1.f, 0.f)), Alpha.x),
			FCLerp(FCHash12(Seed, Cell + MakeFloat2(0.f, 1.f)), FCHash12(Seed, Cell + MakeFloat2(1.f, 1.f)), Alpha.x),
			Alpha.y);

	return Result * 2.f - 1.f;
}

// Returns the raw gradient noise value, roughly [-0.7, 0.7]
FORCEINLINE float FCPerlin2D_Raw(const uint32 Seed, const float2 Position) {
	const float2 Cell = floor(Position);
	const float2 Local = Position - Cell;
	const float2 Alpha = Local * Local * Local * (10.f + Local * (6.f * Local - 15.f));

	const float NoiseA = dot(normalize(FCHash22(Seed, Cell + MakeFloat2(0.f, 0.f)) - 0.5f), Local - MakeFloat2(0.f, 0.f));
	const float NoiseB = dot(normalize(FCHash22(Seed, Cell + MakeFloat2(1.f, 0.f)) - 0.5f), Local - MakeFloat2(1.f, 0.f));
	const float NoiseC = dot(normalize(FCHash22(Seed, Cell + MakeFloat2(0.f, 1.f)) - 0.5f), Local - MakeFloat2(0.f, 1.f));
	const float NoiseD = dot(normalize(FCHash22(Seed, Cell + MakeFloat2(1.f, 1.f)) - 0.5f), Local - MakeFloat2(1.f, 1.f));

	return FCLerp(FCLerp(NoiseA, NoiseB, Alpha.x), FCLerp(NoiseC, NoiseD, Alpha.x), Alpha.y);
}
FORCEINLINE float FCPerlin2D(const uint32 Seed, const float2 Position) {
	return FCPerlin2D_Raw(Seed, Position) * 1.4f;
}

// Gradient noise with analytic derivative, from https://iquilezles.org/articles/gradientnoise/
FORCEINLINE float FCPerlinDeriv2D(const uint32 Seed, const float2 Position, varying float2 *OutGradient) {
	const float2 Cell = floor(Position);
	const float2 Local = Position - Cell;

	const float2 Alpha = Local * Local * Local * (Local * (Local * 6.f - 15.f) + 10.f);
	const float2 DeltaAlpha = 30.f * Local * Local * (Local * (Local - 2.f) + 1.f);

	const float2 GradientA = FCHash22(Seed, Cell + MakeFloat2(0.f, 0.f)) * 2.f - 1.f;
	const float2 GradientB = FCHash22(Seed, Cell + MakeFloat2(1.f, 0.f)) * 2.f - 1.f;
	const float2 GradientC = FCHash22(Seed, Cell + MakeFloat2(0.f, 1.f)) * 2.f - 1.f;
	const float2 GradientD = FCHash22(Seed, Cell + MakeFloat2(1.f, 1.f)) * 2.f - 1.f;

	const float ValueA = dot(GradientA, Local - MakeFloat2(0.f, 0.f));
	const float ValueB = dot(GradientB, Local - MakeFloat2(1.f, 0.f));
	const float ValueC = dot(GradientC, Local - MakeFloat2(0.f, 1.f));
	const float ValueD = dot(GradientD, Local - MakeFloat2(1.f, 1.f));

	*OutGradient =
			GradientA +
			Alpha.x * (GradientB - GradientA) +
			Alpha.y * (GradientC - GradientA) +
			Alpha.x * Alpha.y * (GradientA - GradientB - GradientC + GradientD) +
			DeltaAlpha * (MakeFloat2(Alpha.y, Alpha.x) * (ValueA - ValueB - ValueC + ValueD) + MakeFloat2(ValueB, ValueC) - ValueA);

	return ValueA +
			Alpha.x * (ValueB - ValueA) +
			Alpha.y * (ValueC - ValueA) +
			Alpha.x * Alpha.y * (ValueA - ValueB - ValueC + ValueD);
}

FORCEINLINE float FCSimplex2D(const uint32 Seed, const float2 Position) {
	const float2 Cell = floor(Position + (Position.x + Position.y) * 0.366025f);
	const float2 PositionA = Position - Cell + (Cell.x + Cell.y) * 0.211324f;

	const float Mask = PositionA.x >= PositionA.y ? 1.f : 0.f;
	const float2 PositionB = PositionA - MakeFloat2(Mask, 1.f - Mask) + 0.211324f;
	const float2 PositionC = PositionA - 0.577351f;

	float AlphaA = max(0.5f - dot(PositionA, PositionA), 0.f);
	float AlphaB = max(0.5f - dot(PositionB, PositionB), 0.f);
	float AlphaC = max(0.5f - dot(PositionC, PositionC), 0.f);

	AlphaA = AlphaA * AlphaA * AlphaA * AlphaA;
	AlphaB = AlphaB * AlphaB * AlphaB * AlphaB;
	AlphaC = AlphaC * AlphaC * AlphaC * AlphaC;

	const float Result =
			AlphaA * dot(PositionA, FCHash22(Seed, Cell) - 0.5f) +
			AlphaB * dot(PositionB, FCHash22(Seed, Cell + MakeFloat2(Mask, 1.f - Mask)) - 0.5f) +
			AlphaC * dot(PositionC, FCHash22(Seed, Cell + 1.f) - 0.5f);

	return Result * 140.f;
}

FORCEINLINE float FCWorley2D(const uint32 Seed, const float2 Position) {
	const float2 Cell = floor(Position);
	const float2 Local = Position - Cell;

	float Distance = 1e6f;
	for (uniform int32 IndexX = -1; IndexX <= 1; IndexX++) {
		for (uniform int32 IndexY = -1; IndexY <= 1; IndexY++) {
			const float2 Offset = MakeFloat2(IndexX, IndexY);
			const float2 Delta = Local - Offset - FCHash22(Seed, Cell + Offset);
			Distance = min(Distance, dot(Delta, Delta));
		}
	}

	return (1.f - sqrt(Distance)) * 2.f - 1.f;
}

FORCEINLINE float FCVoronoi2D(const uint32 Seed, const float2 Position, const float Smoothness) {
	const float Sharpness = 1.f / max(Smoothness, 0.001f);

	const float2 Cell = floor(Position);
	const float2 Local = Position - Cell;

	float ValueSum = 0.f;
	float WeightSum = 0.f;
	for (uniform int32 IndexX = -1; IndexX <= 1; IndexX++) {
		for (uniform int32 IndexY = -1; IndexY <= 1; IndexY++) {
			const float2 Offset = MakeFloat2(IndexX, IndexY);
			const float3 Random = FCHash32(Seed, Cell + Offset);
			const float Distance = length(Offset - Local + MakeFloat2(Random.x, Random.y));
			const float Weight = pow(FCSmoothstep(1.414f, 0.f, Distance), Sharpness);
			ValueSum += Random.z * Weight;
			WeightSum += Weight;
		}
	}

	return ValueSum / max(WeightSum, 1e-6f) * 2.f - 1.f;
}

// High-pass filtered white noise, from https://www.shadertoy.com/view/tllcR2
FORCEINLINE float FCBlue2D(const uint32 Seed, const float2 Position) {
	const float2 Cell = floor(Position);

	float Sum = 0.f;
	for (uniform int32 IndexX = -1; IndexX <= 1; IndexX++) {
		for (uniform int32 IndexY = -1; IndexY <= 1; IndexY++) {
			Sum += FCHash12(Seed, Cell + MakeFloat2(IndexX, IndexY));
		}
	}

	return (0.9f * (1.125f * FCHash12(Seed, Cell) - Sum / 8.f) + 0.5f) * 2.f - 1.f;
}

// Hilbert curve based low-discrepancy noise, modified from https://www.shadertoy.com/view/3tB3z3
FORCEINLINE float FCHilbertBlue2D(const uint32 Seed, const float2 Position) {
	const uint32 SeedHash = (Seed ^ 0x9E3779B9u) * 3141592653u;

	int32 X = ((int32)floor(Position.x) + (int32)(SeedHash & 511u)) & 511;
	int32 Y = ((int32)floor(Position.y) + (int32)((SeedHash >> 9) & 511u)) & 511;

	int32 Result = 0;
	for (uniform int32 Mask = 512 >> 1; Mask > 0; Mask >>= 1) {
		const int32 RegionX = (X & Mask) != 0 ? 1 : 0;
		const int32 RegionY = (Y & Mask) != 0 ? 1 : 0;
		Result += Mask * Mask * ((RegionX << 1) | (RegionX ^ RegionY));

		const int32 Flip = ((X ^ Y) * (1 - RegionY)) ^ ((Mask - 1) * (RegionX & (1 - RegionY)));
		X ^= Flip;
		Y ^= Flip;
	}
	Result = Result & 262143;

	// Compute in double: fract of golden ratio times a large index needs more than float precision
	const double GoldenIndex = FC_GOLDEN_RATIO * (double)Result;
	return (float)(GoldenIndex - floor(GoldenIndex)) * 2.f - 1.f;
}

// Impact crater rings, modified from https://www.shadertoy.com/view/XsGBDt
FORCEINLINE float FCCrater2D(const uint32 Seed, const float2 Position) {
	const float2 Cell = floor(Position);
	const float2 Local = Position - Cell;

	float ValueSum = 0.f;
	float WeightSum = 0.f;
	for (uniform int32 IndexX = -2; IndexX <= 2; IndexX++) {
		for (uniform int32 IndexY = -2; IndexY <= 2; IndexY++) {
			const float2 Offset = MakeFloat2(IndexX, IndexY);
			const float2 Random = FCHash22(Seed, Cell + Offset);
			const float Distance = length(Local - Offset - Random);
			const float Weight = exp(-4.f * Distance);
			ValueSum += Weight * sin(6.28318530718f * sqrt(max(Distance, 0.06f)));
			WeightSum += Weight;
		}
	}

	return abs(ValueSum / max(WeightSum, 1e-6f)) * 2.f - 1.f;
}

FORCEINLINE float FCGabor2D(const uint32 Seed, const float2 Position) {
	const uniform float Frequency = 8.f;

	const float2 Cell = floor(Position);
	float2 Alpha = Position - Cell;
	Alpha = Alpha * Alpha * (3.f - 2.f * Alpha);

	return FCLerp(
			FCLerp(
					sin(Frequency * dot(Position, FCHash22(Seed, Cell + MakeFloat2(0.f, 0.f)))),
					sin(Frequency * dot(Position, FCHash22(Seed, Cell + MakeFloat2(1.f, 0.f)))),
					Alpha.x),
			FCLerp(
					sin(Frequency * dot(Position, FCHash22(Seed, Cell + MakeFloat2(0.f, 1.f)))),
					sin(Frequency * dot(Position, FCHash22(Seed, Cell + MakeFloat2(1.f, 1.f)))),
					Alpha.x),
			Alpha.y);
}

// Magnitude of the finite-difference curl of gradient noise
FORCEINLINE float FCCurl2D(const uint32 Seed, const float2 Position) {
	const uniform float Epsilon = 0.1f;

	const float2 CurlValue = MakeFloat2(
									 FCPerlin2D_Raw(Seed, Position + MakeFloat2(Epsilon, 0.f)) - FCPerlin2D_Raw(Seed, Position - MakeFloat2(Epsilon, 0.f)),
									 FCPerlin2D_Raw(Seed, Position + MakeFloat2(0.f, Epsilon)) - FCPerlin2D_Raw(Seed, Position - MakeFloat2(0.f, Epsilon))) /
			Epsilon * 0.5f * 0.7f;

	return length(CurlValue) / 1.414f * 2.f - 1.f;
}

// Single layer of thin wavy lines, inspired from https://www.shadertoy.com/view/4syXRD
FORCEINLINE float FCScratchLayer2D(const uint32 Seed, const float2 Position, const float Smoothness) {
	const uniform float Thickness = 0.02f;
	const uniform float Wavyness = 0.5f;

	const float2 Cell = floor(Position);
	const float2 Random = FCHash22(Seed, Cell) * MakeFloat2(3104.f, 554.f);

	float2 Local = (Position - Cell) * 2.f - 1.f;

	const float SinAngle = sin(Random.x + Random.y);
	const float CosAngle = cos(Random.x + Random.y);
	Local = Local * CosAngle + MakeFloat2(-Local.y, Local.x) * SinAngle;
	Local = Local + sin(Random.x - Random.y);

	float Line = abs(Local.x - cos(Random.x + Local.y * 1.57f) * Wavyness);
	Line = FCSmoothstep(Thickness + Smoothness, Thickness - Smoothness, Line);
	Line *= Local.y * 0.5f + 0.5f;

	return Line;
}
FORCEINLINE float FCScratch2D(const uint32 Seed, const float2 Position, const float Smoothness) {
	float2 Local = Position;
	float Width = max(Smoothness, 0.001f);

	float Scratches = 0.f;
	for (uniform int32 Index = 0; Index < 8; Index++) {
		Scratches = max(Scratches, FCScratchLayer2D(Seed, Local, Width));
		Local = MakeFloat2(
						Local.x + 0.7f * Local.y,
						-0.7f * Local.x + Local.y) -
				12.31f;
		Width *= 1.22f;
	}

	return Scratches * 2.f - 1.f;
}

// Rotated sine wavelets, from https://www.shadertoy.com/view/wsBfzK
FORCEINLINE float FCWavelet2D(const uint32 Seed, const float2 Position, const float Phase) {
	const uniform float Scale = 1.24f;

	float2 Local = Position + FCSeedOffset2(Seed);

	float Value = 0.f;
	float Frequency = 1.f;
	float WeightSum = 0.f;
	for (uniform int32 Index = 0; Index < 4; Index++) {
		const float2 Scaled = Local * Frequency;

		float2 Random = FCFract(floor(Scaled) * MakeFloat2(123.34f, 233.53f));
		Random = Random + dot(Random, Random + 23.234f);
		const float Angle = FCFract(Random.x * Random.y) * 1e3f;

		const float2 Centered = FCFract(Scaled) - 0.5f;
		const float SinAngle = sin(Angle);
		const float CosAngle = cos(Angle);
		const float2 Rotated = MakeFloat2(
				Centered.x * CosAngle + Centered.y * SinAngle,
				-Centered.x * SinAngle + Centered.y * CosAngle);

		Value += sin(Rotated.x * 10.f + Phase) * FCSmoothstep(0.25f, 0.f, dot(Rotated, Rotated)) / Frequency;

		Local = MakeFloat2(
						0.54f * Local.x - 0.84f * Local.y,
						0.84f * Local.x + 0.54f * Local.y) +
				(uniform float)Index;

		WeightSum += 1.f / Frequency;
		Frequency *= Scale;
	}

	return Value / WeightSum;
}

FORCEINLINE float3 FCGullies2D(const uint32 Seed, const float2 Position, const float2 Slope) {
	const float2 SideDirection = MakeFloat2(-Slope.y, Slope.x) * 3.14159265f;

	const float2 Cell = floor(Position);
	const float2 Local = Position - Cell;

	float2 HeightSlope = MakeFloat2(0.f, 0.f);
	float WeightSum = 0.f;
	for (uniform int32 IndexX = -1; IndexX <= 2; IndexX++) {
		for (uniform int32 IndexY = -1; IndexY <= 2; IndexY++) {
			const float2 Offset = MakeFloat2(IndexX, IndexY);
			const float2 Delta = Local - Offset - FCHash22(Seed, Cell + Offset) + 0.5f;
			const float DistanceSquared = dot(Delta, Delta);
			const float Weight = max(0.f, exp(-DistanceSquared * 2.f) - 0.01111f);
			WeightSum += Weight;

			const float Along = dot(Delta, SideDirection);
			HeightSlope = HeightSlope + MakeFloat2(cos(Along), -sin(Along)) * Weight;
		}
	}

	WeightSum = max(WeightSum, 1e-6f);
	return MakeFloat3(
				   HeightSlope.x,
				   HeightSlope.y * SideDirection.x,
				   HeightSlope.y * SideDirection.y) /
			WeightSum;
}
// Gradient noise with slope-following gullies, modified from https://www.shadertoy.com/view/sf23W1
FORCEINLINE float FCErosion2D(const uint32 Seed, const float2 Position) {
	float2 Gradient;
	float Value = FCPerlinDeriv2D(Seed, Position, &Gradient);

	float Strength = 0.25f;
	float Frequency = 8.f;
	float Total = 1.f;
	for (uniform int32 Index = 0; Index < 4; Index++) {
		const float SlopeLengthSquared = max(dot(Gradient, Gradient), 1e-12f);
		const float3 Gully = FCGullies2D(Seed, Position * Frequency, Gradient * pow(SlopeLengthSquared, -0.25f));

		Value += Gully.x * Strength;
		Gradient = Gradient + MakeFloat2(Gully.y, Gully.z) * Strength * Frequency;

		Total += Strength;
		Strength *= 0.5f;
		Frequency *= 2.f;
	}

	return Value / Total;
}

FORCEINLINE float FCPaper2D(const uint32 Seed, const float2 Position) {
	float2 Local = Position;

	float2 Sum = MakeFloat2(0.f, 0.f);
	float WeightSum = 0.f;
	float Weight = 1.f;
	for (uniform int32 Index = 0; Index < 10; Index++) {
		float2 Gradient;
		FCPerlinDeriv2D(Seed, Local, &Gradient);

		Sum = Sum + FCClamp01(Gradient * 0.5f + 0.5f) * Weight;
		WeightSum += Weight;
		Weight *= 0.8f;
		Local = Local * 2.f;
	}
	Sum = Sum / WeightSum;

	return (length(Sum) / 1.414f * 0.6f + 0.4f) * 2.f - 1.f;
}

FORCEINLINE float FCStone2D(const uint32 Seed, const float2 Position) {
	float2 WarpGradient = MakeFloat2(0.f, 0.f);
	{
		float2 Local = Position;
		float Weight = 1.f;
		for (uniform int32 Index = 0; Index < 6; Index++) {
			float2 Gradient;
			FCPerlinDeriv2D(Seed, Local, &Gradient);

			WarpGradient = WarpGradient + Gradient * Weight;
			Weight *= 0.5f;
			Local = Local * 2.f;
		}
	}

	float2 Local = Position + WarpGradient * 0.4f;

	float Sum = 0.f;
	float WeightSum = 0.f;
	float Weight = 1.f;
	for (uniform int32 Index = 0; Index < 6; Index++) {
		Sum += (FCPerlin2D_Raw(Seed, Local) * 0.7f + 0.5f) * Weight;
		WeightSum += Weight;
		Weight *= 0.5f;
		Local = Local * 2.f;
	}

	return Sum / WeightSum * 2.f - 1.f;
}

FORCEINLINE float FCWool2D(const uint32 Seed, const float2 Position) {
	float2 Local = Position;

	float2 Sum = MakeFloat2(0.f, 0.f);
	float WeightSum = 0.f;
	float Weight = 1.f;
	for (uniform int32 Index = 0; Index < 6; Index++) {
		float2 Gradient;
		FCPerlinDeriv2D(Seed, Local, &Gradient);

		Sum = Sum + Gradient * Weight;
		WeightSum += Weight;
		Weight *= 0.5f;
		Local = Local * 2.f;
	}
	Sum = Sum / WeightSum;

	return max(abs(Sum.x), abs(Sum.y)) * 2.f - 1.f;
}

// Interleaved gradient noise
FORCEINLINE float FCInterleavedGradient2D(const uint32 Seed, const float2 Position) {
	const float2 Local = Position + FCSeedOffset2(Seed);
	return FCFract(52.9829189f * FCFract(dot(Local, MakeFloat2(0.06711056f, 0.00583715f)))) * 2.f - 1.f;
}

///////////////////////////////////////////////////////////////////////////////
// 3D noises
///////////////////////////////////////////////////////////////////////////////

FORCEINLINE float FCValue3D(const uint32 Seed, const float3 Position) {
	const float3 Cell = floor(Position);
	float3 Alpha = Position - Cell;
	Alpha = Alpha * Alpha * (3.f - 2.f * Alpha);

	const float Result = FCLerp(
			FCLerp(
					FCLerp(FCHash13(Seed, Cell + MakeFloat3(0.f, 0.f, 0.f)), FCHash13(Seed, Cell + MakeFloat3(1.f, 0.f, 0.f)), Alpha.x),
					FCLerp(FCHash13(Seed, Cell + MakeFloat3(0.f, 1.f, 0.f)), FCHash13(Seed, Cell + MakeFloat3(1.f, 1.f, 0.f)), Alpha.x),
					Alpha.y),
			FCLerp(
					FCLerp(FCHash13(Seed, Cell + MakeFloat3(0.f, 0.f, 1.f)), FCHash13(Seed, Cell + MakeFloat3(1.f, 0.f, 1.f)), Alpha.x),
					FCLerp(FCHash13(Seed, Cell + MakeFloat3(0.f, 1.f, 1.f)), FCHash13(Seed, Cell + MakeFloat3(1.f, 1.f, 1.f)), Alpha.x),
					Alpha.y),
			Alpha.z);

	return Result * 2.f - 1.f;
}

// Returns the raw gradient noise value, roughly [-0.7, 0.7]
FORCEINLINE float FCPerlin3D_Raw(const uint32 Seed, const float3 Position) {
	const float3 Cell = floor(Position);
	const float3 Local = Position - Cell;
	const float3 Alpha = Local * Local * Local * (10.f + Local * (6.f * Local - 15.f));

#define FC_CORNER(X, Y, Z) dot(normalize(FCHash33(Seed, Cell + MakeFloat3(X, Y, Z)) - 0.5f), Local - MakeFloat3(X, Y, Z))
	const float NoiseA = FC_CORNER(0.f, 0.f, 0.f);
	const float NoiseB = FC_CORNER(1.f, 0.f, 0.f);
	const float NoiseC = FC_CORNER(0.f, 1.f, 0.f);
	const float NoiseD = FC_CORNER(1.f, 1.f, 0.f);
	const float NoiseE = FC_CORNER(0.f, 0.f, 1.f);
	const float NoiseF = FC_CORNER(1.f, 0.f, 1.f);
	const float NoiseG = FC_CORNER(0.f, 1.f, 1.f);
	const float NoiseH = FC_CORNER(1.f, 1.f, 1.f);
#undef FC_CORNER

	const float LayerA = FCLerp(FCLerp(NoiseA, NoiseB, Alpha.x), FCLerp(NoiseC, NoiseD, Alpha.x), Alpha.y);
	const float LayerB = FCLerp(FCLerp(NoiseE, NoiseF, Alpha.x), FCLerp(NoiseG, NoiseH, Alpha.x), Alpha.y);

	return FCLerp(LayerA, LayerB, Alpha.z);
}
FORCEINLINE float FCPerlin3D(const uint32 Seed, const float3 Position) {
	return FCPerlin3D_Raw(Seed, Position) * 1.4f;
}

// Gradient noise with analytic derivative, from https://iquilezles.org/articles/gradientnoise/
FORCEINLINE float FCPerlinDeriv3D(const uint32 Seed, const float3 Position, varying float3 *OutGradient) {
	const float3 Cell = floor(Position);
	const float3 Local = Position - Cell;

	const float3 Alpha = Local * Local * Local * (Local * (Local * 6.f - 15.f) + 10.f);
	const float3 DeltaAlpha = 30.f * Local * Local * (Local * (Local - 2.f) + 1.f);

	const float3 GradientA = FCHash33(Seed, Cell + MakeFloat3(0.f, 0.f, 0.f)) * 2.f - 1.f;
	const float3 GradientB = FCHash33(Seed, Cell + MakeFloat3(1.f, 0.f, 0.f)) * 2.f - 1.f;
	const float3 GradientC = FCHash33(Seed, Cell + MakeFloat3(0.f, 1.f, 0.f)) * 2.f - 1.f;
	const float3 GradientD = FCHash33(Seed, Cell + MakeFloat3(1.f, 1.f, 0.f)) * 2.f - 1.f;
	const float3 GradientE = FCHash33(Seed, Cell + MakeFloat3(0.f, 0.f, 1.f)) * 2.f - 1.f;
	const float3 GradientF = FCHash33(Seed, Cell + MakeFloat3(1.f, 0.f, 1.f)) * 2.f - 1.f;
	const float3 GradientG = FCHash33(Seed, Cell + MakeFloat3(0.f, 1.f, 1.f)) * 2.f - 1.f;
	const float3 GradientH = FCHash33(Seed, Cell + MakeFloat3(1.f, 1.f, 1.f)) * 2.f - 1.f;

	const float ValueA = dot(GradientA, Local - MakeFloat3(0.f, 0.f, 0.f));
	const float ValueB = dot(GradientB, Local - MakeFloat3(1.f, 0.f, 0.f));
	const float ValueC = dot(GradientC, Local - MakeFloat3(0.f, 1.f, 0.f));
	const float ValueD = dot(GradientD, Local - MakeFloat3(1.f, 1.f, 0.f));
	const float ValueE = dot(GradientE, Local - MakeFloat3(0.f, 0.f, 1.f));
	const float ValueF = dot(GradientF, Local - MakeFloat3(1.f, 0.f, 1.f));
	const float ValueG = dot(GradientG, Local - MakeFloat3(0.f, 1.f, 1.f));
	const float ValueH = dot(GradientH, Local - MakeFloat3(1.f, 1.f, 1.f));

	const float CornerSum = -ValueA + ValueB + ValueC - ValueD + ValueE - ValueF - ValueG + ValueH;
	const float3 GradientCornerSum =
			-1.f * GradientA + GradientB + GradientC - GradientD + GradientE - GradientF - GradientG + GradientH;

	const float3 AlphaYZX = MakeFloat3(Alpha.y, Alpha.z, Alpha.x);
	const float3 AlphaZXY = MakeFloat3(Alpha.z, Alpha.x, Alpha.y);

	*OutGradient =
			GradientA +
			Alpha.x * (GradientB - GradientA) +
			Alpha.y * (GradientC - GradientA) +
			Alpha.z * (GradientE - GradientA) +
			Alpha.x * Alpha.y * (GradientA - GradientB - GradientC + GradientD) +
			Alpha.y * Alpha.z * (GradientA - GradientC - GradientE + GradientG) +
			Alpha.z * Alpha.x * (GradientA - GradientB - GradientE + GradientF) +
			Alpha.x * Alpha.y * Alpha.z * GradientCornerSum +
			DeltaAlpha * (MakeFloat3(ValueB, ValueC, ValueE) - ValueA +
						  AlphaYZX * MakeFloat3(ValueA - ValueB - ValueC + ValueD, ValueA - ValueC - ValueE + ValueG, ValueA - ValueB - ValueE + ValueF) +
						  AlphaZXY * MakeFloat3(ValueA - ValueB - ValueE + ValueF, ValueA - ValueB - ValueC + ValueD, ValueA - ValueC - ValueE + ValueG) +
						  AlphaYZX * AlphaZXY * CornerSum);

	return ValueA +
			Alpha.x * (ValueB - ValueA) +
			Alpha.y * (ValueC - ValueA) +
			Alpha.z * (ValueE - ValueA) +
			Alpha.x * Alpha.y * (ValueA - ValueB - ValueC + ValueD) +
			Alpha.y * Alpha.z * (ValueA - ValueC - ValueE + ValueG) +
			Alpha.z * Alpha.x * (ValueA - ValueB - ValueE + ValueF) +
			Alpha.x * Alpha.y * Alpha.z * CornerSum;
}

FORCEINLINE float FCSimplex3D(const uint32 Seed, const float3 Position) {
	const float3 Skewed = floor(Position + (Position.x + Position.y + Position.z) * (1.f / 3.f));
	const float3 PositionA = Position - Skewed + (Skewed.x + Skewed.y + Skewed.z) * (1.f / 6.f);

	const float EdgeX = PositionA.x >= PositionA.y ? 1.f : 0.f;
	const float EdgeY = PositionA.y >= PositionA.z ? 1.f : 0.f;
	const float EdgeZ = PositionA.z >= PositionA.x ? 1.f : 0.f;

	// e = step(0, x - x.yzx); i1 = e * (1 - e.zxy); i2 = 1 - e.zxy * (1 - e)
	const float3 Edge = MakeFloat3(EdgeX, EdgeY, EdgeZ);
	const float3 Corner1 = Edge * MakeFloat3(1.f - EdgeZ, 1.f - EdgeX, 1.f - EdgeY);
	const float3 Corner2 = MakeFloat3(1.f, 1.f, 1.f) - (MakeFloat3(EdgeZ, EdgeX, EdgeY) * (MakeFloat3(1.f, 1.f, 1.f) - Edge));

	const float3 PositionB = PositionA - Corner1 + (1.f / 6.f);
	const float3 PositionC = PositionA - Corner2 + (1.f / 3.f);
	const float3 PositionD = PositionA - 0.5f;

	float AlphaA = max(0.6f - dot(PositionA, PositionA), 0.f);
	float AlphaB = max(0.6f - dot(PositionB, PositionB), 0.f);
	float AlphaC = max(0.6f - dot(PositionC, PositionC), 0.f);
	float AlphaD = max(0.6f - dot(PositionD, PositionD), 0.f);

	AlphaA = AlphaA * AlphaA * AlphaA * AlphaA;
	AlphaB = AlphaB * AlphaB * AlphaB * AlphaB;
	AlphaC = AlphaC * AlphaC * AlphaC * AlphaC;
	AlphaD = AlphaD * AlphaD * AlphaD * AlphaD;

	const float Result =
			AlphaA * dot(FCHash33(Seed, Skewed) - 0.5f, PositionA) +
			AlphaB * dot(FCHash33(Seed, Skewed + Corner1) - 0.5f, PositionB) +
			AlphaC * dot(FCHash33(Seed, Skewed + Corner2) - 0.5f, PositionC) +
			AlphaD * dot(FCHash33(Seed, Skewed + 1.f) - 0.5f, PositionD);

	return Result * 52.f;
}

FORCEINLINE float FCWorley3D(const uint32 Seed, const float3 Position) {
	const float3 Cell = floor(Position);
	const float3 Local = Position - Cell;

	float Distance = 1e6f;
	for (uniform int32 IndexX = -1; IndexX <= 1; IndexX++) {
		for (uniform int32 IndexY = -1; IndexY <= 1; IndexY++) {
			for (uniform int32 IndexZ = -1; IndexZ <= 1; IndexZ++) {
				const float3 Offset = MakeFloat3(IndexX, IndexY, IndexZ);
				const float3 Delta = Local - Offset - FCHash33(Seed, Cell + Offset);
				Distance = min(Distance, dot(Delta, Delta));
			}
		}
	}

	return (1.f - sqrt(Distance)) * 2.f - 1.f;
}

FORCEINLINE float FCVoronoi3D(const uint32 Seed, const float3 Position, const float Smoothness) {
	const float Sharpness = 1.f / max(Smoothness, 0.001f);

	const float3 Cell = floor(Position);
	const float3 Local = Position - Cell;

	float ValueSum = 0.f;
	float WeightSum = 0.f;
	for (uniform int32 IndexX = -1; IndexX <= 1; IndexX++) {
		for (uniform int32 IndexY = -1; IndexY <= 1; IndexY++) {
			for (uniform int32 IndexZ = -1; IndexZ <= 1; IndexZ++) {
				const float3 Offset = MakeFloat3(IndexX, IndexY, IndexZ);
				const float3 Random = FCHash33(Seed, Cell + Offset);
				const float Value = FCHash13(Seed ^ 0x9E3779B9u, Cell + Offset);
				const float Distance = length(Offset - Local + Random);
				const float Weight = pow(FCSmoothstep(1.732f, 0.f, Distance), Sharpness);
				ValueSum += Value * Weight;
				WeightSum += Weight;
			}
		}
	}

	return ValueSum / max(WeightSum, 1e-6f) * 2.f - 1.f;
}

// High-pass filtered white noise, 3D extension of https://www.shadertoy.com/view/tllcR2
FORCEINLINE float FCBlue3D(const uint32 Seed, const float3 Position) {
	const float3 Cell = floor(Position);

	float Sum = 0.f;
	for (uniform int32 IndexX = -1; IndexX <= 1; IndexX++) {
		for (uniform int32 IndexY = -1; IndexY <= 1; IndexY++) {
			for (uniform int32 IndexZ = -1; IndexZ <= 1; IndexZ++) {
				Sum += FCHash13(Seed, Cell + MakeFloat3(IndexX, IndexY, IndexZ));
			}
		}
	}

	return (0.9f * (1.125f * FCHash13(Seed, Cell) - Sum / 26.f) + 0.5f) * 2.f - 1.f;
}

// 3D Hilbert curve index using Skilling's transpose algorithm, 6 bits per axis (64x64x64 grid)
FORCEINLINE float FCHilbertBlue3D(const uint32 Seed, const float3 Position) {
	const uint32 SeedHash = (Seed ^ 0x9E3779B9u) * 3141592653u;

	int32 X = ((int32)floor(Position.x) + (int32)(SeedHash & 63u)) & 63;
	int32 Y = ((int32)floor(Position.y) + (int32)((SeedHash >> 6) & 63u)) & 63;
	int32 Z = ((int32)floor(Position.z) + (int32)((SeedHash >> 12) & 63u)) & 63;

	// Axes to transpose (John Skilling, "Programming the Hilbert curve")
	for (uniform int32 Mask = 64 >> 1; Mask > 1; Mask >>= 1) {
		const uniform int32 Lower = Mask - 1;

		if ((X & Mask) != 0) {
			X ^= Lower;
		} else {
			const int32 Swap = X & Lower;
			X ^= Swap;
		}

		if ((Y & Mask) != 0) {
			X ^= Lower;
		} else {
			const int32 Swap = (X ^ Y) & Lower;
			X ^= Swap;
			Y ^= Swap;
		}

		if ((Z & Mask) != 0) {
			X ^= Lower;
		} else {
			const int32 Swap = (X ^ Z) & Lower;
			X ^= Swap;
			Z ^= Swap;
		}
	}

	// Gray encode
	Y ^= X;
	Z ^= Y;

	int32 Gray = 0;
	for (uniform int32 Mask = 64 >> 1; Mask > 1; Mask >>= 1) {
		if ((Z & Mask) != 0) {
			Gray ^= Mask - 1;
		}
	}
	X ^= Gray;
	Y ^= Gray;
	Z ^= Gray;

	// Interleave the transposed bits into a single index
	int32 Result = 0;
	for (uniform int32 Bit = 5; Bit >= 0; Bit--) {
		Result = (Result << 1) | ((X >> Bit) & 1);
		Result = (Result << 1) | ((Y >> Bit) & 1);
		Result = (Result << 1) | ((Z >> Bit) & 1);
	}

	// Compute in double: fract of golden ratio times a large index needs more than float precision
	const double GoldenIndex = FC_GOLDEN_RATIO * (double)Result;
	return (float)(GoldenIndex - floor(GoldenIndex)) * 2.f - 1.f;
}

// Impact crater shells, 3D extension of https://www.shadertoy.com/view/XsGBDt
FORCEINLINE float FCCrater3D(const uint32 Seed, const float3 Position) {
	const float3 Cell = floor(Position);
	const float3 Local = Position - Cell;

	float ValueSum = 0.f;
	float WeightSum = 0.f;
	for (uniform int32 IndexX = -1; IndexX <= 1; IndexX++) {
		for (uniform int32 IndexY = -1; IndexY <= 1; IndexY++) {
			for (uniform int32 IndexZ = -1; IndexZ <= 1; IndexZ++) {
				const float3 Offset = MakeFloat3(IndexX, IndexY, IndexZ);
				const float3 Random = FCHash33(Seed, Cell + Offset);
				const float Distance = length(Local - Offset - Random);
				const float Weight = exp(-4.f * Distance);
				ValueSum += Weight * sin(6.28318530718f * sqrt(max(Distance, 0.06f)));
				WeightSum += Weight;
			}
		}
	}

	return abs(ValueSum / max(WeightSum, 1e-6f)) * 2.f - 1.f;
}

FORCEINLINE float FCGabor3D(const uint32 Seed, const float3 Position) {
	const uniform float Frequency = 8.f;

	const float3 Cell = floor(Position);
	float3 Alpha = Position - Cell;
	Alpha = Alpha * Alpha * (3.f - 2.f * Alpha);

#define FC_CORNER(X, Y, Z) sin(Frequency * dot(Position, FCHash33(Seed, Cell + MakeFloat3(X, Y, Z))))
	const float LayerA = FCLerp(
			FCLerp(FC_CORNER(0.f, 0.f, 0.f), FC_CORNER(1.f, 0.f, 0.f), Alpha.x),
			FCLerp(FC_CORNER(0.f, 1.f, 0.f), FC_CORNER(1.f, 1.f, 0.f), Alpha.x),
			Alpha.y);
	const float LayerB = FCLerp(
			FCLerp(FC_CORNER(0.f, 0.f, 1.f), FC_CORNER(1.f, 0.f, 1.f), Alpha.x),
			FCLerp(FC_CORNER(0.f, 1.f, 1.f), FC_CORNER(1.f, 1.f, 1.f), Alpha.x),
			Alpha.y);
#undef FC_CORNER

	return FCLerp(LayerA, LayerB, Alpha.z);
}

// Magnitude of the finite-difference gradient of gradient noise
FORCEINLINE float FCCurl3D(const uint32 Seed, const float3 Position) {
	const uniform float Epsilon = 0.1f;

	const float3 CurlValue = MakeFloat3(
									 FCPerlin3D_Raw(Seed, Position + MakeFloat3(Epsilon, 0.f, 0.f)) - FCPerlin3D_Raw(Seed, Position - MakeFloat3(Epsilon, 0.f, 0.f)),
									 FCPerlin3D_Raw(Seed, Position + MakeFloat3(0.f, Epsilon, 0.f)) - FCPerlin3D_Raw(Seed, Position - MakeFloat3(0.f, Epsilon, 0.f)),
									 FCPerlin3D_Raw(Seed, Position + MakeFloat3(0.f, 0.f, Epsilon)) - FCPerlin3D_Raw(Seed, Position - MakeFloat3(0.f, 0.f, Epsilon))) /
			Epsilon * 0.5f * 0.7f;

	return length(CurlValue) / 1.732f * 2.f - 1.f;
}

// Single layer of thin wavy strands, 3D extension of https://www.shadertoy.com/view/4syXRD
FORCEINLINE float FCScratchLayer3D(const uint32 Seed, const float3 Position, const float Smoothness) {
	const uniform float Thickness = 0.02f;
	const uniform float Wavyness = 0.5f;

	const float3 Cell = floor(Position);
	const float3 Random = FCHash33(Seed, Cell) * MakeFloat3(3104.f, 554.f, 1234.f);

	float3 Local = (Position - Cell) * 2.f - 1.f;

	// Random orientation: rotate in XY then in YZ
	{
		const float SinAngle = sin(Random.x + Random.y);
		const float CosAngle = cos(Random.x + Random.y);
		Local = MakeFloat3(
				Local.x * CosAngle - Local.y * SinAngle,
				Local.x * SinAngle + Local.y * CosAngle,
				Local.z);
	}
	{
		const float SinAngle = sin(Random.y + Random.z);
		const float CosAngle = cos(Random.y + Random.z);
		Local = MakeFloat3(
				Local.x,
				Local.y * CosAngle - Local.z * SinAngle,
				Local.y * SinAngle + Local.z * CosAngle);
	}
	Local = Local + sin(Random.x - Random.y);

	// A wavy strand along Y: constrain both X and Z to make a line instead of a plane
	const float LineX = abs(Local.x - cos(Random.x + Local.y * 1.57f) * Wavyness);
	const float LineZ = abs(Local.z - sin(Random.z + Local.y * 1.57f) * Wavyness);

	float Line = FCSmoothstep(Thickness + Smoothness, Thickness - Smoothness, max(LineX, LineZ));
	Line *= Local.y * 0.5f + 0.5f;

	return Line;
}
FORCEINLINE float FCScratch3D(const uint32 Seed, const float3 Position, const float Smoothness) {
	float3 Local = Position;
	float Width = max(Smoothness, 0.001f);

	float Scratches = 0.f;
	for (uniform int32 Index = 0; Index < 8; Index++) {
		Scratches = max(Scratches, FCScratchLayer3D(Seed, Local, Width));

		Local = MakeFloat3(
				Local.x + 0.7f * Local.y,
				-0.7f * Local.x + Local.y,
				Local.z);
		Local = MakeFloat3(
						Local.x,
						Local.y + 0.7f * Local.z,
						-0.7f * Local.y + Local.z) -
				12.31f;

		Width *= 1.22f;
	}

	return Scratches * 2.f - 1.f;
}

// Rotated sine wavelets, 3D extension of https://www.shadertoy.com/view/wsBfzK
FORCEINLINE float FCWavelet3D(const uint32 Seed, const float3 Position, const float Phase) {
	const uniform float Scale = 1.24f;

	float3 Local = Position + FCSeedOffset3(Seed);

	float Value = 0.f;
	float Frequency = 1.f;
	float WeightSum = 0.f;
	for (uniform int32 Index = 0; Index < 4; Index++) {
		const float3 Scaled = Local * Frequency;

		float3 Random = FCFract(floor(Scaled) * MakeFloat3(123.34f, 233.53f, 314.15f));
		Random = Random + dot(Random, Random + 23.234f);
		const float AngleA = FCFract(Random.x * Random.y) * 1e3f;
		const float AngleB = FCFract(Random.y * Random.z) * 1e3f;

		float3 Rotated = FCFract(Scaled) - 0.5f;
		{
			const float SinAngle = sin(AngleA);
			const float CosAngle = cos(AngleA);
			Rotated = MakeFloat3(
					Rotated.x * CosAngle + Rotated.y * SinAngle,
					-Rotated.x * SinAngle + Rotated.y * CosAngle,
					Rotated.z);
		}
		{
			const float SinAngle = sin(AngleB);
			const float CosAngle = cos(AngleB);
			Rotated = MakeFloat3(
					Rotated.x,
					Rotated.y * CosAngle + Rotated.z * SinAngle,
					-Rotated.y * SinAngle + Rotated.z * CosAngle);
		}

		Value += sin(Rotated.x * 10.f + Phase) * FCSmoothstep(0.25f, 0.f, dot(Rotated, Rotated)) / Frequency;

		Local = MakeFloat3(
				0.54f * Local.x - 0.84f * Local.y,
				0.84f * Local.x + 0.54f * Local.y,
				Local.z);
		Local = MakeFloat3(
						Local.x,
						0.54f * Local.y - 0.84f * Local.z,
						0.84f * Local.y + 0.54f * Local.z) +
				(uniform float)Index;

		WeightSum += 1.f / Frequency;
		Frequency *= Scale;
	}

	return Value / WeightSum;
}

struct FGullies3DResult {
	float Value;
	float3 Gradient;
};

// 3D extension of the erosion gullies: carve perpendicular to the local slope
FORCEINLINE FGullies3DResult FCGullies3D(const uint32 Seed, const float3 Position, const float3 Slope) {
	// Stable direction perpendicular to the slope: cross with the axis least aligned with it
	const float3 AbsSlope = abs(Slope);
	float3 Axis;
	if (AbsSlope.x <= AbsSlope.y && AbsSlope.x <= AbsSlope.z) {
		Axis = MakeFloat3(1.f, 0.f, 0.f);
	} else if (AbsSlope.y <= AbsSlope.z) {
		Axis = MakeFloat3(0.f, 1.f, 0.f);
	} else {
		Axis = MakeFloat3(0.f, 0.f, 1.f);
	}
	const float3 SideDirection = cross(Slope, Axis) * 3.14159265f;

	const float3 Cell = floor(Position);
	const float3 Local = Position - Cell;

	float2 HeightSlope = MakeFloat2(0.f, 0.f);
	float WeightSum = 0.f;
	for (uniform int32 IndexX = -1; IndexX <= 2; IndexX++) {
		for (uniform int32 IndexY = -1; IndexY <= 2; IndexY++) {
			for (uniform int32 IndexZ = -1; IndexZ <= 2; IndexZ++) {
				const float3 Offset = MakeFloat3(IndexX, IndexY, IndexZ);
				const float3 Delta = Local - Offset - FCHash33(Seed, Cell + Offset) + 0.5f;
				const float DistanceSquared = dot(Delta, Delta);
				const float Weight = max(0.f, exp(-DistanceSquared * 2.f) - 0.01111f);
				WeightSum += Weight;

				const float Along = dot(Delta, SideDirection);
				HeightSlope = HeightSlope + MakeFloat2(cos(Along), -sin(Along)) * Weight;
			}
		}
	}

	WeightSum = max(WeightSum, 1e-6f);

	FGullies3DResult Result;
	Result.Value = HeightSlope.x / WeightSum;
	Result.Gradient = SideDirection * (HeightSlope.y / WeightSum);
	return Result;
}
// Gradient noise with slope-following gullies, 3D extension of https://www.shadertoy.com/view/sf23W1
FORCEINLINE float FCErosion3D(const uint32 Seed, const float3 Position) {
	float3 Gradient;
	float Value = FCPerlinDeriv3D(Seed, Position, &Gradient);

	float Strength = 0.25f;
	float Frequency = 8.f;
	float Total = 1.f;
	for (uniform int32 Index = 0; Index < 4; Index++) {
		const float SlopeLengthSquared = max(dot(Gradient, Gradient), 1e-12f);
		const FGullies3DResult Gully = FCGullies3D(Seed, Position * Frequency, Gradient * pow(SlopeLengthSquared, -0.25f));

		Value += Gully.Value * Strength;
		Gradient = Gradient + Gully.Gradient * (Strength * Frequency);

		Total += Strength;
		Strength *= 0.5f;
		Frequency *= 2.f;
	}

	return Value / Total;
}

FORCEINLINE float FCPaper3D(const uint32 Seed, const float3 Position) {
	float3 Local = Position;

	float3 Sum = MakeFloat3(0.f, 0.f, 0.f);
	float WeightSum = 0.f;
	float Weight = 1.f;
	for (uniform int32 Index = 0; Index < 10; Index++) {
		float3 Gradient;
		FCPerlinDeriv3D(Seed, Local, &Gradient);

		Sum = Sum + FCClamp01(Gradient * 0.5f + 0.5f) * Weight;
		WeightSum += Weight;
		Weight *= 0.8f;
		Local = Local * 2.f;
	}
	Sum = Sum / WeightSum;

	return (length(Sum) / 1.732f * 0.6f + 0.4f) * 2.f - 1.f;
}

FORCEINLINE float FCStone3D(const uint32 Seed, const float3 Position) {
	float3 WarpGradient = MakeFloat3(0.f, 0.f, 0.f);
	{
		float3 Local = Position;
		float Weight = 1.f;
		for (uniform int32 Index = 0; Index < 6; Index++) {
			float3 Gradient;
			FCPerlinDeriv3D(Seed, Local, &Gradient);

			WarpGradient = WarpGradient + Gradient * Weight;
			Weight *= 0.5f;
			Local = Local * 2.f;
		}
	}

	float3 Local = Position + WarpGradient * 0.4f;

	float Sum = 0.f;
	float WeightSum = 0.f;
	float Weight = 1.f;
	for (uniform int32 Index = 0; Index < 6; Index++) {
		Sum += (FCPerlin3D_Raw(Seed, Local) * 0.7f + 0.5f) * Weight;
		WeightSum += Weight;
		Weight *= 0.5f;
		Local = Local * 2.f;
	}

	return Sum / WeightSum * 2.f - 1.f;
}

FORCEINLINE float FCWool3D(const uint32 Seed, const float3 Position) {
	float3 Local = Position;

	float3 Sum = MakeFloat3(0.f, 0.f, 0.f);
	float WeightSum = 0.f;
	float Weight = 1.f;
	for (uniform int32 Index = 0; Index < 6; Index++) {
		float3 Gradient;
		FCPerlinDeriv3D(Seed, Local, &Gradient);

		Sum = Sum + Gradient * Weight;
		WeightSum += Weight;
		Weight *= 0.5f;
		Local = Local * 2.f;
	}
	Sum = Sum / WeightSum;

	return FCMax3(abs(Sum.x), abs(Sum.y), abs(Sum.z)) * 2.f - 1.f;
}

// Interleaved gradient noise
FORCEINLINE float FCInterleavedGradient3D(const uint32 Seed, const float3 Position) {
	const float3 Local = Position + FCSeedOffset3(Seed);
	return FCFract(52.9829189f * FCFract(dot(Local, MakeFloat3(0.06711056f, 0.00583715f, 0.00974572f)))) * 2.f - 1.f;
}

///////////////////////////////////////////////////////////////////////////////
// Type dispatch — used by both the ISPC kernels and the scalar octave loop
///////////////////////////////////////////////////////////////////////////////

FORCEINLINE float FCSampleNoise2D(
		const uniform int32 Type,
		const uint32 Seed,
		const float2 Position,
		const float VoronoiSmoothness,
		const float WaveletPhase,
		const float ScratchSmoothness) {
	switch (Type) {
		default:
		case FC_NOISE_PERLIN:
			return FCPerlin2D(Seed, Position);
		case FC_NOISE_SIMPLEX:
			return FCSimplex2D(Seed, Position);
		case FC_NOISE_VALUE:
			return FCValue2D(Seed, Position);
		case FC_NOISE_WORLEY:
			return FCWorley2D(Seed, Position);
		case FC_NOISE_VORONOI:
			return FCVoronoi2D(Seed, Position, VoronoiSmoothness);
		case FC_NOISE_BLUE:
			return FCBlue2D(Seed, Position);
		case FC_NOISE_HILBERT_BLUE:
			return FCHilbertBlue2D(Seed, Position);
		case FC_NOISE_CRATER:
			return FCCrater2D(Seed, Position);
		case FC_NOISE_GABOR:
			return FCGabor2D(Seed, Position);
		case FC_NOISE_CURL:
			return FCCurl2D(Seed, Position);
		case FC_NOISE_SCRATCH:
			return FCScratch2D(Seed, Position, ScratchSmoothness);
		case FC_NOISE_WAVELET:
			return FCWavelet2D(Seed, Position, WaveletPhase);
		case FC_NOISE_EROSION:
			return FCErosion2D(Seed, Position);
		case FC_NOISE_PAPER:
			return FCPaper2D(Seed, Position);
		case FC_NOISE_STONE:
			return FCStone2D(Seed, Position);
		case FC_NOISE_WOOL:
			return FCWool2D(Seed, Position);
		case FC_NOISE_INTERLEAVED_GRADIENT:
			return FCInterleavedGradient2D(Seed, Position);
	}
}

FORCEINLINE float FCSampleNoise3D(
		const uniform int32 Type,
		const uint32 Seed,
		const float3 Position,
		const float VoronoiSmoothness,
		const float WaveletPhase,
		const float ScratchSmoothness) {
	switch (Type) {
		default:
		case FC_NOISE_PERLIN:
			return FCPerlin3D(Seed, Position);
		case FC_NOISE_SIMPLEX:
			return FCSimplex3D(Seed, Position);
		case FC_NOISE_VALUE:
			return FCValue3D(Seed, Position);
		case FC_NOISE_WORLEY:
			return FCWorley3D(Seed, Position);
		case FC_NOISE_VORONOI:
			return FCVoronoi3D(Seed, Position, VoronoiSmoothness);
		case FC_NOISE_BLUE:
			return FCBlue3D(Seed, Position);
		case FC_NOISE_HILBERT_BLUE:
			return FCHilbertBlue3D(Seed, Position);
		case FC_NOISE_CRATER:
			return FCCrater3D(Seed, Position);
		case FC_NOISE_GABOR:
			return FCGabor3D(Seed, Position);
		case FC_NOISE_CURL:
			return FCCurl3D(Seed, Position);
		case FC_NOISE_SCRATCH:
			return FCScratch3D(Seed, Position, ScratchSmoothness);
		case FC_NOISE_WAVELET:
			return FCWavelet3D(Seed, Position, WaveletPhase);
		case FC_NOISE_EROSION:
			return FCErosion3D(Seed, Position);
		case FC_NOISE_PAPER:
			return FCPaper3D(Seed, Position);
		case FC_NOISE_STONE:
			return FCStone3D(Seed, Position);
		case FC_NOISE_WOOL:
			return FCWool3D(Seed, Position);
		case FC_NOISE_INTERLEAVED_GRADIENT:
			return FCInterleavedGradient3D(Seed, Position);
	}
}

// Seed evolution between octaves — matches the VCET/UE implementation
FORCEINLINE uniform int32 FCNextOctaveSeed(const uniform int32 Seed) {
	return (Seed * 196314165) + 907633515;
}

///////////////////////////////////////////////////////////////////////////////
// Terrain diffusion — ported from the VCET Unreal plugin's TerrainDiffusion
// kernels (MIT, same origin as the noise above).
//
// Hierarchical terrain heightfield built on the gradient noises above.
// AestheticBias blends two shaping strategies per octave:
// - 0 (Realistic): fBm with derivative-based erosion damping — the running
//   slope of previous octaves suppresses later detail, carving smooth valleys
//   into steep areas (Inigo Quilez's "fbm with derivatives" erosion trick)
// - 1 (Fantastical): ridged multifractal — folded noise with amplitude
//   feedback, producing sharp creases, cliff faces and dramatic peaks
//
// Deterministic: per-octave seeds evolve with FCNextOctaveSeed, so identical
// inputs give identical terrain on every machine. O(1) random access.
//
// Godot adaptations vs the UE original:
// - Y is the polar axis (Godot is Y-up; UE used Z)
// - Positions/scales are in world units (meters), not cm — pure convention,
//   the math is scale-free
///////////////////////////////////////////////////////////////////////////////

#ifndef ISPC
FORCEINLINE float select(const bool Cond, const float A, const float B) {
	return Cond ? A : B;
}
FORCEINLINE float acos(const float v) {
	return ::acosf(v);
}
#endif

// Per-stage seed derivation, all uniform
#define FC_TERRAIN_SEED_WARP_X(Seed) ((uniform int32)((uniform uint32)(Seed) ^ 0x36E1A2C5u))
#define FC_TERRAIN_SEED_WARP_Y(Seed) ((uniform int32)((uniform uint32)(Seed) ^ 0x9C2779B9u))
#define FC_TERRAIN_SEED_WARP_Z(Seed) ((uniform int32)((uniform uint32)(Seed) ^ 0x5851F42Du))
#define FC_TERRAIN_SEED_MOUNTAIN(Seed) ((uniform int32)((uniform uint32)(Seed) ^ 0x7F4A7C15u))
#define FC_TERRAIN_SEED_CONTINENT(Seed) ((uniform int32)((uniform uint32)(Seed) ^ 0x2545F491u))
#define FC_TERRAIN_SEED_CANYON(Seed) ((uniform int32)((uniform uint32)(Seed) ^ 0x68E31DA4u))
#define FC_CLIMATE_SEED_TEMPERATURE(Seed) ((uniform int32)((uniform uint32)(Seed) ^ 0x1B873593u))
#define FC_CLIMATE_SEED_MOISTURE(Seed) ((uniform int32)((uniform uint32)(Seed) ^ 0xCC9E2D51u))
#define FC_RIVER_SEED_FLOW_X(Seed) ((uniform int32)((uniform uint32)(Seed) ^ 0xA3B195A8u))
#define FC_RIVER_SEED_FLOW_Y(Seed) ((uniform int32)((uniform uint32)(Seed) ^ 0xFB9C3D21u))
#define FC_RIVER_SEED_TRUNK(Seed) ((uniform int32)((uniform uint32)(Seed) ^ 0x4C957F3Bu))
#define FC_RIVER_SEED_TRIB(Seed) ((uniform int32)((uniform uint32)(Seed) ^ 0xD8A12E67u))

// 3-octave Perlin fBm in roughly [-1, 1], used for regional masks and climate
FORCEINLINE float FCTerrainRegionNoise2D(const uniform int32 InSeed, const float2 Position) {
	uniform int32 Seed = InSeed;
	const uniform uint32 SeedA = (uniform uint32)Seed;
	Seed = FCNextOctaveSeed(Seed);
	const uniform uint32 SeedB = (uniform uint32)Seed;
	Seed = FCNextOctaveSeed(Seed);
	const uniform uint32 SeedC = (uniform uint32)Seed;

	return (FCPerlin2D(SeedA, Position) + FCPerlin2D(SeedB, Position * 2.f) * 0.5f +
			FCPerlin2D(SeedC, Position * 4.f) * 0.25f) /
			1.75f;
}

FORCEINLINE float FCTerrainRegionNoise3D(const uniform int32 InSeed, const float3 Position) {
	uniform int32 Seed = InSeed;
	const uniform uint32 SeedA = (uniform uint32)Seed;
	Seed = FCNextOctaveSeed(Seed);
	const uniform uint32 SeedB = (uniform uint32)Seed;
	Seed = FCNextOctaveSeed(Seed);
	const uniform uint32 SeedC = (uniform uint32)Seed;

	return (FCPerlin3D(SeedA, Position) + FCPerlin3D(SeedB, Position * 2.f) * 0.5f +
			FCPerlin3D(SeedC, Position * 4.f) * 0.25f) /
			1.75f;
}

// Returns a normalized height in roughly [-1, 1]. Position is pre-divided by FeatureScale.
FORCEINLINE float FCInfiniteTerrain2D(
		const uniform int32 InSeed,
		float2 Position,
		const uniform int32 NumOctaves,
		const float Lacunarity,
		const float Gain,
		const float AestheticBias) {
	varying float Sum = 0.f;
	varying float AmplitudeSum = 0.f;
	varying float Amplitude = 1.f;
	varying float2 GradientSum = MakeFloat2(0.f, 0.f);
	varying float RidgeWeight = 1.f;
	uniform int32 Seed = InSeed;

	for (uniform int32 OctaveIndex = 0; OctaveIndex < NumOctaves; OctaveIndex++) {
		const uniform uint32 OctaveSeed = (uniform uint32)Seed;

		float2 Gradient;
		const float Raw = FCPerlinDeriv2D(OctaveSeed, Position, &Gradient) * 1.4f;

		// Erosion-like damping: accumulated slope suppresses later octaves
		GradientSum = GradientSum + Gradient;
		const float Erosion = 1.f / (1.f + dot(GradientSum, GradientSum));
		const float SmoothValue = Raw * Erosion;

		// Ridge filtering with multifractal amplitude feedback
		float Ridge = 1.f - abs(Raw);
		Ridge = Ridge * Ridge * RidgeWeight;
		RidgeWeight = clamp(Ridge * 2.f, 0.f, 1.f);
		const float RidgeValue = Ridge * 2.f - 1.f;

		const float Value = FCLerp(SmoothValue, RidgeValue, AestheticBias);

		const float NewSum = Sum + Value * Amplitude;
		Sum = select(FCIsFinite(NewSum), NewSum, Sum);

		AmplitudeSum += Amplitude;
		Amplitude = Amplitude * Gain;
		Position = Position * Lacunarity;
		Seed = FCNextOctaveSeed(Seed);
	}

	return Sum / (AmplitudeSum == 0.f ? 1.f : AmplitudeSum);
}

// 3D counterpart, sampled in 3D space so it is seamless on any closed surface.
FORCEINLINE float FCInfiniteTerrain3D(
		const uniform int32 InSeed,
		float3 Position,
		const uniform int32 NumOctaves,
		const float Lacunarity,
		const float Gain,
		const float AestheticBias) {
	varying float Sum = 0.f;
	varying float AmplitudeSum = 0.f;
	varying float Amplitude = 1.f;
	varying float3 GradientSum = MakeFloat3(0.f, 0.f, 0.f);
	varying float RidgeWeight = 1.f;
	uniform int32 Seed = InSeed;

	for (uniform int32 OctaveIndex = 0; OctaveIndex < NumOctaves; OctaveIndex++) {
		const uniform uint32 OctaveSeed = (uniform uint32)Seed;

		float3 Gradient;
		const float Raw = FCPerlinDeriv3D(OctaveSeed, Position, &Gradient) * 1.4f;

		GradientSum = GradientSum + Gradient;
		const float Erosion = 1.f / (1.f + dot(GradientSum, GradientSum));
		const float SmoothValue = Raw * Erosion;

		float Ridge = 1.f - abs(Raw);
		Ridge = Ridge * Ridge * RidgeWeight;
		RidgeWeight = clamp(Ridge * 2.f, 0.f, 1.f);
		const float RidgeValue = Ridge * 2.f - 1.f;

		const float Value = FCLerp(SmoothValue, RidgeValue, AestheticBias);

		const float NewSum = Sum + Value * Amplitude;
		Sum = select(FCIsFinite(NewSum), NewSum, Sum);

		AmplitudeSum += Amplitude;
		Amplitude = Amplitude * Gain;
		Position = Position * Lacunarity;
		Seed = FCNextOctaveSeed(Seed);
	}

	return Sum / (AmplitudeSum == 0.f ? 1.f : AmplitudeSum);
}

// Optional post-shaping stages layered on the base terrain, all in normalized
// height space [-1, 1]. Every stage is an exact no-op when its Blend/Strength
// is 0, so the base terrain is unchanged unless a control is enabled.
struct FCTerrainShaping {
	varying float WarpStrength; // world units of positional warp
	varying float WarpScale; // feature scale of the warp field
	varying float MountainBlend; // 0-1
	varying float MountainScale; // spacing of mountain range bands
	varying float ContinentBlend; // 0-1
	varying float ContinentScale; // size of landmasses
	varying float IslandBias; // -1 (archipelago) .. +1 (mostly land)
	varying float CanyonBlend; // 0-1
	varying float CanyonScale; // spacing of canyon networks
	varying float TerraceStrength; // 0-1
	varying float TerraceCount; // number of terrace levels
};

// Soft height quantization into mesas/plateaus, shared by the 2D/3D shaping
FORCEINLINE float FCApplyTerrainTerraces(float Height, const FCTerrainShaping Shaping) {
	const float Count = clamp(Shaping.TerraceCount, 1.f, 64.f);
	const float Level = (Height + 1.f) * 0.5f * Count;
	const float Cell = floor(Level);
	const float Alpha = Level - Cell;
	const float AlphaQuad = Alpha * Alpha * Alpha * Alpha;
	const float InvAlpha = 1.f - Alpha;
	const float InvAlphaQuad = InvAlpha * InvAlpha * InvAlpha * InvAlpha;
	const float SharpAlpha = AlphaQuad / max(AlphaQuad + InvAlphaQuad, 1e-8f);
	const float Terraced = (Cell + SharpAlpha) / Count * 2.f - 1.f;
	return FCLerp(Height, Terraced, clamp(Shaping.TerraceStrength, 0.f, 1.f));
}

// Applies the shaping stages to a normalized height. Position is in world
// units (pre-FeatureScale), matching the masks' own scales.
FORCEINLINE float FCApplyTerrainShaping2D(
		const uniform int32 Seed,
		const float2 Position,
		float Height,
		const FCTerrainShaping Shaping) {
	// Mountain ranges: concentrate relief into sharpened ridge bands
	if (Shaping.MountainBlend != 0.f) {
		float Band = 1.f -
				abs(FCPerlin2D((uniform uint32)FC_TERRAIN_SEED_MOUNTAIN(Seed),
						Position / max(Shaping.MountainScale, 1e-3f)));
		Band = clamp(Band, 0.f, 1.f);
		Band = Band * Band * Band;
		Height = Height * FCLerp(1.f, Band, clamp(Shaping.MountainBlend, 0.f, 1.f));
	}

	// Continents & islands: landmass mask with a detailed ocean floor
	if (Shaping.ContinentBlend != 0.f) {
		const float Continentalness = FCTerrainRegionNoise2D(
				FC_TERRAIN_SEED_CONTINENT(Seed), Position / max(Shaping.ContinentScale, 1e-3f));
		const float Land = FCSmoothstep(-0.2f, 0.2f, Continentalness + Shaping.IslandBias);
		const float OceanHeight = -0.55f + Height * 0.15f;
		const float Shaped = FCLerp(OceanHeight, Height, Land);
		Height = FCLerp(Height, Shaped, clamp(Shaping.ContinentBlend, 0.f, 1.f));
	}

	// Canyons: steep-walled channels carved down to a canyon floor
	if (Shaping.CanyonBlend != 0.f) {
		const float Channel = 1.f -
				FCSmoothstep(0.02f, 0.18f,
						abs(FCPerlin2D((uniform uint32)FC_TERRAIN_SEED_CANYON(Seed),
								Position / max(Shaping.CanyonScale, 1e-3f))));
		const float CarveTo = min(Height, -0.45f); // never raises ocean floors
		Height = FCLerp(Height, CarveTo, Channel * clamp(Shaping.CanyonBlend, 0.f, 1.f));
	}

	// Terraces: soft height quantization into mesas/plateaus
	if (Shaping.TerraceStrength != 0.f) {
		Height = FCApplyTerrainTerraces(Height, Shaping);
	}

	return clamp(Height, -1.f, 1.f);
}

FORCEINLINE float FCApplyTerrainShaping3D(
		const uniform int32 Seed,
		const float3 Position,
		float Height,
		const FCTerrainShaping Shaping) {
	if (Shaping.MountainBlend != 0.f) {
		float Band = 1.f -
				abs(FCPerlin3D((uniform uint32)FC_TERRAIN_SEED_MOUNTAIN(Seed),
						Position / max(Shaping.MountainScale, 1e-3f)));
		Band = clamp(Band, 0.f, 1.f);
		Band = Band * Band * Band;
		Height = Height * FCLerp(1.f, Band, clamp(Shaping.MountainBlend, 0.f, 1.f));
	}

	if (Shaping.ContinentBlend != 0.f) {
		const float Continentalness = FCTerrainRegionNoise3D(
				FC_TERRAIN_SEED_CONTINENT(Seed), Position / max(Shaping.ContinentScale, 1e-3f));
		const float Land = FCSmoothstep(-0.2f, 0.2f, Continentalness + Shaping.IslandBias);
		const float OceanHeight = -0.55f + Height * 0.15f;
		const float Shaped = FCLerp(OceanHeight, Height, Land);
		Height = FCLerp(Height, Shaped, clamp(Shaping.ContinentBlend, 0.f, 1.f));
	}

	if (Shaping.CanyonBlend != 0.f) {
		const float Channel = 1.f -
				FCSmoothstep(0.02f, 0.18f,
						abs(FCPerlin3D((uniform uint32)FC_TERRAIN_SEED_CANYON(Seed),
								Position / max(Shaping.CanyonScale, 1e-3f))));
		const float CarveTo = min(Height, -0.45f);
		Height = FCLerp(Height, CarveTo, Channel * clamp(Shaping.CanyonBlend, 0.f, 1.f));
	}

	if (Shaping.TerraceStrength != 0.f) {
		Height = FCApplyTerrainTerraces(Height, Shaping);
	}

	return clamp(Height, -1.f, 1.f);
}

// Full per-point 2D terrain height: warp + base fBm + shaping. Returns height
// in world units ([-Amplitude, Amplitude]).
FORCEINLINE float FCTerrainHeight2D(
		const uniform int32 Seed,
		float2 Position,
		const float Amplitude,
		const float FeatureScale,
		const float Lacunarity,
		const float Gain,
		const float AestheticBias,
		const uniform int32 NumOctaves,
		const FCTerrainShaping Shaping) {
	// Domain warp in world space, ahead of both the base terrain and the masks
	if (Shaping.WarpStrength != 0.f) {
		const float2 WarpPosition = Position / max(Shaping.WarpScale, 1e-3f);
		Position = Position +
				MakeFloat2(FCPerlin2D((uniform uint32)FC_TERRAIN_SEED_WARP_X(Seed), WarpPosition),
						FCPerlin2D((uniform uint32)FC_TERRAIN_SEED_WARP_Y(Seed), WarpPosition)) *
						Shaping.WarpStrength;
	}

	varying float Height = FCInfiniteTerrain2D(
			Seed, Position / max(FeatureScale, 1e-6f), NumOctaves, Lacunarity, Gain,
			clamp(AestheticBias, 0.f, 1.f));

	Height = FCApplyTerrainShaping2D(Seed, Position, Height, Shaping);

	return Height * Amplitude;
}

// Full per-point 3D planetary terrain. Samples along the unit direction of
// Position scaled to the planet surface, so the terrain is seamless on the
// whole sphere with no pole singularities or longitude seams. Returns the
// radial surface distance from the planet center: PlanetRadius + height.
FORCEINLINE float FCTerrainHeight3D(
		const uniform int32 Seed,
		const float3 InPosition,
		const float Amplitude,
		const float FeatureScale,
		const float Lacunarity,
		const float Gain,
		const float AestheticBias,
		const uniform int32 NumOctaves,
		const uniform float PlanetRadius,
		const FCTerrainShaping Shaping) {
	const float3 Direction = InPosition / max(length(InPosition), 1e-6f);

	// Surface-space position: the unit direction scaled to the planet
	// surface, so all scales stay surface arc lengths in world units
	float3 SurfacePosition = Direction * PlanetRadius;

	// Domain warp in surface space, ahead of both terrain and masks
	if (Shaping.WarpStrength != 0.f) {
		const float3 WarpPosition = SurfacePosition / max(Shaping.WarpScale, 1e-3f);
		SurfacePosition = SurfacePosition +
				MakeFloat3(FCPerlin3D((uniform uint32)FC_TERRAIN_SEED_WARP_X(Seed), WarpPosition),
						FCPerlin3D((uniform uint32)FC_TERRAIN_SEED_WARP_Y(Seed), WarpPosition),
						FCPerlin3D((uniform uint32)FC_TERRAIN_SEED_WARP_Z(Seed), WarpPosition)) *
						Shaping.WarpStrength;
	}

	varying float Height = FCInfiniteTerrain3D(
			Seed, SurfacePosition / max(FeatureScale, 1e-6f), NumOctaves, Lacunarity, Gain,
			clamp(AestheticBias, 0.f, 1.f));

	Height = FCApplyTerrainShaping3D(Seed, SurfacePosition, Height, Shaping);

	return PlanetRadius + Height * Amplitude;
}

// Multi-output results are returned by value (structs of varying members).
// Pointer out-params through a uniform pointer trigger ISPC's "all program
// instances writing to the same location" undefined behavior.
struct FCClimateResult {
	varying float Temperature;
	varying float Moisture;
	varying float NormalizedHeight;
};

// Climate fields from a terrain height + position, for material blending
// (biome masks, snow lines, vegetation density). All outputs normalized [0,1].
// Flat worlds have no latitude; regional noise provides climate zones.
FORCEINLINE FCClimateResult FCTerrainClimate2D(
		const uniform int32 Seed,
		const float2 Position,
		const float Height,
		const float Amplitude,
		const float InSeaLevel,
		const float ClimateScale,
		const float ElevationCooling,
		const float TemperatureVariation) {
	const float2 ClimatePosition = Position / max(ClimateScale, 1e-3f);
	const float NormalizedHeight = clamp(Height / max(Amplitude, 1e-3f), -1.f, 1.f);
	const float SeaLevel = clamp(InSeaLevel, -1.f, 1.f);
	const float AboveSea = max(NormalizedHeight - SeaLevel, 0.f) / max(1.f - SeaLevel, 1e-3f);

	const float TemperatureRegion =
			0.5f + 0.5f * FCTerrainRegionNoise2D(FC_CLIMATE_SEED_TEMPERATURE(Seed), ClimatePosition);
	const float BaseTemperature =
			FCLerp(0.6f, TemperatureRegion, clamp(TemperatureVariation, 0.f, 1.f));

	const float MoistureRegion = 0.5f +
			0.5f * FCTerrainRegionNoise2D(FC_CLIMATE_SEED_MOISTURE(Seed), ClimatePosition * 1.7f);

	FCClimateResult Result;
	Result.Temperature =
			clamp(BaseTemperature - AboveSea * clamp(ElevationCooling, 0.f, 1.f), 0.f, 1.f);
	Result.Moisture = clamp(MoistureRegion * (1.f - 0.35f * AboveSea), 0.f, 1.f);
	Result.NormalizedHeight = NormalizedHeight * 0.5f + 0.5f;
	return Result;
}

// Planetary climate: latitude gradient (Y is the polar axis in Godot) plus
// regional noise, cooled with elevation above sea level.
FORCEINLINE FCClimateResult FCTerrainClimate3D(
		const uniform int32 Seed,
		const float3 Position,
		const float RadialDistance,
		const float Amplitude,
		const float InSeaLevel,
		const float ClimateScale,
		const float ElevationCooling,
		const float TemperatureVariation,
		const uniform float PlanetRadius) {
	const float3 Direction = Position / max(length(Position), 1e-6f);
	const float3 ClimatePosition = Direction * (PlanetRadius / max(ClimateScale, 1e-3f));

	const float TerrainHeight = RadialDistance - PlanetRadius;
	const float NormalizedHeight = clamp(TerrainHeight / max(Amplitude, 1e-3f), -1.f, 1.f);
	const float SeaLevel = clamp(InSeaLevel, -1.f, 1.f);
	const float AboveSea = max(NormalizedHeight - SeaLevel, 0.f) / max(1.f - SeaLevel, 1e-3f);

	// Latitude gradient: cos^2(latitude) — 1 at the equator, 0 at the poles
	const float LatitudeTemperature = 1.f - Direction.y * Direction.y;
	const float TemperatureNoise =
			FCTerrainRegionNoise3D(FC_CLIMATE_SEED_TEMPERATURE(Seed), ClimatePosition);
	const float BaseTemperature =
			LatitudeTemperature + TemperatureNoise * 0.25f * clamp(TemperatureVariation, 0.f, 1.f);

	const float MoistureRegion = 0.5f +
			0.5f * FCTerrainRegionNoise3D(FC_CLIMATE_SEED_MOISTURE(Seed), ClimatePosition * 1.7f);

	FCClimateResult Result;
	Result.Temperature =
			clamp(BaseTemperature - AboveSea * clamp(ElevationCooling, 0.f, 1.f), 0.f, 1.f);
	Result.Moisture = clamp(MoistureRegion * (1.f - 0.35f * AboveSea), 0.f, 1.f);
	Result.NormalizedHeight = NormalizedHeight * 0.5f + 0.5f;
	return Result;
}

// Projects positions on/around a spherical planet into a 2D tangent frame
// centered on a stamp direction, using an azimuthal equidistant projection:
// radial surface distances from the stamp center are exact arc lengths, so a
// stamp keeps its intended size anywhere on the planet (poles included).
// Outputs: Local = tangent-plane coords ((0,0) at center), UV = Local remapped
// to [0,1] across StampSize, Strength = radial falloff (1 center, 0 edge).
struct FCSphereStampResult {
	varying float LocalX;
	varying float LocalY;
	varying float UVX;
	varying float UVY;
	varying float Strength;
};

FORCEINLINE FCSphereStampResult FCSphereStamp(
		const float3 Position,
		const float3 StampCenter,
		const float InStampSize,
		const float Falloff,
		const float RotationDegrees,
		const uniform float PlanetRadius) {
	const float3 Direction = Position / max(length(Position), 1e-6f);

	const float CenterLength = length(StampCenter);
	float3 Center = MakeFloat3(0.f, 1.f, 0.f);
	if (CenterLength > 1e-6f) {
		Center = StampCenter / CenterLength;
	}

	// Tangent basis aligned to the local surface up (= Center direction),
	// with a reference-axis fallback when the stamp sits at a pole (Y-up)
	float3 Reference = MakeFloat3(0.f, 1.f, 0.f);
	if (abs(Center.y) > 0.99f) {
		Reference = MakeFloat3(1.f, 0.f, 0.f);
	}
	float3 East = cross(Reference, Center);
	East = East / max(length(East), 1e-6f);
	const float3 North = cross(Center, East);

	// In-plane rotation of the stamp around its center
	const float RotationRad = RotationDegrees * (3.14159265f / 180.f);
	const float CosRotation = cos(RotationRad);
	const float SinRotation = sin(RotationRad);
	const float3 RotatedEast = East * CosRotation + North * SinRotation;
	const float3 RotatedNorth = North * CosRotation - East * SinRotation;

	// Azimuthal equidistant projection
	const float CosTheta = clamp(dot(Direction, Center), -1.f, 1.f);
	const float ArcDistance = acos(CosTheta) * PlanetRadius;

	const float3 Tangential = Direction - Center * CosTheta;
	const float TangentialLength = length(Tangential);
	float2 PlaneDirection = MakeFloat2(0.f, 0.f);
	if (TangentialLength > 1e-8f) {
		PlaneDirection = MakeFloat2(dot(Tangential, RotatedEast), dot(Tangential, RotatedNorth)) /
				TangentialLength;
	}

	const float2 Local = PlaneDirection * ArcDistance;
	const float StampSize = max(InStampSize, 1e-3f);

	// Radial falloff: full strength inside, smooth fade over the outer
	// Falloff fraction of the stamp radius
	const float NormalizedRadius = ArcDistance / (StampSize * 0.5f);
	const float FalloffFraction = clamp(Falloff, 1e-4f, 1.f);

	FCSphereStampResult Result;
	Result.LocalX = Local.x;
	Result.LocalY = Local.y;
	Result.UVX = Local.x / StampSize + 0.5f;
	Result.UVY = Local.y / StampSize + 0.5f;
	Result.Strength = 1.f - FCSmoothstep(1.f - FalloffFraction, 1.f, NormalizedRadius);
	return Result;
}

// Returns the distance to the nearest Worley cell edge in [0, ~0.7].
// Cells are jittered by FlowWarp to make channels follow terrain contours.
FORCEINLINE float FCRiverEdgeDist2D(const uint32 Seed, const float2 Position, const float2 FlowWarp) {
	const float2 WarpedPos = Position + FlowWarp;
	const float2 Cell = floor(WarpedPos);
	const float2 Local = WarpedPos - Cell;

	// Find the two nearest feature points (F1 and F2) — their midpoint is the edge
	float F1 = 1e6f;
	float F2 = 1e6f;
	for (uniform int32 Ix = -2; Ix <= 2; Ix++) {
		for (uniform int32 Iy = -2; Iy <= 2; Iy++) {
			const float2 Offset = MakeFloat2(Ix, Iy);
			const float2 Feature = Offset + FCHash22(Seed, Cell + Offset);
			const float Dist = length(Local - Feature);
			if (Dist < F1) {
				F2 = F1;
				F1 = Dist;
			} else if (Dist < F2) {
				F2 = Dist;
			}
		}
	}
	// Edge distance: midpoint between F1 and F2 minus F1 == (F2-F1)*0.5
	return (F2 - F1) * 0.5f;
}

// River channel mask and depth from TerrainClimate outputs using
// gradient-directed Worley networks. O(1) per point:
// 1. Sample the terrain gradient via derivative Perlin for a flow direction.
// 2. Warp the sample position along that flow direction (FlowStrength).
// 3. Two Worley passes (trunk + tributary scale), threshold the edge distance.
// 4. Widen channels toward sea level (coastal deltas).
// 5. Zero the mask below sea level (rivers end at the ocean).
// Outputs: RiverMask [0,1] (1 inside channel), RiverDepth [0,1] (1 at center).
struct FCRiversResult {
	varying float Mask;
	varying float Depth;
};

FORCEINLINE FCRiversResult FCTerrainRivers2D(
		const uniform int32 Seed,
		const float2 Position,
		const float NormalizedHeight,
		const float InSeaLevel,
		const float RiverScale,
		const float RiverWidth,
		const float FlowStrength) {
	const float H = clamp(NormalizedHeight, 0.f, 1.f);
	const float Sea = clamp(InSeaLevel, 0.01f, 0.99f);
	const float RivS = max(RiverScale, 1.f);
	const float Width = clamp(RiverWidth, 0.01f, 1.f);
	const float Flow = clamp(FlowStrength, 0.f, 1.f);

	// Gradient warp: sample a 2-octave Perlin derivative field
	const float2 GradPos = Position / (RivS * 2.f);
	float2 Grad0;
	float2 Grad1;
	FCPerlinDeriv2D((uniform uint32)FC_RIVER_SEED_FLOW_X(Seed), GradPos, &Grad0);
	FCPerlinDeriv2D((uniform uint32)FC_RIVER_SEED_FLOW_Y(Seed), GradPos * 1.7f, &Grad1);
	const float2 FlowWarp = (Grad0 + Grad1 * 0.5f) * Flow;

	// Worley edge distances at trunk and tributary scales
	const float2 TrunkPos = Position / (RivS * 2.5f);
	const float2 TribPos = Position / RivS;

	const float TrunkEdge =
			FCRiverEdgeDist2D((uniform uint32)FC_RIVER_SEED_TRUNK(Seed), TrunkPos, FlowWarp * 0.4f);
	const float TribEdge =
			FCRiverEdgeDist2D((uniform uint32)FC_RIVER_SEED_TRIB(Seed), TribPos, FlowWarp);

	// Adaptive width: rivers widen near sea level (coastal delta)
	const float LandT = clamp((H - Sea) / (1.f - Sea), 0.f, 1.f);
	const float CoastBoost = FCSmoothstep(0.25f, 0.f, LandT);
	const float EffWidth = clamp(Width + CoastBoost * Width * 0.6f, 0.01f, 0.99f);

	// Threshold: edge dist < half-width → inside channel
	// Trunk channels are wider, tributaries narrower
	const float TrunkHalf = EffWidth * 0.30f;
	const float TribHalf = EffWidth * 0.18f;

	const float TrunkMask = FCSmoothstep(TrunkHalf, TrunkHalf * 0.4f, TrunkEdge);
	const float TribMask = FCSmoothstep(TribHalf, TribHalf * 0.4f, TribEdge);

	// Depth: 1 at channel centre, 0 at bank
	const float TrunkDepth = clamp(1.f - TrunkEdge / max(TrunkHalf, 1e-4f), 0.f, 1.f);
	const float TribDepth = clamp(1.f - TribEdge / max(TribHalf, 1e-4f), 0.f, 1.f);

	// Merge: trunk takes priority; tributaries don't override trunk channels
	const float RiverMask = max(TrunkMask, TribMask);
	const float RiverDepth = max(TrunkDepth * TrunkMask, TribDepth * TribMask);

	// Zero below sea level — rivers end at the ocean
	const float LandMask = H >= Sea ? 1.f : 0.f;

	FCRiversResult Result;
	Result.Mask = RiverMask * LandMask;
	Result.Depth = RiverDepth * LandMask;
	return Result;
}

// Biome ids produced by FCTerrainMaterialBlend, encoded as floats:
// 0 DeepOcean, 1 ShallowOcean, 2 Coast, 3 River, 4 Tropical, 5 Forest,
// 6 Plains, 7 Desert, 8 Tundra, 9 Mountain, 10 Snow.
//
// Converts TerrainClimate outputs into a dominant biome id plus soft feature
// masks for material selection. The UE original also emitted a blended RGBA
// color; that is intentionally dropped here — terrain color is a flat solid
// per material type in this engine, mapped from the biome id downstream.
struct FCMaterialBlendResult {
	varying float BiomeId;
	varying float OceanMask;
	varying float CoastMask;
	varying float RiverFeatureMask;
	varying float VegetationMask;
	varying float DesertMask;
	varying float TundraMask;
	varying float MountainMask;
	varying float SnowMask;
};

FORCEINLINE FCMaterialBlendResult FCTerrainMaterialBlend(
		const float InNormalizedHeight,
		const float InTemperature,
		const float InMoisture,
		const float InSeaLevel,
		const float InRiverMask,
		const float InBiomeContrast) {
	const float H = clamp(InNormalizedHeight, 0.f, 1.f);
	const float Tmp = clamp(InTemperature, 0.f, 1.f);
	const float Mst = clamp(InMoisture, 0.f, 1.f);
	const float Sea = clamp(InSeaLevel, 0.01f, 0.99f);
	const float RivM = clamp(InRiverMask, 0.f, 1.f);
	const float Contrast = clamp(InBiomeContrast, 0.f, 1.f);

	const float Sharpness = FCLerp(0.25f, 1.f, Contrast);
	const float LandMask = H >= Sea ? 1.f : 0.f;
	const float OceanMask = 1.f - LandMask;
	const float LandT = clamp((H - Sea) / max(1.f - Sea, 1e-3f), 0.f, 1.f);
	const float Warmth = Tmp;
	const float Dryness = 1.f - Mst;

	// Feature masks (all [0,1])
	const float CoastMask =
			LandMask * (1.f - FCSmoothstep(0.03f, FCLerp(0.16f, 0.08f, Contrast), LandT));
	const float MountainMask = LandMask *
			FCSmoothstep(FCLerp(0.68f, 0.60f, Contrast), FCLerp(0.84f, 0.74f, Contrast), LandT);
	// Thresholds below synced from VCET's VoxelNode_TerrainMaterialBlend (VCETProceduralNoiseNodesImpl.ispc,
	// ~line 2688-2796, read 2026-08-16) to keep this port's biome bands in line with VCET's latest tuning.
	// VCET's own inline changelog for these: the SnowStart floor was raised 0.52->0.78 so cold climates
	// still get a visible "bare rock, not yet snowy" band above Mountain's onset before Snow kicks in; the
	// Desert warmth thresholds were raised 0.48/0.66->0.70/0.90 so ordinary mild-temperate warmth no longer
	// saturates to desert. VCET also has a CoastWidth multiplier on the Coast/Mountain edges that this
	// function has no equivalent input for, so those two lines are left as this codebase's existing
	// Contrast-only formulation (unaffected by the sync).
	const float SnowStart = FCLerp(0.78f, 0.95f, Warmth);
	const float SnowMask =
			LandMask * FCSmoothstep(SnowStart - 0.10f * Sharpness, SnowStart + 0.06f, LandT);
	const float DesertMask = LandMask * FCSmoothstep(0.70f, 0.90f, Warmth) *
			FCSmoothstep(0.42f, 0.74f, Dryness) * (1.f - SnowMask) * (1.f - CoastMask);
	const float TundraMask = LandMask * FCSmoothstep(0.48f, 0.78f, 1.f - Warmth) *
			(1.f - 0.55f * MountainMask) * (1.f - SnowMask) * (1.f - DesertMask);
	const float VegetationMask = LandMask * FCSmoothstep(0.30f, 0.68f, Mst) *
			(1.f - DesertMask) * (1.f - 0.75f * SnowMask);
	const float RiverFeatureMask = LandMask * RivM;

	// Ocean depth split for shallow/deep classes
	const float OceanDepth = OceanMask * clamp((Sea - H) / max(Sea, 1e-3f), 0.f, 1.f);
	const float ShallowOceanMask = OceanMask * (1.f - FCSmoothstep(0.25f, 0.55f, OceanDepth));
	const float DeepOceanMask = OceanMask * (1.f - ShallowOceanMask);

	// Land subclasses used for id switching
	const float TropicalMask = LandMask * FCSmoothstep(0.65f, 0.45f, Warmth) *
			FCSmoothstep(0.55f, 0.70f, Mst) * (1.f - MountainMask) * (1.f - SnowMask);
	const float ForestMask = VegetationMask * (1.f - TropicalMask) * (1.f - TundraMask) * (1.f - MountainMask);
	const float PlainsMask = LandMask *
			(1.f -
					max(CoastMask,
							max(DesertMask,
									max(TundraMask,
											max(MountainMask,
													max(SnowMask,
															max(RiverFeatureMask,
																	max(TropicalMask, ForestMask))))))));

	// Dominant biome id
	varying float BiomeId = 6.f;
	varying float Best = PlainsMask;

	if (DeepOceanMask > Best) {
		Best = DeepOceanMask;
		BiomeId = 0.f;
	}
	if (ShallowOceanMask > Best) {
		Best = ShallowOceanMask;
		BiomeId = 1.f;
	}
	if (CoastMask > Best) {
		Best = CoastMask;
		BiomeId = 2.f;
	}
	if (RiverFeatureMask > Best) {
		Best = RiverFeatureMask;
		BiomeId = 3.f;
	}
	if (TropicalMask > Best) {
		Best = TropicalMask;
		BiomeId = 4.f;
	}
	if (ForestMask > Best) {
		Best = ForestMask;
		BiomeId = 5.f;
	}
	if (DesertMask > Best) {
		Best = DesertMask;
		BiomeId = 7.f;
	}
	if (TundraMask > Best) {
		Best = TundraMask;
		BiomeId = 8.f;
	}
	if (MountainMask > Best) {
		Best = MountainMask;
		BiomeId = 9.f;
	}
	if (SnowMask > Best) {
		BiomeId = 10.f;
	}

	FCMaterialBlendResult Result;
	Result.BiomeId = BiomeId;
	Result.OceanMask = OceanMask;
	Result.CoastMask = CoastMask;
	Result.RiverFeatureMask = RiverFeatureMask;
	Result.VegetationMask = VegetationMask;
	Result.DesertMask = DesertMask;
	Result.TundraMask = TundraMask;
	Result.MountainMask = MountainMask;
	Result.SnowMask = SnowMask;
	return Result;
}

// MPL-2.0 erosion filter, kept in its own file
#include "voxel_erosion_filter_impl.h"

#ifndef ISPC
#undef uniform
#undef varying
#undef FORCEINLINE
#undef FC_GOLDEN_RATIO
} // namespace zylann::procedural_noise
#endif
