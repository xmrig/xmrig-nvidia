#pragma once

#include <stdint.h>

__device__ __forceinline__ uint64_t get_reciprocal_heavy(uint32_t a)
{
	const uint32_t shift = __clz(a);
	a <<= shift;

	const float a_hi = __uint_as_float((a >> 8) + 1 + ((126U + 31U) << 23));
	const float a_lo = __int2float_rn((a & 0xFF) - 256);

	float r;
	asm("rcp.approx.f32 %0, %1;" : "=f"(r) : "f"(a_hi));

	const uint32_t tmp0 = __float_as_uint(r);
	const uint32_t tmp1 = tmp0 + ((shift + 2 + 64U) << 23);
	const float r_scaled = __uint_as_float(tmp1);

	const float h = __fmaf_rn(a_lo, r, __fmaf_rn(a_hi, r, -1.0f));

	const float r_scaled_hi = __uint_as_float(tmp1 & ~uint32_t(4095));
	const float h_hi = __uint_as_float(__float_as_uint(h) & ~uint32_t(4095));

	const float r_scaled_lo = r_scaled - r_scaled_hi;
	const float h_lo = h - h_hi;

	const float x1 = h_hi * r_scaled_hi;
	const float x2 = __fmaf_rn(h_lo, r_scaled, h_hi * r_scaled_lo);

	const int64_t h1 = __float2ll_rn(x1);
	const int32_t h2 = __float2int_ru(x2) - __float2int_rd(h * (x1 + x2));

	return (uint64_t(tmp0 & 0xFFFFFF) << (shift + 9)) - ((h1 + h2) >> 2);
}

__device__ __forceinline__ uint64_t fast_div_heavy(int64_t _a, int32_t _b)
{
	const uint64_t a = abs(_a);
	const uint32_t b = abs(_b);
	uint64_t q = __umul64hi(a, get_reciprocal_heavy(b));

	const int64_t tmp = a - q * b;
	const int32_t overshoot = (tmp < 0) ? 1 : 0;
	const int32_t undershoot = (tmp >= b) ? 1 : 0;
	q += undershoot - overshoot;

	return ((((int32_t*) &_a)[1] ^ _b) < 0) ? -q : q;
}
