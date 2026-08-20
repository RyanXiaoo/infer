// f16.h — fp16 (IEEE 754 binary16) and bf16 (bfloat16) <-> fp32 conversions.
//
// Plain C++, no CUDA. This is the price of the "no CUDA types in shared headers"
// rule: CPU code stores half-precision values as uint16_t and converts through
// these helpers to do arithmetic. GPU code uses __half/__nv_bfloat16 natively
// inside kernels/ and never leaks those types up here.
//
// Layouts (both 16 bits, differ in the exponent/mantissa split):
//   fp16: 1 sign | 5 exponent (bias 15)  | 10 mantissa
//   bf16: 1 sign | 8 exponent (bias 127) |  7 mantissa
//
// bf16 is fp32 with the low 16 mantissa bits chopped off — same exponent field —
// which is why bf16 -> fp32 is a single 16-bit left shift, and why bf16 has fp32's
// range but only ~2-3 decimal digits of precision.

#pragma once

#include <cstdint>
#include <cstring>

namespace f16 {

inline float bits_to_f32(uint32_t u) {
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

inline uint32_t f32_to_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

// ---------------------------------------------------------------- bf16

inline float bf16_to_f32(uint16_t h) {
    return bits_to_f32(static_cast<uint32_t>(h) << 16);
}

inline uint16_t f32_to_bf16(float f) {
    uint32_t u = f32_to_bits(f);
    // NaN must stay NaN: truncation could zero all mantissa bits and turn a NaN
    // into an infinity. Force a quiet-NaN mantissa bit instead.
    if ((u & 0x7fffffffu) > 0x7f800000u) {
        return static_cast<uint16_t>((u >> 16) | 0x0040u);
    }
    // Round to nearest even: add 0x7fff plus the lowest surviving bit, then chop.
    uint32_t rounding_bias = 0x7fffu + ((u >> 16) & 1u);
    return static_cast<uint16_t>((u + rounding_bias) >> 16);
}

// ---------------------------------------------------------------- fp16

inline float f16_to_f32(uint16_t h) {
    uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;

    if (exp == 0x1fu) {                       // inf / NaN
        return bits_to_f32(sign | 0x7f800000u | (mant << 13));
    }
    if (exp == 0) {
        if (mant == 0) return bits_to_f32(sign);   // +/- 0
        // Subnormal: value = mant * 2^-24. Normalize by shifting the leading 1
        // up into the implicit-bit position, adjusting the exponent as we go.
        exp = 127 - 15 + 1;
        while ((mant & 0x400u) == 0) {
            mant <<= 1;
            exp--;
        }
        mant &= 0x3ffu;                            // drop the now-implicit leading 1
        return bits_to_f32(sign | (exp << 23) | (mant << 13));
    }
    // Normal number: rebias exponent 15 -> 127, widen mantissa 10 -> 23 bits.
    return bits_to_f32(sign | ((exp + 127 - 15) << 23) | (mant << 13));
}

inline uint16_t f32_to_f16(float f) {
    uint32_t u    = f32_to_bits(f);
    uint32_t sign = (u >> 16) & 0x8000u;
    uint32_t exp  = (u >> 23) & 0xffu;
    uint32_t mant = u & 0x7fffffu;

    if (exp == 0xffu) {                       // inf / NaN
        // Keep a NaN payload bit set so NaN doesn't collapse to inf.
        uint16_t m = mant ? static_cast<uint16_t>((mant >> 13) | 1u) : 0u;
        return static_cast<uint16_t>(sign | 0x7c00u | m);
    }

    int e = static_cast<int>(exp) - 127 + 15;  // rebias to fp16

    if (e >= 0x1f) {                          // overflow -> inf
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    if (e <= 0) {                             // fp16 subnormal or underflow to zero
        if (e < -10) return static_cast<uint16_t>(sign);   // too small even for subnormal
        // Build the subnormal: restore the implicit leading 1, shift right by the
        // exponent deficit, round to nearest even on the bits shifted out.
        mant |= 0x800000u;
        uint32_t shift = static_cast<uint32_t>(14 - e);    // 14..24
        uint32_t half  = 1u << (shift - 1);
        uint32_t out   = mant >> shift;
        uint32_t rem   = mant & ((1u << shift) - 1);
        if (rem > half || (rem == half && (out & 1u))) out++;
        return static_cast<uint16_t>(sign | out);
    }

    // Normal case: round 23-bit mantissa to 10 bits, nearest even.
    uint32_t out = static_cast<uint32_t>(e) << 10 | (mant >> 13);
    uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (out & 1u))) out++;  // may carry into exp: correct
    return static_cast<uint16_t>(sign | out);
}

} // namespace f16
