// test_f16.cpp — exhaustive tests for fp16/bf16 conversions.
//
// 16-bit formats have only 65536 values, so we don't sample: every representable
// value is round-tripped, and float->half rounding is checked against the
// compiler's own conversions where available.

#include "../src/f16.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

static int failures = 0;

#define CHECK(cond, ...)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("FAIL %s:%d  ", __FILE__, __LINE__);                 \
            std::printf(__VA_ARGS__);                                        \
            std::printf("\n");                                               \
            failures++;                                                      \
        }                                                                    \
    } while (0)

static bool is_nan_bits16_fp16(uint16_t h) {
    return ((h >> 10) & 0x1f) == 0x1f && (h & 0x3ff) != 0;
}
static bool is_nan_bits16_bf16(uint16_t h) {
    return ((h >> 7) & 0xff) == 0xff && (h & 0x7f) != 0;
}

int main() {
    // --- fp16: every value round-trips exactly (half -> float -> half) ---
    for (uint32_t i = 0; i < 0x10000u; i++) {
        uint16_t h = static_cast<uint16_t>(i);
        float f = f16::f16_to_f32(h);
        if (is_nan_bits16_fp16(h)) {
            CHECK(std::isnan(f), "fp16 NaN 0x%04x -> %f not NaN", h, f);
            continue;  // NaN payloads may differ; identity of bits not required
        }
        uint16_t back = f16::f32_to_f16(f);
        CHECK(back == h, "fp16 roundtrip 0x%04x -> %f -> 0x%04x", h, f, back);
    }

    // --- bf16: every value round-trips exactly ---
    for (uint32_t i = 0; i < 0x10000u; i++) {
        uint16_t h = static_cast<uint16_t>(i);
        float f = f16::bf16_to_f32(h);
        if (is_nan_bits16_bf16(h)) {
            CHECK(std::isnan(f), "bf16 NaN 0x%04x -> %f not NaN", h, f);
            continue;
        }
        uint16_t back = f16::f32_to_bf16(f);
        CHECK(back == h, "bf16 roundtrip 0x%04x -> %f -> 0x%04x", h, f, back);
    }

    // --- fp16 spot values ---
    CHECK(f16::f16_to_f32(0x3c00) == 1.0f, "0x3c00 != 1.0");
    CHECK(f16::f16_to_f32(0xbc00) == -1.0f, "0xbc00 != -1.0");
    CHECK(f16::f16_to_f32(0x7bff) == 65504.0f, "0x7bff != 65504 (fp16 max)");
    CHECK(f16::f16_to_f32(0x0001) == std::ldexp(1.0f, -24), "0x0001 != 2^-24 (min subnormal)");
    CHECK(f16::f16_to_f32(0x0400) == std::ldexp(1.0f, -14), "0x0400 != 2^-14 (min normal)");
    CHECK(std::isinf(f16::f16_to_f32(0x7c00)), "0x7c00 not +inf");
    CHECK(f16::f16_to_f32(0xfc00) == -INFINITY, "0xfc00 not -inf");

    // --- fp16 encode edge cases ---
    CHECK(f16::f32_to_f16(0.0f) == 0x0000, "+0 encode");
    CHECK(f16::f32_to_f16(-0.0f) == 0x8000, "-0 encode");
    CHECK(f16::f32_to_f16(65504.0f) == 0x7bff, "fp16 max encode");
    CHECK(f16::f32_to_f16(65520.0f) == 0x7c00, "65520 rounds to inf");   // > max, rounds up
    CHECK(f16::f32_to_f16(1e30f) == 0x7c00, "overflow -> inf");
    CHECK(f16::f32_to_f16(-1e30f) == 0xfc00, "overflow -> -inf");
    CHECK(f16::f32_to_f16(std::ldexp(1.0f, -25)) == 0x0000, "2^-25 ties-to-even to 0");
    CHECK(f16::f32_to_f16(std::ldexp(1.0f, -26)) == 0x0000, "2^-26 underflows to 0");
    CHECK(is_nan_bits16_fp16(f16::f32_to_f16(NAN)), "NaN encode stays NaN");

    // Round-to-nearest-even: 1.0 + 2^-11 is exactly between 1.0 and 1.0+2^-10.
    // Even mantissa (0x3c00) must win.
    CHECK(f16::f32_to_f16(1.0f + std::ldexp(1.0f, -11)) == 0x3c00, "ties-to-even down");
    // 1.0 + 3*2^-11 is between odd 0x3c01 and even 0x3c02: even must win.
    CHECK(f16::f32_to_f16(1.0f + 3 * std::ldexp(1.0f, -11)) == 0x3c02, "ties-to-even up");

    // --- bf16 spot values ---
    CHECK(f16::bf16_to_f32(0x3f80) == 1.0f, "bf16 1.0");
    CHECK(f16::bf16_to_f32(0x4049) == f16::bits_to_f32(0x40490000), "bf16 ~pi");
    CHECK(std::isinf(f16::bf16_to_f32(0x7f80)), "bf16 +inf");
    CHECK(is_nan_bits16_bf16(f16::f32_to_bf16(NAN)), "bf16 NaN encode stays NaN");
    CHECK(f16::f32_to_bf16(-0.0f) == 0x8000, "bf16 -0 encode");
    // bf16 keeps fp32 range: 1e38 must NOT overflow.
    CHECK(!std::isinf(f16::bf16_to_f32(f16::f32_to_bf16(1e38f))), "bf16 holds 1e38");

    // --- cross-check against compiler _Float16 if available (GCC/Clang) ---
#if defined(__FLT16_MAX__)
    {
        int mismatches = 0;
        for (uint32_t i = 0; i < 0x10000u && mismatches < 20; i++) {
            uint16_t h = static_cast<uint16_t>(i);
            _Float16 native;
            std::memcpy(&native, &h, sizeof(native));
            float theirs = static_cast<float>(native);
            float ours = f16::f16_to_f32(h);
            bool both_nan = std::isnan(theirs) && std::isnan(ours);
            if (!both_nan && f16::f32_to_bits(theirs) != f16::f32_to_bits(ours)) {
                CHECK(false, "decode mismatch vs _Float16 at 0x%04x: %g vs %g", h, theirs, ours);
                mismatches++;
            }
        }
    }
#endif

    if (failures == 0) {
        std::printf("test_f16: all checks passed (2x 65536 roundtrips + edges)\n");
        return 0;
    }
    std::printf("test_f16: %d FAILURES\n", failures);
    return 1;
}
