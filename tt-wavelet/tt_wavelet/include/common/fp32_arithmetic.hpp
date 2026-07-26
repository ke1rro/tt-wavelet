// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <bit>
#include <cmath>
#include <cstdint>

namespace ttwv {

/**
 * Arithmetic contract used by the scalar lifting oracle.
 *
 * Wormhole and Blackhole SFPMAD are partially fused FP32 operations rather
 * than IEEE-754 FMAs.  The architecture models below are derived from the
 * bit-perfect Apache-2.0 models in:
 *
 *   tt-isa-documentation/Miscellaneous/FMA/fma.c
 */
enum class Fp32Arithmetic : uint8_t {
    kIeee,
    kWormholeSfpu,
    kBlackholeSfpu,
};

namespace fp32_arithmetic_detail {

struct UnpackedNoDenorm {
    int32_t exponent;
    uint32_t mantissa;
};

[[nodiscard]] constexpr UnpackedNoDenorm unpack_no_denorms(const uint32_t value) noexcept {
    const int32_t exponent = static_cast<int32_t>((value >> 23) & 255U);
    const uint32_t mantissa = exponent == 0 ? 0U : (value & 0x7FFFFFU) ^ 0x800000U;
    return {.exponent = exponent, .mantissa = mantissa};
}

template <typename Integer>
[[nodiscard]] constexpr Integer semi_sticky_shift(Integer value, const int32_t amount) noexcept {
    if (amount >= static_cast<int32_t>(sizeof(Integer) * 8)) {
        return 0;
    }
    const Integer original = value;
    value >>= amount;
    if (value != 0) {
        value |= static_cast<Integer>((value << amount) != original);
    }
    return value;
}

[[nodiscard]] inline uint32_t fma_model_blackhole(uint32_t x, uint32_t y, uint32_t z) noexcept {
    const auto x_unpacked = unpack_no_denorms(x);
    const auto y_unpacked = unpack_no_denorms(y);
    const auto z_unpacked = unpack_no_denorms(z);
    int32_t x_e = x_unpacked.exponent;
    int32_t y_e = y_unpacked.exponent;
    int32_t z_e = z_unpacked.exponent;
    uint32_t x_m = x_unpacked.mantissa;
    uint32_t y_m = y_unpacked.mantissa;
    uint32_t z_m = z_unpacked.mantissa;
    const uint32_t z_sign = z & 0x80000000U;

    const uint32_t p_sign = (x ^ y) & 0x80000000U;
    uint64_t p_m = static_cast<uint64_t>(x_m) * static_cast<uint64_t>(y_m);
    int32_t p_e = x_e + y_e - 23 - 127;

    p_m <<= 3;
    z_m <<= 3;
    p_m = (p_m >> 23) | ((p_m & 0x7FFFFFU) != 0);
    p_e += 23;

    if (x_e == 255 || y_e == 255 || p_e >= 255 || z_e == 255) {
        if ((x_e == 255 && (x_m != 0x800000U || y_m == 0)) || (y_e == 255 && (y_m != 0x800000U || x_m == 0)) ||
            (z_e == 255 && z_m != 0x4000000U) || (z_e == 255 && (x_e == 255 || y_e == 255) && z_sign != p_sign)) {
            return 0x7FC00000U;
        }
        if (z_e == 255) {
            return z;
        }
        return p_sign | 0x7F800000U;
    }

    if (p_m == 0 || p_e < 0) {
        return z_m != 0 ? z : z_sign & p_sign;
    }

    const int32_t r_e_initial = p_e > z_e ? p_e : z_e;
    if (p_e < r_e_initial) {
        p_m = semi_sticky_shift(p_m, r_e_initial - p_e);
    }
    if (z_e < r_e_initial) {
        z_m = semi_sticky_shift(z_m, r_e_initial - z_e);
    }
    const uint32_t r_sign = p_m >= z_m ? p_sign : z_sign;
    if (z_sign != r_sign) {
        z_m = ~z_m;
    }
    if (p_sign != r_sign) {
        p_m = ~p_m;
    }
    uint32_t r_m = static_cast<uint32_t>(z_m + p_m + (p_sign != z_sign));
    if (r_m == 0) {
        return z_sign & p_sign;
    }

    const int32_t normalization = 5 - __builtin_clz(r_m);
    int32_t r_e = r_e_initial + normalization;
    if (r_e >= 255) {
        return r_sign | 0x7F800000U;
    }
    int32_t shift = normalization;
    if (r_e <= 0) {
        ++shift;
        r_e = 0;
    }
    if (shift <= 0) {
        r_m <<= -shift;
    } else {
        r_m = (r_m >> shift) | ((r_m & static_cast<uint32_t>(shift | 1)) != 0);
    }

    uint32_t result = (static_cast<uint32_t>(r_e) << 23) + ((r_m >> 3) & 0x7FFFFFU);
    result += (((r_m & 7U) + (result & 1U)) > 4U);
    if ((result >> 23) == 0) {
        result = 0;
    }
    return r_sign | result;
}

[[nodiscard]] inline uint32_t fma_model_wormhole(uint32_t x, uint32_t y, uint32_t z) noexcept {
    const auto x_unpacked = unpack_no_denorms(x);
    const auto y_unpacked = unpack_no_denorms(y);
    const auto z_unpacked = unpack_no_denorms(z);
    int32_t x_e = x_unpacked.exponent;
    int32_t y_e = y_unpacked.exponent;
    int32_t z_e = z_unpacked.exponent;
    uint32_t x_m = x_unpacked.mantissa;
    uint32_t y_m = y_unpacked.mantissa;
    uint32_t z_m = z_unpacked.mantissa;
    const uint32_t z_sign = z & 0x80000000U;

    const uint32_t p_sign = (x ^ y) & 0x80000000U;
    uint64_t p_m = static_cast<uint64_t>(x_m) * static_cast<uint64_t>(y_m);
    int32_t p_e = x_e + y_e - 23 - 127;

    p_m <<= 3;
    z_m <<= 3;
    p_m = (p_m >> 23) | ((p_m & 0x7FFFFFU) != 0);
    p_e += 23;

    uint32_t nan_result = 0;
    if (x_e == 255 || y_e == 255 || p_e >= 255 || z_e == 255) {
        if ((x_e == 255 && (x_m != 0x800000U || y_m == 0)) || (y_e == 255 && (y_m != 0x800000U || x_m == 0)) ||
            (z_e == 255 && z_m == 0x4000000U && (x_e == 255 || y_e == 255 || p_e >= 255) && z_sign != p_sign)) {
            nan_result = p_sign | 0x7F800001U;
        } else if (z_e == 255 && z_m != 0x4000000U) {
            nan_result = z_sign | 0x7F800001U;
        } else if (z_e == 255) {
            return z;
        } else {
            return p_sign | 0x7F800000U;
        }
        if (p_e > 255) {
            p_e = 255;
        }
    }

    if (p_m == 0 || p_e < 0) {
        if (nan_result != 0) {
            p_m = 0;
            p_e = 0;
        } else {
            return z_m != 0 ? z : 0U;
        }
    }

    const int32_t r_e_initial = p_e > z_e ? p_e : z_e;
    if (p_e < r_e_initial) {
        p_m = semi_sticky_shift(p_m, r_e_initial - p_e);
    }
    if (z_e < r_e_initial) {
        z_m = semi_sticky_shift(z_m, r_e_initial - z_e);
    }
    const uint32_t r_sign = p_m >= z_m ? p_sign : z_sign;
    if (z_sign != r_sign) {
        z_m = ~z_m;
    }
    if (p_sign != r_sign) {
        p_m = ~p_m;
    }
    uint32_t r_m = static_cast<uint32_t>(z_m + p_m + (p_sign != z_sign));
    if (r_m == 0) {
        return nan_result;
    }

    const int32_t normalization = 5 - __builtin_clz(r_m);
    const int32_t r_e = r_e_initial + normalization;
    if (r_e >= 255) {
        return nan_result != 0 ? nan_result : (r_sign | 0x7F800000U);
    }
    if (r_e < 0) {
        return nan_result;
    }
    if (normalization <= 0) {
        r_m <<= -normalization;
    } else {
        r_m = (r_m >> normalization) | (r_m & 1U);
    }

    uint32_t result = (static_cast<uint32_t>(r_e) << 23) + ((r_m >> 3) & 0x7FFFFFU);
    result += (((r_m & 7U) + (result & 1U)) > 4U);
    if ((result >> 23) == 0) {
        return nan_result;
    }
    return (nan_result != 0 ? nan_result : r_sign) | result;
}

}  // namespace fp32_arithmetic_detail

[[nodiscard]] inline float fp32_fma(
    const float multiplicand, const float multiplier, const float addend, const Fp32Arithmetic arithmetic) noexcept {
    if (arithmetic == Fp32Arithmetic::kIeee) {
        return std::fma(multiplicand, multiplier, addend);
    }
    const uint32_t x = std::bit_cast<uint32_t>(multiplicand);
    const uint32_t y = std::bit_cast<uint32_t>(multiplier);
    const uint32_t z = std::bit_cast<uint32_t>(addend);
    const uint32_t result = arithmetic == Fp32Arithmetic::kWormholeSfpu
                                ? fp32_arithmetic_detail::fma_model_wormhole(x, y, z)
                                : fp32_arithmetic_detail::fma_model_blackhole(x, y, z);
    return std::bit_cast<float>(result);
}

[[nodiscard]] inline float fp32_multiply(const float lhs, const float rhs, const Fp32Arithmetic arithmetic) noexcept {
    return fp32_fma(lhs, rhs, 0.0F, arithmetic);
}

}  // namespace ttwv
