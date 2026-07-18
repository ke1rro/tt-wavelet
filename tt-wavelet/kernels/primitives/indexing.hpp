#pragma once

#include <cstdint>

#define ALWI inline __attribute__((always_inline))

namespace ttwv::kernels::primitives {

ALWI uint32_t positive_mod(const int32_t value, const uint32_t modulus) {
    if (modulus == 0) {
        return 0;
    }

    const int32_t signed_modulus = static_cast<int32_t>(modulus);
    const int32_t remainder = value % signed_modulus;
    return static_cast<uint32_t>(remainder < 0 ? remainder + signed_modulus : remainder);
}

struct SignedPeriodIndex {
    int64_t quotient;
    uint64_t remainder;
};

ALWI SignedPeriodIndex decompose_signed_period(const int32_t value, const uint64_t period) {
    const int64_t signed_value = static_cast<int64_t>(value);
    const int64_t signed_period = static_cast<int64_t>(period);
    int64_t quotient = signed_value / signed_period;
    int64_t remainder = signed_value % signed_period;
    if (remainder < 0) {
        remainder += signed_period;
        --quotient;
    }
    return SignedPeriodIndex{.quotient = quotient, .remainder = static_cast<uint64_t>(remainder)};
}

ALWI uint32_t symmetric_index(const int32_t index, const uint32_t length) {
    if (length <= 1) {
        return 0;
    }

    const uint64_t period = static_cast<uint64_t>(length) * 2U;
    const uint64_t reflected = decompose_signed_period(index, period).remainder;
    return reflected < length ? static_cast<uint32_t>(reflected) : static_cast<uint32_t>(period - 1U - reflected);
}

ALWI uint32_t reflect_index(const int32_t index, const uint32_t length) {
    if (length <= 1) {
        return 0;
    }

    const uint64_t last_index = static_cast<uint64_t>(length - 1U);
    const uint64_t period = 2U * last_index;
    const uint64_t reflected = decompose_signed_period(index, period).remainder;
    return reflected <= last_index ? static_cast<uint32_t>(reflected) : static_cast<uint32_t>(period - reflected);
}

struct AntisymmetricIndex {
    uint32_t source_index;
    bool negate;
};

ALWI AntisymmetricIndex antisymmetric_index(const int32_t index, const uint32_t length) {
    if (length <= 1) {
        const auto period_index = decompose_signed_period(index, 2U);
        return AntisymmetricIndex{
            .source_index = 0,
            .negate = period_index.remainder != 0,
        };
    }

    const uint64_t segment = static_cast<uint64_t>(length);
    const uint64_t period = 4U * segment;
    const uint64_t reflected = decompose_signed_period(index, period).remainder;
    if (reflected < segment) {
        return AntisymmetricIndex{.source_index = static_cast<uint32_t>(reflected), .negate = false};
    }
    if (reflected < 2U * segment) {
        return AntisymmetricIndex{.source_index = static_cast<uint32_t>(2U * segment - 1U - reflected), .negate = true};
    }
    if (reflected < 3U * segment) {
        return AntisymmetricIndex{.source_index = static_cast<uint32_t>(reflected - 2U * segment), .negate = false};
    }
    return AntisymmetricIndex{.source_index = static_cast<uint32_t>(4U * segment - 1U - reflected), .negate = true};
}

}  // namespace ttwv::kernels::primitives
