#pragma once

#include <cstdint>

namespace ttwv::kernels::utils {

inline __attribute__((always_inline)) uint32_t positive_mod(const int32_t value, const uint32_t modulus) {
    if (modulus == 0) {
        return 0;
    }

    const int32_t signed_modulus = static_cast<int32_t>(modulus);
    const int32_t remainder = value % signed_modulus;
    return static_cast<uint32_t>(remainder < 0 ? remainder + signed_modulus : remainder);
}

inline __attribute__((always_inline)) uint32_t symmetric_index(const int32_t index, const uint32_t length) {
    if (length <= 1) {
        return 0;
    }

    const uint64_t period = static_cast<uint64_t>(length) * 2;
    const uint32_t reflected = positive_mod(index, static_cast<uint32_t>(period > 0xFFFFFFFFU ? 0xFFFFFFFFU : period));
    return reflected < length ? reflected : static_cast<uint32_t>(period) - 1U - reflected;
}

inline __attribute__((always_inline)) uint32_t
even_stick_count(const uint32_t padded_length, const uint32_t stick_width) {
    return ((padded_length + 1) / 2 + stick_width - 1) / stick_width;
}

inline __attribute__((always_inline)) uint32_t
odd_stick_count(const uint32_t padded_length, const uint32_t stick_width) {
    return (padded_length / 2 + stick_width - 1) / stick_width;
}

}  // namespace ttwv::kernels::utils
