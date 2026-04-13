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

ALWI uint32_t symmetric_index(const int32_t index, const uint32_t length) {
    if (length <= 1) {
        return 0;
    }

    const uint64_t period = static_cast<uint64_t>(length) * 2;
    const uint32_t reflected = positive_mod(index, static_cast<uint32_t>(period > 0xFFFFFFFFU ? 0xFFFFFFFFU : period));
    return reflected < length ? reflected : static_cast<uint32_t>(period) - 1U - reflected;
}

}  // namespace ttwv::kernels::primitives
