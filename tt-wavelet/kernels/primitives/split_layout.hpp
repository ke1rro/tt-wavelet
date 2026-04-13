#pragma once

#include <cstdint>

#define ALWI inline __attribute__((always_inline))

namespace ttwv::kernels::primitives {

ALWI uint32_t even_stick_count(const uint32_t padded_length, const uint32_t stick_width) {
    return ((padded_length + 1) / 2 + stick_width - 1) / stick_width;
}

ALWI uint32_t odd_stick_count(const uint32_t padded_length, const uint32_t stick_width) {
    return (padded_length / 2 + stick_width - 1) / stick_width;
}

ALWI uint32_t split_length(const uint32_t padded_length, const uint32_t split_phase) {
    return split_phase == 0U ? ((padded_length + 1U) / 2U) : (padded_length / 2U);
}

}  // namespace ttwv::kernels::primitives
