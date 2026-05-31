#pragma once

#include <cstdint>

#define ALWI inline __attribute__((always_inline))

namespace ttwv::kernels::primitives {

ALWI uint32_t ceil_div(const uint32_t value, const uint32_t divisor) { return (value + divisor - 1) / divisor; }

ALWI uint32_t stick_count_for_elements(const uint32_t element_count, const uint32_t stick_width) {
    return ceil_div(element_count, stick_width);
}

}  // namespace ttwv::kernels::primitives
