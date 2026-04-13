#pragma once

#include <cstdint>

#define ALWI inline __attribute__((always_inline))

namespace ttwv::kernels::primitives {

ALWI int32_t stencil_left_pad(const uint32_t stencil_k) { return 17 - static_cast<int32_t>(stencil_k); }

}  // namespace ttwv::kernels::primitives
