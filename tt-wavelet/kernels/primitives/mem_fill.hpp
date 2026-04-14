#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#define ALWI inline __attribute__((always_inline))

namespace ttwv::kernels::primitives {

ALWI void fill_zeros(float* ptr, const uint32_t n) { __builtin_memset(ptr, 0, n * sizeof(float)); }

}  // namespace ttwv::kernels::primitives
