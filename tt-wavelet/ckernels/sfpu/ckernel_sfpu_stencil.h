#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

using namespace sfpi;

#ifdef TRISC_MATH

namespace ckernel::sfpu {

/**
 * @brief Per-face generic stencil reduction kernel.
 *
 * Calling convention:
 * out = sum_i(input_i)
 *
 * This is the default generic stencil face implementation used when no
 * operation-specific coefficients are provided.
 *
 * @param dst_input_indices Destination-register indices for input taps.
 * @param filter_len Number of taps in @p dst_input_indices.
 * @param dst_index_out Destination-register index where the result is written.
 */
sfpi_inline void calculate_stencil_face(
    const uint32_t* dst_input_indices, uint32_t filter_len, uint32_t dst_index_out) {
    constexpr uint32_t dst_tile_size_sfpi = 32;

#pragma GCC unroll 0
    for (int d = 0; d < 8; ++d) {
        vFloat result = vFloat(0.0f);

#pragma GCC unroll 0
        for (uint32_t i = 0; i < filter_len; ++i) {
            result = result + dst_reg[dst_input_indices[i] * dst_tile_size_sfpi];
        }

        dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        dst_reg++;
    }
}

}  // namespace ckernel::sfpu

#endif  // TRISC_MATH
