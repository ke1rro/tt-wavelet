#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

using namespace sfpi;

#ifdef TRISC_MATH

namespace ckernel::sfpu {

/**
 * @brief Per-face stencil accumulate kernel.
 *
 * Calling convention:
 * out = base + sum_i(input_i)
 *
 * @param dst_index_base Destination-register index for the base tile.
 * @param dst_input_indices Destination-register indices for input taps.
 * @param filter_len Number of taps in @p dst_input_indices.
 * @param dst_index_out Destination-register index where the result is written.
 */
sfpi_inline void calculate_stencil_acc_face(
    uint32_t dst_index_base, const uint32_t* dst_input_indices, uint32_t filter_len, uint32_t dst_index_out) {
    sfpi::vFloat result = dst_reg[dst_index_base];

#pragma GCC unroll 0
    for (uint32_t i = 0; i < filter_len; ++i) {
        result = result + dst_reg[dst_input_indices[i]];
    }

    dst_reg[dst_index_out] = result;
}

}  // namespace ckernel::sfpu

#endif  // TRISC_MATH
