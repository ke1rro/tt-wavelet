#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

using namespace sfpi;

#ifdef TRISC_MATH

namespace ckernel::sfpu {

/**
 * @brief Per-face stencil MAC kernel.
 *
 * Calling convention:
 * out = base + sum_i(coefficients[i] * input_i)
 *
 * @param dst_index_base Destination-register index for the base tile.
 * @param dst_input_indices Destination-register indices for input taps.
 * @param filter_len Number of taps in @p dst_input_indices and @p coefficients.
 * @param dst_index_out Destination-register index where the result is written.
 * @param coefficients Runtime scalar coefficients, one value per tap.
 */
sfpi_inline void calculate_stencil_mac_face(
    uint32_t dst_index_base,
    const uint32_t* dst_input_indices,
    const uint32_t filter_len,
    uint32_t dst_index_out,
    const float* coefficients) {
    constexpr uint32_t dst_tile_size_sfpi = 32;

#pragma GCC unroll 0
    for (int d = 0; d < 8; ++d) {
        vFloat result = dst_reg[dst_index_base * dst_tile_size_sfpi];

#pragma GCC unroll 0
        for (uint32_t i = 0; i < filter_len; ++i) {
            vFloat input = dst_reg[dst_input_indices[i] * dst_tile_size_sfpi];
            vFloat coeff = vFloat(coefficients[i]);
            result = result + coeff * input;
        }

        dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        dst_reg++;
    }
}

}  // namespace ckernel::sfpu

#endif  // TRISC_MATH