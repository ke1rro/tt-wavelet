#pragma once

#include <cstdint>

#include "ckernel_defs.h"
#include "sfpi.h"

#ifdef TRISC_MATH

namespace ckernel::sfpu {

/**
 * @brief Per-face affine stencil kernel.
 *
 * Calling convention:
 * out = alpha * base + beta * sum_i(coefficients[i] * input[i])
 *
 * @param dst_index_base Destination-register index for the base tile.
 * @param dst_input_indices Destination-register indices for stencil input tiles.
 * @param filter_len Number of stencil taps.
 * @param dst_index_out Destination-register index where the result is written.
 * @param coefficients Scalar stencil coefficients.
 * @param alpha Scalar multiplier for the base tile.
 * @param beta Scalar multiplier for the stencil contribution.
 */
sfpi_inline void calculate_stencil_affine_face(
    uint32_t dst_index_base,
    const uint32_t* dst_input_indices,
    uint32_t filter_len,
    uint32_t dst_index_out,
    const float* coefficients,
    float alpha,
    float beta) {
    sfpi::vFloat alpha_v = sfpi::vFloat(alpha);
    sfpi::vFloat beta_v = sfpi::vFloat(beta);
    sfpi::vFloat result = alpha_v * dst_reg[dst_index_base];

#pragma GCC unroll 0
    for (uint32_t i = 0; i < filter_len; ++i) {
        sfpi::vFloat input = dst_reg[dst_input_indices[i]];
        sfpi::vFloat coeff = sfpi::vFloat(coefficients[i]);
        result = result + (beta_v * coeff) * input;
    }

    dst_reg[dst_index_out] = result;
}

}  // namespace ckernel::sfpu

#endif  // TRISC_MATH
