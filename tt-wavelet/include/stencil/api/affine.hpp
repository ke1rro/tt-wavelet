#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "llk_defs.h"
#include "stencil/llk/llk_math_eltwise_stencil_sfpu_params.hpp"

/**
 * @brief Stencil affine API.
 *
 * Calling convention:
 * out = alpha * base + beta * sum_i(coefficients[i] * input_i)
 */
template <bool APPROXIMATE, typename PolicyT, typename Callable>
inline void _llk_math_eltwise_stencil_affine_sfpu_(
    Callable&& sfpu_func,
    uint32_t dst_index_base,
    const uint32_t* dst_input_indices,
    uint32_t filter_len,
    uint32_t dst_index_out,
    const float* coefficients,
    float alpha,
    float beta,
    int vector_mode = static_cast<int>(ckernel::VectorMode::RC)) {
    ckernel::stencil::_llk_math_eltwise_stencil_sfpu_params_<APPROXIMATE, PolicyT>(
        sfpu_func,
        dst_index_base,
        dst_input_indices,
        filter_len,
        dst_index_out,
        coefficients,
        alpha,
        beta,
        vector_mode);
}

template <bool APPROXIMATE, size_t FILTER_LEN, typename PolicyT, typename Callable>
inline void _llk_math_eltwise_stencil_affine_sfpu_(
    Callable&& sfpu_func,
    uint32_t dst_index_base,
    const std::array<uint32_t, FILTER_LEN>& dst_input_indices,
    uint32_t dst_index_out,
    const std::array<float, FILTER_LEN>& coefficients,
    float alpha,
    float beta,
    int vector_mode = static_cast<int>(ckernel::VectorMode::RC)) {
    ckernel::stencil::_llk_math_eltwise_stencil_sfpu_params_<APPROXIMATE, FILTER_LEN, PolicyT>(
        sfpu_func, dst_index_base, dst_input_indices, dst_index_out, coefficients, alpha, beta, vector_mode);
}