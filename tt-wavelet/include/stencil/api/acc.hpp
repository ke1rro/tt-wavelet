#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "llk_defs.h"
#include "stencil/llk/llk_math_eltwise_stencil_sfpu_params.hpp"

/**
 * @brief Stencil-accumulate API.
 *
 * Calling convention:
 * out = base + f(input_0 ... input_{L-1})
 */
template <bool APPROXIMATE, typename PolicyT, typename Callable>
inline void _llk_math_eltwise_stencil_acc_sfpu_(
    Callable&& sfpu_func,
    uint32_t dst_index_base,
    const uint32_t* dst_input_indices,
    uint32_t filter_len,
    uint32_t dst_index_out,
    int vector_mode = static_cast<int>(ckernel::VectorMode::RC)) {
    ckernel::stencil::_llk_math_eltwise_stencil_sfpu_params_<APPROXIMATE, PolicyT>(
        sfpu_func, dst_index_base, dst_input_indices, filter_len, dst_index_out, vector_mode);
}

template <bool APPROXIMATE, size_t FILTER_LEN, typename PolicyT, typename Callable>
inline void _llk_math_eltwise_stencil_acc_sfpu_(
    Callable&& sfpu_func,
    uint32_t dst_index_base,
    const std::array<uint32_t, FILTER_LEN>& dst_input_indices,
    uint32_t dst_index_out,
    int vector_mode = static_cast<int>(ckernel::VectorMode::RC)) {
    ckernel::stencil::_llk_math_eltwise_stencil_sfpu_params_<APPROXIMATE, FILTER_LEN, PolicyT>(
        sfpu_func, dst_index_base, dst_input_indices, dst_index_out, vector_mode);
}