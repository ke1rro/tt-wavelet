#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "llk_defs.h"
#include "tt_wave/sfpu/ckernel_sfpu_stencil_mac.h"
#include "tt_wave/stencil/llk/mac.h"

template <bool APPROXIMATE, typename PolicyT>
inline void stencil_mac_tile(
    uint32_t dst_index_base,
    const uint32_t* dst_input_indices,
    uint32_t filter_len,
    uint32_t dst_index_out,
    int vector_mode,
    const float* coefficients) {
    MATH((_llk_math_eltwise_stencil_mac_sfpu_<APPROXIMATE, PolicyT>(
        ckernel::sfpu::calculate_stencil_mac_face,
        dst_index_base,
        dst_input_indices,
        filter_len,
        dst_index_out,
        vector_mode,
        coefficients)));
}

template <bool APPROXIMATE, typename PolicyT>
inline void stencil_mac_tile(
    uint32_t dst_index_base,
    const uint32_t* dst_input_indices,
    uint32_t filter_len,
    uint32_t dst_index_out,
    const float* coefficients) {
    stencil_mac_tile<APPROXIMATE, PolicyT>(
        dst_index_base,
        dst_input_indices,
        filter_len,
        dst_index_out,
        static_cast<int>(ckernel::VectorMode::RC),
        coefficients);
}

template <bool APPROXIMATE, size_t FILTER_LEN, typename PolicyT>
inline void stencil_mac_tile(
    uint32_t dst_index_base,
    const std::array<uint32_t, FILTER_LEN>& dst_input_indices,
    uint32_t dst_index_out,
    int vector_mode,
    const std::array<float, FILTER_LEN>& coefficients) {
    MATH((_llk_math_eltwise_stencil_mac_sfpu_<APPROXIMATE, FILTER_LEN, PolicyT>(
        ckernel::sfpu::calculate_stencil_mac_face,
        dst_index_base,
        dst_input_indices,
        dst_index_out,
        vector_mode,
        coefficients)));
}

template <bool APPROXIMATE, size_t FILTER_LEN, typename PolicyT>
inline void stencil_mac_tile(
    const uint32_t dst_index_base,
    const std::array<uint32_t, FILTER_LEN>& dst_input_indices,
    const uint32_t dst_index_out,
    const std::array<float, FILTER_LEN>& coefficients) {
    stencil_mac_tile<APPROXIMATE, FILTER_LEN, PolicyT>(
        dst_index_base, dst_input_indices, dst_index_out, static_cast<int>(ckernel::VectorMode::RC), coefficients);
}
