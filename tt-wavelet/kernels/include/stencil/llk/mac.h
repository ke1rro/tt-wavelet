#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "../policy.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "params.h"

template <bool APPROXIMATE, typename PolicyT, typename Callable>
inline void _llk_math_eltwise_stencil_mac_sfpu_(
    Callable&& sfpu_func,
    uint32_t dst_index_base,
    const uint32_t* dst_input_indices,
    uint32_t filter_len,
    uint32_t dst_index_out,
    int vector_mode,
    const float* coefficients) {
    ckernel::stencil::assert_taps_within_policy<PolicyT>(filter_len);
    LLK_ASSERT((coefficients != nullptr), ckernel::stencil::kErrNullCoefficients);

    ckernel::stencil::_llk_math_eltwise_stencil_sfpu_params_<APPROXIMATE, PolicyT>(
        sfpu_func, dst_index_base, dst_input_indices, filter_len, dst_index_out, vector_mode, coefficients);
}

template <bool APPROXIMATE, size_t FILTER_LEN, typename PolicyT, typename Callable>
inline void _llk_math_eltwise_stencil_mac_sfpu_(
    Callable&& sfpu_func,
    uint32_t dst_index_base,
    const std::array<uint32_t, FILTER_LEN>& dst_input_indices,
    uint32_t dst_index_out,
    int vector_mode,
    const std::array<float, FILTER_LEN>& coefficients) {
    static_assert(FILTER_LEN > 0, "FILTER_LEN must be > 0");
    static_assert(FILTER_LEN <= PolicyT::max_taps_per_pass, "FILTER_LEN exceeds per-pass stencil policy capacity");

    ckernel::stencil::_llk_math_eltwise_stencil_sfpu_params_<APPROXIMATE, FILTER_LEN, PolicyT>(
        sfpu_func, dst_index_base, dst_input_indices, dst_index_out, vector_mode, coefficients.data());
}
