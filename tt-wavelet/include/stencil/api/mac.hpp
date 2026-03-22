#pragma once

#include <cstdint>

#include "llk_defs.h"

template <bool APPROXIMATE, typename PolicyT, typename Callable, typename... Args>
void _llk_math_eltwise_stencil_mac_sfpu_(
    Callable&& sfpu_func,
    uint32_t dst_index_base,
    const uint32_t* dst_input_indices,
    uint32_t filter_len,
    uint32_t dst_index_out,
    int vector_mode = static_cast<int>(ckernel::VectorMode::RC),
    Args&&... args);
