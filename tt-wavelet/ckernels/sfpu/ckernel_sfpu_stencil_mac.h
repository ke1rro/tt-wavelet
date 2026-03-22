#pragma once

#include <cstdint>

#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu {

template <typename CoeffAccessor>
inline void calculate_stencil_mac_face(
    std::uint32_t dst_index_base,
    const std::uint32_t* dst_input_indices,
    const std::uint32_t filter_len,
    std::uint32_t dst_index_out,
    const CoeffAccessor& coeffs) {
    vFloat result = dst_reg[dst_index_base];

    for (std::uint32_t i{0}; i < filter_len; ++i) {
        vFloat input{dst_reg[dst_input_indices[i]]};
        vFloat coeff{coeffs[i]};
        result = result + coeff * input;
    }

    dst_reg[dst_index_out] = result;
}

}  // namespace ckernel::sfpu