#pragma once

#include <cstdint>

namespace ckernel::sfpu {

template <typename... Args>
inline void calculate_stencil_mac_face(
    std::uint32_t dst_index_base,
    const std::uint32_t* dst_input_indices,
    std::uint32_t filter_len,
    std::uint32_t dst_index_out,
    Args&&... args) {
    (void)dst_index_base;
    (void)dst_input_indices;
    (void)filter_len;
    (void)dst_index_out;
    (void)sizeof...(args);
}

}  // namespace ckernel::sfpu
