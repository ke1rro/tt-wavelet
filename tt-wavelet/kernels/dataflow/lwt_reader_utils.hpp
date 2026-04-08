#pragma once

#include <cstdint>



#include "../utils/boundary.hpp"
#include "../utils/fill.hpp"
#include "../utils/stick_read_cache.hpp"

namespace ttwv::kernels::utils {

template <typename SrcAccessor>
ALWI float read_fused_lwt_split_value(
    const SrcAccessor& src,
    StickReadCache& cache,
    const uint32_t input_length,
    const uint32_t padded_length,
    const uint32_t left_pad,
    const uint32_t split_phase,
    const int32_t split_index) {
    if (input_length == 0) {
        return 0.0F;
    }

    const uint32_t logical_split_length = split_length(padded_length, split_phase);
    if (split_index < 0 || static_cast<uint32_t>(split_index) >= logical_split_length) {
        return 0.0F;
    }

    const uint32_t padded_index = static_cast<uint32_t>(split_index) * 2U + split_phase;
    const int32_t source_logical_index = static_cast<int32_t>(padded_index) - static_cast<int32_t>(left_pad);
    const uint32_t source_index = symmetric_index(source_logical_index, input_length);
    return read_source_value(src, cache, source_index);
}

template <typename SrcAccessor>
ALWI void push_fused_lwt_stream_stick(
    const SrcAccessor& src,
    StickReadCache& cache,
    const uint32_t cb_id,
    const uint32_t stick_width,
    const uint32_t input_length,
    const uint32_t padded_length,
    const uint32_t left_pad,
    const uint32_t split_phase,
    const int32_t source_offset,
    const int32_t logical_base_index) {
    cb_reserve_back(cb_id, 1);
    auto* ptr = reinterpret_cast<float*>(get_write_ptr(cb_id));
    fill_zeros(ptr, stick_width);

    for (uint32_t lane = 0; lane < stick_width; ++lane) {
        const int32_t split_index = logical_base_index + static_cast<int32_t>(lane) + source_offset;
        ptr[lane] = read_fused_lwt_split_value(
            src, cache, input_length, padded_length, left_pad, split_phase, split_index);
    }

    cb_push_back(cb_id, 1);
}

template <typename SrcAccessor>
ALWI void push_fused_lwt_tile_pair(
    const SrcAccessor& src,
    StickReadCache& cache,
    const uint32_t cb_halo,
    const uint32_t cb_cur,
    const uint32_t stick_nbytes,
    const uint32_t stick_width,
    const uint32_t input_length,
    const uint32_t padded_length,
    const uint32_t left_pad,
    const uint32_t split_phase,
    const int32_t source_offset,
    const uint32_t stencil_k,
    const uint32_t tile_index) {
    const int32_t current_base = static_cast<int32_t>(tile_index * stick_width) - stencil_left_pad(stencil_k);
    const int32_t halo_base = current_base - static_cast<int32_t>(stick_width);

    push_fused_lwt_stream_stick(
        src,
        cache,
        cb_halo,
        stick_width,
        input_length,
        padded_length,
        left_pad,
        split_phase,
        source_offset,
        halo_base);
    push_zero_sticks(cb_halo, stick_nbytes, 31);

    push_fused_lwt_stream_stick(
        src,
        cache,
        cb_cur,
        stick_width,
        input_length,
        padded_length,
        left_pad,
        split_phase,
        source_offset,
        current_base);
    push_zero_sticks(cb_cur, stick_nbytes, 31);
}

}  // namespace ttwv::kernels::utils
