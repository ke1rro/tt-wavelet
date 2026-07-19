#pragma once

#include <cstdint>

#include "../../tt_wavelet/include/device_protocol/lwt_config.hpp"
#include "../../tt_wavelet/include/lifting/step.hpp"
#include "api/dataflow/dataflow_api.h"
#include "workspace_layout.hpp"

namespace ttwv::kernels::primitives {

template <bool TileNative, typename DstAccessor>
ALWI void write_reconstructed_signal(
    const DstAccessor& dst,
    const uint32_t cb_interleave,
    const uint32_t left_pad,
    const uint32_t even_addr,
    const uint32_t even_offset,
    const uint32_t even_begin,
    const uint32_t odd_addr,
    const uint32_t odd_offset,
    const uint32_t odd_begin,
    const uint32_t output_begin,
    const uint32_t output_length) {
    const auto* even = reinterpret_cast<volatile tt_l1_ptr float*>(even_addr);
    const auto* odd = reinterpret_cast<volatile tt_l1_ptr float*>(odd_addr);
    const uint32_t output_end = output_begin + output_length;
    const uint32_t first_stick = output_begin / ttwv::kStickWidth;
    const uint32_t stick_count = (output_length + ttwv::kStickWidth - 1U) / ttwv::kStickWidth;

    for (uint32_t local_stick = 0; local_stick < stick_count; ++local_stick) {
        cb_reserve_back(cb_interleave, 1);
        auto* staging = reinterpret_cast<float*>(get_write_ptr(cb_interleave));
        const uint32_t signal_base = output_begin + local_stick * ttwv::kStickWidth;
#pragma unroll
        for (uint32_t lane = 0; lane < ttwv::kStickWidth; ++lane) {
            const uint32_t signal_index = signal_base + lane;
            float value = 0.0F;
            if (signal_index < output_end) {
                const uint32_t padded_index = left_pad + signal_index;
                const uint32_t split_index = padded_index / 2U;
                if ((padded_index & 1U) == 0) {
                    const uint32_t logical_index = even_offset + split_index - even_begin;
                    value = even[workspace_physical_index<TileNative>(logical_index)];
                } else {
                    const uint32_t logical_index = odd_offset + split_index - odd_begin;
                    value = odd[workspace_physical_index<TileNative>(logical_index)];
                }
            }
            staging[lane] = value;
        }
        noc_async_write(
            get_write_ptr(cb_interleave),
            dst.get_noc_addr(first_stick + local_stick),
            ttwv::device_protocol::kStickBytes);
        noc_async_write_barrier();
        cb_push_back(cb_interleave, 1);
        cb_wait_front(cb_interleave, 1);
        cb_pop_front(cb_interleave, 1);
    }
}

[[nodiscard]] ALWI float read_direct_output_value(
    const uint32_t output_tiles, const uint32_t tile_bytes, const uint32_t logical_index) {
    constexpr uint32_t row_elements =
        ttwv::device_protocol::kLwtOutputBlocksPerRow * ttwv::device_protocol::kLwtHalfStickElements;
    const uint32_t row = logical_index / row_elements;
    const uint32_t in_row = logical_index - row * row_elements;
    const uint32_t block = in_row / ttwv::device_protocol::kLwtHalfStickElements;
    const uint32_t lane = in_row - block * ttwv::device_protocol::kLwtHalfStickElements;
    const auto* tile = reinterpret_cast<volatile tt_l1_ptr float*>(output_tiles + block * tile_bytes);
    return tile[row * ttwv::device_protocol::kLwtHalfStickElements + lane];
}

template <bool TileNative, typename DstAccessor>
ALWI void write_direct_interleaved_signal(
    const DstAccessor& dst,
    const uint32_t cb_output,
    const uint32_t cb_interleave,
    const uint32_t tile_bytes,
    const uint32_t left_pad,
    const uint32_t route_type,
    const uint32_t updated_group_count,
    const uint32_t even_addr,
    const uint32_t even_offset,
    const uint32_t even_begin,
    const uint32_t odd_addr,
    const uint32_t odd_offset,
    const uint32_t odd_begin,
    const uint32_t output_begin,
    const uint32_t output_length) {
    constexpr uint32_t split_group_elements = ttwv::device_protocol::kLwtGroupOutputElements;
    constexpr uint32_t signal_group_elements = ttwv::device_protocol::kIlwtGroupOutputElements;
    const bool updates_even = route_type == static_cast<uint32_t>(ttwv::StepType::kUpdate);
    const auto* even = reinterpret_cast<volatile tt_l1_ptr float*>(even_addr);
    const auto* odd = reinterpret_cast<volatile tt_l1_ptr float*>(odd_addr);
    const uint32_t output_group_count = (output_length + signal_group_elements - 1U) / signal_group_elements;

    for (uint32_t group = 0; group < output_group_count; ++group) {
        const bool has_updated_values = group < updated_group_count;
        uint32_t output_tiles = 0;
        if (has_updated_values) {
            cb_wait_front(cb_output, 3);
            output_tiles = get_read_ptr(cb_output);
        }

        const uint32_t group_signal_offset = group * signal_group_elements;
        const uint32_t group_output_length = output_length - group_signal_offset < signal_group_elements
                                                 ? output_length - group_signal_offset
                                                 : signal_group_elements;
        const uint32_t stick_count = (group_output_length + ttwv::kStickWidth - 1U) / ttwv::kStickWidth;
        const uint32_t first_stick = (output_begin + group_signal_offset) / ttwv::kStickWidth;

        for (uint32_t local_stick = 0; local_stick < stick_count; ++local_stick) {
            cb_reserve_back(cb_interleave, 1);
            auto* staging = reinterpret_cast<float*>(get_write_ptr(cb_interleave));
            const uint32_t local_signal_base = group_signal_offset + local_stick * ttwv::kStickWidth;
#pragma unroll
            for (uint32_t lane = 0; lane < ttwv::kStickWidth; ++lane) {
                const uint32_t local_signal_index = local_signal_base + lane;
                float value = 0.0F;
                if (local_signal_index < output_length) {
                    const uint32_t signal_index = output_begin + local_signal_index;
                    const uint32_t padded_index = left_pad + signal_index;
                    const uint32_t split_index = padded_index / 2U;
                    const bool is_even = (padded_index & 1U) == 0;
                    if (is_even == updates_even) {
                        const uint32_t updated_begin = updates_even ? even_begin : odd_begin;
                        const uint32_t local_updated_index = split_index - updated_begin;
                        const uint32_t group_updated_index = local_updated_index - group * split_group_elements;
                        value = has_updated_values
                                    ? read_direct_output_value(output_tiles, tile_bytes, group_updated_index)
                                    : 0.0F;
                    } else if (is_even) {
                        const uint32_t logical_index = even_offset + split_index - even_begin;
                        value = even[workspace_physical_index<TileNative>(logical_index)];
                    } else {
                        const uint32_t logical_index = odd_offset + split_index - odd_begin;
                        value = odd[workspace_physical_index<TileNative>(logical_index)];
                    }
                }
                staging[lane] = value;
            }
            noc_async_write(
                get_write_ptr(cb_interleave),
                dst.get_noc_addr(first_stick + local_stick),
                ttwv::device_protocol::kStickBytes);
            noc_async_write_barrier();
            cb_push_back(cb_interleave, 1);
            cb_wait_front(cb_interleave, 1);
            cb_pop_front(cb_interleave, 1);
        }

        if (has_updated_values) {
            cb_pop_front(cb_output, 3);
        }
    }
}

}  // namespace ttwv::kernels::primitives
