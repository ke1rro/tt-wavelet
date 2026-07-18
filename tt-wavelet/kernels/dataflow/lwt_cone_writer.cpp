#include <cstdint>

#include "../../tt_wavelet/include/lifting/step.hpp"
#include "api/dataflow/dataflow_api.h"
#include "lwt_tile_row_major_utils.hpp"

namespace {

template <typename ConfigAccessor>
ALWI const uint32_t* load_route_config(
    const ConfigAccessor& config, const uint32_t config_addr, const uint32_t cb_config, const uint32_t page_index) {
    const auto page_accessor = TensorAccessor(config, config_addr, ttwv::device_protocol::kRouteConfigPageBytes);
    cb_reserve_back(cb_config, 1);
    noc_async_read(
        page_accessor.get_noc_addr(page_index), get_write_ptr(cb_config), ttwv::device_protocol::kRouteConfigPageBytes);
    noc_async_read_barrier();
    cb_push_back(cb_config, 1);
    cb_wait_front(cb_config, 1);
    return reinterpret_cast<const uint32_t*>(get_read_ptr(cb_config));
}

template <typename DstAccessor>
ALWI void write_dram_half_block(
    const DstAccessor& dst,
    const uint32_t tile_addr,
    const uint32_t row,
    const uint32_t local_output_index,
    const uint32_t output_offset,
    const uint32_t output_length) {
    if (local_output_index >= output_length) {
        return;
    }
    const uint32_t destination_index = output_offset + local_output_index;
    const uint32_t destination_stick = destination_index / ttwv::kStickWidth;
    const uint32_t destination_lane = destination_index % ttwv::kStickWidth;
    const uint32_t source_offset = row * ttwv::device_protocol::kLwtHalfStickBytes;
    noc_async_write(
        tile_addr + source_offset,
        dst.get_noc_addr(destination_stick) + destination_lane * sizeof(float),
        ttwv::device_protocol::kLwtHalfStickBytes);
}

template <typename DstAccessor>
ALWI void write_dram_output_groups(
    const DstAccessor& dst,
    const uint32_t cb_output,
    const uint32_t tile_bytes,
    const uint32_t output_offset,
    const uint32_t output_length,
    const uint32_t group_count) {
    for (uint32_t group = 0; group < group_count; ++group) {
        cb_wait_front(cb_output, 3);
        const uint32_t output_tiles = get_read_ptr(cb_output);
        const uint32_t group_base = group * ttwv::device_protocol::kLwtGroupOutputElements;

        for (uint32_t row = 0; row < ttwv::device_protocol::kLwtRowsPerGroup; ++row) {
            const uint32_t row_base = group_base + row * ttwv::device_protocol::kLwtOutputBlocksPerRow *
                                                       ttwv::device_protocol::kLwtHalfStickElements;
            for (uint32_t block = 0; block < ttwv::device_protocol::kLwtOutputBlocksPerRow; ++block) {
                write_dram_half_block(
                    dst,
                    output_tiles + block * tile_bytes,
                    row,
                    row_base + block * ttwv::device_protocol::kLwtHalfStickElements,
                    output_offset,
                    output_length);
            }
        }
        // Bound the number of outstanding NoC writes.  A long cone route can
        // contain hundreds of groups, while the output CB pages must not be
        // released until all writes sourcing those pages have completed.
        noc_async_write_barrier();
        cb_pop_front(cb_output, 3);
    }
}

template <bool UseNocLocalWrite>
ALWI void write_local_half_block(
    const uint32_t dst_addr,
    const uint32_t tile_addr,
    const uint32_t row,
    const uint32_t local_output_index,
    const uint32_t output_offset,
    const uint32_t output_length) {
    if (local_output_index >= output_length) {
        return;
    }

    const uint32_t source_index = row * ttwv::device_protocol::kLwtHalfStickElements;
    const uint32_t destination_index = output_offset + local_output_index;
    if constexpr (UseNocLocalWrite) {
        noc_async_write_one_packet_with_state(
            tile_addr + source_index * static_cast<uint32_t>(sizeof(float)),
            dst_addr + destination_index * static_cast<uint32_t>(sizeof(float)));
    } else {
        auto* dst = reinterpret_cast<volatile tt_l1_ptr float*>(dst_addr);
        const auto* src = reinterpret_cast<volatile tt_l1_ptr float*>(tile_addr);
#pragma unroll
        for (uint32_t lane = 0; lane < ttwv::device_protocol::kLwtHalfStickElements; ++lane) {
            dst[destination_index + lane] = src[source_index + lane];
        }
    }
}

template <bool UseNocLocalWrite, bool TileNative>
ALWI void write_local_output_groups(
    const uint32_t dst_addr,
    const uint32_t cb_output,
    const uint32_t tile_bytes,
    const uint32_t output_offset,
    const uint32_t output_length,
    const uint32_t group_count) {
    constexpr uint32_t group_elements = ttwv::device_protocol::kLwtGroupOutputElements;
    constexpr uint32_t blocks_per_group = ttwv::device_protocol::kLwtOutputBlocksPerRow;
    const uint32_t group_bytes = blocks_per_group * tile_bytes;
    if constexpr (UseNocLocalWrite) {
        const uint32_t write_bytes = TileNative ? tile_bytes : ttwv::device_protocol::kLwtHalfStickBytes;
        noc_async_write_one_packet_set_state(get_noc_addr(dst_addr), write_bytes);
    }

    for (uint32_t group = 0; group < group_count; ++group) {
        cb_wait_front(cb_output, 3);
        const uint32_t output_tiles = get_read_ptr(cb_output);
        const uint32_t group_base = group * ttwv::device_protocol::kLwtGroupOutputElements;
        if constexpr (TileNative) {
            const uint32_t destination_index = output_offset + group_base;
            const uint32_t destination_group = destination_index / group_elements;
            const uint32_t destination_addr = dst_addr + destination_group * group_bytes;
            if constexpr (UseNocLocalWrite) {
#pragma unroll
                for (uint32_t block = 0; block < blocks_per_group; ++block) {
                    noc_async_write_one_packet_with_state(
                        output_tiles + block * tile_bytes, destination_addr + block * tile_bytes);
                }
            } else {
                auto* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(destination_addr);
                const auto* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(output_tiles);
#pragma unroll 4
                for (uint32_t word = 0; word < group_bytes / sizeof(uint32_t); ++word) {
                    dst[word] = src[word];
                }
            }
        } else {
            for (uint32_t row = 0; row < ttwv::device_protocol::kLwtRowsPerGroup; ++row) {
                const uint32_t row_base =
                    group_base + row * blocks_per_group * ttwv::device_protocol::kLwtHalfStickElements;
                for (uint32_t block = 0; block < blocks_per_group; ++block) {
                    write_local_half_block<UseNocLocalWrite>(
                        dst_addr,
                        output_tiles + block * tile_bytes,
                        row,
                        row_base + block * ttwv::device_protocol::kLwtHalfStickElements,
                        output_offset,
                        output_length);
                }
            }
        }
        if constexpr (UseNocLocalWrite) {
            // Once the three tile-page writes have departed, the NoC no longer
            // reads these CB pages.  The route-level barrier below still waits
            // for completion before the next lifting route consumes workspace.
            noc_async_writes_flushed();
        }
        cb_pop_front(cb_output, 3);
    }
}

template <bool TileNative>
[[nodiscard]] ALWI uint32_t workspace_physical_index(const uint32_t logical_index) {
    if constexpr (!TileNative) {
        return logical_index;
    }
    constexpr uint32_t group_elements = ttwv::device_protocol::kLwtGroupOutputElements;
    constexpr uint32_t block_elements = ttwv::device_protocol::kLwtHalfStickElements;
    constexpr uint32_t blocks_per_row = ttwv::device_protocol::kLwtOutputBlocksPerRow;
    constexpr uint32_t narrow_tile_elements = ttwv::device_protocol::kLwtNarrowTileElements;
    const uint32_t group = logical_index / group_elements;
    const uint32_t in_group = logical_index - group * group_elements;
    const uint32_t row = in_group / (blocks_per_row * block_elements);
    const uint32_t in_row = in_group - row * blocks_per_row * block_elements;
    const uint32_t block = in_row / block_elements;
    const uint32_t lane = in_row - block * block_elements;
    return group * group_elements + block * narrow_tile_elements + row * block_elements + lane;
}

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

[[nodiscard]] ALWI float read_fused_output_value(
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
ALWI void write_fused_reconstructed_signal(
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
                        // Chunk boundaries are aligned to 3072 signal values,
                        // so each output group maps to one 1536-value split
                        // group in the final route's three narrow pages.
                        const uint32_t updated_begin = updates_even ? even_begin : odd_begin;
                        const uint32_t local_updated_index = split_index - updated_begin;
                        const uint32_t group_updated_index = local_updated_index - group * split_group_elements;
                        value = has_updated_values
                                    ? read_fused_output_value(output_tiles, tile_bytes, group_updated_index)
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

}  // namespace

void kernel_main() {
    const uint32_t route_config_addr = get_arg_val<uint32_t>(0);
    const uint32_t chunk_begin = get_arg_val<uint32_t>(1);
    const uint32_t chunk_count = get_arg_val<uint32_t>(2);
    const uint32_t route_count = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_config = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output = get_compile_time_arg_val(1);
    constexpr uint32_t cb_sync = get_compile_time_arg_val(2);
    constexpr bool use_noc_local_write = get_compile_time_arg_val(3) != 0;
    constexpr bool tile_native_workspace = get_compile_time_arg_val(4) != 0;
    constexpr bool inverse = get_compile_time_arg_val(5) != 0;
    constexpr uint32_t cb_interleave = get_compile_time_arg_val(6);
    constexpr uint32_t tile_bytes = get_tile_size(cb_output);
    constexpr auto config_args = TensorAccessorArgs<7>();
    constexpr auto final_args = TensorAccessorArgs<config_args.next_compile_time_args_offset()>();

    if constexpr (inverse) {
        const uint32_t chunk_config_addr = get_arg_val<uint32_t>(4);
        const uint32_t output_addr = get_arg_val<uint32_t>(5);
        const uint32_t left_pad = get_arg_val<uint32_t>(6);
        const auto output = TensorAccessor(final_args, output_addr, ttwv::device_protocol::kStickBytes);
        for (uint32_t local_chunk = 0; local_chunk < chunk_count; ++local_chunk) {
            const uint32_t global_chunk = chunk_begin + local_chunk;
            uint32_t chunk_words[ttwv::device_protocol::kConeChunkConfigWordCount];
            const uint32_t* loaded_chunk = load_route_config(config_args, chunk_config_addr, cb_config, global_chunk);
#pragma unroll
            for (uint32_t word = 0; word < ttwv::device_protocol::kConeChunkConfigWordCount; ++word) {
                chunk_words[word] = loaded_chunk[word];
            }
            cb_pop_front(cb_config, 1);

            bool fused_interleave_written = false;
            for (uint32_t route_index = 0; route_index < route_count; ++route_index) {
                const uint32_t config_index = global_chunk * route_count + route_index;
                const uint32_t* route = load_route_config(config_args, route_config_addr, cb_config, config_index);
                const uint32_t route_flags = route[ttwv::device_protocol::kRouteFlags];
                const bool fused_interleave = (route_flags & ttwv::device_protocol::kRouteFlagIlwtFinalInterleave) != 0;
                if (fused_interleave) {
                    write_fused_reconstructed_signal<tile_native_workspace>(
                        output,
                        cb_output,
                        cb_interleave,
                        tile_bytes,
                        left_pad,
                        route[ttwv::device_protocol::kRouteType],
                        route[ttwv::device_protocol::kRouteGroupCount],
                        chunk_words[ttwv::device_protocol::kIlwtFinalEvenAddr],
                        chunk_words[ttwv::device_protocol::kIlwtFinalEvenOffset],
                        chunk_words[ttwv::device_protocol::kIlwtFinalEvenBegin],
                        chunk_words[ttwv::device_protocol::kIlwtFinalOddAddr],
                        chunk_words[ttwv::device_protocol::kIlwtFinalOddOffset],
                        chunk_words[ttwv::device_protocol::kIlwtFinalOddBegin],
                        chunk_words[ttwv::device_protocol::kIlwtOutputBegin],
                        chunk_words[ttwv::device_protocol::kIlwtOutputLength]);
                    fused_interleave_written = true;
                } else {
                    write_local_output_groups<use_noc_local_write, tile_native_workspace>(
                        route[ttwv::device_protocol::kRouteOutputAddr],
                        cb_output,
                        tile_bytes,
                        route[ttwv::device_protocol::kRouteOutputOffset],
                        route[ttwv::device_protocol::kRouteOutputLength],
                        route[ttwv::device_protocol::kRouteGroupCount]);
                }
                noc_async_write_barrier();
                cb_pop_front(cb_config, 1);
                if (route_index + 1 < route_count) {
                    cb_reserve_back(cb_sync, 1);
                    cb_push_back(cb_sync, 1);
                }
            }

            if (!fused_interleave_written) {
                write_reconstructed_signal<tile_native_workspace>(
                    output,
                    cb_interleave,
                    left_pad,
                    chunk_words[ttwv::device_protocol::kIlwtFinalEvenAddr],
                    chunk_words[ttwv::device_protocol::kIlwtFinalEvenOffset],
                    chunk_words[ttwv::device_protocol::kIlwtFinalEvenBegin],
                    chunk_words[ttwv::device_protocol::kIlwtFinalOddAddr],
                    chunk_words[ttwv::device_protocol::kIlwtFinalOddOffset],
                    chunk_words[ttwv::device_protocol::kIlwtFinalOddBegin],
                    chunk_words[ttwv::device_protocol::kIlwtOutputBegin],
                    chunk_words[ttwv::device_protocol::kIlwtOutputLength]);
            }
            if (local_chunk + 1 < chunk_count) {
                cb_reserve_back(cb_sync, 1);
                cb_push_back(cb_sync, 1);
            }
        }
    } else {
        const uint32_t local_route_count = chunk_count * route_count;
        uint32_t flattened_route = 0;
        for (uint32_t local_chunk = 0; local_chunk < chunk_count; ++local_chunk) {
            const uint32_t global_chunk = chunk_begin + local_chunk;
            for (uint32_t route_index = 0; route_index < route_count; ++route_index, ++flattened_route) {
                const uint32_t config_index = global_chunk * route_count + route_index;
                const uint32_t* route = load_route_config(config_args, route_config_addr, cb_config, config_index);
                const uint32_t output_addr = route[ttwv::device_protocol::kRouteOutputAddr];
                const uint32_t output_length = route[ttwv::device_protocol::kRouteOutputLength];
                const uint32_t output_offset = route[ttwv::device_protocol::kRouteOutputOffset];
                const uint32_t group_count = route[ttwv::device_protocol::kRouteGroupCount];
                const uint32_t route_flags = route[ttwv::device_protocol::kRouteFlags];
                const bool final_dram = (route_flags & ttwv::device_protocol::kRouteFlagFinalDram) != 0;
                if (final_dram) {
                    const auto dst = TensorAccessor(final_args, output_addr, ttwv::device_protocol::kStickBytes);
                    write_dram_output_groups(dst, cb_output, tile_bytes, output_offset, output_length, group_count);
                } else {
                    write_local_output_groups<use_noc_local_write, tile_native_workspace>(
                        output_addr, cb_output, tile_bytes, output_offset, output_length, group_count);
                }

                noc_async_write_barrier();
                cb_pop_front(cb_config, 1);
                if (flattened_route + 1 < local_route_count) {
                    cb_reserve_back(cb_sync, 1);
                    cb_push_back(cb_sync, 1);
                }
            }
        }
    }
}
