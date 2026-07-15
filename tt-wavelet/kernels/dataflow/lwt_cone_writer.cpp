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
    const uint32_t col,
    const uint32_t local_output_index,
    const uint32_t output_offset,
    const uint32_t output_length) {
    if (local_output_index >= output_length) {
        return;
    }
    const uint32_t destination_index = output_offset + local_output_index;
    const uint32_t destination_stick = destination_index / ttwv::kStickWidth;
    const uint32_t destination_lane = destination_index % ttwv::kStickWidth;
    const uint32_t source_offset =
        ttwv::kernels::primitives::tile_offset(row, col) * static_cast<uint32_t>(sizeof(float));
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
        cb_wait_front(cb_output, 2);
        const uint32_t output_full_tile = get_read_ptr(cb_output);
        const uint32_t output_tail_tile = output_full_tile + tile_bytes;
        const uint32_t group_base = group * ttwv::device_protocol::kLwtGroupOutputElements;

        for (uint32_t row = 0; row < ttwv::device_protocol::kLwtRowsPerGroup; ++row) {
            const uint32_t row_base = group_base + row * ttwv::device_protocol::kLwtOutputBlocksPerRow *
                                                       ttwv::device_protocol::kLwtHalfStickElements;
            write_dram_half_block(dst, output_full_tile, row, 0, row_base, output_offset, output_length);
            write_dram_half_block(
                dst,
                output_full_tile,
                row,
                ttwv::device_protocol::kLwtHalfStickElements,
                row_base + ttwv::device_protocol::kLwtHalfStickElements,
                output_offset,
                output_length);
            write_dram_half_block(
                dst,
                output_tail_tile,
                row,
                0,
                row_base + 2 * ttwv::device_protocol::kLwtHalfStickElements,
                output_offset,
                output_length);
        }
        // Bound the number of outstanding NoC writes.  A long cone route can
        // contain hundreds of groups, while the output CB pages must not be
        // released until all writes sourcing those pages have completed.
        noc_async_write_barrier();
        cb_pop_front(cb_output, 2);
    }
}

ALWI void write_local_half_block(
    const uint32_t dst_addr,
    const uint32_t tile_addr,
    const uint32_t row,
    const uint32_t col,
    const uint32_t local_output_index,
    const uint32_t output_offset,
    const uint32_t output_length) {
    if (local_output_index >= output_length) {
        return;
    }

    auto* dst = reinterpret_cast<volatile tt_l1_ptr float*>(dst_addr);
    const auto* src = reinterpret_cast<volatile tt_l1_ptr float*>(tile_addr);
    const uint32_t source_index = ttwv::kernels::primitives::tile_offset(row, col);
    const uint32_t destination_index = output_offset + local_output_index;
#pragma unroll
    for (uint32_t lane = 0; lane < ttwv::device_protocol::kLwtHalfStickElements; ++lane) {
        dst[destination_index + lane] = src[source_index + lane];
    }
}

ALWI void write_local_output_groups(
    const uint32_t dst_addr,
    const uint32_t cb_output,
    const uint32_t tile_bytes,
    const uint32_t output_offset,
    const uint32_t output_length,
    const uint32_t group_count) {
    for (uint32_t group = 0; group < group_count; ++group) {
        cb_wait_front(cb_output, 2);
        const uint32_t output_full_tile = get_read_ptr(cb_output);
        const uint32_t output_tail_tile = output_full_tile + tile_bytes;
        const uint32_t group_base = group * ttwv::device_protocol::kLwtGroupOutputElements;

        for (uint32_t row = 0; row < ttwv::device_protocol::kLwtRowsPerGroup; ++row) {
            const uint32_t row_base = group_base + row * ttwv::device_protocol::kLwtOutputBlocksPerRow *
                                                       ttwv::device_protocol::kLwtHalfStickElements;
            write_local_half_block(dst_addr, output_full_tile, row, 0, row_base, output_offset, output_length);
            write_local_half_block(
                dst_addr,
                output_full_tile,
                row,
                ttwv::device_protocol::kLwtHalfStickElements,
                row_base + ttwv::device_protocol::kLwtHalfStickElements,
                output_offset,
                output_length);
            write_local_half_block(
                dst_addr,
                output_tail_tile,
                row,
                0,
                row_base + 2 * ttwv::device_protocol::kLwtHalfStickElements,
                output_offset,
                output_length);
        }
        cb_pop_front(cb_output, 2);
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
    constexpr uint32_t tile_bytes = get_tile_size(cb_output);
    constexpr auto config_args = TensorAccessorArgs<3>();
    constexpr auto final_args = TensorAccessorArgs<config_args.next_compile_time_args_offset()>();

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
                write_local_output_groups(
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
