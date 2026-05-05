#include <cstdint>

#include "../primitives/tile_row_major.hpp"
#include "api/dataflow/dataflow_api.h"

namespace row_major = ttwv::kernels::primitives;

void kernel_main() {
    const uint32_t num_steps = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_output = get_compile_time_arg_val(0);
    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(1);
    constexpr uint32_t cb_sync = get_compile_time_arg_val(2);
    constexpr bool output_is_local_l1 = get_compile_time_arg_val(3) != 0;
    constexpr uint32_t tile_nbytes = get_tile_size(cb_output);
    constexpr uint32_t stick_width = stick_nbytes / sizeof(float);
    constexpr auto dst_args = TensorAccessorArgs<4>();
    constexpr auto tile_dst_args = TensorAccessorArgs<dst_args.next_compile_time_args_offset()>();

    for (uint32_t step = 0; step < num_steps; ++step) {
        const uint32_t arg_base = 1 + step * 5;
        const uint32_t output_addr = get_arg_val<uint32_t>(arg_base + 0);
        const uint32_t output_length = get_arg_val<uint32_t>(arg_base + 1);
        const uint32_t output_group_count = get_arg_val<uint32_t>(arg_base + 2);
        const uint32_t tile_output_addr = get_arg_val<uint32_t>(arg_base + 3);
        const uint32_t write_tile_sidecar = get_arg_val<uint32_t>(arg_base + 4);

        const auto dst = TensorAccessor(dst_args, output_addr, stick_nbytes);
        const auto tile_dst = TensorAccessor(tile_dst_args, tile_output_addr, tile_nbytes);

        for (uint32_t group = 0; group < output_group_count; ++group) {
            cb_wait_front(cb_output, 2);
            const uint32_t output_full_tile = get_read_ptr(cb_output);
            const uint32_t output_tail_tile = output_full_tile + tile_nbytes;
            const uint32_t group_base = group * row_major::kLwtGroupOutputElements;

            if (write_tile_sidecar != 0) {
                noc_async_write_tile(group * 2, tile_dst, output_full_tile);
                noc_async_write_tile(group * 2 + 1, tile_dst, output_tail_tile);
                noc_async_write_barrier();
            }

            for (uint32_t row = 0; row < row_major::kLwtRowsPerGroup; ++row) {
                const uint32_t row_base =
                    group_base + row * row_major::kLwtOutputBlocksPerRow * row_major::kLwtHalfStickElements;
                if constexpr (output_is_local_l1) {
                    row_major::write_lwt_half_block_local_l1(
                        output_addr, stick_nbytes, output_full_tile, row, 0, row_base, output_length, stick_width);
                    row_major::write_lwt_half_block_local_l1(
                        output_addr,
                        stick_nbytes,
                        output_full_tile,
                        row,
                        row_major::kLwtHalfStickElements,
                        row_base + row_major::kLwtHalfStickElements,
                        output_length,
                        stick_width);
                    row_major::write_lwt_half_block_local_l1(
                        output_addr,
                        stick_nbytes,
                        output_tail_tile,
                        row,
                        0,
                        row_base + 2 * row_major::kLwtHalfStickElements,
                        output_length,
                        stick_width);
                } else {
                    row_major::write_lwt_half_block(
                        dst, output_full_tile, row, 0, row_base, output_length, stick_width);
                    row_major::write_lwt_half_block(
                        dst,
                        output_full_tile,
                        row,
                        row_major::kLwtHalfStickElements,
                        row_base + row_major::kLwtHalfStickElements,
                        output_length,
                        stick_width);
                    row_major::write_lwt_half_block(
                        dst,
                        output_tail_tile,
                        row,
                        0,
                        row_base + 2 * row_major::kLwtHalfStickElements,
                        output_length,
                        stick_width);
                    noc_async_write_barrier();
                }
            }

            cb_pop_front(cb_output, 2);
        }

        cb_reserve_back(cb_sync, 1);
        cb_push_back(cb_sync, 1);
    }
}
