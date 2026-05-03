#include <cstdint>

#include "../primitives/tile_row_major.hpp"
#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t kHalfStickElements = 16;
constexpr uint32_t kHalfStickBytes = kHalfStickElements * sizeof(float);
constexpr uint32_t kRowsPerGroup = 32;
constexpr uint32_t kOutputBlocksPerRow = 3;
constexpr uint32_t kGroupOutputElements = kRowsPerGroup * kOutputBlocksPerRow * kHalfStickElements;

template <typename DstAccessor>
void write_half_block(
    const DstAccessor& dst,
    const uint32_t tile_addr,
    const uint32_t row,
    const uint32_t col,
    const uint32_t output_index,
    const uint32_t output_length,
    const uint32_t stick_width) {
    if (output_index >= output_length) {
        return;
    }

    const uint32_t dst_stick = output_index / stick_width;
    const uint32_t dst_lane = output_index % stick_width;
    const uint32_t src_offset = ttwv::kernels::primitives::tile_offset(row, col) * static_cast<uint32_t>(sizeof(float));
    const uint64_t noc_addr = dst.get_noc_addr(dst_stick) + dst_lane * sizeof(float);
    noc_async_write(tile_addr + src_offset, noc_addr, kHalfStickBytes);
}

}  // namespace

void kernel_main() {
    const uint32_t num_steps = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_output = get_compile_time_arg_val(0);
    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(1);
    constexpr uint32_t cb_sync = get_compile_time_arg_val(2);
    constexpr uint32_t tile_nbytes = get_tile_size(cb_output);
    constexpr uint32_t stick_width = stick_nbytes / sizeof(float);
    constexpr auto dst_args = TensorAccessorArgs<3>();

    for (uint32_t step = 0; step < num_steps; ++step) {
        const uint32_t arg_base = 1 + step * 3;
        const uint32_t output_addr = get_arg_val<uint32_t>(arg_base + 0);
        const uint32_t output_length = get_arg_val<uint32_t>(arg_base + 1);
        const uint32_t output_group_count = get_arg_val<uint32_t>(arg_base + 2);

        const auto dst = TensorAccessor(dst_args, output_addr, stick_nbytes);

        for (uint32_t group = 0; group < output_group_count; ++group) {
            cb_wait_front(cb_output, 2);
            const uint32_t output_full_tile = get_read_ptr(cb_output);
            const uint32_t output_tail_tile = output_full_tile + tile_nbytes;
            const uint32_t group_base = group * kGroupOutputElements;

            for (uint32_t row = 0; row < kRowsPerGroup; ++row) {
                const uint32_t row_base = group_base + row * kOutputBlocksPerRow * kHalfStickElements;
                write_half_block(dst, output_full_tile, row, 0, row_base, output_length, stick_width);
                write_half_block(
                    dst,
                    output_full_tile,
                    row,
                    kHalfStickElements,
                    row_base + kHalfStickElements,
                    output_length,
                    stick_width);
                write_half_block(
                    dst, output_tail_tile, row, 0, row_base + 2 * kHalfStickElements, output_length, stick_width);
                noc_async_write_barrier();
            }

            cb_pop_front(cb_output, 2);
        }

        cb_reserve_back(cb_sync, 1);
        cb_push_back(cb_sync, 1);
    }
}
