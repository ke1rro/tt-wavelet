#include <cstdint>

#include "../primitives/tile_row_major.hpp"
#include "api/dataflow/dataflow_api.h"

namespace row_major = ttwv::kernels::primitives;

namespace {

constexpr uint32_t kSpliceAdvance = 32 * 48;
constexpr uint32_t kTilesPerSplice = 2;

ALWI float read_chain_value(const uint32_t cb_id, const uint32_t tile_nbytes, const uint32_t source_index) {
    const uint32_t splice = source_index / kSpliceAdvance;
    const uint32_t in_splice = source_index % kSpliceAdvance;
    const uint32_t row = in_splice / 48;
    const uint32_t column = in_splice % 48;
    const uint32_t tile = column / 32;
    const uint32_t tile_column = column % 32;
    const uint32_t tile_addr = get_read_ptr(cb_id) + (splice * kTilesPerSplice + tile) * tile_nbytes;
    const auto* values = reinterpret_cast<const volatile tt_l1_ptr float*>(tile_addr);
    return values[row_major::tile_offset(row, tile_column)];
}

template <typename Accessor>
ALWI void write_dense_stream(
    const Accessor& dst,
    const uint32_t cb_source,
    const uint32_t cb_scratch,
    const uint32_t tile_nbytes,
    const uint32_t stick_width,
    const uint32_t source_offset,
    const uint32_t destination_stick_start,
    const uint32_t output_length) {
    const uint32_t stick_nbytes = stick_width * sizeof(float);
    const uint32_t output_sticks = (output_length + stick_width - 1) / stick_width;
    for (uint32_t stick = 0; stick < output_sticks; ++stick) {
        cb_reserve_back(cb_scratch, 1);
        auto* scratch = reinterpret_cast<volatile tt_l1_ptr float*>(get_write_ptr(cb_scratch));
        const uint32_t destination_start = stick * stick_width;
        for (uint32_t lane = 0; lane < stick_width; ++lane) {
            const uint32_t output_index = destination_start + lane;
            scratch[lane] = output_index < output_length
                                ? read_chain_value(cb_source, tile_nbytes, source_offset + output_index)
                                : 0.0F;
        }
        noc_async_write(get_write_ptr(cb_scratch), dst.get_noc_addr(destination_stick_start + stick), stick_nbytes);
        noc_async_write_barrier();
        cb_push_back(cb_scratch, 1);
        cb_wait_front(cb_scratch, 1);
        cb_pop_front(cb_scratch, 1);
    }
}

}  // namespace

void kernel_main() {
    const uint32_t splice_count = get_arg_val<uint32_t>(0);
    const uint32_t approximation_addr = get_arg_val<uint32_t>(1);
    const uint32_t detail_addr = get_arg_val<uint32_t>(2);
    const uint32_t approximation_offset = get_arg_val<uint32_t>(3);
    const uint32_t detail_offset = get_arg_val<uint32_t>(4);
    const uint32_t output_stick_start = get_arg_val<uint32_t>(5);
    const uint32_t output_length = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_even = get_named_compile_time_arg_val("cb_even");
    constexpr uint32_t cb_odd = get_named_compile_time_arg_val("cb_odd");
    constexpr uint32_t cb_done = get_named_compile_time_arg_val("cb_done");
    constexpr uint32_t cb_scratch = get_named_compile_time_arg_val("cb_scratch");
    constexpr bool final_swapped = get_named_compile_time_arg_val("final_swapped") != 0;
    constexpr uint32_t stick_width = get_named_compile_time_arg_val("stick_width");
    constexpr uint32_t tile_nbytes = get_tile_size(cb_even);
    constexpr auto approximation_args = TensorAccessorArgs<0>();
    constexpr auto detail_args = TensorAccessorArgs<approximation_args.next_compile_time_args_offset()>();
    const auto approximation = TensorAccessor(approximation_args, approximation_addr, stick_width * sizeof(float));
    const auto detail = TensorAccessor(detail_args, detail_addr, stick_width * sizeof(float));

    constexpr uint32_t approximation_cb = final_swapped ? cb_odd : cb_even;
    constexpr uint32_t detail_cb = final_swapped ? cb_even : cb_odd;
    cb_wait_front(cb_done, 1);
    cb_wait_front(approximation_cb, splice_count * kTilesPerSplice);
    cb_wait_front(detail_cb, splice_count * kTilesPerSplice);

    write_dense_stream(
        approximation,
        approximation_cb,
        cb_scratch,
        tile_nbytes,
        stick_width,
        approximation_offset,
        output_stick_start,
        output_length);
    write_dense_stream(
        detail, detail_cb, cb_scratch, tile_nbytes, stick_width, detail_offset, output_stick_start, output_length);

    cb_pop_front(cb_done, 1);
    cb_pop_front(approximation_cb, splice_count * kTilesPerSplice);
    cb_pop_front(detail_cb, splice_count * kTilesPerSplice);
}
