#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "pad_split_1d_reader_utils.hpp"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t input_length = get_arg_val<uint32_t>(1);
    const uint32_t padded_length = get_arg_val<uint32_t>(2);
    const uint32_t left_pad = get_arg_val<uint32_t>(3);
    const uint32_t padded_begin = get_arg_val<uint32_t>(4);
    const uint32_t padded_end = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_even = get_named_compile_time_arg_val("cb_even");
    constexpr uint32_t cb_odd = get_named_compile_time_arg_val("cb_odd");
    constexpr uint32_t cb_cache = get_named_compile_time_arg_val("cb_cache");
    constexpr auto src_args = TensorAccessorArgs<0>();
    const auto src = TensorAccessor(src_args, src_addr, ttwv::device_protocol::kStickBytes);

    ttwv::kernels::primitives::StickReadCache read_cache{
        cb_cache,
        ttwv::device_protocol::kStickBytes,
        ttwv::kStickWidth,
        ttwv::device_protocol::kPadSplitCacheStickCount,
        ttwv::kernels::primitives::kInvalidStick,
        0,
        false};

    const uint32_t local_length = padded_end - padded_begin;
    const uint32_t local_even_elements = (local_length + 1 - (padded_begin & 1U)) / 2;
    const uint32_t local_odd_elements = local_length - local_even_elements;
    const uint32_t local_even_sticks =
        ttwv::kernels::primitives::stick_count_for_elements(local_even_elements, ttwv::kStickWidth);
    const uint32_t local_odd_sticks =
        ttwv::kernels::primitives::stick_count_for_elements(local_odd_elements, ttwv::kStickWidth);

    auto even_writer =
        ttwv::kernels::primitives::make_output_stick_writer(cb_even, ttwv::kStickWidth, local_even_sticks);
    auto odd_writer = ttwv::kernels::primitives::make_output_stick_writer(cb_odd, ttwv::kStickWidth, local_odd_sticks);

    const uint32_t left_end = ttwv::kernels::primitives::min_u32(left_pad, padded_length);
    const uint32_t interior_end = ttwv::kernels::primitives::min_u32(left_pad + input_length, padded_length);

    uint32_t out_idx = padded_begin;

    if (out_idx < left_end && out_idx < padded_end) {
        const uint32_t region_end = ttwv::kernels::primitives::min_u32(left_end, padded_end);
        out_idx =
            ttwv::kernels::primitives::push_split_range<ttwv::kernels::primitives::SplitRangeReadMode::kSymmetric>(
                src, read_cache, even_writer, odd_writer, input_length, left_pad, out_idx, region_end);
    }

    if (out_idx < interior_end && out_idx < padded_end) {
        const uint32_t region_end = ttwv::kernels::primitives::min_u32(interior_end, padded_end);
        out_idx = ttwv::kernels::primitives::push_split_range<ttwv::kernels::primitives::SplitRangeReadMode::kInterior>(
            src, read_cache, even_writer, odd_writer, input_length, left_pad, out_idx, region_end);
    }

    if (out_idx < padded_length && out_idx < padded_end) {
        const uint32_t region_end = padded_end;
        ttwv::kernels::primitives::push_split_range<ttwv::kernels::primitives::SplitRangeReadMode::kSymmetric>(
            src, read_cache, even_writer, odd_writer, input_length, left_pad, out_idx, region_end);
    }

    ttwv::kernels::primitives::flush_partial_output_stick(even_writer);
    ttwv::kernels::primitives::flush_partial_output_stick(odd_writer);

    ttwv::kernels::primitives::release_cache(read_cache);
}
