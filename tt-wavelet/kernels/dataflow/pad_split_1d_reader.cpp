#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "pad_split_1d_reader_utils.hpp"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t input_length = get_arg_val<uint32_t>(1);
    const uint32_t padded_length = get_arg_val<uint32_t>(2);
    const uint32_t left_pad = get_arg_val<uint32_t>(3);

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

    const uint32_t even_stick_count = ttwv::kernels::primitives::even_stick_count(padded_length, ttwv::kStickWidth);
    const uint32_t odd_stick_count = ttwv::kernels::primitives::odd_stick_count(padded_length, ttwv::kStickWidth);
    const uint32_t pair_count = padded_length / 2;

    auto even_writer =
        ttwv::kernels::primitives::make_output_stick_writer(cb_even, ttwv::kStickWidth, even_stick_count);
    auto odd_writer = ttwv::kernels::primitives::make_output_stick_writer(cb_odd, ttwv::kStickWidth, odd_stick_count);

    for (uint32_t pair = 0; pair < pair_count; ++pair) {
        ttwv::kernels::primitives::push_output_value(
            even_writer,
            ttwv::kernels::primitives::read_padded_symmetric_value(src, read_cache, input_length, left_pad, pair * 2));
        ttwv::kernels::primitives::push_output_value(
            odd_writer,
            ttwv::kernels::primitives::read_padded_symmetric_value(
                src, read_cache, input_length, left_pad, pair * 2 + 1));
    }

    if (padded_length & 1U) {
        ttwv::kernels::primitives::push_output_value(
            even_writer,
            ttwv::kernels::primitives::read_padded_symmetric_value(
                src, read_cache, input_length, left_pad, padded_length - 1));
    }

    ttwv::kernels::primitives::flush_partial_output_stick(even_writer);
    ttwv::kernels::primitives::flush_partial_output_stick(odd_writer);

    ttwv::kernels::primitives::release_cache(read_cache);
}
