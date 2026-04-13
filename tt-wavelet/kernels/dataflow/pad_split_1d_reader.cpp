#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "pad_split_1d_reader_utils.hpp"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t input_length = get_arg_val<uint32_t>(1);
    const uint32_t padded_length = get_arg_val<uint32_t>(2);
    const uint32_t left_pad = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_even = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_odd = get_compile_time_arg_val(1);
    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(2);
    constexpr uint32_t cb_id_cache = get_compile_time_arg_val(3);
    constexpr uint32_t stick_width = get_compile_time_arg_val(4);
    constexpr auto src_args = TensorAccessorArgs<5>();
    const auto src = TensorAccessor(src_args, src_addr, stick_nbytes);

    ttwv::kernels::primitives::StickReadCache read_cache{
        cb_id_cache, stick_nbytes, stick_width, ttwv::kernels::primitives::kInvalidStick, false};

    const uint32_t even_stick_count = ttwv::kernels::primitives::even_stick_count(padded_length, stick_width);
    const uint32_t odd_stick_count = ttwv::kernels::primitives::odd_stick_count(padded_length, stick_width);
    const uint32_t pair_count = padded_length / 2;

    auto even_writer = ttwv::kernels::primitives::make_output_stick_writer(cb_id_even, stick_width, even_stick_count);
    auto odd_writer = ttwv::kernels::primitives::make_output_stick_writer(cb_id_odd, stick_width, odd_stick_count);

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
