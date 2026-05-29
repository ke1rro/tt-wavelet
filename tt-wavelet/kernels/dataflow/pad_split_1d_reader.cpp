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

    auto even_writer =
        ttwv::kernels::primitives::make_output_stick_writer(cb_even, ttwv::kStickWidth, even_stick_count);
    auto odd_writer = ttwv::kernels::primitives::make_output_stick_writer(cb_odd, ttwv::kStickWidth, odd_stick_count);

    const uint32_t left_end = ttwv::kernels::primitives::min_u32(left_pad, padded_length);
    const uint32_t interior_end = ttwv::kernels::primitives::min_u32(left_pad + input_length, padded_length);

    // Process left edge, interior, and right edge to skip symmetric indexing in the interior.
    {
        uint32_t out_idx = 0;
        if ((out_idx & 1U) && out_idx < left_end) {
            ttwv::kernels::primitives::push_output_value(
                odd_writer,
                ttwv::kernels::primitives::read_padded_symmetric_value(
                    src, read_cache, input_length, left_pad, out_idx));
            out_idx++;
        }
        for (; out_idx + 1 < left_end; out_idx += 2) {
            ttwv::kernels::primitives::push_output_value(
                even_writer,
                ttwv::kernels::primitives::read_padded_symmetric_value(
                    src, read_cache, input_length, left_pad, out_idx));
            ttwv::kernels::primitives::push_output_value(
                odd_writer,
                ttwv::kernels::primitives::read_padded_symmetric_value(
                    src, read_cache, input_length, left_pad, out_idx + 1));
        }
        if (out_idx < left_end) {
            ttwv::kernels::primitives::push_output_value(
                even_writer,
                ttwv::kernels::primitives::read_padded_symmetric_value(
                    src, read_cache, input_length, left_pad, out_idx));
        }
    }

    {
        uint32_t out_idx = left_end;
        if ((out_idx & 1U) && out_idx < interior_end) {
            const uint32_t source_index = out_idx - left_pad;
            ttwv::kernels::primitives::push_output_value(
                odd_writer, ttwv::kernels::primitives::read_source_value(src, read_cache, source_index));
            out_idx++;
        }
        for (; out_idx + 1 < interior_end; out_idx += 2) {
            const uint32_t source_index = out_idx - left_pad;
            ttwv::kernels::primitives::push_output_value(
                even_writer, ttwv::kernels::primitives::read_source_value(src, read_cache, source_index));
            ttwv::kernels::primitives::push_output_value(
                odd_writer, ttwv::kernels::primitives::read_source_value(src, read_cache, source_index + 1));
        }
        if (out_idx < interior_end) {
            const uint32_t source_index = out_idx - left_pad;
            ttwv::kernels::primitives::push_output_value(
                even_writer, ttwv::kernels::primitives::read_source_value(src, read_cache, source_index));
        }
    }

    {
        uint32_t out_idx = interior_end;
        if ((out_idx & 1U) && out_idx < padded_length) {
            ttwv::kernels::primitives::push_output_value(
                odd_writer,
                ttwv::kernels::primitives::read_padded_symmetric_value(
                    src, read_cache, input_length, left_pad, out_idx));
            out_idx++;
        }
        for (; out_idx + 1 < padded_length; out_idx += 2) {
            ttwv::kernels::primitives::push_output_value(
                even_writer,
                ttwv::kernels::primitives::read_padded_symmetric_value(
                    src, read_cache, input_length, left_pad, out_idx));
            ttwv::kernels::primitives::push_output_value(
                odd_writer,
                ttwv::kernels::primitives::read_padded_symmetric_value(
                    src, read_cache, input_length, left_pad, out_idx + 1));
        }
        if (out_idx < padded_length) {
            ttwv::kernels::primitives::push_output_value(
                even_writer,
                ttwv::kernels::primitives::read_padded_symmetric_value(
                    src, read_cache, input_length, left_pad, out_idx));
        }
    }

    ttwv::kernels::primitives::flush_partial_output_stick(even_writer);
    ttwv::kernels::primitives::flush_partial_output_stick(odd_writer);

    ttwv::kernels::primitives::release_cache(read_cache);
}
