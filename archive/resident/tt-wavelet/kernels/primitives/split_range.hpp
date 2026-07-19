#pragma once

#include <cstdint>

#include "stick_cache.hpp"
#include "stick_writer.hpp"

#define ALWI inline __attribute__((always_inline))

namespace ttwv::kernels::primitives {

enum class SplitRangeReadMode : uint32_t {
    kSymmetric,
    kInterior,
};

template <SplitRangeReadMode mode, typename SrcAccessor>
ALWI float read_split_range_value(
    const SrcAccessor& src,
    StickReadCache& read_cache,
    const uint32_t input_length,
    const uint32_t left_pad,
    const uint32_t out_idx) {
    if constexpr (mode == SplitRangeReadMode::kInterior) {
        return read_source_value(src, read_cache, out_idx - left_pad);
    }

    return read_padded_symmetric_value(src, read_cache, input_length, left_pad, out_idx);
}

template <SplitRangeReadMode mode, typename SrcAccessor>
ALWI uint32_t push_split_range(
    const SrcAccessor& src,
    StickReadCache& read_cache,
    OutputStickWriter& even_writer,
    OutputStickWriter& odd_writer,
    const uint32_t input_length,
    const uint32_t left_pad,
    uint32_t out_idx,
    const uint32_t region_end) {
    if ((out_idx & 1U) && out_idx < region_end) {
        push_output_value(odd_writer, read_split_range_value<mode>(src, read_cache, input_length, left_pad, out_idx));
        out_idx++;
    }

    for (; out_idx + 1 < region_end; out_idx += 2) {
        push_output_value(even_writer, read_split_range_value<mode>(src, read_cache, input_length, left_pad, out_idx));
        push_output_value(
            odd_writer, read_split_range_value<mode>(src, read_cache, input_length, left_pad, out_idx + 1));
    }

    if (out_idx < region_end) {
        push_output_value(even_writer, read_split_range_value<mode>(src, read_cache, input_length, left_pad, out_idx));
        out_idx++;
    }

    return out_idx;
}

}  // namespace ttwv::kernels::primitives
