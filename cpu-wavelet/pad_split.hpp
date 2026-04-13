#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "cpu-wavelet/padding.hpp"
#include "tt-wavelet/tt_wavelet/include/pad_split/layout.hpp"

namespace ttwv::cpu {

struct SplitResult {
    std::vector<float> even;
    std::vector<float> odd;
};

[[nodiscard]] inline SplitResult materialize_reference_pad_split(
    const std::span<const float> input, const PadSplit1DLayout& layout) {
    const Pad1DLayout pad_layout = make_pad_1d_layout(layout.input, 0, layout.pad_config);
    const std::vector<float> padded = materialize_reference_padding(input, pad_layout);

    SplitResult result{
        .even = std::vector<float>(layout.output.even.physical_length(), 0.0F),
        .odd = std::vector<float>(layout.output.odd.physical_length(), 0.0F)};

    for (size_t i = 0; i < layout.output.even.length; ++i) {
        result.even[i] = padded[i * 2];
    }
    for (size_t i = 0; i < layout.output.odd.length; ++i) {
        result.odd[i] = padded[i * 2 + 1];
    }

    return result;
}

}  // namespace ttwv::cpu
