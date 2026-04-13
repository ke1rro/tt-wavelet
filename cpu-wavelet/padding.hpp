#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

#include "cpu-wavelet/boundary.hpp"
#include "tt-wavelet/tt_wavelet/include/common/padding.hpp"

namespace ttwv::cpu {

[[nodiscard]] constexpr std::optional<size_t> source_index_for_output(
    const Pad1DLayout& layout, const size_t output_index) noexcept {
    return boundary_index(
        static_cast<int64_t>(output_index) - static_cast<int64_t>(layout.config.left),
        layout.input.length,
        layout.config.mode);
}

[[nodiscard]] inline std::vector<float> materialize_reference_padding(
    const std::span<const float> input, const Pad1DLayout& layout) {
    std::vector<float> output(layout.output.physical_length(), 0.0F);

    for (size_t output_index = 0; output_index < layout.output.length; ++output_index) {
        const auto source_index{source_index_for_output(layout, output_index)};
        output[output_index] = source_index.has_value() ? input[*source_index] : 0.0F;
    }

    return output;
}

[[nodiscard]] inline std::vector<float> materialize_reference_padding(
    const std::span<const float> input, const Pad1DConfig config) {
    const auto layout = make_pad_1d_layout(
        SignalBuffer{.dram_address = 0, .length = input.size(), .stick_width = 32, .element_size_bytes = sizeof(float)},
        0,
        config);
    return materialize_reference_padding(input, layout);
}

}  // namespace ttwv::cpu
