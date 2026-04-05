#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "tt_wavelet/include/padding.hpp"
#include "tt_wavelet/include/split.hpp"
#include "tt_wavelet/include/signal.hpp"

namespace ttwv {


struct PadSplit1DLayout {
    SignalBuffer input{};
    Pad1DConfig pad_config{};
    Signal output{}; ///< Even/odd output buffers.

    /**
     * @return Physical length of padded signal
     */
[[nodiscard]] constexpr size_t padded_length() const noexcept {
        return input.length + static_cast<size_t>(pad_config.left) + static_cast<size_t>(pad_config.right);
    }
/**
 * @return Number of sticks in the signal divided into.
 */
    [[nodiscard]] constexpr size_t even_stick_count() const noexcept { return output.even.stick_count(); }
    [[nodiscard]] constexpr size_t odd_stick_count() const noexcept { return output.odd.stick_count(); }
};


[[nodiscard]] constexpr PadSplit1DLayout make_pad_split_1d_layout(const SignalBuffer& input, const uint64_t even_addr, const uint64_t odd_addr, const Pad1DConfig config) noexcept {
    const size_t length {input.length + static_cast<size_t>(config.left) + static_cast<size_t>(config.right) };
    const size_t even_len {ceil_div(length, size_t{2})};
    const size_t odd_len {length / 2};

    return PadSplit1DLayout{
    .input = input,
    .pad_config = config,
    .output = Signal{.even = SignalBuffer{.dram_address = even_addr, .length = even_len, .stick_width = input.stick_width, .element_size_bytes = input.element_size_bytes},
        .odd = SignalBuffer{.dram_address = odd_addr, .length = odd_len, .stick_width = input.stick_width, .element_size_bytes = input.element_size_bytes}}};
}
/**
 * @brief Temp function will be removed
 */
[[nodiscard]] inline SplitResult materialize_reference_pad_split(
    const std::span<const float> input, const PadSplit1DLayout& layout) {
    const Pad1DLayout pad_layout = make_pad_1d_layout(layout.input, 0, layout.pad_config);
    const std::vector<float> padded = materialize_reference_padding(input, pad_layout);

    SplitResult result{
        .even = std::vector<float>(layout.output.even.physical_length(), 0.0F),
        .odd = std::vector<float>(layout.output.odd.physical_length(), 0.0F)};

    for (size_t i = 0; i < layout.output.even.length; ++i) {
        result.even[i] = padded[i * 2]; // NOLINT
    }
    for (size_t i = 0; i < layout.output.odd.length; ++i) {
        result.odd[i] = padded[i * 2 + 1]; // NOLINT
    }

    return result;
}





}