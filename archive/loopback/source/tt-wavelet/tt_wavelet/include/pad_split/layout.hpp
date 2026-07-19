#pragma once

#include <cstddef>
#include <cstdint>

#include "tt_wavelet/include/common/padding.hpp"
#include "tt_wavelet/include/common/signal.hpp"

namespace ttwv {

struct PadSplit1DLayout {
    SignalBuffer input{};
    Pad1DConfig pad_config{};
    Signal output{};  ///< Even/odd output buffers.

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

[[nodiscard]] constexpr PadSplit1DLayout make_pad_split_1d_layout(
    const SignalBuffer& input, const uint64_t even_addr, const uint64_t odd_addr, const Pad1DConfig config) noexcept {
    const size_t length = input.length + static_cast<size_t>(config.left) + static_cast<size_t>(config.right);

    return PadSplit1DLayout{
        .input = input, .pad_config = config, .output = make_split_signal(input, length, even_addr, odd_addr)};
}

}  // namespace ttwv
