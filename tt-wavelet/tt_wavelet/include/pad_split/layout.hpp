#pragma once

#include <cstddef>
#include <cstdint>

#include "tt_wavelet/include/common/padding.hpp"
#include "tt_wavelet/include/common/signal.hpp"

namespace ttwv {

struct PadSplit1DLayout {
    SignalBuffer input{};
    Pad1DConfig pad_config{};
    Signal output{};

    [[nodiscard]] constexpr size_t padded_length() const noexcept {
        return input.length + static_cast<size_t>(pad_config.left) + static_cast<size_t>(pad_config.right);
    }
};

[[nodiscard]] constexpr PadSplit1DLayout make_pad_split_1d_layout(
    const SignalBuffer& input, const uint64_t even_addr, const uint64_t odd_addr, const Pad1DConfig config) noexcept {
    const size_t length = input.length + static_cast<size_t>(config.left) + static_cast<size_t>(config.right);

    return PadSplit1DLayout{
        .input = input, .pad_config = config, .output = make_split_signal(input, length, even_addr, odd_addr)};
}

}  // namespace ttwv
