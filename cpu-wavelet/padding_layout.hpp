#pragma once

#include <cstddef>
#include <cstdint>

#include "tt-wavelet/tt_wavelet/include/common/padding.hpp"
#include "tt-wavelet/tt_wavelet/include/common/signal.hpp"

namespace ttwv::cpu {

/**
 * @brief CPU reference layout of a 1D padding operation.
 */
struct Pad1DLayout {
    SignalBuffer input{};   ///< Descriptor of the unpadded input signal in DRAM-like layout.
    SignalBuffer output{};  ///< Descriptor of the padded output signal in DRAM-like layout.
    Pad1DConfig config{};   ///< Padding configuration (mode, left, right).

    /**
     * @brief Number of output sticks required to cover the full padded signal.
     * @return output.stick_count()
     */
    [[nodiscard]] constexpr size_t output_stick_count() const noexcept { return output.stick_count(); }
};

/**
 * @brief Constructs a @ref Pad1DLayout for CPU reference materialization.
 *
 * The output buffer length is derived as input.length + config.left + config.right,
 * while preserving stick geometry (stick width and element size).
 */
[[nodiscard]] constexpr Pad1DLayout make_pad_1d_layout(
    const SignalBuffer& input, const uint64_t output_dram_address, const Pad1DConfig config) noexcept {
    return Pad1DLayout{
        .input = input,
        .output =
            SignalBuffer{
                .dram_address = output_dram_address,
                .length = input.length + static_cast<size_t>(config.left) + static_cast<size_t>(config.right),
                .stick_width = input.stick_width,
                .element_size_bytes = input.element_size_bytes},
        .config = config};
}

}  // namespace ttwv::cpu
