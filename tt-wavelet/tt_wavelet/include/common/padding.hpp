#pragma once

#include <cstddef>
#include <cstdint>

#include "tt_wavelet/include/common/boundary.hpp"
#include "tt_wavelet/include/common/signal.hpp"

namespace ttwv {

/**
 * @brief Parameters describing signal padding 1D.
 *
 * Specifies the extension mode and the number of samples to prepend/append.
 * The actual output length is input.length + left + right.
 */
struct Pad1DConfig {
    BoundaryMode mode{BoundaryMode::kSymmetric};  ///< Signal extension mode at both boundaries.
    uint32_t left{0};                             ///< Number of samples to prepend (left pad).
    uint32_t right{0};                            ///< Number of samples to append (right pad).
};

/**
 * @brief Describes layout of a 1D padding operation.
 */
struct Pad1DLayout {
    SignalBuffer input{};   ///< Descriptor of the unpadded input signal in DRAM.
    SignalBuffer output{};  ///< Descriptor of the padded output signal in DRAM.
    Pad1DConfig config{};   ///< Padding configuration (mode, left, right).

    /**
     * @brief Number of output sticks required to cover the full padded signal.
     * @return output.stick_count()
     */
    [[nodiscard]] constexpr size_t output_stick_count() const noexcept { return output.stick_count(); }
};

/**
 * @brief Constructs a @ref Pad1DLayout from an input buffer descriptor and padding config.
 *
 * The output buffer length is derived automatically as input.length + config.left + config.right.
 * The output buffer inherits the same stick geometry (width, element size) as the input.
 *
 * @param input              Descriptor of the input signal (must have a valid @c dram_address).
 * @param output_dram_address DRAM base address where the padded output will be written.
 * @param config             Padding mode and left/right sample counts.
 * @return Fully populated @ref Pad1DLayout.
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

}  // namespace ttwv
