#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "tt_wavelet/include/signal.hpp"

namespace ttwv {

/**
 * @brief Layout configuration for splitting a 1D signal into even and odd indices.
 */
struct Split1DLayout {
    SignalBuffer input{};     ///< Original interleaved signal.
    Signal output{};          ///< Even/odd output buffers.

    [[nodiscard]] constexpr size_t even_stick_count() const noexcept { return output.even.stick_count(); }
    [[nodiscard]] constexpr size_t odd_stick_count() const noexcept { return output.odd.stick_count(); }
};

/**
 * @brief Constructs a @ref Split1DLayout from an input descriptor.
 *
 * @param input       Descriptor of the original signal.
 * @param even_addr   DRAM base address for the even split output.
 * @param odd_addr    DRAM base address for the odd split output.
 * @return Fully populated @ref Split1DLayout.
 */
[[nodiscard]] constexpr Split1DLayout make_split_1d_layout(
    const SignalBuffer& input,
    const uint64_t even_addr,
    const uint64_t odd_addr) noexcept {

    const size_t even_len = ceil_div(input.length, size_t{2});
    const size_t odd_len = input.length / 2;

    return Split1DLayout{
        .input = input,
        .output =
            Signal{
                .even =
                    SignalBuffer{
                        .dram_address = even_addr,
                        .length = even_len,
                        .stick_width = input.stick_width,
                        .element_size_bytes = input.element_size_bytes},
                .odd =
                    SignalBuffer{
                        .dram_address = odd_addr,
                        .length = odd_len,
                        .stick_width = input.stick_width,
                        .element_size_bytes = input.element_size_bytes}}};
}

/**
 * @brief Result vectors for a split operation on the host.
 */
struct SplitResult {
    std::vector<float> even;
    std::vector<float> odd;
};

}  // namespace ttwv
