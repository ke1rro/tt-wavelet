#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "tt_wavelet/include/boundary.hpp"
#include "tt_wavelet/include/signal.hpp"

namespace ttwv {

/**
 * @brief Parameters describing how to pad a 1D signal.
 *
 * Specifies the extension mode and the number of samples to prepend/append.
 * The actual output length is `input.length + left + right`.
 */
struct Pad1DConfig {
    BoundaryMode mode{BoundaryMode::Symmetric};  ///< Signal extension mode at both boundaries.
    uint32_t left{0};                            ///< Number of samples to prepend (left pad).
    uint32_t right{0};                           ///< Number of samples to append (right pad).
};

/**
 * @brief Describes the full memory layout of a 1D padding operation.
 *
 * Bundles the input and output @ref SignalBuffer descriptors together with the
 * @ref Pad1DConfig
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

/**
 * @brief Maps an output sample index back to the corresponding source index in the input signal.
 *
 * Accounts for the left padding offset: output index 0 corresponds to logical input index
 * -config.left, which is then resolved via @ref boundary_index according to the configured mode.
 *
 * @param layout       The padding layout (input/output descriptors and config).
 * @param output_index Sample index in the padded output (0-based).
 * @return Source index in [0, input.length), or std::nullopt if the output sample is zero
 *         (e.g. @ref BoundaryMode::Zero).
 */
[[nodiscard]] constexpr std::optional<size_t> source_index_for_output(
    const Pad1DLayout& layout, const size_t output_index) noexcept {
    return boundary_index(
        static_cast<int64_t>(output_index) - static_cast<int64_t>(layout.config.left),
        layout.input.length,
        layout.config.mode);
}

/**
 * @brief Materializes the padded signal into a host-side vector (reference implementation).
 *
 * Iterates over every output index, resolves its source via @ref source_index_for_output,
 * and copies the corresponding input sample (or writes 0.0 for @ref BoundaryMode::Zero).
 * The output vector is allocated to @ref SignalBuffer::physical_length to ensure the last
 * stick is zero-padded.
 *
 * @note This is a host-only reference path used for testing. Device padding is handled by
 *       the kernels in `kernels/dataflow/`.
 *
 * @param input  View of the input signal samples.
 * @param layout Fully constructed padding layout.
 * @return Zero-padded output vector of size `layout.output.physical_length()`.
 */
[[nodiscard]] inline std::vector<float> materialize_reference_padding(
    const std::span<const float> input, const Pad1DLayout& layout) {
    std::vector<float> output(layout.output.physical_length(), 0.0F);

    for (size_t output_index = 0; output_index < layout.output.length; ++output_index) {
        const auto source_index{source_index_for_output(layout, output_index)};
        output[output_index] = source_index.has_value() ? input[*source_index] : 0.0F;
    }

    return output;
}

/**
 * @brief Convenience overload that constructs the layout internally from a flat input span.
 *
 * Creates a temporary @ref Pad1DLayout with stick_width = 32 and element_size_bytes = 4
 * (fp32), then delegates to @ref materialize_reference_padding(const std::span<const float>, const Pad1DLayout&).
 *
 * @param input  View of the input signal samples.
 * @param config Padding configuration to apply.
 * @return Padded output vector (host reference).
 */
[[nodiscard]] inline std::vector<float> materialize_reference_padding(
    const std::span<const float> input, const Pad1DConfig config) {
    const auto layout = make_pad_1d_layout(
        SignalBuffer{.dram_address = 0, .length = input.size(), .stick_width = 32, .element_size_bytes = sizeof(float)},
        0,
        config);
    return materialize_reference_padding(input, layout);
}

}  // namespace ttwv
