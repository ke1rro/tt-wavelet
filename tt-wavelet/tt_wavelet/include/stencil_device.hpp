#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

#include "tt-metalium/host_api.hpp"

namespace ttwv {

/**
 * @brief Describes a 1D stencil convolution to run on device.
 *
 * The stencil computes: g[i] = sum_{j=0}^{k-1} h[j] * f[i - j]
 * The signal is padded by (17 - k) zeros on the left for tile alignment.
 */
struct StencilConfig {
    std::vector<float> coefficients;  ///< Filter coefficients h[0..k-1]
    uint32_t signal_length{0};        ///< Number of signal samples (after pad-split)
    uint32_t stick_width{32};         ///< Samples per stick
    uint32_t element_size_bytes{sizeof(float)};

    [[nodiscard]] uint32_t k() const noexcept { return static_cast<uint32_t>(coefficients.size()); }
    [[nodiscard]] uint32_t pad_amount() const noexcept { return 17 - k(); }
    [[nodiscard]] uint32_t num_tiles() const noexcept { return 1; }  // Single tile for short signals
    [[nodiscard]] uint32_t stick_bytes() const noexcept { return stick_width * element_size_bytes; }
    [[nodiscard]] uint32_t aligned_stick_bytes(uint32_t alignment = 32) const noexcept {
        return ((stick_bytes() + alignment - 1) / alignment) * alignment;
    }
    [[nodiscard]] uint32_t output_stick_count() const noexcept {
        return (signal_length + stick_width - 1) / stick_width;
    }
};

/**
 * @brief Bundle of program + kernel handles for the stencil operation.
 */
struct StencilDeviceProgram {
    tt::tt_metal::Program program;
    tt::tt_metal::KernelHandle reader;
    tt::tt_metal::KernelHandle compute;
    tt::tt_metal::KernelHandle writer;
};

/**
 * @brief Creates a TT-Metal program that runs the stencil convolution.
 *
 * @param kernel_root  Root path where kernels/ directory lives.
 * @param core         Core coordinate to run on.
 * @param input_buffer Buffer containing the input signal (e.g. even stream from pad-split).
 * @param output_buffer Buffer for the stencil output.
 * @param config       Stencil configuration (coefficients, signal length).
 * @return Program bundle ready to enqueue.
 */
[[nodiscard]] StencilDeviceProgram create_stencil_program(
    const std::filesystem::path& kernel_root,
    const tt::tt_metal::CoreCoord& core,
    const tt::tt_metal::Buffer& input_buffer,
    const tt::tt_metal::Buffer& output_buffer,
    const StencilConfig& config);

/**
 * @brief Host-side reference implementation for stencil convolution.
 *
 * Computes g[i] = sum_{j=0}^{k-1} h[j] * f[i-j] with symmetric boundary,
 * after applying 17-k padding and extracting the valid region.
 *
 * @param signal       Input signal samples.
 * @param coefficients Filter coefficients.
 * @return Convolution result (same length as input signal).
 */
[[nodiscard]] std::vector<float> reference_stencil_1d(
    const std::vector<float>& signal, const std::vector<float>& coefficients);

}  // namespace ttwv
