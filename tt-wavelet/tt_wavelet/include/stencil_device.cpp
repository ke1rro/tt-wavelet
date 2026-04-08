#pragma once

#include <cstdint>
#include <filesystem>
#include <array>

#include "tt-metalium/host_api.hpp"
namespace ttwv{

template<size_t K>
struct StencilConfig{
    std::array<float, K> step_coefficients;  ///< Filter coefficients h[0..k-1]
    uint32_t signal_length{0};        ///< Number of signal samples (after pad-split)
    uint32_t stick_width{32};         ///< Samples per stick
    uint32_t element_size_bytes{sizeof(float)};

    [[nodiscard]] uint32_t coeff_count() const noexcept { return K; }
    [[nodiscard]] uint32_t pad_amount() const noexcept { return 17 - coeff_count(); }
    [[nodiscard]] uint32_t num_tiles() const noexcept { return 1; }
    [[nodiscard]] uint32_t stick_bytes() const noexcept { return stick_width * element_size_bytes; }
    [[nodiscard]] uint32_t aligned_stick_bytes(uint32_t alignment = 32) const noexcept {
        return ((stick_bytes() + alignment - 1) / alignment) * alignment;
    }
    [[nodiscard]] uint32_t output_stick_count() const noexcept {
        return (signal_length + stick_width - 1) / stick_width;
};
} //namespace ttwv