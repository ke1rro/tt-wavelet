#pragma once

#include <cstddef>

#include "tt-wavelet/tt_wavelet/include/common/signal.hpp"

namespace ttwv::cpu {

/**
 * @brief Number of scalar slots materialized in host memory for a DRAM-like signal buffer.
 *
 * This matches the physical stick footprint used by CPU reference paths.
 */
[[nodiscard]] constexpr size_t physical_length(const SignalBuffer& buffer) noexcept {
    return buffer.stick_count() * static_cast<size_t>(buffer.stick_width);
}

}  // namespace ttwv::cpu
