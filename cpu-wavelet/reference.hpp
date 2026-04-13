#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "tt_wavelet/include/lifting/device.hpp"

namespace ttwv::debug {

struct HostLiftingReference {
    std::vector<float> even;
    std::vector<float> odd;
    int even_shift{0};
    int odd_shift{0};
};

[[nodiscard]] bool matches_reference(
    const std::vector<float>& device_values, const std::vector<float>& reference, size_t logical_length);

[[nodiscard]] HostLiftingReference materialize_reference_forward_lifting(
    std::span<const float> input, const LiftingPreprocessDeviceProgram& bundle);

[[nodiscard]] std::vector<float> canonicalize_forward_output(
    const std::vector<float>& values,
    int signal_shift,
    int direct_shift,
    size_t output_length,
    const char* label);

}  // namespace ttwv::debug
