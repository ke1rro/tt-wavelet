#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

#include "tt-wavelet/tt_wavelet/include/common/boundary.hpp"

namespace ttwv::cpu {

[[nodiscard]] constexpr size_t positive_mod(const int64_t value, const size_t modulus) noexcept {
    if (modulus == 0) {
        return 0;
    }

    const auto signed_modulus{static_cast<int64_t>(modulus)};
    const auto remainder{value % signed_modulus};
    return static_cast<size_t>(remainder < 0 ? remainder + signed_modulus : remainder);
}

[[nodiscard]] constexpr size_t symmetric_index(const int64_t index, const size_t length) noexcept {
    if (length <= 1) {
        return 0;
    }

    const auto period{length * 2};
    const auto reflected{positive_mod(index, period)};
    return reflected < length ? reflected : period - 1 - reflected;
}

[[nodiscard]] constexpr std::optional<size_t> boundary_index(
    const int64_t index, const size_t length, const BoundaryMode mode) noexcept {
    if (length == 0) {
        return std::nullopt;
    }

    switch (mode) {
        case BoundaryMode::kZero: return std::nullopt;
        case BoundaryMode::kConstant:
            if (index < 0) {
                return size_t{0};
            }
            return static_cast<size_t>(index) >= length ? length - 1 : static_cast<size_t>(index);
        case BoundaryMode::kSymmetric: return symmetric_index(index, length);
        case BoundaryMode::kPeriodic: return positive_mod(index, length);
    }

    return std::nullopt;
}

}  // namespace ttwv::cpu
