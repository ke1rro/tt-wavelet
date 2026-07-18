#pragma once

#include <cstdint>

namespace ttwv {

/**
 * @brief Signal extension mode used when accessing out-of-bounds indices during padding.
 *
 * @note Resident pad/split materializes the extension; ConeStreamed maps it
 *       while loading each dependency cone.
 *
 * Mirrors the non-periodization modes supported by PyWavelets.
 */
enum class BoundaryMode : uint8_t {
    kZero = 0,           ///< Pad with zeros
    kConstant = 1,       ///< Clamp to edge value
    kSymmetric = 2,      ///< Half-sample symmetric reflection
    kPeriodic = 3,       ///< Wrap around the original signal
    kAntisymmetric = 4,  ///< Half-sample symmetric reflection with alternating sign
    kSmooth = 5,         ///< Extrapolate the first edge difference as a straight line
    kAntireflect = 6,    ///< Whole-sample reflection, antisymmetric about each edge value
    kReflect = 7,        ///< Whole-sample symmetric reflection
};

[[nodiscard]] constexpr bool is_cone_boundary_mode(const BoundaryMode mode) noexcept {
    switch (mode) {
        case BoundaryMode::kZero:
        case BoundaryMode::kConstant:
        case BoundaryMode::kSymmetric:
        case BoundaryMode::kPeriodic:
        case BoundaryMode::kAntisymmetric:
        case BoundaryMode::kSmooth:
        case BoundaryMode::kAntireflect:
        case BoundaryMode::kReflect: return true;
    }
    return false;
}

[[nodiscard]] constexpr bool boundary_mode_requires_multiple_samples(const BoundaryMode mode) noexcept {
    return mode == BoundaryMode::kReflect || mode == BoundaryMode::kAntireflect;
}

}  // namespace ttwv
