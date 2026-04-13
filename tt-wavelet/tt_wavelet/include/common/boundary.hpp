#pragma once

#include <cstdint>

namespace ttwv {

/**
 * @brief Signal extension mode used when accessing out-of-bounds indices during padding.
 *
 * @note Padding in this library is a physical operation.
 *
 * Mirrors the modes supported by NumPy / PyWavelets: kZero, kConstant, kSymmetric, kPeriodic
 */
enum class BoundaryMode : uint8_t {
    kZero = 0,       ///< Pad with zeros
    kConstant = 1,   ///< Clamp to edge value
    kSymmetric = 2,  ///< Mirror reflection
    kPeriodic = 3    ///< Wrap-around
};

}  // namespace ttwv
