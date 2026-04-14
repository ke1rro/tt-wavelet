#pragma once

#include <cstdint>

#include "tt_wavelet/include/common/boundary.hpp"

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

}  // namespace ttwv
