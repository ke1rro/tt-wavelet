#pragma once

#include <cstdint>

#include "mem_fill.hpp"

#define ALWI inline __attribute__((always_inline))

namespace ttwv::kernels::primitives {

constexpr uint32_t kTileWidth = 32;
constexpr uint32_t kTileHeight = 32;
constexpr uint32_t kFaceWidth = 16;
constexpr uint32_t kFaceHeight = 16;
constexpr uint32_t kFaceElements = kFaceWidth * kFaceHeight;
constexpr uint32_t kTileScalars = kTileWidth * kTileHeight;

[[nodiscard]] ALWI constexpr uint32_t tile_row0_offset(const uint32_t col) {
    return col < kFaceWidth ? col : kFaceElements + (col - kFaceWidth);
}

ALWI void clear_tile(float* tile_ptr) { fill_zeros(tile_ptr, kTileScalars); }

ALWI void store_row0_value(float* tile_ptr, const uint32_t col, const float value) {
    tile_ptr[tile_row0_offset(col)] = value;
}

[[nodiscard]] ALWI float load_row0_value(const float* tile_ptr, const uint32_t col) {
    return tile_ptr[tile_row0_offset(col)];
}

}  // namespace ttwv::kernels::primitives
