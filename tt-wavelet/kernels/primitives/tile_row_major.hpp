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

[[nodiscard]] ALWI constexpr uint32_t tile_offset(const uint32_t row, const uint32_t col) {
    const uint32_t face_row = row / kFaceHeight;
    const uint32_t face_col = col / kFaceWidth;
    const uint32_t face = face_row * 2U + face_col;
    return face * kFaceElements + (row % kFaceHeight) * kFaceWidth + (col % kFaceWidth);
}

ALWI void clear_tile(float* tile_ptr) { fill_zeros(tile_ptr, kTileScalars); }

ALWI void store_row0_value(float* tile_ptr, const uint32_t col, const float value) {
    tile_ptr[tile_row0_offset(col)] = value;
}

ALWI void store_tile_value(float* tile_ptr, const uint32_t row, const uint32_t col, const float value) {
    tile_ptr[tile_offset(row, col)] = value;
}

}  // namespace ttwv::kernels::primitives
