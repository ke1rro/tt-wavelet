#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "mem_fill.hpp"

#define ALWI inline __attribute__((always_inline))

namespace ttwv::kernels::primitives {

constexpr uint32_t kTileWidth = 32;
constexpr uint32_t kTileHeight = 32;
constexpr uint32_t kFaceWidth = 16;
constexpr uint32_t kFaceHeight = 16;
constexpr uint32_t kFaceElements = kFaceWidth * kFaceHeight;
constexpr uint32_t kTileScalars = kTileWidth * kTileHeight;
constexpr uint32_t kLwtRowsPerGroup = 32;
constexpr uint32_t kLwtOutputBlocksPerRow = 3;
constexpr uint32_t kLwtHalfStickElements = 16;
constexpr uint32_t kLwtHalfStickBytes = kLwtHalfStickElements * sizeof(float);
constexpr uint32_t kLwtGroupOutputElements = kLwtRowsPerGroup * kLwtOutputBlocksPerRow * kLwtHalfStickElements;

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

template <typename DstAccessor>
ALWI void write_lwt_half_block(
    const DstAccessor& dst,
    const uint32_t tile_addr,
    const uint32_t row,
    const uint32_t col,
    const uint32_t output_index,
    const uint32_t output_length,
    const uint32_t stick_width) {
    if (output_index >= output_length) {
        return;
    }

    const uint32_t dst_stick = output_index / stick_width;
    const uint32_t dst_lane = output_index % stick_width;
    const uint32_t src_offset = tile_offset(row, col) * static_cast<uint32_t>(sizeof(float));
    const uint64_t noc_addr = dst.get_noc_addr(dst_stick) + dst_lane * sizeof(float);
    noc_async_write(tile_addr + src_offset, noc_addr, kLwtHalfStickBytes);
}

}  // namespace ttwv::kernels::primitives
