#pragma once

#include <cstdint>

#include "../../tt_wavelet/include/device_protocol/lwt_config.hpp"
#include "api/dataflow/dataflow_api.h"
#include "mem_fill.hpp"

#define ALWI inline __attribute__((always_inline))

namespace ttwv::kernels::primitives {

constexpr uint32_t kBlobInvalid = 0xFFFFFFFFU;
constexpr uint32_t kBlobElements = 4096;
constexpr uint32_t kBlobBytes = kBlobElements * sizeof(float);
static_assert(kBlobElements % 16 == 0, "Blob elements must be a multiple of 16 for noc operations.");

constexpr uint32_t kTileWidth = 32;
constexpr uint32_t kTileHeight = 32;
constexpr uint32_t kFaceWidth = 16;
constexpr uint32_t kFaceHeight = 16;
constexpr uint32_t kFaceElements = kFaceWidth * kFaceHeight;
constexpr uint32_t kTileScalars = kTileWidth * kTileHeight;

// FROM tt-metal SOURCE CODE
ALWI uint32_t get_tilized_idx(uint32_t h, uint32_t w) {
    using namespace tt::constants;
    // Get local coordinates within the tile
    uint32_t local_row = h % TILE_HEIGHT;
    uint32_t local_col = w % TILE_WIDTH;
    // Determine the index offset based on which quadrant we're in
    uint32_t offset = 0;
    // If we're in the right half (columns beyond FACE_WIDTH)
    if (local_col >= FACE_WIDTH) {
        local_col -= FACE_WIDTH;
        offset += FACE_HEIGHT * FACE_WIDTH;  // Right face offset
    }
    // If we're in the bottom half (rows beyond FACE_WIDTH)
    if (local_row >= FACE_WIDTH) {
        local_row -= FACE_WIDTH;
        offset += FACE_HEIGHT * TILE_WIDTH;  // Bottom face offset
    }
    // Final index within the tile
    uint32_t index = offset + local_row * FACE_WIDTH + local_col;
    return index;
}

ALWI uint32_t get_splicized_idx(const uint32_t row, const uint32_t col) {
    return (col / 32) * 1024 + get_tilized_idx(row, col);
}

}  // namespace ttwv::kernels::primitives
