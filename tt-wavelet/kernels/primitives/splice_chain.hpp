#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tile_row_major.hpp"

namespace ttwv::kernels::primitives::splice {

constexpr uint32_t kTileHeight = 32;
constexpr uint32_t kHStickElements = 16;
constexpr uint32_t kSpliceTiles = 2;
constexpr uint32_t kSpliceRows = kTileHeight;
constexpr uint32_t kSpliceHSticksPerRow = 4;
constexpr uint32_t kSpliceWidth = kSpliceHSticksPerRow * kHStickElements;
constexpr uint32_t kSpliceOverlap = kHStickElements;
constexpr uint32_t kSpliceRowStride = kSpliceWidth - kSpliceOverlap;
constexpr uint32_t kDefaultCacheBytes = 2048;

struct StreamReadState {
    uint32_t source_length;
    uint32_t source_start;
    uint32_t local_length;
    uint32_t prefix_length;
    uint32_t cache_l1_addr;
    uint32_t cache_bytes;
    uint32_t cache_start_index;
    uint32_t cache_valid_bytes;
    bool cache_valid;
};

ALWI bool is_aligned_16(const void* ptr) { return (reinterpret_cast<uintptr_t>(ptr) & 0xFU) == 0; }

ALWI bool is_multiple_of_16(const uint32_t bytes) { return (bytes & 0xFU) == 0; }

ALWI uint32_t min_u32(const uint32_t lhs, const uint32_t rhs) { return lhs < rhs ? lhs : rhs; }

ALWI float* tile_hstick_addr(float* tile0, float* tile1, const uint32_t row, const uint32_t splice_hstick) {
    float* tile = splice_hstick < 2 ? tile0 : tile1;
    const uint32_t local_hstick = splice_hstick & 1U;
    return tile + ttwv::kernels::primitives::tile_offset(row, local_hstick * kHStickElements);
}

ALWI void copy_fp32(float* dst, const float* src, const uint32_t count) {
    const uint32_t bytes = count * sizeof(float);
    if (is_aligned_16(dst) && is_aligned_16(src) && is_multiple_of_16(bytes)) {
        __builtin_memcpy(dst, src, bytes);
        return;
    }

    for (uint32_t lane = 0; lane < count; ++lane) {
        dst[lane] = src[lane];
    }
}

template <typename Accessor>
ALWI float read_signal_value(const Accessor& accessor, StreamReadState& state, const uint32_t index) {
    if (index >= state.local_length) {
        return 0.0F;
    }

    if (index < state.prefix_length) {
        return 0.0F;
    }

    const uint32_t source_index = state.source_start + index - state.prefix_length;
    if (source_index >= state.source_length) {
        return 0.0F;
    }
    const uint32_t values_per_page = state.cache_bytes / sizeof(float);
    const uint32_t page_start = (source_index / values_per_page) * values_per_page;
    if (!state.cache_valid || page_start != state.cache_start_index) {
        noc_async_read(accessor.get_noc_addr(page_start / values_per_page), state.cache_l1_addr, state.cache_bytes);
        noc_async_read_barrier();
        state.cache_start_index = page_start;
        state.cache_valid_bytes = state.cache_bytes;
        state.cache_valid = true;
    }

    const auto* cached_signal = reinterpret_cast<const volatile tt_l1_ptr float*>(state.cache_l1_addr);
    return cached_signal[source_index - page_start];
}

template <typename Accessor>
ALWI void build_splice_chain(
    const uint32_t cb_id,
    const Accessor& accessor,
    StreamReadState& state,
    const uint32_t splice_number,
    const uint32_t tile_nbytes) {
    for (uint32_t splice = 0; splice < splice_number; ++splice) {
        cb_reserve_back(cb_id, kSpliceTiles);
        const uint32_t tiles_addr = get_write_ptr(cb_id);
        auto* tile0 = reinterpret_cast<float*>(tiles_addr);
        auto* tile1 = reinterpret_cast<float*>(tiles_addr + tile_nbytes);

        for (uint32_t row = 0; row < kSpliceRows; ++row) {
            const uint32_t global_row = splice * kSpliceRows + row;
            for (uint32_t hstick = 0; hstick < kSpliceHSticksPerRow; ++hstick) {
                float* dst = tile_hstick_addr(tile0, tile1, row, hstick);
                const uint32_t logical_index = global_row * kSpliceRowStride + hstick * kHStickElements;
                for (uint32_t lane = 0; lane < kHStickElements; ++lane) {
                    dst[lane] = read_signal_value(accessor, state, logical_index + lane);
                }
            }
        }

        cb_push_back(cb_id, kSpliceTiles);
    }
}

}  // namespace ttwv::kernels::primitives::splice
