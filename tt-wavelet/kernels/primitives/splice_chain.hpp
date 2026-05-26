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
    uint32_t length;
    uint32_t next_index;
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
ALWI void load_signal_l1(const Accessor& accessor, StreamReadState& state) {
    const uint32_t logical_bytes = state.length * sizeof(float);

    noc_async_read(accessor.get_noc_addr(0), state.cache_l1_addr, logical_bytes);
    noc_async_read_barrier();

    state.cache_start_index = 0;
    state.cache_valid_bytes = min_u32(state.cache_bytes, logical_bytes);
    state.cache_valid = state.cache_valid_bytes == logical_bytes;
}

template <typename Accessor>
ALWI void build_splice_chain(
    const uint32_t cb_id,
    const Accessor& accessor,
    StreamReadState& state,
    const uint32_t splice_number,
    const uint32_t tile_nbytes) {
    if (!state.cache_valid) {
        load_signal_l1(accessor, state);
    }

    const auto* cached_signal = reinterpret_cast<const float*>(state.cache_l1_addr);
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
                const uint32_t src_index = logical_index % state.length;
                const uint32_t contiguous = min_u32(kHStickElements, state.length - src_index);

                copy_fp32(dst, cached_signal + src_index, contiguous);
                if (contiguous != kHStickElements) {
                    copy_fp32(dst + contiguous, cached_signal, kHStickElements - contiguous);
                }
                state.next_index = (src_index + kHStickElements) % state.length;
            }
        }

        cb_push_back(cb_id, kSpliceTiles);
    }
}

}  // namespace ttwv::kernels::primitives::splice
