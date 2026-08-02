// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/operations/wavelet/common/signal_extension.hpp"
#include "api/dataflow/dataflow_api.h"
#define ALWI inline __attribute__((always_inline))

// Wormhole NCRISC has a 16 KiB instruction region.  Keep the shared interior
// test inline, but emit one callable copy of the larger boundary-only path
// instead of duplicating it at the even and odd initialization sites.
#if defined(ARCH_WORMHOLE)
#define LWT_BOUNDARY_CALLABLE __attribute__((noinline))
#else
#define LWT_BOUNDARY_CALLABLE ALWI
#endif

namespace ttnn::operations::wavelet::kernels::primitives {

constexpr uint32_t kInvalidStick = 0xFFFFFFFFU;

struct StickReadCache {
    uint32_t cb_id;
    uint32_t stick_nbytes;
    uint32_t stick_width;
    uint32_t stick_capacity;
    uint32_t cached_stick_id;
    uint32_t cached_stick_count;
    uint32_t source_page;
    bool valid;
};

ALWI uint32_t min_u32(const uint32_t lhs, const uint32_t rhs) { return lhs < rhs ? lhs : rhs; }

ALWI bool cache_contains_stick(const StickReadCache& cache, const uint32_t source_stick) {
    return cache.valid && source_stick >= cache.cached_stick_id &&
           source_stick < cache.cached_stick_id + cache.cached_stick_count;
}

template <typename SrcAccessor>
ALWI void cache_source_sticks(
    const SrcAccessor& src, StickReadCache& cache, const uint32_t source_stick, const uint32_t source_length) {
    if (cache.valid) {
        cb_pop_front(cache.cb_id, cache.cached_stick_count);
    }

    const uint32_t source_stick_count = (source_length + cache.stick_width - 1U) / cache.stick_width;
    const uint32_t available_sticks = source_stick_count - source_stick;
    const uint32_t reserve_sticks = min_u32(cache.stick_capacity, available_sticks);

    cb_reserve_back(cache.cb_id, reserve_sticks);
    const uint32_t cache_l1_addr = get_write_ptr(cache.cb_id);
#pragma GCC unroll 8
    for (uint32_t i = 0; i < reserve_sticks; ++i) {
        const uint32_t stick = source_stick + i;
        const uint32_t stick_begin = stick * cache.stick_width;
        const uint32_t valid_elements = min_u32(cache.stick_width, source_length - stick_begin);
        // Rank-1 TTNN row-major tensors use one exact-size page.  The final
        // logical stick therefore has no guaranteed 32-byte tail padding;
        // read only the bytes owned by the tensor and leave the zero-filled
        // remainder of the cache stick untouched.
        const uint32_t read_bytes = valid_elements * sizeof(float);
        auto* cache_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cache_l1_addr + i * cache.stick_nbytes);
#pragma GCC unroll 8
        for (uint32_t word = 0; word < cache.stick_nbytes / sizeof(uint32_t); ++word) {
            cache_words[word] = 0;
        }
        noc_async_read(
            src.get_noc_addr(cache.source_page, stick * cache.stick_nbytes),
            cache_l1_addr + i * cache.stick_nbytes,
            read_bytes);
    }
    noc_async_read_barrier();
    cb_push_back(cache.cb_id, reserve_sticks);
    cb_wait_front(cache.cb_id, reserve_sticks);

    cache.cached_stick_id = source_stick;
    cache.cached_stick_count = reserve_sticks;
    cache.valid = true;
}

template <typename SrcAccessor>
ALWI float read_source_value(
    const SrcAccessor& src, StickReadCache& cache, const uint32_t source_index, const uint32_t source_length) {
    const uint32_t source_stick = source_index / cache.stick_width;
    const uint32_t source_lane = source_index % cache.stick_width;
    if (!cache_contains_stick(cache, source_stick)) {
        cache_source_sticks(src, cache, source_stick, source_length);
    }

    const auto* cached_values = reinterpret_cast<const float*>(get_read_ptr(cache.cb_id));
    const uint32_t cached_offset = source_stick - cache.cached_stick_id;
    return cached_values[cached_offset * cache.stick_width + source_lane];
}

template <typename SrcAccessor>
LWT_BOUNDARY_CALLABLE float read_extended_source_value(
    const SrcAccessor& src, StickReadCache& cache, const uint32_t source_index, const uint32_t source_length) {
    return read_source_value(src, cache, source_index, source_length);
}

template <typename SrcAccessor>
struct CachedSourceReader {
    const SrcAccessor& src;
    StickReadCache& cache;
    uint32_t source_length;

    ALWI float operator()(const uint32_t source_index) const {
        return read_extended_source_value(src, cache, source_index, source_length);
    }
};

template <ttnn::operations::wavelet::BoundaryMode Mode, typename SrcAccessor>
LWT_BOUNDARY_CALLABLE float read_extended_value(
    const SrcAccessor& src,
    StickReadCache& cache,
    const uint32_t input_length,
    const uint32_t left_pad,
    const uint32_t out_idx) {
    static_assert(
        ttnn::operations::wavelet::is_supported_lwt_boundary_mode(Mode), "Unsupported compile-time boundary mode");
    const CachedSourceReader<SrcAccessor> reader{
        .src = src,
        .cache = cache,
        .source_length = input_length,
    };
    if constexpr (Mode == ttnn::operations::wavelet::BoundaryMode::kAntireflect) {
        // The host contract bounds the padded signal to INT32_MAX.  Use the
        // compact 32-bit decomposition here: 64-bit division expands beyond
        // Wormhole's 16 KiB NCRISC text region, while producing the same
        // antireflect index and affine correction for the supported domain.
        const int32_t logical = static_cast<int32_t>(out_idx) - static_cast<int32_t>(left_pad);
        const auto extended = ttnn::operations::wavelet::make_antireflect_index_i32(logical, input_length);
        return ttnn::operations::wavelet::evaluate_antireflect_index_i32(extended, input_length, reader);
    } else {
        const int64_t logical = static_cast<int64_t>(out_idx) - static_cast<int64_t>(left_pad);
        const ttnn::operations::wavelet::ExtendedIndex extended =
            ttnn::operations::wavelet::make_extended_index<Mode>(logical, input_length);
        return ttnn::operations::wavelet::evaluate_extended_index<Mode>(extended, input_length, reader);
    }
}

template <ttnn::operations::wavelet::BoundaryMode Mode, typename SrcAccessor>
ALWI float read_padded_value(
    const SrcAccessor& src,
    StickReadCache& cache,
    const uint32_t input_length,
    const uint32_t left_pad,
    const uint32_t out_idx) {
    static_assert(
        ttnn::operations::wavelet::is_supported_lwt_boundary_mode(Mode), "Unsupported compile-time boundary mode");

    // All supported modes share this direct interior path.  On Wormhole the
    // bounded prefix/suffix calls one out-of-line specialization so its larger
    // extension arithmetic is not duplicated in the reader binary.
    if (out_idx >= left_pad) {
        const uint32_t source_index = out_idx - left_pad;
        if (source_index < input_length) {
            return read_source_value(src, cache, source_index, input_length);
        }
    }
    return read_extended_value<Mode>(src, cache, input_length, left_pad, out_idx);
}

ALWI void release_cache(StickReadCache& cache) {
    if (!cache.valid) {
        return;
    }

    cb_pop_front(cache.cb_id, cache.cached_stick_count);
    cache.cached_stick_id = kInvalidStick;
    cache.cached_stick_count = 0;
    cache.valid = false;
}

}  // namespace ttnn::operations::wavelet::kernels::primitives

#undef LWT_BOUNDARY_CALLABLE
