#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "indexing.hpp"
#define ALWI inline __attribute__((always_inline))

namespace ttwv::kernels::primitives {

constexpr uint32_t kInvalidStick = 0xFFFFFFFFU;

struct StickReadCache {
    uint32_t cb_id;
    uint32_t stick_nbytes;
    uint32_t stick_width;
    uint32_t stick_capacity;
    uint32_t cached_stick_id;
    uint32_t cached_stick_count;
    bool valid;
};

ALWI uint32_t min_u32(const uint32_t lhs, const uint32_t rhs) { return lhs < rhs ? lhs : rhs; }

ALWI uint32_t max_u32(const uint32_t lhs, const uint32_t rhs) { return lhs > rhs ? lhs : rhs; }

ALWI bool cache_contains_stick(const StickReadCache& cache, const uint32_t source_stick) {
    return cache.valid && source_stick >= cache.cached_stick_id &&
           source_stick < cache.cached_stick_id + cache.cached_stick_count;
}

template <typename SrcAccessor>
ALWI void cache_source_sticks(
    const SrcAccessor& src, StickReadCache& cache, const uint32_t source_stick, const uint32_t source_stick_count) {
    if (cache.valid) {
        cb_pop_front(cache.cb_id, cache.cached_stick_count);
    }

    const uint32_t available_sticks = source_stick < source_stick_count ? source_stick_count - source_stick : 1;
    const uint32_t reserve_sticks = min_u32(max_u32(cache.stick_capacity, 1), available_sticks);

    cb_reserve_back(cache.cb_id, reserve_sticks);
    const uint32_t cache_l1_addr = get_write_ptr(cache.cb_id);
#pragma unroll 8
    for (uint32_t i = 0; i < reserve_sticks; ++i) {
        const uint64_t src_noc_addr = src.get_noc_addr(source_stick + i);
        noc_async_read(src_noc_addr, cache_l1_addr + i * cache.stick_nbytes, cache.stick_nbytes);
    }
    noc_async_read_barrier();
    cb_push_back(cache.cb_id, reserve_sticks);
    cb_wait_front(cache.cb_id, reserve_sticks);

    cache.cached_stick_id = source_stick;
    cache.cached_stick_count = reserve_sticks;
    cache.valid = true;
}

template <typename SrcAccessor>
ALWI void cache_source_stick(const SrcAccessor& src, StickReadCache& cache, const uint32_t source_stick) {
    cache_source_sticks(src, cache, source_stick, source_stick + 1);
}

template <typename SrcAccessor>
ALWI float read_padded_symmetric_value(
    const SrcAccessor& src,
    StickReadCache& cache,
    const uint32_t input_length,
    const uint32_t left_pad,
    const uint32_t out_idx) {
    if (input_length == 0) {
        return 0.0F;
    }

    const int32_t logical = static_cast<int32_t>(out_idx) - static_cast<int32_t>(left_pad);
    const uint32_t source_index = symmetric_index(logical, input_length);
    const uint32_t source_stick = source_index / cache.stick_width;
    const uint32_t source_lane = source_index % cache.stick_width;
    const uint32_t source_stick_count = (input_length + cache.stick_width - 1) / cache.stick_width;

    if (!cache_contains_stick(cache, source_stick)) {
        cache_source_sticks(src, cache, source_stick, source_stick_count);
    }

    const auto* cached_values = reinterpret_cast<const float*>(get_read_ptr(cache.cb_id));
    const uint32_t cached_offset = source_stick - cache.cached_stick_id;
    return cached_values[cached_offset * cache.stick_width + source_lane];
}

template <typename SrcAccessor>
ALWI float read_source_value(const SrcAccessor& src, StickReadCache& cache, const uint32_t source_index) {
    const uint32_t source_stick = source_index / cache.stick_width;
    const uint32_t source_lane = source_index % cache.stick_width;

    if (!cache_contains_stick(cache, source_stick)) {
        cache_source_stick(src, cache, source_stick);
    }

    const auto* cached_values = reinterpret_cast<const float*>(get_read_ptr(cache.cb_id));
    const uint32_t cached_offset = source_stick - cache.cached_stick_id;
    return cached_values[cached_offset * cache.stick_width + source_lane];
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

}  // namespace ttwv::kernels::primitives
