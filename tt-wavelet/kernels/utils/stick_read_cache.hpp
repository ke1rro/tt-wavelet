#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "boundary.hpp"
#define ALWI inline __attribute__((always_inline))

namespace ttwv::kernels::utils {

constexpr uint32_t kInvalidStick = 0xFFFFFFFFU;

struct StickReadCache {
    uint32_t cb_id;
    uint32_t stick_nbytes;
    uint32_t stick_width;
    uint32_t cached_stick_id;
    bool valid;
};

template <typename SrcAccessor>
ALWI void cache_source_stick(
    const SrcAccessor& src, StickReadCache& cache, const uint32_t source_stick) {
    if (cache.valid) {
        cb_pop_front(cache.cb_id, 1);
    }

    cb_reserve_back(cache.cb_id, 1);
    const uint32_t cache_l1_addr = get_write_ptr(cache.cb_id);
    const uint64_t src_noc_addr = src.get_noc_addr(source_stick);
    noc_async_read(src_noc_addr, cache_l1_addr, cache.stick_nbytes);
    noc_async_read_barrier();
    cb_push_back(cache.cb_id, 1);
    cb_wait_front(cache.cb_id, 1);

    cache.cached_stick_id = source_stick;
    cache.valid = true;
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

    if (!cache.valid || source_stick != cache.cached_stick_id) {
        cache_source_stick(src, cache, source_stick);
    }

    const auto* cached_values = reinterpret_cast<const float*>(get_read_ptr(cache.cb_id));
    return cached_values[source_lane];
}

template <typename SrcAccessor>
ALWI float read_source_value(const SrcAccessor& src, StickReadCache& cache, const uint32_t source_index) {
    const uint32_t source_stick = source_index / cache.stick_width;
    const uint32_t source_lane = source_index % cache.stick_width;

    if (!cache.valid || source_stick != cache.cached_stick_id) {
        cache_source_stick(src, cache, source_stick);
    }

    const auto* cached_values = reinterpret_cast<const float*>(get_read_ptr(cache.cb_id));
    return cached_values[source_lane];
}

template <typename SrcAccessor>
ALWI float read_shifted_symmetric_value(
    const SrcAccessor& src,
    StickReadCache& cache,
    const uint32_t input_length,
    const int32_t logical_index,
    const int32_t source_offset) {
    if (input_length == 0) {
        return 0.0F;
    }

    const int32_t source_logical_index = logical_index + source_offset;
    const uint32_t source_index = symmetric_index(source_logical_index, input_length);
    return read_source_value(src, cache, source_index);
}

ALWI void release_cache(StickReadCache& cache) {
    if (!cache.valid) {
        return;
    }

    cb_pop_front(cache.cb_id, 1);
    cache.cached_stick_id = kInvalidStick;
    cache.valid = false;
}

}  // namespace ttwv::kernels::utils
