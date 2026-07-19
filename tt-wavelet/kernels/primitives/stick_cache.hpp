#pragma once

#include <cstdint>

#include "../../tt_wavelet/include/common/boundary.hpp"
#include "api/dataflow/dataflow_api.h"
#include "indexing.hpp"
#define ALWI inline __attribute__((always_inline))

// Wormhole NCRISC has a 16 KiB instruction region.  Keep the shared interior
// test inline, but emit one callable copy of the larger boundary-only path
// instead of duplicating it at the even and odd initialization sites.
#if defined(ARCH_WORMHOLE)
#define TTWV_BOUNDARY_SLOW_PATH __attribute__((noinline))
#else
#define TTWV_BOUNDARY_SLOW_PATH ALWI
#endif

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

    const uint32_t available_sticks = source_stick_count - source_stick;
    const uint32_t reserve_sticks = min_u32(cache.stick_capacity, available_sticks);

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

template <typename SrcAccessor>
ALWI float read_source_value(
    const SrcAccessor& src, StickReadCache& cache, const uint32_t source_index, const uint32_t source_length) {
    const uint32_t source_stick = source_index / cache.stick_width;
    const uint32_t source_lane = source_index % cache.stick_width;
    const uint32_t source_stick_count = (source_length + cache.stick_width - 1) / cache.stick_width;

    if (!cache_contains_stick(cache, source_stick)) {
        cache_source_sticks(src, cache, source_stick, source_stick_count);
    }

    const auto* cached_values = reinterpret_cast<const float*>(get_read_ptr(cache.cb_id));
    const uint32_t cached_offset = source_stick - cache.cached_stick_id;
    return cached_values[cached_offset * cache.stick_width + source_lane];
}

template <typename SrcAccessor>
TTWV_BOUNDARY_SLOW_PATH float read_extended_source_value(
    const SrcAccessor& src, StickReadCache& cache, const uint32_t source_index, const uint32_t source_length) {
    return read_source_value(src, cache, source_index, source_length);
}

template <ttwv::BoundaryMode Mode, typename SrcAccessor>
TTWV_BOUNDARY_SLOW_PATH float read_extended_value(
    const SrcAccessor& src,
    StickReadCache& cache,
    const uint32_t input_length,
    const uint32_t left_pad,
    const uint32_t out_idx) {
    static_assert(ttwv::is_cone_boundary_mode(Mode), "Unsupported compile-time boundary mode");

    if constexpr (Mode == ttwv::BoundaryMode::kZero) {
        return 0.0F;
    } else if constexpr (Mode == ttwv::BoundaryMode::kConstant) {
        const uint32_t source_index = out_idx < left_pad ? 0U : input_length - 1U;
        return read_extended_source_value(src, cache, source_index, input_length);
    } else if constexpr (Mode == ttwv::BoundaryMode::kSymmetric) {
        const int32_t logical = static_cast<int32_t>(out_idx) - static_cast<int32_t>(left_pad);
        return read_extended_source_value(src, cache, symmetric_index(logical, input_length), input_length);
    } else if constexpr (Mode == ttwv::BoundaryMode::kReflect) {
        const int32_t logical = static_cast<int32_t>(out_idx) - static_cast<int32_t>(left_pad);
        return read_extended_source_value(src, cache, reflect_index(logical, input_length), input_length);
    } else if constexpr (Mode == ttwv::BoundaryMode::kPeriodic) {
        const int32_t logical = static_cast<int32_t>(out_idx) - static_cast<int32_t>(left_pad);
        return read_extended_source_value(src, cache, positive_mod(logical, input_length), input_length);
    } else if constexpr (Mode == ttwv::BoundaryMode::kAntisymmetric) {
        const int32_t logical = static_cast<int32_t>(out_idx) - static_cast<int32_t>(left_pad);
        const AntisymmetricIndex mapped = antisymmetric_index(logical, input_length);
        const float value = read_extended_source_value(src, cache, mapped.source_index, input_length);
        return mapped.negate ? -value : value;
    } else if constexpr (Mode == ttwv::BoundaryMode::kSmooth) {
        const int32_t logical = static_cast<int32_t>(out_idx) - static_cast<int32_t>(left_pad);
        if (input_length == 1U) {
            // A one-sample DWT has no edge difference. PyWavelets' DWT treats
            // the missing slope as zero (its public pad helper behaves
            // differently for this degenerate input).
            return read_extended_source_value(src, cache, 0U, input_length);
        }
        const bool left = logical < 0;
        const uint32_t edge_index = left ? 0U : input_length - 1U;
        const uint32_t neighbor_index = left ? 1U : input_length - 2U;
        const int32_t distance = left ? -logical : logical - static_cast<int32_t>(input_length - 1U);
        const float edge = read_extended_source_value(src, cache, edge_index, input_length);
        const float neighbor = read_extended_source_value(src, cache, neighbor_index, input_length);
        return edge + static_cast<float>(distance) * (edge - neighbor);
    } else {
        static_assert(Mode == ttwv::BoundaryMode::kAntireflect);
        if (input_length == 1U) {
            return read_extended_source_value(src, cache, 0U, input_length);
        }
        const int32_t logical = static_cast<int32_t>(out_idx) - static_cast<int32_t>(left_pad);
        const uint32_t last_index = input_length - 1U;
        const uint64_t period = 2U * static_cast<uint64_t>(last_index);
        const SignedPeriodIndex mapped = decompose_signed_period(logical, period);
        const bool reflected = mapped.remainder > last_index;
        const uint32_t source_index =
            reflected ? static_cast<uint32_t>(period - mapped.remainder) : static_cast<uint32_t>(mapped.remainder);
        const float source_value = read_extended_source_value(src, cache, source_index, input_length);
        const float first = read_extended_source_value(src, cache, 0U, input_length);
        const float last = read_extended_source_value(src, cache, last_index, input_length);
        const float base = reflected ? 2.0F * last - source_value : source_value;
        return base + static_cast<float>(2 * mapped.quotient) * (last - first);
    }
}

template <ttwv::BoundaryMode Mode, typename SrcAccessor>
ALWI float read_padded_value(
    const SrcAccessor& src,
    StickReadCache& cache,
    const uint32_t input_length,
    const uint32_t left_pad,
    const uint32_t out_idx) {
    static_assert(ttwv::is_cone_boundary_mode(Mode), "Unsupported compile-time boundary mode");

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

template <typename SrcAccessor>
ALWI float read_padded_symmetric_value(
    const SrcAccessor& src,
    StickReadCache& cache,
    const uint32_t input_length,
    const uint32_t left_pad,
    const uint32_t out_idx) {
    return read_padded_value<ttwv::BoundaryMode::kSymmetric>(src, cache, input_length, left_pad, out_idx);
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

#undef TTWV_BOUNDARY_SLOW_PATH
