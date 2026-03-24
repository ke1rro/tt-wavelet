#pragma once

#include <cstdint>

#include "llk_assert.h"

namespace ckernel::stencil {

inline constexpr char kErrFilterLenExceedsPolicy[] = "filter_len exceeds per-pass stencil policy capacity";

template <uint32_t UsableDstTiles, uint32_t NeedsBaseTile, uint32_t NeedsOutputTile, uint32_t NeedsTempTiles = 0>
struct Policy {
    static constexpr uint32_t usable_dst_tiles = UsableDstTiles;
    static constexpr uint32_t needs_base_tile = NeedsBaseTile;
    static constexpr uint32_t needs_output_tile = NeedsOutputTile;
    static constexpr uint32_t needs_temp_tiles = NeedsTempTiles;

    static_assert(usable_dst_tiles > 0, "usable_dst_tiles must be > 0");
    static_assert(needs_output_tile > 0, "needs_output_tile must be > 0");
    static_assert(
        usable_dst_tiles >= (needs_base_tile + needs_output_tile + needs_temp_tiles),
        "Policy reserves more tiles than available");

    static constexpr uint32_t max_taps_per_pass =
        usable_dst_tiles - needs_base_tile - needs_output_tile - needs_temp_tiles;

    static_assert(max_taps_per_pass > 0, "Policy leaves no room for stencil taps");
};

template <uint32_t UsableDstTiles = 8, uint32_t NeedsTempTiles = 0>
using StencilGenericPolicy = Policy<UsableDstTiles, 0, 1, NeedsTempTiles>;

template <uint32_t UsableDstTiles = 8, uint32_t NeedsTempTiles = 0>
using StencilAccPolicy = Policy<UsableDstTiles, 0, 1, NeedsTempTiles>;

template <uint32_t UsableDstTiles = 8, uint32_t NeedsTempTiles = 0>
using StencilMacPolicy = Policy<UsableDstTiles, 1, 1, NeedsTempTiles>;

template <uint32_t UsableDstTiles = 8, uint32_t NeedsTempTiles = 1>
using StencilAffinePolicy = Policy<UsableDstTiles, 1, 1, NeedsTempTiles>;

/** @brief Runtime tap-count check against the selected stencil policy. */
template <typename PolicyT>
void assert_taps_within_policy(const uint32_t filter_len) {
    LLK_ASSERT((filter_len <= PolicyT::max_taps_per_pass), kErrFilterLenExceedsPolicy);
}

}  // namespace ckernel::stencil
