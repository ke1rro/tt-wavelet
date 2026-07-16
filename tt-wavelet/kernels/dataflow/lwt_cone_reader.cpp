#include <cstdint>

#include "../../tt_wavelet/include/lifting/step.hpp"
#include "api/dataflow/dataflow_api.h"
#include "lwt_readers_utils.hpp"
#include "lwt_tile_row_major_utils.hpp"

namespace {

constexpr uint32_t kStepPredict = static_cast<uint32_t>(ttwv::StepType::kPredict);
constexpr uint32_t kStepUpdate = static_cast<uint32_t>(ttwv::StepType::kUpdate);
constexpr uint32_t kStepScaleEven = static_cast<uint32_t>(ttwv::StepType::kScaleEven);
constexpr uint32_t kStepScaleOdd = static_cast<uint32_t>(ttwv::StepType::kScaleOdd);

constexpr uint32_t kBlockElements = ttwv::device_protocol::kLwtHalfStickElements;
constexpr uint32_t kRowsPerGroup = ttwv::device_protocol::kLwtRowsPerGroup;
constexpr uint32_t kOutputBlocksPerRow = ttwv::device_protocol::kLwtOutputBlocksPerRow;
constexpr uint32_t kGroupOutputElements = ttwv::device_protocol::kLwtGroupOutputElements;
constexpr uint32_t kSourcePackedElements = kGroupOutputElements + kBlockElements;
constexpr uint32_t kNarrowTileElements = ttwv::device_protocol::kLwtNarrowTileElements;
constexpr uint32_t kNarrowTileBytes = ttwv::device_protocol::kLwtNarrowTileBytes;
constexpr uint32_t kGroupOutputBytes = kGroupOutputElements * sizeof(float);

struct WorkspaceIndexCursor {
    uint32_t group{0};
    uint32_t row{0};
    uint32_t block{0};
    uint32_t lane{0};
    uint32_t physical{0};

    ALWI explicit WorkspaceIndexCursor(const uint32_t logical_index) {
        group = logical_index / kGroupOutputElements;
        const uint32_t group_index = logical_index - group * kGroupOutputElements;
        row = group_index / (kOutputBlocksPerRow * kBlockElements);
        const uint32_t row_index = group_index - row * kOutputBlocksPerRow * kBlockElements;
        block = row_index / kBlockElements;
        lane = row_index - block * kBlockElements;
        physical = group * kGroupOutputElements + block * kNarrowTileElements + row * kBlockElements + lane;
    }

    ALWI void advance() {
        ++lane;
        ++physical;
        if (lane == kBlockElements) {
            lane = 0;
            ++block;
            if (block == kOutputBlocksPerRow) {
                block = 0;
                ++row;
                if (row == kRowsPerGroup) {
                    row = 0;
                    ++group;
                }
            }
            physical = group * kGroupOutputElements + block * kNarrowTileElements + row * kBlockElements;
        }
    }

    // Advance by one logical 16-element block while preserving the lane.  In
    // tile-native storage this is usually a jump to another narrow tile, not a
    // contiguous physical increment.
    ALWI void advance_block() {
        ++block;
        if (block == kOutputBlocksPerRow) {
            block = 0;
            ++row;
            if (row == kRowsPerGroup) {
                row = 0;
                ++group;
            }
        }
        physical = group * kGroupOutputElements + block * kNarrowTileElements + row * kBlockElements + lane;
    }
};

ALWI void read_workspace_block(const volatile tt_l1_ptr float* src, WorkspaceIndexCursor& cursor, float* dst) {
    // A logical 16-element block crosses at most one physical narrow-tile
    // block boundary.  Copy the two contiguous segments and update the cursor
    // once, avoiding per-element lane/block state in the hot dense path.
    const uint32_t initial_lane = cursor.lane;
    const uint32_t first_count = kBlockElements - initial_lane;
    for (uint32_t index = 0; index < first_count; ++index) {
        dst[index] = src[cursor.physical + index];
    }

    cursor.lane = 0;
    cursor.advance_block();
    for (uint32_t index = 0; index < initial_lane; ++index) {
        dst[first_count + index] = src[cursor.physical + index];
    }
    cursor.lane = initial_lane;
    cursor.physical += initial_lane;
}

template <bool BoundsChecked>
ALWI void read_workspace_block(
    const volatile tt_l1_ptr float* src, const int64_t logical_start, const uint32_t logical_end, float* dst) {
    if constexpr (BoundsChecked) {
        const uint32_t zero_prefix =
            logical_start < 0
                ? static_cast<uint32_t>((-logical_start) < kBlockElements ? -logical_start : kBlockElements)
                : 0;
        const uint32_t valid_start = logical_start < 0 ? 0U : static_cast<uint32_t>(logical_start);
        WorkspaceIndexCursor cursor(valid_start);
#pragma unroll
        for (uint32_t lane = 0; lane < kBlockElements; ++lane) {
            const bool valid = lane >= zero_prefix && valid_start + lane - zero_prefix < logical_end;
            dst[lane] = valid ? src[cursor.physical] : 0.0F;
            if (valid) {
                cursor.advance();
            }
        }
    } else {
        WorkspaceIndexCursor cursor(static_cast<uint32_t>(logical_start));
#pragma unroll
        for (uint32_t lane = 0; lane < kBlockElements; ++lane) {
            dst[lane] = src[cursor.physical];
            cursor.advance();
        }
    }
}

ALWI void read_aligned_source_group(
    const uint32_t source_addr,
    const uint32_t logical_start,
    const uint32_t src_tiles01_addr,
    const uint32_t src_tiles23_addr) {
    const uint32_t physical_group_addr = source_addr + logical_start * sizeof(float);
    noc_async_read(get_noc_addr(physical_group_addr), src_tiles01_addr, kNarrowTileBytes);
    noc_async_read(
        get_noc_addr(physical_group_addr + kNarrowTileBytes), src_tiles01_addr + kNarrowTileBytes, kNarrowTileBytes);
    noc_async_read(get_noc_addr(physical_group_addr + 2 * kNarrowTileBytes), src_tiles23_addr, kNarrowTileBytes);

    // The fourth source tile is the first tile shifted up by one logical
    // 48-element row.  Its final row comes from the following native group.
    noc_async_read(
        get_noc_addr(physical_group_addr + kBlockElements * sizeof(float)),
        src_tiles23_addr + kNarrowTileBytes,
        kNarrowTileBytes - kBlockElements * sizeof(float));
    noc_async_read(
        get_noc_addr(physical_group_addr + kGroupOutputBytes),
        src_tiles23_addr + 2 * kNarrowTileBytes - kBlockElements * sizeof(float),
        kBlockElements * sizeof(float));
}

ALWI void read_aligned_output_group(
    const uint32_t source_addr, const uint32_t logical_start, const uint32_t narrow_tiles_addr) {
    const uint32_t physical_group_addr = source_addr + logical_start * sizeof(float);
#pragma unroll
    for (uint32_t block = 0; block < kOutputBlocksPerRow; ++block) {
        noc_async_read(
            get_noc_addr(physical_group_addr + block * kNarrowTileBytes),
            narrow_tiles_addr + block * kNarrowTileBytes,
            kNarrowTileBytes);
    }
}

template <typename ConfigAccessor>
ALWI const uint32_t* load_config_page(
    const ConfigAccessor& config, const uint32_t config_addr, const uint32_t cb_config, const uint32_t page_index) {
    const auto page_accessor = TensorAccessor(config, config_addr, ttwv::device_protocol::kRouteConfigPageBytes);
    cb_reserve_back(cb_config, 1);
    noc_async_read(
        page_accessor.get_noc_addr(page_index), get_write_ptr(cb_config), ttwv::device_protocol::kRouteConfigPageBytes);
    noc_async_read_barrier();
    cb_push_back(cb_config, 1);
    cb_wait_front(cb_config, 1);
    return reinterpret_cast<const uint32_t*>(get_read_ptr(cb_config));
}

template <bool TileNative, typename InputAccessor>
ALWI void initialize_cone(
    const InputAccessor& input,
    const uint32_t even_addr,
    const uint32_t odd_addr,
    const uint32_t cb_input_cache,
    const uint32_t input_length,
    const uint32_t left_pad,
    const uint32_t even_begin,
    const uint32_t even_length,
    const uint32_t odd_begin,
    const uint32_t odd_length) {
    ttwv::kernels::primitives::StickReadCache input_cache{
        cb_input_cache,
        ttwv::device_protocol::kStickBytes,
        ttwv::kStickWidth,
        ttwv::device_protocol::kLwtCacheStickCount,
        ttwv::kernels::primitives::kInvalidStick,
        0,
        false};
    auto* even_dst = reinterpret_cast<volatile tt_l1_ptr float*>(even_addr);
    auto* odd_dst = reinterpret_cast<volatile tt_l1_ptr float*>(odd_addr);
    uint32_t even_written = 0;
    uint32_t odd_written = 0;
    WorkspaceIndexCursor even_cursor(0);
    WorkspaceIndexCursor odd_cursor(0);

    const uint32_t even_end = even_begin + even_length;
    const uint32_t odd_end = odd_begin + odd_length;
    const uint32_t split_begin = even_begin < odd_begin ? even_begin : odd_begin;
    const uint32_t split_end = even_end > odd_end ? even_end : odd_end;
    for (uint32_t split_index = split_begin; split_index < split_end; ++split_index) {
        if (split_index >= even_begin && split_index < even_end) {
            const uint32_t padded_index = 2U * split_index;
            const float value = ttwv::kernels::primitives::read_padded_symmetric_value(
                input, input_cache, input_length, left_pad, padded_index);
            if constexpr (TileNative) {
                even_dst[even_cursor.physical] = value;
                even_cursor.advance();
            } else {
                even_dst[even_written++] = value;
            }
        }
        if (split_index >= odd_begin && split_index < odd_end) {
            const uint32_t padded_index = 2U * split_index + 1U;
            const float value = ttwv::kernels::primitives::read_padded_symmetric_value(
                input, input_cache, input_length, left_pad, padded_index);
            if constexpr (TileNative) {
                odd_dst[odd_cursor.physical] = value;
                odd_cursor.advance();
            } else {
                odd_dst[odd_written++] = value;
            }
        }
    }

    ttwv::kernels::primitives::release_cache(input_cache);
}

template <bool BoundsChecked>
ALWI void fill_source_row_major(
    const volatile tt_l1_ptr float* src,
    float* src_tiles01,
    float* src_tiles23,
    const uint32_t source_end,
    const uint32_t source_offset,
    const uint32_t source_left_pad,
    const uint32_t group_base) {
    for (uint32_t block = 0; block < 4; ++block) {
        auto* narrow_tile =
            block < 2 ? src_tiles01 + block * kNarrowTileElements : src_tiles23 + (block - 2) * kNarrowTileElements;
        for (uint32_t row = 0; row < kRowsPerGroup; ++row) {
            auto* tile_row = narrow_tile + row * kBlockElements;
            const int64_t logical_start = static_cast<int64_t>(source_offset) + group_base +
                                          (row * kOutputBlocksPerRow + block) * kBlockElements - source_left_pad;
#pragma unroll
            for (uint32_t lane = 0; lane < kBlockElements; ++lane) {
                const int64_t logical_index = logical_start + lane;
                if constexpr (BoundsChecked) {
                    tile_row[lane] = logical_index >= 0 && static_cast<uint64_t>(logical_index) < source_end
                                         ? src[static_cast<uint32_t>(logical_index)]
                                         : 0.0F;
                } else {
                    tile_row[lane] = src[static_cast<uint32_t>(logical_index)];
                }
            }
        }
    }
}

template <bool BoundsChecked>
ALWI void fill_output_row_major(
    const volatile tt_l1_ptr float* src,
    float* narrow_tiles,
    const uint32_t source_end,
    const uint32_t source_offset,
    const uint32_t output_length,
    const uint32_t group_base) {
    for (uint32_t row = 0; row < kRowsPerGroup; ++row) {
        for (uint32_t block = 0; block < kOutputBlocksPerRow; ++block) {
            auto* tile_block = narrow_tiles + block * kNarrowTileElements + row * kBlockElements;
            const uint32_t output_index = group_base + (row * kOutputBlocksPerRow + block) * kBlockElements;
#pragma unroll
            for (uint32_t lane = 0; lane < kBlockElements; ++lane) {
                const uint32_t local_index = output_index + lane;
                const uint32_t logical_index = source_offset + local_index;
                if constexpr (BoundsChecked) {
                    tile_block[lane] =
                        local_index < output_length && logical_index < source_end ? src[logical_index] : 0.0F;
                } else {
                    tile_block[lane] = src[logical_index];
                }
            }
        }
    }
}

template <bool BoundsChecked>
ALWI void fill_source_narrow_tiles(
    const volatile tt_l1_ptr float* src,
    float* src_tiles01,
    float* src_tiles23,
    const uint32_t source_end,
    const uint32_t source_offset,
    const uint32_t source_left_pad,
    const uint32_t group_base) {
    if constexpr (!BoundsChecked) {
        // The four destination tiles traverse the same logical rows at four
        // different block offsets.  Keep one persistent cursor per tile so a
        // dense group pays four index decompositions instead of 4*32.
        for (uint32_t block = 0; block < 4; ++block) {
            auto* narrow_tile =
                block < 2 ? src_tiles01 + block * kNarrowTileElements : src_tiles23 + (block - 2) * kNarrowTileElements;
            const uint32_t logical_start = source_offset + group_base + block * kBlockElements - source_left_pad;
            WorkspaceIndexCursor cursor(logical_start);
            for (uint32_t row = 0; row < kRowsPerGroup; ++row) {
                read_workspace_block(src, cursor, narrow_tile + row * kBlockElements);
                // Reading the block already advanced once.  Skip the remaining
                // two blocks in this 48-element logical row.
                cursor.advance_block();
                cursor.advance_block();
            }
        }
        return;
    }

    for (uint32_t block = 0; block < 4; ++block) {
        auto* narrow_tile =
            block < 2 ? src_tiles01 + block * kNarrowTileElements : src_tiles23 + (block - 2) * kNarrowTileElements;
        for (uint32_t row = 0; row < kRowsPerGroup; ++row) {
            auto* tile_row = narrow_tile + row * kBlockElements;
            const int64_t logical_start = static_cast<int64_t>(source_offset) + group_base +
                                          (row * kOutputBlocksPerRow + block) * kBlockElements - source_left_pad;
            read_workspace_block<BoundsChecked>(src, logical_start, source_end, tile_row);
        }
    }
}

template <bool BoundsChecked>
ALWI void fill_output_narrow_tiles(
    const volatile tt_l1_ptr float* src,
    float* narrow_tiles,
    const uint32_t source_end,
    const uint32_t source_offset,
    const uint32_t output_length,
    const uint32_t group_base) {
    if constexpr (!BoundsChecked) {
        // Destination order is block-major inside each logical row.  A single
        // cursor walks the complete dense logical group while remapping only
        // the physical tile boundary transitions.
        WorkspaceIndexCursor cursor(source_offset + group_base);
        for (uint32_t row = 0; row < kRowsPerGroup; ++row) {
            for (uint32_t block = 0; block < kOutputBlocksPerRow; ++block) {
                auto* tile_block = narrow_tiles + block * kNarrowTileElements + row * kBlockElements;
                read_workspace_block(src, cursor, tile_block);
            }
        }
        return;
    }

    for (uint32_t row = 0; row < kRowsPerGroup; ++row) {
        for (uint32_t block = 0; block < kOutputBlocksPerRow; ++block) {
            auto* tile_block = narrow_tiles + block * kNarrowTileElements + row * kBlockElements;
            const uint32_t output_index = group_base + (row * kOutputBlocksPerRow + block) * kBlockElements;
            const uint32_t logical_end =
                source_offset + output_length < source_end ? source_offset + output_length : source_end;
            read_workspace_block<BoundsChecked>(
                src, static_cast<int64_t>(source_offset) + output_index, logical_end, tile_block);
        }
    }
}

template <bool TileNative>
ALWI void emit_predict_update_tiles(
    const uint32_t source_addr,
    const uint32_t base_addr,
    const uint32_t cb_src_tile0,
    const uint32_t cb_src_tile1,
    const uint32_t cb_base_tile,
    const uint32_t source_end,
    const uint32_t base_end,
    const uint32_t output_length,
    const uint32_t group_count,
    const uint32_t source_offset,
    const uint32_t base_offset,
    const uint32_t source_left_pad) {
    const auto* src = reinterpret_cast<volatile tt_l1_ptr float*>(source_addr);
    const auto* base = reinterpret_cast<volatile tt_l1_ptr float*>(base_addr);

    for (uint32_t group = 0; group < group_count; ++group) {
        const uint32_t group_base = group * kGroupOutputElements;
        cb_reserve_back(cb_src_tile0, 2);
        auto* src_tiles01 = reinterpret_cast<float*>(get_write_ptr(cb_src_tile0));
        cb_reserve_back(cb_src_tile1, 2);
        auto* src_tiles23 = reinterpret_cast<float*>(get_write_ptr(cb_src_tile1));

        bool source_is_dense = group_base >= source_left_pad;
        uint32_t source_begin = 0;
        if (source_is_dense) {
            const uint64_t candidate_begin = static_cast<uint64_t>(source_offset) + group_base - source_left_pad;
            source_is_dense = candidate_begin + kSourcePackedElements <= source_end;
            source_begin = static_cast<uint32_t>(candidate_begin);
        }
        bool needs_read_barrier = false;
        if constexpr (TileNative) {
            if (source_is_dense && source_begin % kGroupOutputElements == 0) {
                read_aligned_source_group(
                    source_addr, source_begin, get_write_ptr(cb_src_tile0), get_write_ptr(cb_src_tile1));
                needs_read_barrier = true;
            } else if (source_is_dense) {
                fill_source_narrow_tiles<false>(
                    src, src_tiles01, src_tiles23, source_end, source_offset, source_left_pad, group_base);
            } else {
                fill_source_narrow_tiles<true>(
                    src, src_tiles01, src_tiles23, source_end, source_offset, source_left_pad, group_base);
            }
        } else if (source_is_dense) {
            fill_source_row_major<false>(
                src, src_tiles01, src_tiles23, source_end, source_offset, source_left_pad, group_base);
        } else {
            fill_source_row_major<true>(
                src, src_tiles01, src_tiles23, source_end, source_offset, source_left_pad, group_base);
        }

        cb_reserve_back(cb_base_tile, 3);
        auto* base_tiles = reinterpret_cast<float*>(get_write_ptr(cb_base_tile));
        const bool base_is_dense = static_cast<uint64_t>(group_base) + kGroupOutputElements <= output_length &&
                                   static_cast<uint64_t>(base_offset) + group_base + kGroupOutputElements <= base_end;
        const uint32_t base_begin = base_offset + group_base;
        if constexpr (TileNative) {
            if (base_is_dense && base_begin % kGroupOutputElements == 0) {
                read_aligned_output_group(base_addr, base_begin, get_write_ptr(cb_base_tile));
                needs_read_barrier = true;
            } else if (base_is_dense) {
                fill_output_narrow_tiles<false>(base, base_tiles, base_end, base_offset, output_length, group_base);
            } else {
                fill_output_narrow_tiles<true>(base, base_tiles, base_end, base_offset, output_length, group_base);
            }
        } else if (base_is_dense) {
            fill_output_row_major<false>(base, base_tiles, base_end, base_offset, output_length, group_base);
        } else {
            fill_output_row_major<true>(base, base_tiles, base_end, base_offset, output_length, group_base);
        }
        if (needs_read_barrier) {
            noc_async_read_barrier();
        }

        cb_push_back(cb_src_tile0, 2);
        cb_push_back(cb_src_tile1, 2);
        cb_push_back(cb_base_tile, 3);
    }
}

template <bool TileNative>
ALWI void emit_scale_tiles(
    const uint32_t source_addr,
    const uint32_t cb_scale_tile,
    const uint32_t source_end,
    const uint32_t source_offset,
    const uint32_t output_length,
    const uint32_t group_count) {
    const auto* src = reinterpret_cast<volatile tt_l1_ptr float*>(source_addr);

    for (uint32_t group = 0; group < group_count; ++group) {
        const uint32_t group_base = group * kGroupOutputElements;
        cb_reserve_back(cb_scale_tile, 3);
        auto* scale_tiles = reinterpret_cast<float*>(get_write_ptr(cb_scale_tile));
        const bool source_is_dense =
            static_cast<uint64_t>(group_base) + kGroupOutputElements <= output_length &&
            static_cast<uint64_t>(source_offset) + group_base + kGroupOutputElements <= source_end;
        const uint32_t source_begin = source_offset + group_base;
        if constexpr (TileNative) {
            if (source_is_dense && source_begin % kGroupOutputElements == 0) {
                read_aligned_output_group(source_addr, source_begin, get_write_ptr(cb_scale_tile));
                noc_async_read_barrier();
            } else if (source_is_dense) {
                fill_output_narrow_tiles<false>(src, scale_tiles, source_end, source_offset, output_length, group_base);
            } else {
                fill_output_narrow_tiles<true>(src, scale_tiles, source_end, source_offset, output_length, group_base);
            }
        } else if (source_is_dense) {
            fill_output_row_major<false>(src, scale_tiles, source_end, source_offset, output_length, group_base);
        } else {
            fill_output_row_major<true>(src, scale_tiles, source_end, source_offset, output_length, group_base);
        }
        cb_push_back(cb_scale_tile, 3);
    }
}

}  // namespace

void kernel_main() {
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t input_length = get_arg_val<uint32_t>(1);
    const uint32_t left_pad = get_arg_val<uint32_t>(2);
    const uint32_t initial_even_addr = get_arg_val<uint32_t>(3);
    const uint32_t initial_odd_addr = get_arg_val<uint32_t>(4);
    const uint32_t chunk_config_addr = get_arg_val<uint32_t>(5);
    const uint32_t route_config_addr = get_arg_val<uint32_t>(6);
    const uint32_t chunk_begin = get_arg_val<uint32_t>(7);
    const uint32_t chunk_count = get_arg_val<uint32_t>(8);
    const uint32_t route_count = get_arg_val<uint32_t>(9);

    constexpr uint32_t cb_config = get_compile_time_arg_val(0);
    constexpr uint32_t cb_src_tile0 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_src_tile1 = get_compile_time_arg_val(2);
    constexpr uint32_t cb_base_tile = get_compile_time_arg_val(3);
    constexpr uint32_t cb_input_cache = get_compile_time_arg_val(4);
    constexpr uint32_t cb_sync = get_compile_time_arg_val(5);
    constexpr bool tile_native_workspace = get_compile_time_arg_val(6) != 0;
    constexpr auto config_args = TensorAccessorArgs<7>();
    constexpr auto input_args = TensorAccessorArgs<config_args.next_compile_time_args_offset()>();

    const auto input = TensorAccessor(input_args, input_addr, ttwv::device_protocol::kStickBytes);

    bool first_local_route = true;
    for (uint32_t local_chunk = 0; local_chunk < chunk_count; ++local_chunk) {
        if (!first_local_route) {
            cb_wait_front(cb_sync, 1);
            cb_pop_front(cb_sync, 1);
        }

        const uint32_t global_chunk = chunk_begin + local_chunk;
        const uint32_t* chunk = load_config_page(config_args, chunk_config_addr, cb_config, global_chunk);
        const uint32_t initial_even_begin = chunk[ttwv::device_protocol::kConeInitialEvenBegin];
        const uint32_t initial_even_length = chunk[ttwv::device_protocol::kConeInitialEvenLength];
        const uint32_t initial_odd_begin = chunk[ttwv::device_protocol::kConeInitialOddBegin];
        const uint32_t initial_odd_length = chunk[ttwv::device_protocol::kConeInitialOddLength];
        cb_pop_front(cb_config, 1);

        initialize_cone<tile_native_workspace>(
            input,
            initial_even_addr,
            initial_odd_addr,
            cb_input_cache,
            input_length,
            left_pad,
            initial_even_begin,
            initial_even_length,
            initial_odd_begin,
            initial_odd_length);

        for (uint32_t route_index = 0; route_index < route_count; ++route_index) {
            if (route_index > 0) {
                cb_wait_front(cb_sync, 1);
                cb_pop_front(cb_sync, 1);
            }
            const uint32_t config_index = global_chunk * route_count + route_index;
            const uint32_t* route = load_config_page(config_args, route_config_addr, cb_config, config_index);
            const uint32_t route_type = route[ttwv::device_protocol::kRouteType];
            const uint32_t source_addr = route[ttwv::device_protocol::kRouteSourceAddr];
            const uint32_t source_end = route[ttwv::device_protocol::kRouteSourceLength];
            const uint32_t base_addr = route[ttwv::device_protocol::kRouteBaseAddr];
            const uint32_t base_end = route[ttwv::device_protocol::kRouteBaseLength];
            const uint32_t output_length = route[ttwv::device_protocol::kRouteOutputLength];
            const uint32_t source_offset = route[ttwv::device_protocol::kRouteSourceOffset];
            const uint32_t base_offset = route[ttwv::device_protocol::kRouteBaseOffset];
            const uint32_t source_left_pad = route[ttwv::device_protocol::kRouteSourceLeftPad];
            const uint32_t group_count = route[ttwv::device_protocol::kRouteGroupCount];

            if (route_type == kStepPredict || route_type == kStepUpdate) {
                emit_predict_update_tiles<tile_native_workspace>(
                    source_addr,
                    base_addr,
                    cb_src_tile0,
                    cb_src_tile1,
                    cb_base_tile,
                    source_end,
                    base_end,
                    output_length,
                    group_count,
                    source_offset,
                    base_offset,
                    source_left_pad);
            } else if (route_type == kStepScaleEven || route_type == kStepScaleOdd) {
                emit_scale_tiles<tile_native_workspace>(
                    source_addr, cb_base_tile, source_end, source_offset, output_length, group_count);
            }
            cb_pop_front(cb_config, 1);
            first_local_route = false;
        }
    }
}
