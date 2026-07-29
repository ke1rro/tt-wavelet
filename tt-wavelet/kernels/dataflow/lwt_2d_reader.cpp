// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "../../tt_wavelet/include/common/signal_extension.hpp"
#include "../../tt_wavelet/include/device_protocol/lwt_2d_config.hpp"
#include "../../tt_wavelet/include/lifting/step.hpp"
#include "../primitives/tile_2d_layout.hpp"
#include "api/dataflow/dataflow_api.h"

namespace {

using ttwv::kernels::primitives::kFaceSide;
using ttwv::kernels::primitives::kTileBytes;
using ttwv::kernels::primitives::kTileElements;
using ttwv::kernels::primitives::kTileSide;
using ttwv::kernels::primitives::tile_element_offset;
using ttwv::kernels::primitives::tile_face_column_offset;
using ttwv::kernels::primitives::tile_face_row_offset;
using ttwv::kernels::primitives::tiled_element_offset;

struct Rect {
    uint32_t y_begin;
    uint32_t y_length;
    uint32_t x_begin;
    uint32_t x_length;
};

struct SplitMetrics {
    uint32_t input_bytes;
    uint32_t local_output_bytes;
    uint32_t noc_read_calls;
    uint32_t noc_read_barriers;
    uint32_t interior_macro_tiles;
    uint32_t boundary_macro_tiles;
};

#ifdef TTWV_CAPTURE_TRANSPORT_METRICS
#define TTWV_TRANSPORT_METRIC(statement) statement
#else
#define TTWV_TRANSPORT_METRIC(statement) \
    do {                                 \
    } while (false)
#endif

struct RouteStagingMetrics {
    uint32_t exact_source_tiles{0};
    uint32_t shifted_source_tiles{0};
    uint32_t generic_source_tiles{0};
    uint32_t exact_base_tiles{0};
    uint32_t shifted_base_tiles{0};
    uint32_t generic_base_tiles{0};
};

#ifdef TTWV_VALIDATE_ROUTE_STAGING
struct StagingValidationMetrics {
    uint32_t validated_tiles{0};
    uint32_t mismatched_words{0};
    uint32_t class_mismatched_words[5]{};
};
#define TTWV_STAGING_VALIDATION_PARAMETER , StagingValidationMetrics& validation_metrics
#define TTWV_STAGING_VALIDATION_ARGUMENT , staging_validation_metrics
#else
#define TTWV_STAGING_VALIDATION_PARAMETER
#define TTWV_STAGING_VALIDATION_ARGUMENT
#endif
#ifdef TTWV_CAPTURE_SPLIT_METRICS
#define TTWV_SPLIT_METRIC(statement) statement
#else
#define TTWV_SPLIT_METRIC(statement)
#endif

#ifdef TTWV_LWT_2D_COMPACT_BOUNDARY_CODE
#define TTWV_BOUNDARY_FUNCTION __attribute__((noinline))
#else
#define TTWV_BOUNDARY_FUNCTION ALWI
#endif

[[nodiscard]] ALWI Rect load_rect(const uint32_t* words, const uint32_t offset) {
    return Rect{
        .y_begin = words[offset + ttwv::device_protocol::kLwt2DRectYBegin],
        .y_length = words[offset + ttwv::device_protocol::kLwt2DRectYLength],
        .x_begin = words[offset + ttwv::device_protocol::kLwt2DRectXBegin],
        .x_length = words[offset + ttwv::device_protocol::kLwt2DRectXLength],
    };
}

[[nodiscard]] ALWI uint32_t aligned_begin(const uint32_t value) { return (value / kTileSide) * kTileSide; }

[[nodiscard]] ALWI uint32_t aligned_end(const uint32_t begin, const uint32_t length) {
    return ((begin + length + kTileSide - 1) / kTileSide) * kTileSide;
}

[[nodiscard]] ALWI bool contains(const Rect& rectangle, const int32_t y, const int32_t x) {
    return y >= static_cast<int32_t>(rectangle.y_begin) && x >= static_cast<int32_t>(rectangle.x_begin) &&
           y < static_cast<int32_t>(rectangle.y_begin + rectangle.y_length) &&
           x < static_cast<int32_t>(rectangle.x_begin + rectangle.x_length);
}

template <typename Accessor>
ALWI void load_config_page(
    const Accessor& accessor,
    const uint32_t address,
    const uint32_t page_bytes,
    const uint32_t page_index,
    const uint32_t cb,
    uint32_t* words,
    const uint32_t word_count) {
    const auto pages = TensorAccessor(accessor, address, page_bytes);
    cb_reserve_back(cb, 1);
    noc_async_read(pages.get_noc_addr(page_index), get_write_ptr(cb), page_bytes);
    noc_async_read_barrier();
    cb_push_back(cb, 1);
    cb_wait_front(cb, 1);
    const auto* loaded = reinterpret_cast<const uint32_t*>(get_read_ptr(cb));
    for (uint32_t word = 0; word < word_count; ++word) {
        words[word] = loaded[word];
    }
    cb_pop_front(cb, 1);
}

template <typename Accessor>
ALWI void preload_config_pages(
    const Accessor& accessor,
    const uint32_t address,
    const uint32_t page_bytes,
    const uint32_t page_begin,
    const uint32_t page_count,
    const uint32_t destination_addr) {
    const auto pages = TensorAccessor(accessor, address, page_bytes);
    for (uint32_t page = 0; page < page_count; ++page) {
        noc_async_read(pages.get_noc_addr(page_begin + page), destination_addr + page * page_bytes, page_bytes);
    }
    noc_async_read_barrier();
}

#ifndef TTWV_LWT_2D_TILED_SPLIT
template <typename InputAccessor>
struct DirectInputColumnReader {
    const InputAccessor& input;
    uint32_t source_y;
    uint32_t input_tile_columns;
    uint32_t scratch_addr;
    SplitMetrics& metrics;

    ALWI float operator()(const uint32_t source_x) const {
        const uint32_t source_tile = (source_y / kTileSide) * input_tile_columns + source_x / kTileSide;
        const uint32_t source_offset =
            tile_element_offset(source_y % kTileSide, source_x % kTileSide) * sizeof(float);
        const uint64_t source_noc_addr = input.get_noc_addr(source_tile) + source_offset;
        const uint32_t scratch_lane = static_cast<uint32_t>(source_noc_addr) & 63U;
        noc_async_read(source_noc_addr, scratch_addr + scratch_lane, sizeof(float));
        TTWV_SPLIT_METRIC(metrics.input_bytes += sizeof(float));
        TTWV_SPLIT_METRIC(++metrics.noc_read_calls);
        noc_async_read_barrier();
        TTWV_SPLIT_METRIC(++metrics.noc_read_barriers);
        return *reinterpret_cast<volatile tt_l1_ptr float*>(scratch_addr + scratch_lane);
    }
};

template <ttwv::BoundaryMode Mode, typename InputAccessor>
struct DirectInputRowReader {
    const InputAccessor& input;
    const ttwv::ExtendedIndex& x_extended;
    uint32_t input_width;
    uint32_t input_tile_columns;
    uint32_t scratch_addr;
    SplitMetrics& metrics;

    ALWI float operator()(const uint32_t source_y) const {
        return ttwv::evaluate_extended_index<Mode>(
            x_extended,
            input_width,
            DirectInputColumnReader<InputAccessor>{
                .input = input,
                .source_y = source_y,
                .input_tile_columns = input_tile_columns,
                .scratch_addr = scratch_addr,
                .metrics = metrics,
            });
    }
};

template <ttwv::BoundaryMode Mode, typename InputAccessor>
ALWI float read_direct_extended_2d(
    const InputAccessor& input,
    const int64_t raw_y,
    const int64_t raw_x,
    const uint32_t input_height,
    const uint32_t input_width,
    const uint32_t input_tile_columns,
    const uint32_t scratch_addr,
    SplitMetrics& metrics) {
    const ttwv::ExtendedIndex y_extended = ttwv::make_extended_index<Mode>(raw_y, input_height);
    const ttwv::ExtendedIndex x_extended = ttwv::make_extended_index<Mode>(raw_x, input_width);
    return ttwv::evaluate_extended_index<Mode>(
        y_extended,
        input_height,
        DirectInputRowReader<Mode, InputAccessor>{
            .input = input,
            .x_extended = x_extended,
            .input_width = input_width,
            .input_tile_columns = input_tile_columns,
            .scratch_addr = scratch_addr,
            .metrics = metrics,
        });
}

template <ttwv::BoundaryMode Mode, typename InputAccessor>
ALWI void initialize_plane(
    const InputAccessor& input,
    const uint32_t input_height,
    const uint32_t input_width,
    const uint32_t input_tile_columns,
    const uint32_t pad_y,
    const uint32_t pad_x,
    const uint32_t parity_y,
    const uint32_t parity_x,
    const Rect& rectangle,
    const uint32_t plane_addr,
    const uint32_t plane_tile_columns,
    const uint32_t noc_scratch_addr,
    SplitMetrics& metrics) {
    const uint32_t y_origin = aligned_begin(rectangle.y_begin);
    const uint32_t x_origin = aligned_begin(rectangle.x_begin);
    const uint32_t storage_height = aligned_end(rectangle.y_begin, rectangle.y_length) - y_origin;
    const uint32_t storage_width = aligned_end(rectangle.x_begin, rectangle.x_length) - x_origin;
    auto* destination = reinterpret_cast<volatile tt_l1_ptr float*>(plane_addr);
    TTWV_SPLIT_METRIC(
        metrics.local_output_bytes +=
        (storage_height * storage_width + rectangle.y_length * rectangle.x_length) * sizeof(float));

    // Every persistent plane page is a complete tile.  Clear tile padding
    // first, then fill only the exact logical dependency rectangle.
    for (uint32_t local_y = 0; local_y < storage_height; ++local_y) {
        for (uint32_t local_x = 0; local_x < storage_width; ++local_x) {
            destination[tiled_element_offset(local_y, local_x, plane_tile_columns)] = 0.0F;
}

}

    for (uint32_t polyphase_y = rectangle.y_begin; polyphase_y < rectangle.y_begin + rectangle.y_length;
         ++polyphase_y) {
        const int32_t raw_y = static_cast<int32_t>(2 * polyphase_y + parity_y) - static_cast<int32_t>(pad_y);
        const uint32_t local_y = polyphase_y - y_origin;
        const int32_t raw_x_begin =
            static_cast<int32_t>(2 * rectangle.x_begin + parity_x) - static_cast<int32_t>(pad_x);
        const int32_t raw_x_end = raw_x_begin + static_cast<int32_t>(2 * (rectangle.x_length - 1));

        // Interior rows are gathered a face segment at a time.  A 32-sample
        // polyphase row maps to alternating values in the raw row, so each
        // burst reads at most 15 contiguous FP32 values and emits up to eight
        // plane values.  The 64-byte aligned scratch window is large enough
        // for the complete burst.
        if (raw_y >= 0 && raw_y < static_cast<int32_t>(input_height) && raw_x_begin >= 0 &&
            raw_x_end < static_cast<int32_t>(input_width)) {
            const uint32_t source_y = static_cast<uint32_t>(raw_y);
            uint32_t local_x = rectangle.x_begin - x_origin;
            uint32_t source_x = static_cast<uint32_t>(raw_x_begin);
            uint32_t remaining = rectangle.x_length;
            while (remaining > 0) {
                const uint32_t source_face_remaining = kFaceSide - source_x % kFaceSide;
                const uint32_t output_count = std::min(remaining, (source_face_remaining + 1) / 2);
                const uint32_t source_count = 2 * output_count - 1;
                const uint32_t source_tile = (source_y / kTileSide) * input_tile_columns + source_x / kTileSide;
                const uint32_t source_offset =
                    tile_element_offset(source_y % kTileSide, source_x % kTileSide) * sizeof(float);
                const uint64_t source_noc_addr = input.get_noc_addr(source_tile) + source_offset;
                const uint32_t scratch_lane = static_cast<uint32_t>(source_noc_addr) & 63U;
                noc_async_read(source_noc_addr, noc_scratch_addr + scratch_lane, source_count * sizeof(float));
                TTWV_SPLIT_METRIC(metrics.input_bytes += source_count * sizeof(float));
                TTWV_SPLIT_METRIC(++metrics.noc_read_calls);
                noc_async_read_barrier();
                TTWV_SPLIT_METRIC(++metrics.noc_read_barriers);
                const auto* staged = reinterpret_cast<volatile tt_l1_ptr float*>(noc_scratch_addr + scratch_lane);
                for (uint32_t value = 0; value < output_count; ++value) {
                    destination[tiled_element_offset(local_y, local_x + value, plane_tile_columns)] = staged[2 * value];
                }
                local_x += output_count;
                source_x += 2 * output_count;
                remaining -= output_count;
            }
            continue;
        }

        // Affine and reflected edges are evaluated only in this conservative
        // boundary path. The dense in-bounds row path above remains mode-free.
        for (uint32_t polyphase_x = rectangle.x_begin; polyphase_x < rectangle.x_begin + rectangle.x_length;
             ++polyphase_x) {
            const int32_t raw_x = static_cast<int32_t>(2 * polyphase_x + parity_x) - static_cast<int32_t>(pad_x);
            destination[tiled_element_offset(local_y, polyphase_x - x_origin, plane_tile_columns)] =
                read_direct_extended_2d<Mode>(
                    input,
                    raw_y,
                    raw_x,
                    input_height,
                    input_width,
                    input_tile_columns,
                    noc_scratch_addr,
                    metrics);
        }
    }
}
#else
template <ttwv::BoundaryMode Mode>
struct SplitSourceTiles {
    static constexpr uint32_t kAxisCapacity =
        Mode == ttwv::BoundaryMode::kSymmetric
            ? ttwv::device_protocol::kLwt2DSymmetricSplitScratchTileRows
            : ttwv::device_protocol::kLwt2DSplitScratchTileRows;
    uint32_t rows[kAxisCapacity];
    uint32_t columns[kAxisCapacity];
    uint32_t row_count;
    uint32_t column_count;
};

struct SourceAxisTileCollector {
    uint32_t* tiles;
    uint32_t& count;
    uint32_t capacity;

    TTWV_BOUNDARY_FUNCTION void operator()(const uint32_t source_index) const {
        const uint32_t source_tile = source_index / kTileSide;
        for (uint32_t index = 0; index < count; ++index) {
            if (tiles[index] == source_tile) {
                return;
            }
        }
        if (count < capacity) {
            tiles[count++] = source_tile;
        }
    }
};

[[nodiscard]] ALWI bool intersects_tile(const Rect& rectangle, const uint32_t tile_y, const uint32_t tile_x) {
    return rectangle.y_begin < tile_y + kTileSide && rectangle.y_begin + rectangle.y_length > tile_y &&
           rectangle.x_begin < tile_x + kTileSide && rectangle.x_begin + rectangle.x_length > tile_x;
}

[[nodiscard]] ALWI bool covers_tile(const Rect& rectangle, const uint32_t tile_y, const uint32_t tile_x) {
    return rectangle.y_begin <= tile_y && rectangle.y_begin + rectangle.y_length >= tile_y + kTileSide &&
           rectangle.x_begin <= tile_x && rectangle.x_begin + rectangle.x_length >= tile_x + kTileSide;
}

template <uint32_t Capacity>
[[nodiscard]] ALWI uint32_t find_index(const uint32_t (&values)[Capacity], const uint32_t count, const uint32_t value) {
    for (uint32_t index = 0; index < count; ++index) {
        if (values[index] == value) {
            return index;
        }
    }
    return 0;
}

template <ttwv::BoundaryMode Mode>
TTWV_BOUNDARY_FUNCTION void collect_boundary_source_axis_tiles(
    uint32_t* tiles,
    uint32_t& count,
    const uint32_t capacity,
    const int32_t raw_begin,
    const uint32_t logical_length) {
    if constexpr (Mode == ttwv::BoundaryMode::kSymmetric) {
        for (uint32_t offset = 0; offset < 2 * kTileSide; ++offset) {
            const uint32_t source_tile =
                ttwv::make_symmetric_index_i32(raw_begin + static_cast<int32_t>(offset), logical_length) / kTileSide;
            bool found = false;
            for (uint32_t index = 0; index < count; ++index) {
                found = found || tiles[index] == source_tile;
            }
            if (!found && count < capacity) {
                tiles[count++] = source_tile;
            }
        }
        return;
    }
    const SourceAxisTileCollector collector{
        .tiles = tiles,
        .count = count,
        .capacity = capacity,
    };
    if constexpr (Mode == ttwv::BoundaryMode::kAntireflect) {
        for (uint32_t offset = 0; offset < 2 * kTileSide; ++offset) {
            const int32_t raw_index = raw_begin + static_cast<int32_t>(offset);
            const ttwv::AntireflectIndexI32 extended =
                ttwv::make_antireflect_index_i32(raw_index, logical_length);
            collector(extended.source_index);
        }
        // Affine extension may use both endpoint values, but their tile IDs
        // are invariant across this whole macro-tile.
        collector(0);
        collector(logical_length - 1U);
    } else if constexpr (Mode == ttwv::BoundaryMode::kSmooth) {
        for (uint32_t offset = 0; offset < 2 * kTileSide; ++offset) {
            const int32_t raw_index = raw_begin + static_cast<int32_t>(offset);
            const ttwv::SmoothIndexI32 extended =
                ttwv::make_smooth_index_i32(raw_index, logical_length);
            ttwv::visit_smooth_source_indices_i32(extended, collector);
        }
    } else {
        for (uint32_t offset = 0; offset < 2 * kTileSide; ++offset) {
            const int32_t raw_index = raw_begin + static_cast<int32_t>(offset);
            const ttwv::ExtendedIndex extended =
                ttwv::make_extended_index<Mode>(raw_index, logical_length);
            ttwv::visit_extended_source_indices<Mode>(extended, logical_length, collector);
        }
    }
}

template <bool Interior, ttwv::BoundaryMode Mode>
ALWI void collect_source_axis_tiles(
    uint32_t* tiles,
    uint32_t& count,
    const uint32_t capacity,
    const int32_t raw_begin,
    const uint32_t logical_length) {
    if constexpr (Interior) {
        const uint32_t source_begin = static_cast<uint32_t>(raw_begin);
        const uint32_t source_end = source_begin + 2 * kTileSide - 1;
        for (uint32_t tile = source_begin / kTileSide; tile <= source_end / kTileSide; ++tile) {
            tiles[count++] = tile;
        }
    } else {
        collect_boundary_source_axis_tiles<Mode>(tiles, count, capacity, raw_begin, logical_length);
    }
}

template <bool Interior, ttwv::BoundaryMode Mode, typename InputAccessor>
[[nodiscard]] ALWI SplitSourceTiles<Mode> stage_split_source_tiles(
    const InputAccessor& input,
    const uint32_t input_height,
    const uint32_t input_width,
    const uint32_t input_tile_columns,
    const int32_t raw_y_begin,
    const int32_t raw_x_begin,
    const uint32_t scratch_addr,
    SplitMetrics& metrics) {
    SplitSourceTiles<Mode> tiles{};
    collect_source_axis_tiles<Interior, Mode>(
        tiles.rows,
        tiles.row_count,
        SplitSourceTiles<Mode>::kAxisCapacity,
        raw_y_begin,
        input_height);
    collect_source_axis_tiles<Interior, Mode>(
        tiles.columns,
        tiles.column_count,
        SplitSourceTiles<Mode>::kAxisCapacity,
        raw_x_begin,
        input_width);

    for (uint32_t tile_y = 0; tile_y < tiles.row_count; ++tile_y) {
        for (uint32_t tile_x = 0; tile_x < tiles.column_count; ++tile_x) {
            const uint32_t source_tile = tiles.rows[tile_y] * input_tile_columns + tiles.columns[tile_x];
            const uint32_t scratch_tile = tile_y * tiles.column_count + tile_x;
            noc_async_read(input.get_noc_addr(source_tile), scratch_addr + scratch_tile * kTileBytes, kTileBytes);
            TTWV_SPLIT_METRIC(metrics.input_bytes += kTileBytes);
            TTWV_SPLIT_METRIC(++metrics.noc_read_calls);
        }
    }
    noc_async_read_barrier();
    TTWV_SPLIT_METRIC(++metrics.noc_read_barriers);
    return tiles;
}

template <ttwv::BoundaryMode Mode>
struct StagedInputColumnReader {
    uint32_t source_y;
    const SplitSourceTiles<Mode>& source_tiles;
    uint32_t scratch_addr;

    TTWV_BOUNDARY_FUNCTION float operator()(const uint32_t source_x) const {
        const uint32_t source_tile_y = find_index(source_tiles.rows, source_tiles.row_count, source_y / kTileSide);
        const uint32_t source_tile_x =
            find_index(source_tiles.columns, source_tiles.column_count, source_x / kTileSide);
        const uint32_t scratch_tile = source_tile_y * source_tiles.column_count + source_tile_x;
        const auto* source =
            reinterpret_cast<volatile tt_l1_ptr float*>(scratch_addr + scratch_tile * kTileBytes);
        return source[tile_element_offset(source_y % kTileSide, source_x % kTileSide)];
    }
};

template <ttwv::BoundaryMode Mode>
struct StagedInputRowReader {
    const ttwv::ExtendedIndex& x_extended;
    uint32_t input_width;
    const SplitSourceTiles<Mode>& source_tiles;
    uint32_t scratch_addr;

    ALWI float operator()(const uint32_t source_y) const {
        return ttwv::evaluate_extended_index<Mode>(
            x_extended,
            input_width,
            StagedInputColumnReader<Mode>{
                .source_y = source_y,
                .source_tiles = source_tiles,
                .scratch_addr = scratch_addr,
            });
    }
};

struct StagedAntireflectInputRowReader {
    const ttwv::AntireflectIndexI32& x_extended;
    uint32_t input_width;
    const SplitSourceTiles<ttwv::BoundaryMode::kAntireflect>& source_tiles;
    uint32_t scratch_addr;

    TTWV_BOUNDARY_FUNCTION float operator()(const uint32_t source_y) const {
        return ttwv::evaluate_antireflect_index_i32(
            x_extended,
            input_width,
            StagedInputColumnReader<ttwv::BoundaryMode::kAntireflect>{
                .source_y = source_y,
                .source_tiles = source_tiles,
                .scratch_addr = scratch_addr,
            });
    }
};

struct StagedSmoothInputRowReader {
    const ttwv::SmoothIndexI32& x_extended;
    const SplitSourceTiles<ttwv::BoundaryMode::kSmooth>& source_tiles;
    uint32_t scratch_addr;

    TTWV_BOUNDARY_FUNCTION float operator()(const uint32_t source_y) const {
        return ttwv::evaluate_smooth_index_i32(
            x_extended,
            StagedInputColumnReader<ttwv::BoundaryMode::kSmooth>{
                .source_y = source_y,
                .source_tiles = source_tiles,
                .scratch_addr = scratch_addr,
            });
    }
};

template <ttwv::BoundaryMode Mode>
[[nodiscard]] ALWI float read_staged_extended_2d(
    const int32_t raw_y,
    const int32_t raw_x,
    const uint32_t input_height,
    const uint32_t input_width,
    const SplitSourceTiles<Mode>& source_tiles,
    const uint32_t scratch_addr) {
    if constexpr (Mode == ttwv::BoundaryMode::kAntireflect) {
        const ttwv::AntireflectIndexI32 y_extended =
            ttwv::make_antireflect_index_i32(raw_y, input_height);
        const ttwv::AntireflectIndexI32 x_extended =
            ttwv::make_antireflect_index_i32(raw_x, input_width);
        return ttwv::evaluate_antireflect_index_i32(
            y_extended,
            input_height,
            StagedAntireflectInputRowReader{
                .x_extended = x_extended,
                .input_width = input_width,
                .source_tiles = source_tiles,
                .scratch_addr = scratch_addr,
            });
    }
    if constexpr (Mode == ttwv::BoundaryMode::kSmooth) {
        const ttwv::SmoothIndexI32 y_extended =
            ttwv::make_smooth_index_i32(raw_y, input_height);
        const ttwv::SmoothIndexI32 x_extended =
            ttwv::make_smooth_index_i32(raw_x, input_width);
        return ttwv::evaluate_smooth_index_i32(
            y_extended,
            StagedSmoothInputRowReader{
                .x_extended = x_extended,
                .source_tiles = source_tiles,
                .scratch_addr = scratch_addr,
            });
    }
    if constexpr (Mode == ttwv::BoundaryMode::kSymmetric) {
        const uint32_t source_y =
            ttwv::make_symmetric_index_i32(raw_y, input_height);
        const uint32_t source_x =
            ttwv::make_symmetric_index_i32(raw_x, input_width);
        return StagedInputColumnReader<Mode>{
            .source_y = source_y,
            .source_tiles = source_tiles,
            .scratch_addr = scratch_addr,
        }(source_x);
    }
    const ttwv::ExtendedIndex y_extended = ttwv::make_extended_index<Mode>(raw_y, input_height);
    const ttwv::ExtendedIndex x_extended = ttwv::make_extended_index<Mode>(raw_x, input_width);
    return ttwv::evaluate_extended_index<Mode>(
        y_extended,
        input_height,
        StagedInputRowReader<Mode>{
            .x_extended = x_extended,
            .input_width = input_width,
            .source_tiles = source_tiles,
            .scratch_addr = scratch_addr,
        });
}

#ifdef TTWV_LWT_2D_COMPACT_BOUNDARY_CODE
#define TTWV_POLYPHASE_TEMPLATE template <bool Interior, ttwv::BoundaryMode Mode>
#define TTWV_POLYPHASE_FUNCTION __attribute__((noinline))
#define TTWV_POLYPHASE_PARITY_PARAMETERS \
    const uint32_t parity_y,              \
    const uint32_t parity_x,
#define TTWV_POLYPHASE_PARITY_Y parity_y
#define TTWV_POLYPHASE_PARITY_X parity_x
#else
#define TTWV_POLYPHASE_TEMPLATE \
    template <bool Interior, ttwv::BoundaryMode Mode, uint32_t ParityY, uint32_t ParityX>
#define TTWV_POLYPHASE_FUNCTION ALWI
#define TTWV_POLYPHASE_PARITY_PARAMETERS
#define TTWV_POLYPHASE_PARITY_Y ParityY
#define TTWV_POLYPHASE_PARITY_X ParityX
#endif

TTWV_POLYPHASE_TEMPLATE
TTWV_POLYPHASE_FUNCTION void write_polyphase_tile(
    const uint32_t input_height,
    const uint32_t input_width,
    const uint32_t pad_y,
    const uint32_t pad_x,
    TTWV_POLYPHASE_PARITY_PARAMETERS
    const Rect& rectangle,
    const uint32_t plane_addr,
    const uint32_t plane_tile_columns,
    const uint32_t tile_y,
    const uint32_t tile_x,
    const SplitSourceTiles<Mode>& source_tiles,
    const uint32_t scratch_addr) {
    if (!intersects_tile(rectangle, tile_y, tile_x)) {
        return;
    }

    const uint32_t rectangle_y_origin = aligned_begin(rectangle.y_begin);
    const uint32_t rectangle_x_origin = aligned_begin(rectangle.x_begin);
    const uint32_t plane_tile_y = (tile_y - rectangle_y_origin) / kTileSide;
    const uint32_t plane_tile_x = (tile_x - rectangle_x_origin) / kTileSide;
    const uint32_t destination_tile_index = plane_tile_y * plane_tile_columns + plane_tile_x;
    auto* destination = reinterpret_cast<volatile tt_l1_ptr float*>(plane_addr + destination_tile_index * kTileBytes);

    const uint32_t y_begin = std::max(rectangle.y_begin, tile_y);
    const uint32_t y_end = std::min(rectangle.y_begin + rectangle.y_length, tile_y + kTileSide);
    const uint32_t x_begin = std::max(rectangle.x_begin, tile_x);
    const uint32_t x_end = std::min(rectangle.x_begin + rectangle.x_length, tile_x + kTileSide);
    for (uint32_t polyphase_y = y_begin; polyphase_y < y_end; ++polyphase_y) {
        const int32_t raw_y =
            2 * static_cast<int32_t>(polyphase_y) + static_cast<int32_t>(TTWV_POLYPHASE_PARITY_Y) -
            static_cast<int32_t>(pad_y);
        for (uint32_t polyphase_x = x_begin; polyphase_x < x_end; ++polyphase_x) {
            const int32_t raw_x =
                2 * static_cast<int32_t>(polyphase_x) + static_cast<int32_t>(TTWV_POLYPHASE_PARITY_X) -
                static_cast<int32_t>(pad_x);
            if constexpr (Interior) {
                const uint32_t source_y = static_cast<uint32_t>(raw_y);
                const uint32_t source_x = static_cast<uint32_t>(raw_x);
                const uint32_t source_tile_y =
                    find_index(source_tiles.rows, source_tiles.row_count, source_y / kTileSide);
                const uint32_t source_tile_x =
                    find_index(source_tiles.columns, source_tiles.column_count, source_x / kTileSide);
                const uint32_t scratch_tile = source_tile_y * source_tiles.column_count + source_tile_x;
                const auto* source =
                    reinterpret_cast<volatile tt_l1_ptr float*>(scratch_addr + scratch_tile * kTileBytes);
                destination[tile_element_offset(polyphase_y - tile_y, polyphase_x - tile_x)] =
                    source[tile_element_offset(source_y % kTileSide, source_x % kTileSide)];
            } else {
                destination[tile_element_offset(polyphase_y - tile_y, polyphase_x - tile_x)] =
                    read_staged_extended_2d<Mode>(
                        raw_y,
                        raw_x,
                        input_height,
                        input_width,
                        source_tiles,
                        scratch_addr);
            }
        }
    }
}

#undef TTWV_POLYPHASE_TEMPLATE
#undef TTWV_POLYPHASE_FUNCTION
#undef TTWV_POLYPHASE_PARITY_PARAMETERS
#undef TTWV_POLYPHASE_PARITY_Y
#undef TTWV_POLYPHASE_PARITY_X

template <bool Interior, ttwv::BoundaryMode Mode, uint32_t ParityY, uint32_t ParityX>
ALWI void write_polyphase_tile_dispatch(
    const uint32_t input_height,
    const uint32_t input_width,
    const uint32_t pad_y,
    const uint32_t pad_x,
    const Rect& rectangle,
    const uint32_t plane_addr,
    const uint32_t plane_tile_columns,
    const uint32_t tile_y,
    const uint32_t tile_x,
    const SplitSourceTiles<Mode>& source_tiles,
    const uint32_t scratch_addr) {
#ifdef TTWV_LWT_2D_COMPACT_BOUNDARY_CODE
    write_polyphase_tile<Interior, Mode>(
        input_height,
        input_width,
        pad_y,
        pad_x,
        ParityY,
        ParityX,
        rectangle,
        plane_addr,
        plane_tile_columns,
        tile_y,
        tile_x,
        source_tiles,
        scratch_addr);
#else
    write_polyphase_tile<Interior, Mode, ParityY, ParityX>(
        input_height,
        input_width,
        pad_y,
        pad_x,
        rectangle,
        plane_addr,
        plane_tile_columns,
        tile_y,
        tile_x,
        source_tiles,
        scratch_addr);
#endif
}

struct SplitSourceColumn {
    uint32_t scratch_tile_byte_offset;
    uint32_t face_column_offset;
};

[[nodiscard]] ALWI uint32_t split_destination_tile_address(
    const Rect& rectangle,
    const uint32_t plane_addr,
    const uint32_t plane_tile_columns,
    const uint32_t tile_y,
    const uint32_t tile_x) {
    const uint32_t rectangle_y_origin = aligned_begin(rectangle.y_begin);
    const uint32_t rectangle_x_origin = aligned_begin(rectangle.x_begin);
    const uint32_t plane_tile_y = (tile_y - rectangle_y_origin) / kTileSide;
    const uint32_t plane_tile_x = (tile_x - rectangle_x_origin) / kTileSide;
    return plane_addr + (plane_tile_y * plane_tile_columns + plane_tile_x) * kTileBytes;
}

template <ttwv::BoundaryMode Mode>
ALWI void write_full_interior_polyphase_tiles(
    const int32_t raw_y_begin,
    const int32_t raw_x_begin,
    const Rect* rectangles,
    const uint32_t* plane_addrs,
    const uint32_t* plane_tile_columns,
    const uint32_t tile_y,
    const uint32_t tile_x,
    const SplitSourceTiles<Mode>& source_tiles,
    const uint32_t scratch_addr) {
    // A complete interior macro-tile is a pure bit permutation. Precompute
    // the tiled source-column mapping once, then traverse the four outputs
    // together. This removes symmetric mapping, source-tile searches,
    // division/modulo, and tile_element_offset() from the 32x32 hot loop.
    SplitSourceColumn even_columns[kTileSide];
    SplitSourceColumn odd_columns[kTileSide];
    uint32_t destination_column_offsets[kTileSide];
    const uint32_t first_source_tile_column = source_tiles.columns[0];
    for (uint32_t column = 0; column < kTileSide; ++column) {
        const uint32_t source_even_x = static_cast<uint32_t>(raw_x_begin + 2 * static_cast<int32_t>(column));
        const uint32_t source_odd_x = source_even_x + 1;
        destination_column_offsets[column] = tile_face_column_offset(column);
        even_columns[column] = SplitSourceColumn{
            .scratch_tile_byte_offset =
                (source_even_x / kTileSide - first_source_tile_column) * kTileBytes,
            .face_column_offset = tile_face_column_offset(source_even_x % kTileSide),
        };
        odd_columns[column] = SplitSourceColumn{
            .scratch_tile_byte_offset =
                (source_odd_x / kTileSide - first_source_tile_column) * kTileBytes,
            .face_column_offset = tile_face_column_offset(source_odd_x % kTileSide),
        };
    }

    auto* ee = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        split_destination_tile_address(
            rectangles[0], plane_addrs[0], plane_tile_columns[0], tile_y, tile_x));
    auto* eo = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        split_destination_tile_address(
            rectangles[1], plane_addrs[1], plane_tile_columns[1], tile_y, tile_x));
    auto* oe = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        split_destination_tile_address(
            rectangles[2], plane_addrs[2], plane_tile_columns[2], tile_y, tile_x));
    auto* oo = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        split_destination_tile_address(
            rectangles[3], plane_addrs[3], plane_tile_columns[3], tile_y, tile_x));

    const uint32_t first_source_tile_row = source_tiles.rows[0];
    const uint32_t scratch_tile_row_stride = source_tiles.column_count * kTileBytes;
    for (uint32_t row = 0; row < kTileSide; ++row) {
        const uint32_t source_even_y = static_cast<uint32_t>(raw_y_begin + 2 * static_cast<int32_t>(row));
        const uint32_t source_odd_y = source_even_y + 1;
        const uint32_t source_even_row_base =
            scratch_addr + (source_even_y / kTileSide - first_source_tile_row) * scratch_tile_row_stride;
        const uint32_t source_odd_row_base =
            scratch_addr + (source_odd_y / kTileSide - first_source_tile_row) * scratch_tile_row_stride;
        const uint32_t source_even_row_offset = tile_face_row_offset(source_even_y % kTileSide);
        const uint32_t source_odd_row_offset = tile_face_row_offset(source_odd_y % kTileSide);
        const uint32_t destination_row_offset = tile_face_row_offset(row);

        for (uint32_t column = 0; column < kTileSide; ++column) {
            const SplitSourceColumn even_column = even_columns[column];
            const SplitSourceColumn odd_column = odd_columns[column];
            const auto* even_row_even_column = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                source_even_row_base + even_column.scratch_tile_byte_offset);
            const auto* even_row_odd_column = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                source_even_row_base + odd_column.scratch_tile_byte_offset);
            const auto* odd_row_even_column = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                source_odd_row_base + even_column.scratch_tile_byte_offset);
            const auto* odd_row_odd_column = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                source_odd_row_base + odd_column.scratch_tile_byte_offset);
            const uint32_t destination_offset = destination_row_offset + destination_column_offsets[column];
            ee[destination_offset] =
                even_row_even_column[source_even_row_offset + even_column.face_column_offset];
            eo[destination_offset] = even_row_odd_column[source_even_row_offset + odd_column.face_column_offset];
            oe[destination_offset] = odd_row_even_column[source_odd_row_offset + even_column.face_column_offset];
            oo[destination_offset] = odd_row_odd_column[source_odd_row_offset + odd_column.face_column_offset];
        }
    }
}

template <bool Interior, ttwv::BoundaryMode Mode, typename InputAccessor>
ALWI void split_macro_tile(
    const InputAccessor& input,
    const uint32_t input_height,
    const uint32_t input_width,
    const uint32_t input_tile_columns,
    const uint32_t pad_y,
    const uint32_t pad_x,
    const Rect* rectangles,
    const uint32_t* plane_addrs,
    const uint32_t* plane_tile_columns,
    const uint32_t tile_y,
    const uint32_t tile_x,
    const uint32_t scratch_addr,
    SplitMetrics& metrics) {
    const int32_t raw_y_begin = 2 * static_cast<int32_t>(tile_y) - static_cast<int32_t>(pad_y);
    const int32_t raw_x_begin = 2 * static_cast<int32_t>(tile_x) - static_cast<int32_t>(pad_x);
    const SplitSourceTiles<Mode> source_tiles = stage_split_source_tiles<Interior, Mode>(
        input, input_height, input_width, input_tile_columns, raw_y_begin, raw_x_begin, scratch_addr, metrics);

    if constexpr (Interior) {
        bool complete = true;
        for (uint32_t plane = 0; plane < 4; ++plane) {
            complete = complete && covers_tile(rectangles[plane], tile_y, tile_x);
        }
        if (complete) {
            write_full_interior_polyphase_tiles<Mode>(
                raw_y_begin,
                raw_x_begin,
                rectangles,
                plane_addrs,
                plane_tile_columns,
                tile_y,
                tile_x,
                source_tiles,
                scratch_addr);
            return;
        }
    }

    write_polyphase_tile_dispatch<Interior, Mode, 0, 0>(
        input_height,
        input_width,
        pad_y,
        pad_x,
        rectangles[0],
        plane_addrs[0],
        plane_tile_columns[0],
        tile_y,
        tile_x,
        source_tiles,
        scratch_addr);
    write_polyphase_tile_dispatch<Interior, Mode, 0, 1>(
        input_height,
        input_width,
        pad_y,
        pad_x,
        rectangles[1],
        plane_addrs[1],
        plane_tile_columns[1],
        tile_y,
        tile_x,
        source_tiles,
        scratch_addr);
    write_polyphase_tile_dispatch<Interior, Mode, 1, 0>(
        input_height,
        input_width,
        pad_y,
        pad_x,
        rectangles[2],
        plane_addrs[2],
        plane_tile_columns[2],
        tile_y,
        tile_x,
        source_tiles,
        scratch_addr);
    write_polyphase_tile_dispatch<Interior, Mode, 1, 1>(
        input_height,
        input_width,
        pad_y,
        pad_x,
        rectangles[3],
        plane_addrs[3],
        plane_tile_columns[3],
        tile_y,
        tile_x,
        source_tiles,
        scratch_addr);
}

template <ttwv::BoundaryMode Mode, typename InputAccessor>
ALWI void initialize_planes_tiled(
    const InputAccessor& input,
    const uint32_t input_height,
    const uint32_t input_width,
    const uint32_t input_tile_columns,
    const uint32_t pad_y,
    const uint32_t pad_x,
    const Rect* rectangles,
    const uint32_t* plane_addrs,
    const uint32_t* plane_tile_columns,
    const uint32_t scratch_addr,
    SplitMetrics& metrics) {
    uint32_t y_begin = rectangles[0].y_begin;
    uint32_t y_end = rectangles[0].y_begin + rectangles[0].y_length;
    uint32_t x_begin = rectangles[0].x_begin;
    uint32_t x_end = rectangles[0].x_begin + rectangles[0].x_length;
    for (uint32_t plane = 1; plane < 4; ++plane) {
        y_begin = std::min(y_begin, rectangles[plane].y_begin);
        y_end = std::max(y_end, rectangles[plane].y_begin + rectangles[plane].y_length);
        x_begin = std::min(x_begin, rectangles[plane].x_begin);
        x_end = std::max(x_end, rectangles[plane].x_begin + rectangles[plane].x_length);
    }
    for (uint32_t plane = 0; plane < 4; ++plane) {
        TTWV_SPLIT_METRIC(
            metrics.local_output_bytes += rectangles[plane].y_length * rectangles[plane].x_length * sizeof(float));
    }

    for (uint32_t tile_y = aligned_begin(y_begin); tile_y < aligned_end(y_begin, y_end - y_begin);
         tile_y += kTileSide) {
        for (uint32_t tile_x = aligned_begin(x_begin); tile_x < aligned_end(x_begin, x_end - x_begin);
             tile_x += kTileSide) {
            bool active = false;
            for (uint32_t plane = 0; plane < 4; ++plane) {
                active = active || intersects_tile(rectangles[plane], tile_y, tile_x);
            }
            if (!active) {
                continue;
            }
            const int32_t raw_y_begin = 2 * static_cast<int32_t>(tile_y) - static_cast<int32_t>(pad_y);
            const int32_t raw_x_begin = 2 * static_cast<int32_t>(tile_x) - static_cast<int32_t>(pad_x);
            const bool interior =
                raw_y_begin >= 0 && raw_x_begin >= 0 &&
                raw_y_begin + static_cast<int32_t>(2 * kTileSide) <= static_cast<int32_t>(input_height) &&
                raw_x_begin + static_cast<int32_t>(2 * kTileSide) <= static_cast<int32_t>(input_width);
            if (interior) {
                TTWV_SPLIT_METRIC(++metrics.interior_macro_tiles);
                split_macro_tile<true, Mode>(
                    input,
                    input_height,
                    input_width,
                    input_tile_columns,
                    pad_y,
                    pad_x,
                    rectangles,
                    plane_addrs,
                    plane_tile_columns,
                    tile_y,
                    tile_x,
                    scratch_addr,
                    metrics);
            } else {
                TTWV_SPLIT_METRIC(++metrics.boundary_macro_tiles);
                split_macro_tile<false, Mode>(
                    input,
                    input_height,
                    input_width,
                    input_tile_columns,
                    pad_y,
                    pad_x,
                    rectangles,
                    plane_addrs,
                    plane_tile_columns,
                    tile_y,
                    tile_x,
                    scratch_addr,
                    metrics);
            }
        }
    }
}
#endif

#ifdef TTWV_ILWT_2D
template <typename BandAccessor>
ALWI void initialize_inverse_band_plane(
    const BandAccessor& band_args,
    const uint32_t band_addr,
    const uint32_t band_height,
    const uint32_t band_width,
    const uint32_t band_tile_columns,
    const int32_t y_internal_offset,
    const int32_t x_internal_offset,
    const Rect& rectangle,
    const uint32_t plane_addr,
    const uint32_t plane_tile_columns,
    const uint32_t scratch_addr,
    const uint32_t zero_tile_addr) {
    const auto band = TensorAccessor(band_args, band_addr, kTileBytes);
    const uint32_t y_origin = aligned_begin(rectangle.y_begin);
    const uint32_t x_origin = aligned_begin(rectangle.x_begin);
    const uint32_t y_end = aligned_end(rectangle.y_begin, rectangle.y_length);
    const uint32_t x_end = aligned_end(rectangle.x_begin, rectangle.x_length);

    for (uint32_t tile_y = y_origin; tile_y < y_end; tile_y += kTileSide) {
        for (uint32_t tile_x = x_origin; tile_x < x_end; tile_x += kTileSide) {
            const uint32_t destination_tile =
                ((tile_y - y_origin) / kTileSide) * plane_tile_columns + (tile_x - x_origin) / kTileSide;
            const uint32_t destination_addr = plane_addr + destination_tile * kTileBytes;
            const int32_t full_canonical_y = static_cast<int32_t>(tile_y) - y_internal_offset;
            const int32_t full_canonical_x = static_cast<int32_t>(tile_x) - x_internal_offset;
            const bool exact_full_tile =
                tile_y >= rectangle.y_begin && tile_y + kTileSide <= rectangle.y_begin + rectangle.y_length &&
                tile_x >= rectangle.x_begin && tile_x + kTileSide <= rectangle.x_begin + rectangle.x_length &&
                full_canonical_y >= 0 && full_canonical_x >= 0 &&
                full_canonical_y % static_cast<int32_t>(kTileSide) == 0 &&
                full_canonical_x % static_cast<int32_t>(kTileSide) == 0 &&
                full_canonical_y + static_cast<int32_t>(kTileSide) <= static_cast<int32_t>(band_height) &&
                full_canonical_x + static_cast<int32_t>(kTileSide) <= static_cast<int32_t>(band_width);
            if (exact_full_tile) {
                const uint32_t source_tile =
                    (static_cast<uint32_t>(full_canonical_y) / kTileSide) * band_tile_columns +
                    static_cast<uint32_t>(full_canonical_x) / kTileSide;
                noc_async_read(band.get_noc_addr(source_tile), destination_addr, kTileBytes);
                noc_async_read_barrier();
                continue;
            }
            noc_async_read(get_noc_addr(zero_tile_addr), destination_addr, kTileBytes);
            noc_async_read_barrier();

            const uint32_t internal_y_begin = std::max(tile_y, rectangle.y_begin);
            const uint32_t internal_y_end = std::min(tile_y + kTileSide, rectangle.y_begin + rectangle.y_length);
            const uint32_t internal_x_begin = std::max(tile_x, rectangle.x_begin);
            const uint32_t internal_x_end = std::min(tile_x + kTileSide, rectangle.x_begin + rectangle.x_length);
            if (internal_y_begin == internal_y_end || internal_x_begin == internal_x_end) {
                continue;
            }

            const int32_t canonical_y_begin = static_cast<int32_t>(internal_y_begin) - y_internal_offset;
            const int32_t canonical_y_end = static_cast<int32_t>(internal_y_end) - y_internal_offset;
            const int32_t canonical_x_begin = static_cast<int32_t>(internal_x_begin) - x_internal_offset;
            const int32_t canonical_x_end = static_cast<int32_t>(internal_x_end) - x_internal_offset;
            ASSERT(canonical_y_begin >= 0 && canonical_x_begin >= 0);
            ASSERT(canonical_y_end <= static_cast<int32_t>(band_height));
            ASSERT(canonical_x_end <= static_cast<int32_t>(band_width));

            const uint32_t source_tile_y_begin = static_cast<uint32_t>(canonical_y_begin) / kTileSide;
            const uint32_t source_tile_y_end =
                (static_cast<uint32_t>(canonical_y_end - 1) / kTileSide) + 1;
            const uint32_t source_tile_x_begin = static_cast<uint32_t>(canonical_x_begin) / kTileSide;
            const uint32_t source_tile_x_end =
                (static_cast<uint32_t>(canonical_x_end - 1) / kTileSide) + 1;
            const uint32_t source_tile_rows = source_tile_y_end - source_tile_y_begin;
            const uint32_t source_tile_columns = source_tile_x_end - source_tile_x_begin;
            ASSERT(source_tile_rows <= ttwv::device_protocol::kLwt2DSplitScratchTileRows);
            ASSERT(source_tile_columns <= ttwv::device_protocol::kLwt2DSplitScratchTileColumns);

            for (uint32_t source_tile_y = 0; source_tile_y < source_tile_rows; ++source_tile_y) {
                for (uint32_t source_tile_x = 0; source_tile_x < source_tile_columns; ++source_tile_x) {
                    const uint32_t source_tile =
                        (source_tile_y_begin + source_tile_y) * band_tile_columns +
                        source_tile_x_begin + source_tile_x;
                    const uint32_t scratch_tile = source_tile_y * source_tile_columns + source_tile_x;
                    noc_async_read(
                        band.get_noc_addr(source_tile),
                        scratch_addr + scratch_tile * kTileBytes,
                        kTileBytes);
                }
            }
            noc_async_read_barrier();

            auto* destination = reinterpret_cast<volatile tt_l1_ptr float*>(destination_addr);
            for (uint32_t internal_y = internal_y_begin; internal_y < internal_y_end; ++internal_y) {
                const uint32_t canonical_y =
                    static_cast<uint32_t>(static_cast<int32_t>(internal_y) - y_internal_offset);
                const uint32_t source_tile_y = canonical_y / kTileSide - source_tile_y_begin;
                for (uint32_t internal_x = internal_x_begin; internal_x < internal_x_end; ++internal_x) {
                    const uint32_t canonical_x =
                        static_cast<uint32_t>(static_cast<int32_t>(internal_x) - x_internal_offset);
                    const uint32_t source_tile_x = canonical_x / kTileSide - source_tile_x_begin;
                    const uint32_t scratch_tile = source_tile_y * source_tile_columns + source_tile_x;
                    const auto* source =
                        reinterpret_cast<volatile tt_l1_ptr float*>(scratch_addr + scratch_tile * kTileBytes);
                    destination[tile_element_offset(internal_y - tile_y, internal_x - tile_x)] =
                        source[tile_element_offset(canonical_y % kTileSide, canonical_x % kTileSide)];
                }
            }
        }
    }
}

template <typename BandAccessor>
ALWI void initialize_inverse_band_planes(
    const BandAccessor& band_args,
    const uint32_t* band_addrs,
    const uint32_t band_height,
    const uint32_t band_width,
    const uint32_t band_tile_columns,
    const int32_t* y_internal_offsets,
    const int32_t* x_internal_offsets,
    const Rect* rectangles,
    const uint32_t* plane_addrs,
    const uint32_t* plane_tile_columns,
    const uint32_t scratch_addr,
    const uint32_t zero_tile_addr) {
    // LL, LH, HL, HH map to (low-y,low-x), (low-y,high-x),
    // (high-y,low-x), and (high-y,high-x), respectively.
    constexpr uint32_t y_stream[4] = {0, 0, 1, 1};
    constexpr uint32_t x_stream[4] = {0, 1, 0, 1};
    for (uint32_t plane = 0; plane < 4; ++plane) {
        initialize_inverse_band_plane(
            band_args,
            band_addrs[plane],
            band_height,
            band_width,
            band_tile_columns,
            y_internal_offsets[y_stream[plane]],
            x_internal_offsets[x_stream[plane]],
            rectangles[plane],
            plane_addrs[plane],
            plane_tile_columns[plane],
            scratch_addr,
            zero_tile_addr);
    }
}
#endif

#ifdef TTWV_CAPTURE_SPLIT_SNAPSHOTS
template <typename SnapshotAccessor>
ALWI void snapshot_initial_planes(
    const SnapshotAccessor& snapshot_args,
    const uint32_t snapshot_addr,
    const uint32_t snapshot_tiles_per_plane,
    const uint32_t global_chunk,
    const Rect* rectangles,
    const uint32_t* plane_addrs,
    const uint32_t* plane_tile_columns) {
    const auto snapshots = TensorAccessor(snapshot_args, snapshot_addr, kTileBytes);
    for (uint32_t plane = 0; plane < 4; ++plane) {
        const uint32_t tile_rows = (aligned_end(rectangles[plane].y_begin, rectangles[plane].y_length) -
                                    aligned_begin(rectangles[plane].y_begin)) /
                                   kTileSide;
        const uint32_t tile_columns = (aligned_end(rectangles[plane].x_begin, rectangles[plane].x_length) -
                                       aligned_begin(rectangles[plane].x_begin)) /
                                      kTileSide;
        const uint32_t destination_base = (global_chunk * 4 + plane) * snapshot_tiles_per_plane;
        for (uint32_t tile_y = 0; tile_y < tile_rows; ++tile_y) {
            for (uint32_t tile_x = 0; tile_x < tile_columns; ++tile_x) {
                const uint32_t source_tile = tile_y * plane_tile_columns[plane] + tile_x;
                const uint32_t destination_tile = destination_base + tile_y * tile_columns + tile_x;
                noc_async_write(
                    plane_addrs[plane] + source_tile * kTileBytes,
                    snapshots.get_noc_addr(destination_tile),
                    kTileBytes);
            }
        }
    }
    noc_async_write_barrier();
}
#endif

[[nodiscard]] ALWI float read_plane(
    const uint32_t plane_addr,
    const uint32_t plane_tile_columns,
    const Rect& stored,
    const int32_t y,
    const int32_t x) {
    if (!contains(stored, y, x)) {
        return 0.0F;
    }
    const uint32_t local_y = static_cast<uint32_t>(y) - aligned_begin(stored.y_begin);
    const uint32_t local_x = static_cast<uint32_t>(x) - aligned_begin(stored.x_begin);
    const auto* source = reinterpret_cast<volatile tt_l1_ptr float*>(plane_addr);
    return source[tiled_element_offset(local_y, local_x, plane_tile_columns)];
}

enum class RouteTileClass : uint32_t {
    kExact,
    kOneAxisShifted,
    kTwoAxisShifted,
    kPartial,
    kEmpty,
};

#ifdef TTWV_VALIDATE_ROUTE_STAGING
[[nodiscard]] ALWI uint32_t read_plane_bits(
    const uint32_t plane_addr,
    const uint32_t plane_tile_columns,
    const Rect& stored,
    const int32_t y,
    const int32_t x) {
    if (!contains(stored, y, x)) {
        return 0;
    }
    const uint32_t local_y = static_cast<uint32_t>(y) - aligned_begin(stored.y_begin);
    const uint32_t local_x = static_cast<uint32_t>(x) - aligned_begin(stored.x_begin);
    const auto* source = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(plane_addr);
    return source[tiled_element_offset(local_y, local_x, plane_tile_columns)];
}

ALWI void validate_staged_tile(
    const uint32_t destination_addr,
    const uint32_t plane_addr,
    const uint32_t plane_tile_columns,
    const Rect& stored,
    const int32_t requested_y,
    const int32_t requested_x,
    const RouteTileClass tile_class,
    StagingValidationMetrics& metrics) {
    const auto* staged = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(destination_addr);
    ++metrics.validated_tiles;
    for (uint32_t row = 0; row < kTileSide; ++row) {
        for (uint32_t column = 0; column < kTileSide; ++column) {
            const uint32_t expected = read_plane_bits(
                plane_addr,
                plane_tile_columns,
                stored,
                requested_y + static_cast<int32_t>(row),
                requested_x + static_cast<int32_t>(column));
            const bool mismatch = staged[tile_element_offset(row, column)] != expected;
            metrics.mismatched_words += mismatch ? 1U : 0U;
            metrics.class_mismatched_words[static_cast<uint32_t>(tile_class)] += mismatch ? 1U : 0U;
        }
    }
}
#define TTWV_VALIDATE_STAGED_TILE(...) validate_staged_tile(__VA_ARGS__)
#else
#define TTWV_VALIDATE_STAGED_TILE(...) \
    do {                               \
    } while (false)
#endif

ALWI void reserve_tile(const uint32_t cb) {
    cb_reserve_back(cb, 1);
    auto* tile = reinterpret_cast<volatile tt_l1_ptr float*>(get_write_ptr(cb));
    for (uint32_t element = 0; element < kTileElements; ++element) {
        tile[element] = 0.0F;
    }
}

enum class StageTileResult : uint32_t {
    kExactPending,
    kBoundedPending,
    kCompleted,
};

#ifdef TTWV_LWT_2D_COMPUTE_ONLY_BENCHMARK
[[nodiscard]] __attribute__((noinline)) StageTileResult
stage_compute_benchmark_tile(const uint32_t cb, const uint32_t zero_tile_addr) {
    cb_reserve_back(cb, 1);
    noc_async_read(get_noc_addr(zero_tile_addr), get_write_ptr(cb), kTileBytes);
    return StageTileResult::kExactPending;
}
#endif

[[nodiscard]] ALWI bool requested_tile_inside(
    const Rect& stored, const int32_t requested_y, const int32_t requested_x) {
    return requested_y >= static_cast<int32_t>(stored.y_begin) && requested_x >= static_cast<int32_t>(stored.x_begin) &&
           requested_y + static_cast<int32_t>(kTileSide) <= static_cast<int32_t>(stored.y_begin + stored.y_length) &&
           requested_x + static_cast<int32_t>(kTileSide) <= static_cast<int32_t>(stored.x_begin + stored.x_length);
}

[[nodiscard]] ALWI RouteTileClass
classify_route_tile(const Rect& stored, const int32_t requested_y, const int32_t requested_x) {
    if (!requested_tile_inside(stored, requested_y, requested_x)) {
        const int32_t requested_y_end = requested_y + static_cast<int32_t>(kTileSide);
        const int32_t requested_x_end = requested_x + static_cast<int32_t>(kTileSide);
        const int32_t stored_y_end = static_cast<int32_t>(stored.y_begin + stored.y_length);
        const int32_t stored_x_end = static_cast<int32_t>(stored.x_begin + stored.x_length);
        const bool intersects = requested_y < stored_y_end && requested_y_end > static_cast<int32_t>(stored.y_begin) &&
                                requested_x < stored_x_end && requested_x_end > static_cast<int32_t>(stored.x_begin);
        return intersects ? RouteTileClass::kPartial : RouteTileClass::kEmpty;
    }
    const bool y_aligned =
        (requested_y - static_cast<int32_t>(aligned_begin(stored.y_begin))) % static_cast<int32_t>(kTileSide) == 0;
    const bool x_aligned =
        (requested_x - static_cast<int32_t>(aligned_begin(stored.x_begin))) % static_cast<int32_t>(kTileSide) == 0;
    if (y_aligned && x_aligned) {
        return RouteTileClass::kExact;
    }
    if (y_aligned != x_aligned) {
        return RouteTileClass::kOneAxisShifted;
    }
    return RouteTileClass::kTwoAxisShifted;
}

[[nodiscard]] ALWI uint32_t route_plane_tile_addr(
    const uint32_t plane_addr,
    const uint32_t plane_tile_columns,
    const Rect& stored,
    const int32_t requested_y,
    const int32_t requested_x) {
    const uint32_t local_y = static_cast<uint32_t>(requested_y) - aligned_begin(stored.y_begin);
    const uint32_t local_x = static_cast<uint32_t>(requested_x) - aligned_begin(stored.x_begin);
    const uint32_t tile_index = (local_y / kTileSide) * plane_tile_columns + local_x / kTileSide;
    return plane_addr + tile_index * kTileBytes;
}

ALWI void copy_contiguous_words(
    volatile tt_l1_ptr uint32_t* destination, const volatile tt_l1_ptr uint32_t* source, const uint32_t word_count) {
    for (uint32_t word = 0; word < word_count; ++word) {
        destination[word] = source[word];
    }
}

ALWI void assemble_one_axis_shifted_tile(
    const uint32_t destination_addr,
    const uint32_t plane_addr,
    const uint32_t plane_tile_columns,
    const Rect& stored,
    const int32_t requested_y,
    const int32_t requested_x) {
    auto* destination = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(destination_addr);
    const auto* source = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(plane_addr);
    const uint32_t stored_y_origin = aligned_begin(stored.y_begin);
    const uint32_t stored_x_origin = aligned_begin(stored.x_begin);
    for (uint32_t row = 0; row < kTileSide; ++row) {
        uint32_t column = 0;
        while (column < kTileSide) {
            const uint32_t source_y = static_cast<uint32_t>(requested_y) + row - stored_y_origin;
            const uint32_t source_x = static_cast<uint32_t>(requested_x) + column - stored_x_origin;
            const uint32_t count = std::min(
                kTileSide - column, std::min(kFaceSide - source_x % kFaceSide, kFaceSide - column % kFaceSide));
            const uint32_t source_offset = tiled_element_offset(source_y, source_x, plane_tile_columns);
            const uint32_t destination_offset = tile_element_offset(row, column);
            copy_contiguous_words(destination + destination_offset, source + source_offset, count);
            column += count;
        }
    }
}

__attribute__((noinline)) void assemble_bounded_tile(
    const uint32_t destination_addr,
    const uint32_t plane_addr,
    const uint32_t plane_tile_columns,
    const Rect& stored,
    const int32_t requested_y,
    const int32_t requested_x) {
    const int32_t valid_y_begin = std::max(requested_y, static_cast<int32_t>(stored.y_begin));
    const int32_t valid_y_end = std::min(
        requested_y + static_cast<int32_t>(kTileSide),
        static_cast<int32_t>(stored.y_begin + stored.y_length));
    const int32_t valid_x_begin = std::max(requested_x, static_cast<int32_t>(stored.x_begin));
    const int32_t valid_x_end = std::min(
        requested_x + static_cast<int32_t>(kTileSide),
        static_cast<int32_t>(stored.x_begin + stored.x_length));
    if (valid_y_begin >= valid_y_end || valid_x_begin >= valid_x_end) {
        return;
    }

    auto* destination = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(destination_addr);
    const auto* source = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(plane_addr);
    const uint32_t stored_y_origin = aligned_begin(stored.y_begin);
    const uint32_t stored_x_origin = aligned_begin(stored.x_begin);
    const uint32_t destination_row_begin = static_cast<uint32_t>(valid_y_begin - requested_y);
    const uint32_t destination_column_begin = static_cast<uint32_t>(valid_x_begin - requested_x);
    const uint32_t valid_width = static_cast<uint32_t>(valid_x_end - valid_x_begin);

    for (uint32_t destination_row = destination_row_begin;
         destination_row < destination_row_begin + static_cast<uint32_t>(valid_y_end - valid_y_begin);
         ++destination_row) {
        const uint32_t source_y =
            static_cast<uint32_t>(valid_y_begin - static_cast<int32_t>(stored_y_origin)) +
            destination_row - destination_row_begin;
        uint32_t copied = 0;
        while (copied < valid_width) {
            const uint32_t destination_column = destination_column_begin + copied;
            const uint32_t source_x =
                static_cast<uint32_t>(valid_x_begin - static_cast<int32_t>(stored_x_origin)) + copied;
            const uint32_t count = std::min(
                valid_width - copied,
                std::min(kFaceSide - source_x % kFaceSide, kFaceSide - destination_column % kFaceSide));
            copy_contiguous_words(
                destination + tile_element_offset(destination_row, destination_column),
                source + tiled_element_offset(source_y, source_x, plane_tile_columns),
                count);
            copied += count;
        }
    }
}

ALWI void count_route_tile(RouteStagingMetrics& metrics, const RouteTileClass tile_class, const bool base_tile) {
    if (base_tile) {
        if (tile_class == RouteTileClass::kExact) {
            TTWV_TRANSPORT_METRIC(++metrics.exact_base_tiles);
        } else if (tile_class == RouteTileClass::kOneAxisShifted) {
            TTWV_TRANSPORT_METRIC(++metrics.shifted_base_tiles);
        } else {
            TTWV_TRANSPORT_METRIC(++metrics.generic_base_tiles);
        }
    } else {
        if (tile_class == RouteTileClass::kExact) {
            TTWV_TRANSPORT_METRIC(++metrics.exact_source_tiles);
        } else if (tile_class == RouteTileClass::kOneAxisShifted) {
            TTWV_TRANSPORT_METRIC(++metrics.shifted_source_tiles);
        } else {
            TTWV_TRANSPORT_METRIC(++metrics.generic_source_tiles);
        }
    }
}

[[nodiscard]] __attribute__((noinline)) StageTileResult stage_optimized_tile(
    const uint32_t cb,
    const uint32_t zero_tile_addr,
    const uint32_t plane_addr,
    const uint32_t plane_tile_columns,
    const Rect& stored,
    const int32_t requested_y,
    const int32_t requested_x,
    const bool base_tile,
    RouteStagingMetrics& metrics TTWV_STAGING_VALIDATION_PARAMETER) {
    const RouteTileClass tile_class = classify_route_tile(stored, requested_y, requested_x);
    count_route_tile(metrics, tile_class, base_tile);
    cb_reserve_back(cb, 1);
    const uint32_t destination_addr = get_write_ptr(cb);
    if (tile_class == RouteTileClass::kExact) {
        const uint32_t source_addr =
            route_plane_tile_addr(plane_addr, plane_tile_columns, stored, requested_y, requested_x);
#ifdef TTWV_LWT_2D_EXACT_L1_COPY
        copy_contiguous_words(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(destination_addr),
            reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(source_addr),
            kTileBytes / sizeof(uint32_t));
        TTWV_VALIDATE_STAGED_TILE(
            destination_addr,
            plane_addr,
            plane_tile_columns,
            stored,
            requested_y,
            requested_x,
            tile_class,
            validation_metrics);
        cb_push_back(cb, 1);
        return StageTileResult::kCompleted;
#else
        noc_async_read(get_noc_addr(source_addr), destination_addr, kTileBytes);
        return StageTileResult::kExactPending;
#endif
    }

    if (tile_class == RouteTileClass::kPartial || tile_class == RouteTileClass::kEmpty) {
        noc_async_read(get_noc_addr(zero_tile_addr), destination_addr, kTileBytes);
        return StageTileResult::kBoundedPending;
    }

    if (tile_class == RouteTileClass::kOneAxisShifted) {
        assemble_one_axis_shifted_tile(
            destination_addr, plane_addr, plane_tile_columns, stored, requested_y, requested_x);
    } else {
        assemble_bounded_tile(destination_addr, plane_addr, plane_tile_columns, stored, requested_y, requested_x);
    }
    TTWV_VALIDATE_STAGED_TILE(
        destination_addr,
        plane_addr,
        plane_tile_columns,
        stored,
        requested_y,
        requested_x,
        tile_class,
        validation_metrics);
    cb_push_back(cb, 1);
    return StageTileResult::kCompleted;
}

__attribute__((noinline)) void finish_pending_tile(
    const StageTileResult result,
    const uint32_t cb,
    const uint32_t plane_addr,
    const uint32_t plane_tile_columns,
    const Rect& stored,
    const int32_t requested_y,
    const int32_t requested_x TTWV_STAGING_VALIDATION_PARAMETER) {
    if (result == StageTileResult::kCompleted) {
        return;
    }
    const uint32_t destination_addr = get_write_ptr(cb);
    if (result == StageTileResult::kBoundedPending) {
        assemble_bounded_tile(destination_addr, plane_addr, plane_tile_columns, stored, requested_y, requested_x);
    }
    TTWV_VALIDATE_STAGED_TILE(
        destination_addr,
        plane_addr,
        plane_tile_columns,
        stored,
        requested_y,
        requested_x,
        classify_route_tile(stored, requested_y, requested_x),
        validation_metrics);
    cb_push_back(cb, 1);
}

[[nodiscard]] ALWI int32_t base_requested_y(const Rect& source, const Rect& output, const uint32_t output_tile_y) {
    const int32_t output_y_origin = static_cast<int32_t>(aligned_begin(output.y_begin) + output_tile_y * kTileSide);
    return static_cast<int32_t>(source.y_begin) + output_y_origin - static_cast<int32_t>(output.y_begin);
}

[[nodiscard]] ALWI int32_t base_requested_x(const Rect& source, const Rect& output, const uint32_t output_tile_x) {
    const int32_t output_x_origin = static_cast<int32_t>(aligned_begin(output.x_begin) + output_tile_x * kTileSide);
    return static_cast<int32_t>(source.x_begin) + output_x_origin - static_cast<int32_t>(output.x_begin);
}

ALWI void stencil_requested_origin(
    const bool vertical,
    const uint32_t source_tile_index,
    const uint32_t coefficient_count,
    const Rect& source,
    const Rect& output,
    const uint32_t output_tile_y,
    const uint32_t output_tile_x,
    int32_t& requested_y,
    int32_t& requested_x) {
    requested_y = base_requested_y(source, output, output_tile_y);
    requested_x = base_requested_x(source, output, output_tile_x);
    if (vertical) {
        requested_y += static_cast<int32_t>(source_tile_index * kTileSide);
    } else {
        requested_x +=
            static_cast<int32_t>(source_tile_index * kTileSide) - static_cast<int32_t>(17 - coefficient_count);
    }
}

ALWI void fill_base_or_scale_tile(
    const uint32_t cb,
    const uint32_t plane_addr,
    const uint32_t plane_tile_columns,
    const Rect& stored,
    const Rect& source,
    const Rect& output,
    const uint32_t output_tile_y,
    const uint32_t output_tile_x) {
    reserve_tile(cb);
    auto* tile = reinterpret_cast<volatile tt_l1_ptr float*>(get_write_ptr(cb));
    const int32_t output_y_origin = static_cast<int32_t>(aligned_begin(output.y_begin) + output_tile_y * kTileSide);
    const int32_t output_x_origin = static_cast<int32_t>(aligned_begin(output.x_begin) + output_tile_x * kTileSide);
    for (uint32_t row = 0; row < kTileSide; ++row) {
        for (uint32_t column = 0; column < kTileSide; ++column) {
            const int32_t output_y = output_y_origin + row;
            const int32_t output_x = output_x_origin + column;
            const int32_t source_y =
                static_cast<int32_t>(source.y_begin) + output_y - static_cast<int32_t>(output.y_begin);
            const int32_t source_x =
                static_cast<int32_t>(source.x_begin) + output_x - static_cast<int32_t>(output.x_begin);
            tile[tile_element_offset(row, column)] =
                read_plane(plane_addr, plane_tile_columns, stored, source_y, source_x);
        }
    }
    cb_push_back(cb, 1);
}

ALWI void fill_stencil_source_tile(
    const uint32_t cb,
    const bool vertical,
    const uint32_t source_tile_index,
    const uint32_t coefficient_count,
    const uint32_t plane_addr,
    const uint32_t plane_tile_columns,
    const Rect& stored,
    const Rect& source,
    const Rect& output,
    const uint32_t output_tile_y,
    const uint32_t output_tile_x) {
    reserve_tile(cb);
    auto* tile = reinterpret_cast<volatile tt_l1_ptr float*>(get_write_ptr(cb));
    const int32_t output_y_origin = static_cast<int32_t>(aligned_begin(output.y_begin) + output_tile_y * kTileSide);
    const int32_t output_x_origin = static_cast<int32_t>(aligned_begin(output.x_begin) + output_tile_x * kTileSide);
    const int32_t alignment = static_cast<int32_t>(17 - coefficient_count);
    for (uint32_t row = 0; row < kTileSide; ++row) {
        for (uint32_t column = 0; column < kTileSide; ++column) {
            int32_t source_y = 0;
            int32_t source_x = 0;
            if (vertical) {
                source_y = static_cast<int32_t>(source.y_begin) + output_y_origin -
                           static_cast<int32_t>(output.y_begin) +
                           static_cast<int32_t>(source_tile_index * kTileSide + row);
                source_x = static_cast<int32_t>(source.x_begin) + output_x_origin + static_cast<int32_t>(column) -
                           static_cast<int32_t>(output.x_begin);
            } else {
                source_y = static_cast<int32_t>(source.y_begin) + output_y_origin + static_cast<int32_t>(row) -
                           static_cast<int32_t>(output.y_begin);
                source_x = static_cast<int32_t>(source.x_begin) + output_x_origin -
                           static_cast<int32_t>(output.x_begin) +
                           static_cast<int32_t>(source_tile_index * kTileSide + column) - alignment;
            }
            tile[tile_element_offset(row, column)] =
                read_plane(plane_addr, plane_tile_columns, stored, source_y, source_x);
        }
    }
    cb_push_back(cb, 1);
}

#ifdef TTWV_CAPTURE_TRANSPORT_METRICS
template <typename MetricAccessor>
ALWI void write_transport_reader_half(
    const MetricAccessor& metric_args,
    const uint32_t metrics_addr,
    const uint32_t metric_pages_per_chunk,
    const uint32_t global_chunk,
    const uint32_t page_in_chunk,
    const uint32_t scratch_addr,
    const uint64_t config_cycles,
    const uint64_t staging_cycles,
    const uint64_t sync_wait_cycles,
    const uint32_t output_tiles,
    const uint32_t axis,
    const uint32_t step_type,
    const uint32_t coefficient_count,
    const RouteStagingMetrics& staging_metrics) {
    auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_addr);
    for (uint32_t word = 0; word < ttwv::device_protocol::kLwt2DTransportMetricWordCount / 2; ++word) {
        words[word] = 0;
    }
    words[ttwv::device_protocol::kLwt2DTransportMetricReaderConfigCyclesLow] = static_cast<uint32_t>(config_cycles);
    words[ttwv::device_protocol::kLwt2DTransportMetricReaderConfigCyclesHigh] =
        static_cast<uint32_t>(config_cycles >> 32);
    words[ttwv::device_protocol::kLwt2DTransportMetricStagingCyclesLow] = static_cast<uint32_t>(staging_cycles);
    words[ttwv::device_protocol::kLwt2DTransportMetricStagingCyclesHigh] = static_cast<uint32_t>(staging_cycles >> 32);
    words[ttwv::device_protocol::kLwt2DTransportMetricSyncWaitCyclesLow] = static_cast<uint32_t>(sync_wait_cycles);
    words[ttwv::device_protocol::kLwt2DTransportMetricSyncWaitCyclesHigh] =
        static_cast<uint32_t>(sync_wait_cycles >> 32);
    words[ttwv::device_protocol::kLwt2DTransportMetricOutputTiles] = output_tiles;
    words[ttwv::device_protocol::kLwt2DTransportMetricAxis] = axis;
    words[ttwv::device_protocol::kLwt2DTransportMetricStepType] = step_type;
    words[ttwv::device_protocol::kLwt2DTransportMetricCoefficientCount] = coefficient_count;
    words[ttwv::device_protocol::kLwt2DTransportMetricExactSourceTiles] = staging_metrics.exact_source_tiles;
    words[ttwv::device_protocol::kLwt2DTransportMetricShiftedSourceTiles] = staging_metrics.shifted_source_tiles;
    words[ttwv::device_protocol::kLwt2DTransportMetricGenericSourceTiles] = staging_metrics.generic_source_tiles;
    words[ttwv::device_protocol::kLwt2DTransportMetricExactBaseTiles] = staging_metrics.exact_base_tiles;
    words[ttwv::device_protocol::kLwt2DTransportMetricShiftedBaseTiles] = staging_metrics.shifted_base_tiles;
    words[ttwv::device_protocol::kLwt2DTransportMetricGenericBaseTiles] = staging_metrics.generic_base_tiles;
    const auto metrics =
        TensorAccessor(metric_args, metrics_addr, ttwv::device_protocol::kLwt2DTransportMetricPageBytes);
    const uint32_t page = global_chunk * metric_pages_per_chunk + page_in_chunk;
    noc_async_write(
        scratch_addr, metrics.get_noc_addr(page), ttwv::device_protocol::kLwt2DTransportMetricHalfPageBytes);
    noc_async_write_barrier();
}
#endif

#ifdef TTWV_VALIDATE_ROUTE_STAGING
template <typename MetricAccessor>
ALWI void write_staging_validation_summary(
    const MetricAccessor& metric_args,
    const uint32_t metrics_addr,
    const uint32_t metric_pages_per_chunk,
    const uint32_t global_chunk,
    const uint32_t page_in_chunk,
    const uint32_t scratch_addr,
    const StagingValidationMetrics& validation) {
    auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_addr);
    for (uint32_t word = 0; word < ttwv::device_protocol::kLwt2DTransportMetricWordCount / 2; ++word) {
        words[word] = 0;
    }
    words[ttwv::device_protocol::kLwt2DTransportMetricValidatedStagingTiles] = validation.validated_tiles;
    words[ttwv::device_protocol::kLwt2DTransportMetricStagingValidationMismatches] = validation.mismatched_words;
    words[ttwv::device_protocol::kLwt2DTransportMetricValidationExactMismatches] =
        validation.class_mismatched_words[static_cast<uint32_t>(RouteTileClass::kExact)];
    words[ttwv::device_protocol::kLwt2DTransportMetricValidationShiftedMismatches] =
        validation.class_mismatched_words[static_cast<uint32_t>(RouteTileClass::kOneAxisShifted)];
    words[ttwv::device_protocol::kLwt2DTransportMetricValidationTwoAxisMismatches] =
        validation.class_mismatched_words[static_cast<uint32_t>(RouteTileClass::kTwoAxisShifted)];
    words[ttwv::device_protocol::kLwt2DTransportMetricValidationPartialMismatches] =
        validation.class_mismatched_words[static_cast<uint32_t>(RouteTileClass::kPartial)];
    words[ttwv::device_protocol::kLwt2DTransportMetricValidationEmptyMismatches] =
        validation.class_mismatched_words[static_cast<uint32_t>(RouteTileClass::kEmpty)];
    const auto metrics =
        TensorAccessor(metric_args, metrics_addr, ttwv::device_protocol::kLwt2DTransportMetricPageBytes);
    const uint32_t page = global_chunk * metric_pages_per_chunk + page_in_chunk;
    noc_async_write(
        scratch_addr, metrics.get_noc_addr(page), ttwv::device_protocol::kLwt2DTransportMetricHalfPageBytes);
    noc_async_write_barrier();
}
#endif

}  // namespace

void kernel_main() {
#ifdef TTWV_ILWT_2D
    uint32_t band_addrs[ttwv::device_protocol::kLwt2DBandCount];
    for (uint32_t band = 0; band < ttwv::device_protocol::kLwt2DBandCount; ++band) {
        band_addrs[band] = get_arg_val<uint32_t>(band);
    }
    const uint32_t input_height = get_arg_val<uint32_t>(4);
    const uint32_t input_width = get_arg_val<uint32_t>(5);
    const uint32_t input_tile_columns = get_arg_val<uint32_t>(6);
    const int32_t y_internal_offsets[2] = {
        static_cast<int32_t>(get_arg_val<uint32_t>(7)),
        static_cast<int32_t>(get_arg_val<uint32_t>(8)),
    };
    const int32_t x_internal_offsets[2] = {
        static_cast<int32_t>(get_arg_val<uint32_t>(9)),
        static_cast<int32_t>(get_arg_val<uint32_t>(10)),
    };
    constexpr uint32_t plane_arg_base = 11;
#else
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t input_height = get_arg_val<uint32_t>(1);
    const uint32_t input_width = get_arg_val<uint32_t>(2);
    const uint32_t input_tile_columns = get_arg_val<uint32_t>(3);
    const uint32_t pad_y = get_arg_val<uint32_t>(4);
    const uint32_t pad_x = get_arg_val<uint32_t>(5);
    constexpr uint32_t plane_arg_base = 6;
#endif
    uint32_t plane_addrs[ttwv::device_protocol::kLwt2DPlaneCount];
    uint32_t plane_tile_columns[ttwv::device_protocol::kLwt2DPlaneCount];
    for (uint32_t slot = 0; slot < ttwv::device_protocol::kLwt2DPlaneCount; ++slot) {
        plane_addrs[slot] = get_arg_val<uint32_t>(plane_arg_base + slot);
        plane_tile_columns[slot] =
            get_arg_val<uint32_t>(plane_arg_base + ttwv::device_protocol::kLwt2DPlaneCount + slot);
    }
    constexpr uint32_t plane_arg_count = 2 * ttwv::device_protocol::kLwt2DPlaneCount;
    const uint32_t chunk_config_addr = get_arg_val<uint32_t>(plane_arg_base + plane_arg_count);
    const uint32_t route_config_addr = get_arg_val<uint32_t>(plane_arg_base + plane_arg_count + 1);
    const uint32_t chunk_begin = get_arg_val<uint32_t>(plane_arg_base + plane_arg_count + 2);
    const uint32_t chunk_count = get_arg_val<uint32_t>(plane_arg_base + plane_arg_count + 3);
    const uint32_t route_count = get_arg_val<uint32_t>(plane_arg_base + plane_arg_count + 4);
#ifdef TTWV_CAPTURE_SPLIT_METRICS
    const uint32_t split_metrics_addr = get_arg_val<uint32_t>(plane_arg_base + plane_arg_count + 5);
    const bool capture_split_metrics = get_arg_val<uint32_t>(plane_arg_base + plane_arg_count + 6) != 0;
#endif
#ifdef TTWV_CAPTURE_SPLIT_SNAPSHOTS
    const uint32_t split_snapshot_addr = get_arg_val<uint32_t>(plane_arg_base + plane_arg_count + 7);
    const uint32_t split_snapshot_tiles_per_plane = get_arg_val<uint32_t>(plane_arg_base + plane_arg_count + 8);
#endif
#if defined(TTWV_CAPTURE_TRANSPORT_METRICS) || defined(TTWV_VALIDATE_ROUTE_STAGING)
    const uint32_t transport_metrics_addr = get_arg_val<uint32_t>(plane_arg_base + plane_arg_count + 9);
    const uint32_t transport_metric_pages_per_chunk = get_arg_val<uint32_t>(plane_arg_base + plane_arg_count + 10);
#endif

    constexpr uint32_t cb_source0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_source1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_base = get_compile_time_arg_val(2);
    constexpr uint32_t cb_sync = get_compile_time_arg_val(3);
    constexpr uint32_t cb_chunk_config = get_compile_time_arg_val(4);
    constexpr uint32_t cb_route_config = get_compile_time_arg_val(5);
    constexpr uint32_t cb_noc_scratch = get_compile_time_arg_val(6);
    constexpr uint32_t cb_route_zero = get_compile_time_arg_val(7);
    constexpr auto input_args = TensorAccessorArgs<8>();
    constexpr auto chunk_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto route_args = TensorAccessorArgs<chunk_args.next_compile_time_args_offset()>();
    constexpr auto metric_args = TensorAccessorArgs<route_args.next_compile_time_args_offset()>();
    constexpr auto snapshot_args = TensorAccessorArgs<metric_args.next_compile_time_args_offset()>();
    constexpr auto transport_metric_args = TensorAccessorArgs<snapshot_args.next_compile_time_args_offset()>();
    constexpr uint32_t boundary_mode_arg_offset = transport_metric_args.next_compile_time_args_offset();
    constexpr auto boundary_mode =
        static_cast<ttwv::BoundaryMode>(get_compile_time_arg_val(boundary_mode_arg_offset));
    constexpr uint32_t split_scratch_bytes = get_compile_time_arg_val(boundary_mode_arg_offset + 1);
    static_assert(ttwv::is_supported_lwt_boundary_mode(boundary_mode), "Unsupported 2D signal-extension mode");
#ifndef TTWV_ILWT_2D
    const auto input = TensorAccessor(input_args, input_addr, kTileBytes);
#endif
    cb_reserve_back(cb_route_zero, 1);
    const uint32_t zero_tile_addr = get_write_ptr(cb_route_zero);
    auto* zero_tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero_tile_addr);
    for (uint32_t word = 0; word < kTileElements; ++word) {
        zero_tile[word] = 0;
    }
    cb_push_back(cb_route_zero, 1);
    const uint32_t noc_scratch_raw = get_write_ptr(cb_noc_scratch);
#if defined(TTWV_LWT_2D_TILED_SPLIT) || defined(TTWV_ILWT_2D)
    const uint32_t noc_scratch_addr = noc_scratch_raw;
#else
    const uint32_t noc_scratch_addr = (noc_scratch_raw + 63U) & ~63U;
#endif
#ifdef TTWV_LWT_2D_PRELOAD_ROUTE_CONFIG
    constexpr uint32_t reader_config_capacity =
        split_scratch_bytes / 2 - ttwv::device_protocol::kLwt2DTransportMetricPageBytes;
    const uint32_t reader_config_addr = noc_scratch_addr;
#endif
#if defined(TTWV_CAPTURE_TRANSPORT_METRICS) || defined(TTWV_VALIDATE_ROUTE_STAGING)
    const uint32_t reader_metric_scratch_addr = noc_scratch_addr + split_scratch_bytes / 2 -
                                                ttwv::device_protocol::kLwt2DTransportMetricPageBytes;
#endif

    for (uint32_t local_chunk = 0; local_chunk < chunk_count; ++local_chunk) {
        const uint32_t global_chunk = chunk_begin + local_chunk;
#ifdef TTWV_CAPTURE_TRANSPORT_METRICS
        const uint64_t chunk_start = get_timestamp();
        uint64_t preload_config_cycles = 0;
#endif
#ifdef TTWV_VALIDATE_ROUTE_STAGING
        StagingValidationMetrics staging_validation_metrics{};
#endif
        uint32_t chunk_words[ttwv::device_protocol::kLwt2DChunkConfigWordCount];
        load_config_page(
            chunk_args,
            chunk_config_addr,
            ttwv::device_protocol::kLwt2DChunkConfigPageBytes,
            global_chunk,
            cb_chunk_config,
            chunk_words,
            ttwv::device_protocol::kLwt2DChunkConfigWordCount);

        Rect stored[ttwv::device_protocol::kLwt2DPlaneCount];
        stored[0] = load_rect(chunk_words, ttwv::device_protocol::kLwt2DInitialEe);
        stored[1] = load_rect(chunk_words, ttwv::device_protocol::kLwt2DInitialEo);
        stored[2] = load_rect(chunk_words, ttwv::device_protocol::kLwt2DInitialOe);
        stored[3] = load_rect(chunk_words, ttwv::device_protocol::kLwt2DInitialOo);
        stored[4] = Rect{};
        SplitMetrics split_metrics{};
#ifdef TTWV_CAPTURE_SPLIT_METRICS
        const uint64_t split_start = get_timestamp();
#endif

#ifdef TTWV_ILWT_2D
        initialize_inverse_band_planes(
            input_args,
            band_addrs,
            input_height,
            input_width,
            input_tile_columns,
            y_internal_offsets,
            x_internal_offsets,
            stored,
            plane_addrs,
            plane_tile_columns,
            noc_scratch_addr,
            zero_tile_addr);
#elif defined(TTWV_LWT_2D_TILED_SPLIT)
        initialize_planes_tiled<boundary_mode>(
            input,
            input_height,
            input_width,
            input_tile_columns,
            pad_y,
            pad_x,
            stored,
            plane_addrs,
            plane_tile_columns,
            noc_scratch_addr,
            split_metrics);
#else
        initialize_plane<boundary_mode>(
            input,
            input_height,
            input_width,
            input_tile_columns,
            pad_y,
            pad_x,
            0,
            0,
            stored[0],
            plane_addrs[0],
            plane_tile_columns[0],
            noc_scratch_addr,
            split_metrics);
        initialize_plane<boundary_mode>(
            input,
            input_height,
            input_width,
            input_tile_columns,
            pad_y,
            pad_x,
            0,
            1,
            stored[1],
            plane_addrs[1],
            plane_tile_columns[1],
            noc_scratch_addr,
            split_metrics);
        initialize_plane<boundary_mode>(
            input,
            input_height,
            input_width,
            input_tile_columns,
            pad_y,
            pad_x,
            1,
            0,
            stored[2],
            plane_addrs[2],
            plane_tile_columns[2],
            noc_scratch_addr,
            split_metrics);
        initialize_plane<boundary_mode>(
            input,
            input_height,
            input_width,
            input_tile_columns,
            pad_y,
            pad_x,
            1,
            1,
            stored[3],
            plane_addrs[3],
            plane_tile_columns[3],
            noc_scratch_addr,
            split_metrics);
#endif
        // The route-zero page is intentionally persistent, but a worker may
        // execute multiple chunks while the split stage reuses nearby L1
        // scratch. Refresh it once per chunk so bounded empty/partial pages
        // never inherit stale words from an earlier chunk.
        for (uint32_t word = 0; word < kTileElements; ++word) {
            zero_tile[word] = 0;
        }
#ifdef TTWV_CAPTURE_SPLIT_METRICS
        const uint64_t split_cycles = get_timestamp() - split_start;
        if (capture_split_metrics) {
            auto* metric_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(noc_scratch_addr);
            for (uint32_t word = 0; word < ttwv::device_protocol::kLwt2DSplitMetricWordCount; ++word) {
                metric_words[word] = 0;
            }
            metric_words[ttwv::device_protocol::kLwt2DSplitMetricCyclesLow] = static_cast<uint32_t>(split_cycles);
            metric_words[ttwv::device_protocol::kLwt2DSplitMetricCyclesHigh] =
                static_cast<uint32_t>(split_cycles >> 32);
            metric_words[ttwv::device_protocol::kLwt2DSplitMetricInputBytes] = split_metrics.input_bytes;
            metric_words[ttwv::device_protocol::kLwt2DSplitMetricLocalOutputBytes] = split_metrics.local_output_bytes;
            metric_words[ttwv::device_protocol::kLwt2DSplitMetricNocReadCalls] = split_metrics.noc_read_calls;
            metric_words[ttwv::device_protocol::kLwt2DSplitMetricNocReadBarriers] = split_metrics.noc_read_barriers;
            metric_words[ttwv::device_protocol::kLwt2DSplitMetricInteriorMacroTiles] =
                split_metrics.interior_macro_tiles;
            metric_words[ttwv::device_protocol::kLwt2DSplitMetricBoundaryMacroTiles] =
                split_metrics.boundary_macro_tiles;
            const auto metric_pages =
                TensorAccessor(metric_args, split_metrics_addr, ttwv::device_protocol::kLwt2DSplitMetricPageBytes);
            noc_async_write(
                noc_scratch_addr,
                metric_pages.get_noc_addr(global_chunk),
                ttwv::device_protocol::kLwt2DSplitMetricPageBytes);
            noc_async_write_barrier();
        }
#endif
#ifdef TTWV_CAPTURE_SPLIT_SNAPSHOTS
        snapshot_initial_planes(
            snapshot_args,
            split_snapshot_addr,
            split_snapshot_tiles_per_plane,
            global_chunk,
            stored,
            plane_addrs,
            plane_tile_columns);
#endif
#ifdef TTWV_LWT_2D_SPLIT_ONLY_BENCHMARK
        continue;
#endif
#ifdef TTWV_LWT_2D_PRELOAD_ROUTE_CONFIG
        ASSERT(route_count * ttwv::device_protocol::kLwt2DRouteConfigPageBytes <= reader_config_capacity);
#ifdef TTWV_CAPTURE_TRANSPORT_METRICS
        const uint64_t preload_config_start = get_timestamp();
#endif
        preload_config_pages(
            route_args,
            route_config_addr,
            ttwv::device_protocol::kLwt2DRouteConfigPageBytes,
            global_chunk * route_count,
            route_count,
            reader_config_addr);
#ifdef TTWV_CAPTURE_TRANSPORT_METRICS
        preload_config_cycles = get_timestamp() - preload_config_start;
#endif
#endif
        for (uint32_t route_index = 0; route_index < route_count; ++route_index) {
#ifdef TTWV_LWT_2D_PRELOAD_ROUTE_CONFIG
            const auto* route_words = reinterpret_cast<const uint32_t*>(
                reader_config_addr + route_index * ttwv::device_protocol::kLwt2DRouteConfigPageBytes);
#ifdef TTWV_CAPTURE_TRANSPORT_METRICS
            constexpr uint64_t config_cycles = 0;
#endif
#else
            uint32_t route_words[ttwv::device_protocol::kLwt2DRouteConfigWordCount];
#ifdef TTWV_CAPTURE_TRANSPORT_METRICS
            const uint64_t config_start = get_timestamp();
#endif
            load_config_page(
                route_args,
                route_config_addr,
                ttwv::device_protocol::kLwt2DRouteConfigPageBytes,
                global_chunk * route_count + route_index,
                cb_route_config,
                route_words,
                ttwv::device_protocol::kLwt2DRouteConfigWordCount);
#ifdef TTWV_CAPTURE_TRANSPORT_METRICS
            const uint64_t config_cycles = get_timestamp() - config_start;
#endif
#endif
            const uint32_t flags = route_words[ttwv::device_protocol::kLwt2DRouteFlags];
            if ((flags & ttwv::device_protocol::kLwt2DRouteFlagMetadataOnly) != 0) {
                continue;
            }
            const bool vertical = route_words[ttwv::device_protocol::kLwt2DRouteAxis] == 0;
            const uint32_t source_slot = route_words[ttwv::device_protocol::kLwt2DRouteSourceSlot];
            const uint32_t base_slot = route_words[ttwv::device_protocol::kLwt2DRouteBaseSlot];
            const uint32_t output_slot = route_words[ttwv::device_protocol::kLwt2DRouteOutputSlot];
            const Rect source = load_rect(route_words, ttwv::device_protocol::kLwt2DRouteSourceRect);
            const Rect base = load_rect(route_words, ttwv::device_protocol::kLwt2DRouteBaseRect);
            const Rect output = load_rect(route_words, ttwv::device_protocol::kLwt2DRouteOutputRect);
            const uint32_t output_tile_rows =
                (aligned_end(output.y_begin, output.y_length) - aligned_begin(output.y_begin)) / kTileSide;
            const uint32_t output_tile_columns =
                (aligned_end(output.x_begin, output.x_length) - aligned_begin(output.x_begin)) / kTileSide;
            const bool scale = (flags & ttwv::device_protocol::kLwt2DRouteFlagScale) != 0;
            const uint32_t coefficient_count =
                scale ? 1 : (vertical ? source.y_length - output.y_length + 1 : source.x_length - output.x_length + 1);
            const uint32_t output_tile_count = output_tile_rows * output_tile_columns;
            RouteStagingMetrics route_staging_metrics{};
#ifdef TTWV_CAPTURE_TRANSPORT_METRICS
            const uint64_t staging_start = get_timestamp();
#endif

            for (uint32_t tile_y = 0; tile_y < output_tile_rows; ++tile_y) {
                for (uint32_t tile_x = 0; tile_x < output_tile_columns; ++tile_x) {
#ifdef TTWV_LWT_2D_OPTIMIZED_ROUTE_STAGING
                    int32_t source0_requested_y = 0;
                    int32_t source0_requested_x = 0;
                    int32_t source1_requested_y = 0;
                    int32_t source1_requested_x = 0;
                    int32_t base_requested_tile_y = 0;
                    int32_t base_requested_tile_x = 0;
                    StageTileResult source0_result = StageTileResult::kCompleted;
                    StageTileResult source1_result = StageTileResult::kCompleted;
                    StageTileResult base_result = StageTileResult::kCompleted;
#ifdef TTWV_LWT_2D_COMPUTE_ONLY_BENCHMARK
                    source0_result = stage_compute_benchmark_tile(cb_source0, zero_tile_addr);
                    if (!scale) {
                        source1_result = stage_compute_benchmark_tile(cb_source1, zero_tile_addr);
                        base_result = stage_compute_benchmark_tile(cb_base, zero_tile_addr);
                    }
#else
                    if (scale) {
                        const int32_t requested_y = base_requested_y(source, output, tile_y);
                        const int32_t requested_x = base_requested_x(source, output, tile_x);
                        source0_requested_y = requested_y;
                        source0_requested_x = requested_x;
                        source0_result = stage_optimized_tile(
                            cb_source0,
                            zero_tile_addr,
                            plane_addrs[source_slot],
                            plane_tile_columns[source_slot],
                            stored[source_slot],
                            requested_y,
                            requested_x,
                            true,
                            route_staging_metrics TTWV_STAGING_VALIDATION_ARGUMENT);
                    } else {
                        int32_t requested_y = 0;
                        int32_t requested_x = 0;
                        stencil_requested_origin(
                            vertical, 0, coefficient_count, source, output, tile_y, tile_x, requested_y, requested_x);
                        source0_requested_y = requested_y;
                        source0_requested_x = requested_x;
                        source0_result = stage_optimized_tile(
                            cb_source0,
                            zero_tile_addr,
                            plane_addrs[source_slot],
                            plane_tile_columns[source_slot],
                            stored[source_slot],
                            requested_y,
                            requested_x,
                            false,
                            route_staging_metrics TTWV_STAGING_VALIDATION_ARGUMENT);
                        stencil_requested_origin(
                            vertical, 1, coefficient_count, source, output, tile_y, tile_x, requested_y, requested_x);
                        source1_requested_y = requested_y;
                        source1_requested_x = requested_x;
                        source1_result = stage_optimized_tile(
                            cb_source1,
                            zero_tile_addr,
                            plane_addrs[source_slot],
                            plane_tile_columns[source_slot],
                            stored[source_slot],
                            requested_y,
                            requested_x,
                            false,
                            route_staging_metrics TTWV_STAGING_VALIDATION_ARGUMENT);
                        requested_y = base_requested_y(base, output, tile_y);
                        requested_x = base_requested_x(base, output, tile_x);
                        base_requested_tile_y = requested_y;
                        base_requested_tile_x = requested_x;
                        base_result = stage_optimized_tile(
                            cb_base,
                            zero_tile_addr,
                            plane_addrs[base_slot],
                            plane_tile_columns[base_slot],
                            stored[base_slot],
                            requested_y,
                            requested_x,
                            true,
                            route_staging_metrics TTWV_STAGING_VALIDATION_ARGUMENT);
                    }
#endif
                    if (source0_result != StageTileResult::kCompleted ||
                        source1_result != StageTileResult::kCompleted || base_result != StageTileResult::kCompleted) {
                        noc_async_read_barrier();
                    }
                    finish_pending_tile(
                        source0_result,
                        cb_source0,
                        plane_addrs[source_slot],
                        plane_tile_columns[source_slot],
                        stored[source_slot],
                        source0_requested_y,
                        source0_requested_x TTWV_STAGING_VALIDATION_ARGUMENT);
                    finish_pending_tile(
                        source1_result,
                        cb_source1,
                        plane_addrs[source_slot],
                        plane_tile_columns[source_slot],
                        stored[source_slot],
                        source1_requested_y,
                        source1_requested_x TTWV_STAGING_VALIDATION_ARGUMENT);
                    finish_pending_tile(
                        base_result,
                        cb_base,
                        plane_addrs[base_slot],
                        plane_tile_columns[base_slot],
                        stored[base_slot],
                        base_requested_tile_y,
                        base_requested_tile_x TTWV_STAGING_VALIDATION_ARGUMENT);
#else
                    if (scale) {
                        count_route_tile(
                            route_staging_metrics,
                            classify_route_tile(
                                stored[source_slot],
                                base_requested_y(source, output, tile_y),
                                base_requested_x(source, output, tile_x)),
                            true);
                        fill_base_or_scale_tile(
                            cb_source0,
                            plane_addrs[source_slot],
                            plane_tile_columns[source_slot],
                            stored[source_slot],
                            source,
                            output,
                            tile_y,
                            tile_x);
                    } else {
                        int32_t requested_y = 0;
                        int32_t requested_x = 0;
                        for (uint32_t source_tile = 0; source_tile < 2; ++source_tile) {
                            stencil_requested_origin(
                                vertical,
                                source_tile,
                                coefficient_count,
                                source,
                                output,
                                tile_y,
                                tile_x,
                                requested_y,
                                requested_x);
                            count_route_tile(
                                route_staging_metrics,
                                classify_route_tile(stored[source_slot], requested_y, requested_x),
                                false);
                        }
                        count_route_tile(
                            route_staging_metrics,
                            classify_route_tile(
                                stored[base_slot],
                                base_requested_y(base, output, tile_y),
                                base_requested_x(base, output, tile_x)),
                            true);
                        fill_stencil_source_tile(
                            cb_source0,
                            vertical,
                            0,
                            coefficient_count,
                            plane_addrs[source_slot],
                            plane_tile_columns[source_slot],
                            stored[source_slot],
                            source,
                            output,
                            tile_y,
                            tile_x);
                        fill_stencil_source_tile(
                            cb_source1,
                            vertical,
                            1,
                            coefficient_count,
                            plane_addrs[source_slot],
                            plane_tile_columns[source_slot],
                            stored[source_slot],
                            source,
                            output,
                            tile_y,
                            tile_x);
                        fill_base_or_scale_tile(
                            cb_base,
                            plane_addrs[base_slot],
                            plane_tile_columns[base_slot],
                            stored[base_slot],
                            base,
                            output,
                            tile_y,
                            tile_x);
                    }
#endif
                }
            }
#ifdef TTWV_CAPTURE_TRANSPORT_METRICS
            const uint64_t staging_cycles = get_timestamp() - staging_start;
            const uint64_t sync_start = get_timestamp();
#endif
            cb_wait_front(cb_sync, 1);
            cb_pop_front(cb_sync, 1);
#ifdef TTWV_CAPTURE_TRANSPORT_METRICS
            const uint64_t sync_wait_cycles = get_timestamp() - sync_start;
            write_transport_reader_half(
                transport_metric_args,
                transport_metrics_addr,
                transport_metric_pages_per_chunk,
                global_chunk,
                route_index,
                reader_metric_scratch_addr,
                config_cycles,
                staging_cycles,
                sync_wait_cycles,
                output_tile_count,
                route_words[ttwv::device_protocol::kLwt2DRouteAxis],
                route_words[ttwv::device_protocol::kLwt2DRouteType],
                coefficient_count,
                route_staging_metrics);
#endif
            stored[output_slot] = output;
        }
        // The final route handshake only guarantees that its result has been
        // persisted to the local plane.  The writer still reads all four
        // terminal bands from those planes.  Do not reuse the workspace for
        // the next chunk until those DRAM writes have completed.
        cb_wait_front(cb_sync, 1);
        cb_pop_front(cb_sync, 1);
#ifdef TTWV_CAPTURE_TRANSPORT_METRICS
        const uint64_t chunk_cycles = get_timestamp() - chunk_start;
        write_transport_reader_half(
            transport_metric_args,
            transport_metrics_addr,
            transport_metric_pages_per_chunk,
            global_chunk,
            route_count,
            reader_metric_scratch_addr,
            preload_config_cycles,
            chunk_cycles,
            0,
            0,
            0,
            0,
            0,
            RouteStagingMetrics{});
#elif defined(TTWV_VALIDATE_ROUTE_STAGING)
        write_staging_validation_summary(
            transport_metric_args,
            transport_metrics_addr,
            transport_metric_pages_per_chunk,
            global_chunk,
            route_count,
            reader_metric_scratch_addr,
            staging_validation_metrics);
#endif
    }
}
