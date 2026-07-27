// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "../../tt_wavelet/include/device_protocol/lwt_2d_config.hpp"
#include "../primitives/tile_2d_layout.hpp"
#include "api/dataflow/dataflow_api.h"

namespace {

using ttwv::kernels::primitives::kFaceSide;
using ttwv::kernels::primitives::kTileBytes;
using ttwv::kernels::primitives::kTileElements;
using ttwv::kernels::primitives::kTileSide;
using ttwv::kernels::primitives::tile_element_offset;
using ttwv::kernels::primitives::tiled_element_offset;

struct Rect {
    uint32_t y_begin;
    uint32_t y_length;
    uint32_t x_begin;
    uint32_t x_length;
};

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

#ifdef TTWV_VALIDATE_ROUTE_PERSISTENCE
struct PersistenceValidationMetrics {
    uint32_t validated_tiles{0};
    uint32_t mismatched_words{0};
};
#define TTWV_PERSISTENCE_VALIDATION_PARAMETER , PersistenceValidationMetrics& validation_metrics
#define TTWV_PERSISTENCE_VALIDATION_ARGUMENT , persistence_validation_metrics
#else
#define TTWV_PERSISTENCE_VALIDATION_PARAMETER
#define TTWV_PERSISTENCE_VALIDATION_ARGUMENT
#endif

#ifdef TTWV_VALIDATE_ROUTE_PERSISTENCE
ALWI void validate_persisted_tile(
    const uint32_t destination_addr, const uint32_t source_addr, PersistenceValidationMetrics& metrics) {
    const auto* destination = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(destination_addr);
    const auto* source = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(source_addr);
    ++metrics.validated_tiles;
    for (uint32_t word = 0; word < kTileBytes / sizeof(uint32_t); ++word) {
        metrics.mismatched_words += destination[word] != source[word] ? 1U : 0U;
    }
}
#define TTWV_VALIDATE_PERSISTED_TILE(...) validate_persisted_tile(__VA_ARGS__)
#else
#define TTWV_VALIDATE_PERSISTED_TILE(...) \
    do {                                  \
    } while (false)
#endif

template <typename SnapshotAccessor>
ALWI void write_local_output(
    const uint32_t cb_output,
    const uint32_t plane_addr,
    const uint32_t plane_tile_columns,
    const Rect& output,
    const SnapshotAccessor& snapshot_args,
    const bool capture_snapshot,
    const uint32_t snapshot_addr,
    const uint32_t snapshot_page_base,
    uint64_t& compute_wait_cycles,
    uint64_t& persistence_cycles TTWV_PERSISTENCE_VALIDATION_PARAMETER) {
    const auto snapshots = TensorAccessor(snapshot_args, snapshot_addr, kTileBytes);
    const uint32_t tile_rows =
        (aligned_end(output.y_begin, output.y_length) - aligned_begin(output.y_begin)) / kTileSide;
    const uint32_t tile_columns =
        (aligned_end(output.x_begin, output.x_length) - aligned_begin(output.x_begin)) / kTileSide;
#ifdef TTWV_LWT_2D_FULL_TILE_PERSISTENCE
    if (!capture_snapshot) {
        const uint32_t tile_count = tile_rows * tile_columns;
        for (uint32_t first_tile = 0; first_tile < tile_count;) {
            const uint32_t read_ptr = get_read_ptr(cb_output);
            const uint32_t fifo_limit = get_local_cb_interface(cb_output).fifo_limit;
            const uint32_t batch = first_tile + 1 < tile_count && read_ptr + 2 * kTileBytes <= fifo_limit ? 2U : 1U;
#ifdef TTWV_CAPTURE_TRANSPORT_METRICS
            const uint64_t wait_start = get_timestamp();
#endif
            cb_wait_front(cb_output, batch);
#ifdef TTWV_CAPTURE_TRANSPORT_METRICS
            compute_wait_cycles += get_timestamp() - wait_start;
            const uint64_t persistence_start = get_timestamp();
#endif
            for (uint32_t tile_in_batch = 0; tile_in_batch < batch; ++tile_in_batch) {
                const uint32_t flat_tile = first_tile + tile_in_batch;
                const uint32_t tile_y = flat_tile / tile_columns;
                const uint32_t tile_x = flat_tile % tile_columns;
                const uint32_t destination_addr = plane_addr + (tile_y * plane_tile_columns + tile_x) * kTileBytes;
                noc_async_write(read_ptr + tile_in_batch * kTileBytes, get_noc_addr(destination_addr), kTileBytes);
            }
            noc_async_write_barrier();
            for (uint32_t tile_in_batch = 0; tile_in_batch < batch; ++tile_in_batch) {
                const uint32_t flat_tile = first_tile + tile_in_batch;
                const uint32_t tile_y = flat_tile / tile_columns;
                const uint32_t tile_x = flat_tile % tile_columns;
                const uint32_t destination_addr = plane_addr + (tile_y * plane_tile_columns + tile_x) * kTileBytes;
                TTWV_VALIDATE_PERSISTED_TILE(
                    destination_addr, read_ptr + tile_in_batch * kTileBytes, validation_metrics);
            }
            cb_pop_front(cb_output, batch);
#ifdef TTWV_CAPTURE_TRANSPORT_METRICS
            persistence_cycles += get_timestamp() - persistence_start;
#endif
            first_tile += batch;
        }
        return;
    }
#endif
    for (uint32_t tile_y = 0; tile_y < tile_rows; ++tile_y) {
        for (uint32_t tile_x = 0; tile_x < tile_columns; ++tile_x) {
#ifdef TTWV_CAPTURE_TRANSPORT_METRICS
            const uint64_t wait_start = get_timestamp();
#endif
            cb_wait_front(cb_output, 1);
#ifdef TTWV_CAPTURE_TRANSPORT_METRICS
            compute_wait_cycles += get_timestamp() - wait_start;
            const uint64_t persistence_start = get_timestamp();
#endif
#ifdef TTWV_LWT_2D_FULL_TILE_PERSISTENCE
            const uint32_t destination_addr = plane_addr + (tile_y * plane_tile_columns + tile_x) * kTileBytes;
            noc_async_write(get_read_ptr(cb_output), get_noc_addr(destination_addr), kTileBytes);
            noc_async_write_barrier();
            TTWV_VALIDATE_PERSISTED_TILE(destination_addr, get_read_ptr(cb_output), validation_metrics);
#else
            auto* destination = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                plane_addr + (tile_y * plane_tile_columns + tile_x) * kTileBytes);
            const auto* source = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_output));
            for (uint32_t word = 0; word < kTileBytes / sizeof(uint32_t); ++word) {
                destination[word] = source[word];
            }
#endif
            cb_pop_front(cb_output, 1);
#ifdef TTWV_CAPTURE_TRANSPORT_METRICS
            persistence_cycles += get_timestamp() - persistence_start;
#endif
            if (capture_snapshot) {
                const uint32_t snapshot_page = snapshot_page_base + tile_y * tile_columns + tile_x;
                noc_async_write(
                    plane_addr + (tile_y * plane_tile_columns + tile_x) * kTileBytes,
                    snapshots.get_noc_addr(snapshot_page),
                    kTileBytes);
                noc_async_write_barrier();
            }
        }
    }
}

struct TerminalWriteMetrics {
    uint32_t exact_tiles{0};
    uint32_t fragmented_tiles{0};
};

template <typename OutputAccessor>
ALWI void write_band_fragmented(
    const OutputAccessor& output_args,
    const uint32_t output_addr,
    const uint32_t output_tile_columns,
    const uint32_t plane_addr,
    const uint32_t plane_tile_columns,
    const Rect& source,
    const uint32_t final_y_begin,
    const uint32_t final_y_length,
    const uint32_t final_x_begin,
    const uint32_t final_x_length,
    const uint32_t noc_scratch_addr) {
    const auto output = TensorAccessor(output_args, output_addr, kTileBytes);
    const uint32_t source_y_origin = aligned_begin(source.y_begin);
    const uint32_t source_x_origin = aligned_begin(source.x_begin);
    for (uint32_t local_y = 0; local_y < final_y_length; ++local_y) {
        uint32_t local_x = 0;
        while (local_x < final_x_length) {
            const uint32_t source_y = source.y_begin + local_y - source_y_origin;
            const uint32_t source_x = source.x_begin + local_x - source_x_origin;
            const uint32_t destination_y = final_y_begin + local_y;
            const uint32_t destination_x = final_x_begin + local_x;
            const uint32_t count = std::min(
                final_x_length - local_x,
                std::min(kFaceSide - source_x % kFaceSide, kFaceSide - destination_x % kFaceSide));
            const uint32_t source_offset = tiled_element_offset(source_y, source_x, plane_tile_columns) * sizeof(float);
            const uint32_t destination_tile =
                (destination_y / kTileSide) * output_tile_columns + destination_x / kTileSide;
            const uint32_t destination_offset =
                tile_element_offset(destination_y % kTileSide, destination_x % kTileSide) * sizeof(float);
            const uint64_t destination_noc_addr = output.get_noc_addr(destination_tile) + destination_offset;
            const uint32_t scratch_lane = static_cast<uint32_t>(destination_noc_addr) & 63U;
            auto* staged = reinterpret_cast<volatile tt_l1_ptr float*>(noc_scratch_addr + scratch_lane);
            const auto* source_values = reinterpret_cast<volatile tt_l1_ptr float*>(plane_addr + source_offset);
            for (uint32_t value = 0; value < count; ++value) {
                staged[value] = source_values[value];
            }
            noc_async_write(noc_scratch_addr + scratch_lane, destination_noc_addr, count * sizeof(float));
            noc_async_write_barrier();
            local_x += count;
        }
    }
}

#ifdef TTWV_LWT_2D_TILED_TERMINAL_WRITES
template <typename OutputAccessor>
[[nodiscard]] ALWI bool write_band_full_tiles(
    const OutputAccessor& output_args,
    const uint32_t output_addr,
    const uint32_t output_tile_columns,
    const uint32_t plane_addr,
    const uint32_t plane_tile_columns,
    const Rect& source,
    const uint32_t final_y_begin,
    const uint32_t final_y_length,
    const uint32_t final_x_begin,
    const uint32_t final_x_length,
    TerminalWriteMetrics& metrics) {
    const bool exact = final_y_begin % kTileSide == 0 && final_x_begin % kTileSide == 0 &&
                       final_y_length % kTileSide == 0 && final_x_length % kTileSide == 0 &&
                       source.y_begin % kTileSide == 0 && source.x_begin % kTileSide == 0 &&
                       source.y_length == final_y_length && source.x_length == final_x_length;
    if (!exact) {
        return false;
    }

    constexpr uint32_t kWriteBatchTiles = 16;
    const auto output = TensorAccessor(output_args, output_addr, kTileBytes);
    const uint32_t tile_rows = final_y_length / kTileSide;
    const uint32_t tile_columns = final_x_length / kTileSide;
    uint32_t outstanding = 0;
    for (uint32_t tile_y = 0; tile_y < tile_rows; ++tile_y) {
        for (uint32_t tile_x = 0; tile_x < tile_columns; ++tile_x) {
            const uint32_t source_tile = tile_y * plane_tile_columns + tile_x;
            const uint32_t destination_tile =
                (final_y_begin / kTileSide + tile_y) * output_tile_columns + final_x_begin / kTileSide + tile_x;
            noc_async_write(plane_addr + source_tile * kTileBytes, output.get_noc_addr(destination_tile), kTileBytes);
            ++metrics.exact_tiles;
            if (++outstanding == kWriteBatchTiles) {
                noc_async_write_barrier();
                outstanding = 0;
            }
        }
    }
    if (outstanding != 0) {
        noc_async_write_barrier();
    }
    return true;
}
#endif

template <typename OutputAccessor>
ALWI void write_band(
    const OutputAccessor& output_args,
    const uint32_t output_addr,
    const uint32_t output_tile_columns,
    const uint32_t plane_addr,
    const uint32_t plane_tile_columns,
    const Rect& source,
    const uint32_t final_y_begin,
    const uint32_t final_y_length,
    const uint32_t final_x_begin,
    const uint32_t final_x_length,
    const uint32_t noc_scratch_addr,
    TerminalWriteMetrics& metrics) {
#ifdef TTWV_LWT_2D_TILED_TERMINAL_WRITES
    if (write_band_full_tiles(
            output_args,
            output_addr,
            output_tile_columns,
            plane_addr,
            plane_tile_columns,
            source,
            final_y_begin,
            final_y_length,
            final_x_begin,
            final_x_length,
            metrics)) {
        return;
    }
#endif
    const uint32_t destination_tile_rows =
        (aligned_end(final_y_begin, final_y_length) - aligned_begin(final_y_begin)) / kTileSide;
    const uint32_t destination_tile_columns =
        (aligned_end(final_x_begin, final_x_length) - aligned_begin(final_x_begin)) / kTileSide;
    metrics.fragmented_tiles += destination_tile_rows * destination_tile_columns;
    write_band_fragmented(
        output_args,
        output_addr,
        output_tile_columns,
        plane_addr,
        plane_tile_columns,
        source,
        final_y_begin,
        final_y_length,
        final_x_begin,
        final_x_length,
        noc_scratch_addr);
}

#ifdef TTWV_ILWT_2D
[[nodiscard]] ALWI float read_plane_value(
    const uint32_t plane_addr,
    const uint32_t plane_tile_columns,
    const Rect& stored,
    const uint32_t y,
    const uint32_t x) {
    ASSERT(y >= stored.y_begin && y < stored.y_begin + stored.y_length);
    ASSERT(x >= stored.x_begin && x < stored.x_begin + stored.x_length);
    const uint32_t local_y = y - aligned_begin(stored.y_begin);
    const uint32_t local_x = x - aligned_begin(stored.x_begin);
    const auto* plane = reinterpret_cast<const volatile tt_l1_ptr float*>(plane_addr);
    return plane[tiled_element_offset(local_y, local_x, plane_tile_columns)];
}

template <typename OutputAccessor>
ALWI void write_interleaved_output(
    const OutputAccessor& output_args,
    const uint32_t output_addr,
    const uint32_t output_tile_columns,
    const uint32_t* plane_addrs,
    const uint32_t* plane_tile_columns,
    const uint32_t* parity_slots,
    const Rect* parity_sources,
    const uint32_t final_y_begin,
    const uint32_t final_y_length,
    const uint32_t final_x_begin,
    const uint32_t final_x_length,
    const uint32_t pad_y,
    const uint32_t pad_x,
    const uint32_t scratch_addr,
    TerminalWriteMetrics& metrics) {
    const auto output = TensorAccessor(output_args, output_addr, kTileBytes);
    const uint32_t tile_y_begin = aligned_begin(final_y_begin);
    const uint32_t tile_y_end = aligned_end(final_y_begin, final_y_length);
    const uint32_t tile_x_begin = aligned_begin(final_x_begin);
    const uint32_t tile_x_end = aligned_end(final_x_begin, final_x_length);
    auto* tile = reinterpret_cast<volatile tt_l1_ptr float*>(scratch_addr);

    for (uint32_t tile_y = tile_y_begin; tile_y < tile_y_end; tile_y += kTileSide) {
        for (uint32_t tile_x = tile_x_begin; tile_x < tile_x_end; tile_x += kTileSide) {
            for (uint32_t element = 0; element < kTileElements; ++element) {
                tile[element] = 0.0F;
            }
            const uint32_t y_end = std::min(tile_y + kTileSide, final_y_begin + final_y_length);
            const uint32_t x_end = std::min(tile_x + kTileSide, final_x_begin + final_x_length);
            for (uint32_t y = std::max(tile_y, final_y_begin); y < y_end; ++y) {
                const uint32_t padded_y = y + pad_y;
                const uint32_t parity_y = padded_y & 1U;
                const uint32_t polyphase_y = padded_y / 2;
                for (uint32_t x = std::max(tile_x, final_x_begin); x < x_end; ++x) {
                    const uint32_t padded_x = x + pad_x;
                    const uint32_t parity_x = padded_x & 1U;
                    const uint32_t polyphase_x = padded_x / 2;
                    const uint32_t parity = 2 * parity_y + parity_x;
                    const uint32_t slot = parity_slots[parity];
                    tile[tile_element_offset(y - tile_y, x - tile_x)] = read_plane_value(
                        plane_addrs[slot],
                        plane_tile_columns[slot],
                        parity_sources[parity],
                        polyphase_y,
                        polyphase_x);
                }
            }
            const uint32_t destination_tile =
                (tile_y / kTileSide) * output_tile_columns + tile_x / kTileSide;
            const bool complete =
                tile_y >= final_y_begin && tile_x >= final_x_begin &&
                tile_y + kTileSide <= final_y_begin + final_y_length &&
                tile_x + kTileSide <= final_x_begin + final_x_length;
            if (complete) {
                noc_async_write(scratch_addr, output.get_noc_addr(destination_tile), kTileBytes);
                ++metrics.exact_tiles;
            } else {
                const uint32_t valid_y_begin = std::max(tile_y, final_y_begin);
                const uint32_t valid_x_begin = std::max(tile_x, final_x_begin);
                for (uint32_t y = valid_y_begin; y < y_end; ++y) {
                    for (uint32_t x = valid_x_begin; x < x_end;) {
                        const uint32_t local_x = x - tile_x;
                        const uint32_t count = std::min(x_end - x, kFaceSide - local_x % kFaceSide);
                        const uint32_t byte_offset =
                            tile_element_offset(y - tile_y, local_x) * sizeof(float);
                        noc_async_write(
                            scratch_addr + byte_offset,
                            output.get_noc_addr(destination_tile) + byte_offset,
                            count * sizeof(float));
                        x += count;
                    }
                }
                ++metrics.fragmented_tiles;
            }
            noc_async_write_barrier();
        }
    }
}
#endif

#ifdef TTWV_CAPTURE_TRANSPORT_METRICS
template <typename MetricAccessor>
ALWI void write_transport_writer_half(
    const MetricAccessor& metric_args,
    const uint32_t metrics_addr,
    const uint32_t metric_pages_per_chunk,
    const uint32_t global_chunk,
    const uint32_t page_in_chunk,
    const uint32_t scratch_addr,
    const uint64_t config_cycles,
    const uint64_t persistence_cycles,
    const uint64_t compute_cycles,
    const uint32_t persistence_tiles,
    const TerminalWriteMetrics terminal_metrics,
    const uint64_t terminal_cycles,
    const uint64_t kernel_cycles) {
    constexpr uint32_t base = ttwv::device_protocol::kLwt2DTransportMetricWriterConfigCyclesLow;
    auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_addr);
    for (uint32_t word = 0; word < ttwv::device_protocol::kLwt2DTransportMetricWordCount / 2; ++word) {
        words[word] = 0;
    }
    words[ttwv::device_protocol::kLwt2DTransportMetricWriterConfigCyclesLow - base] =
        static_cast<uint32_t>(config_cycles);
    words[ttwv::device_protocol::kLwt2DTransportMetricWriterConfigCyclesHigh - base] =
        static_cast<uint32_t>(config_cycles >> 32);
    words[ttwv::device_protocol::kLwt2DTransportMetricPersistenceCyclesLow - base] =
        static_cast<uint32_t>(persistence_cycles);
    words[ttwv::device_protocol::kLwt2DTransportMetricPersistenceCyclesHigh - base] =
        static_cast<uint32_t>(persistence_cycles >> 32);
    words[ttwv::device_protocol::kLwt2DTransportMetricComputeCyclesLow - base] = static_cast<uint32_t>(compute_cycles);
    words[ttwv::device_protocol::kLwt2DTransportMetricComputeCyclesHigh - base] =
        static_cast<uint32_t>(compute_cycles >> 32);
    words[ttwv::device_protocol::kLwt2DTransportMetricPersistenceTiles - base] = persistence_tiles;
    words[ttwv::device_protocol::kLwt2DTransportMetricExactTerminalTiles - base] = terminal_metrics.exact_tiles;
    words[ttwv::device_protocol::kLwt2DTransportMetricFragmentedTerminalTiles - base] =
        terminal_metrics.fragmented_tiles;
    words[ttwv::device_protocol::kLwt2DTransportMetricTerminalWriteCyclesLow - base] =
        static_cast<uint32_t>(terminal_cycles);
    words[ttwv::device_protocol::kLwt2DTransportMetricTerminalWriteCyclesHigh - base] =
        static_cast<uint32_t>(terminal_cycles >> 32);
    words[ttwv::device_protocol::kLwt2DTransportMetricKernelCyclesLow - base] = static_cast<uint32_t>(kernel_cycles);
    words[ttwv::device_protocol::kLwt2DTransportMetricKernelCyclesHigh - base] =
        static_cast<uint32_t>(kernel_cycles >> 32);
    const auto metrics =
        TensorAccessor(metric_args, metrics_addr, ttwv::device_protocol::kLwt2DTransportMetricPageBytes);
    const uint32_t page = global_chunk * metric_pages_per_chunk + page_in_chunk;
    noc_async_write(
        scratch_addr,
        metrics.get_noc_addr(page) + ttwv::device_protocol::kLwt2DTransportMetricHalfPageBytes,
        ttwv::device_protocol::kLwt2DTransportMetricHalfPageBytes);
    noc_async_write_barrier();
}
#endif

#ifdef TTWV_VALIDATE_ROUTE_PERSISTENCE
template <typename MetricAccessor>
ALWI void write_persistence_validation_summary(
    const MetricAccessor& metric_args,
    const uint32_t metrics_addr,
    const uint32_t metric_pages_per_chunk,
    const uint32_t global_chunk,
    const uint32_t page_in_chunk,
    const uint32_t scratch_addr,
    const PersistenceValidationMetrics& validation) {
    constexpr uint32_t base = ttwv::device_protocol::kLwt2DTransportMetricWriterConfigCyclesLow;
    auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_addr);
    for (uint32_t word = 0; word < ttwv::device_protocol::kLwt2DTransportMetricWordCount / 2; ++word) {
        words[word] = 0;
    }
    words[ttwv::device_protocol::kLwt2DTransportMetricValidatedPersistenceTiles - base] = validation.validated_tiles;
    words[ttwv::device_protocol::kLwt2DTransportMetricPersistenceValidationMismatches - base] =
        validation.mismatched_words;
    const auto metrics =
        TensorAccessor(metric_args, metrics_addr, ttwv::device_protocol::kLwt2DTransportMetricPageBytes);
    const uint32_t page = global_chunk * metric_pages_per_chunk + page_in_chunk;
    noc_async_write(
        scratch_addr,
        metrics.get_noc_addr(page) + ttwv::device_protocol::kLwt2DTransportMetricHalfPageBytes,
        ttwv::device_protocol::kLwt2DTransportMetricHalfPageBytes);
    noc_async_write_barrier();
}
#endif

}  // namespace

void kernel_main() {
    uint32_t plane_addrs[ttwv::device_protocol::kLwt2DPlaneCount];
    uint32_t plane_tile_columns[ttwv::device_protocol::kLwt2DPlaneCount];
    for (uint32_t slot = 0; slot < ttwv::device_protocol::kLwt2DPlaneCount; ++slot) {
        plane_addrs[slot] = get_arg_val<uint32_t>(slot);
        plane_tile_columns[slot] = get_arg_val<uint32_t>(ttwv::device_protocol::kLwt2DPlaneCount + slot);
    }
    constexpr uint32_t plane_arg_count = 2 * ttwv::device_protocol::kLwt2DPlaneCount;
    const uint32_t route_config_addr = get_arg_val<uint32_t>(plane_arg_count);
    const uint32_t band_config_addr = get_arg_val<uint32_t>(plane_arg_count + 1);
    uint32_t output_addrs[ttwv::device_protocol::kLwt2DBandCount];
    for (uint32_t band = 0; band < ttwv::device_protocol::kLwt2DBandCount; ++band) {
        output_addrs[band] = get_arg_val<uint32_t>(plane_arg_count + 2 + band);
    }
    const uint32_t output_tile_columns = get_arg_val<uint32_t>(plane_arg_count + 6);
    const uint32_t chunk_begin = get_arg_val<uint32_t>(plane_arg_count + 7);
    const uint32_t chunk_count = get_arg_val<uint32_t>(plane_arg_count + 8);
    const uint32_t route_count = get_arg_val<uint32_t>(plane_arg_count + 9);
    const bool capture_snapshots = get_arg_val<uint32_t>(plane_arg_count + 10) != 0;
    const uint32_t snapshot_addr = get_arg_val<uint32_t>(plane_arg_count + 11);
    const uint32_t snapshot_tiles_per_route = get_arg_val<uint32_t>(plane_arg_count + 12);
#if defined(TTWV_CAPTURE_TRANSPORT_METRICS) || defined(TTWV_VALIDATE_ROUTE_PERSISTENCE)
    const uint32_t transport_metrics_addr = get_arg_val<uint32_t>(plane_arg_count + 13);
    const uint32_t transport_metric_pages_per_chunk = get_arg_val<uint32_t>(plane_arg_count + 14);
#endif
#ifdef TTWV_ILWT_2D
    const uint32_t pad_y = get_arg_val<uint32_t>(plane_arg_count + 15);
    const uint32_t pad_x = get_arg_val<uint32_t>(plane_arg_count + 16);
#endif

    constexpr uint32_t cb_output = get_compile_time_arg_val(0);
    constexpr uint32_t cb_sync = get_compile_time_arg_val(1);
    constexpr uint32_t cb_route_config = get_compile_time_arg_val(2);
    constexpr uint32_t cb_band_config = get_compile_time_arg_val(3);
    constexpr uint32_t cb_noc_scratch = get_compile_time_arg_val(4);
    constexpr auto route_args = TensorAccessorArgs<5>();
    constexpr auto band_args = TensorAccessorArgs<route_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<band_args.next_compile_time_args_offset()>();
    constexpr auto transport_metric_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    const uint32_t noc_scratch_raw = get_write_ptr(cb_noc_scratch);
    const uint32_t noc_scratch_addr = (noc_scratch_raw + 63U) & ~63U;
#if defined(TTWV_CAPTURE_TRANSPORT_METRICS) || defined(TTWV_VALIDATE_ROUTE_PERSISTENCE)
    // Reader and writer execute concurrently and share cb_noc_scratch's L1
    // allocation.  Keep their metric assembly cache lines disjoint so one
    // RISC cannot overwrite the other's in-flight NoC source buffer.
    const uint32_t writer_metric_scratch_addr = noc_scratch_addr + ttwv::device_protocol::kLwt2DSplitScratchBytes -
                                                ttwv::device_protocol::kLwt2DTransportMetricHalfPageBytes;
#endif
#ifdef TTWV_LWT_2D_PRELOAD_ROUTE_CONFIG
    constexpr uint32_t writer_config_capacity =
        ttwv::device_protocol::kLwt2DSplitScratchBytes / 2 - ttwv::device_protocol::kLwt2DTransportMetricPageBytes;
    const uint32_t writer_config_addr = noc_scratch_addr + ttwv::device_protocol::kLwt2DSplitScratchBytes / 2;
#endif

    for (uint32_t local_chunk = 0; local_chunk < chunk_count; ++local_chunk) {
        const uint32_t global_chunk = chunk_begin + local_chunk;
#ifdef TTWV_VALIDATE_ROUTE_PERSISTENCE
        PersistenceValidationMetrics persistence_validation_metrics{};
#endif
#ifdef TTWV_CAPTURE_TRANSPORT_METRICS
        const uint64_t chunk_start = get_timestamp();
        uint64_t preload_config_cycles = 0;
#endif
#ifdef TTWV_LWT_2D_PRELOAD_ROUTE_CONFIG
        // The first packed output proves that split and reader preloading no
        // longer use the shared split scratch. Leave the page in the CB for
        // write_local_output() to consume after descriptor preloading.
        cb_wait_front(cb_output, 1);
        ASSERT(route_count * ttwv::device_protocol::kLwt2DRouteConfigPageBytes <= writer_config_capacity);
#ifdef TTWV_CAPTURE_TRANSPORT_METRICS
        const uint64_t preload_config_start = get_timestamp();
#endif
        preload_config_pages(
            route_args,
            route_config_addr,
            ttwv::device_protocol::kLwt2DRouteConfigPageBytes,
            global_chunk * route_count,
            route_count,
            writer_config_addr);
#ifdef TTWV_CAPTURE_TRANSPORT_METRICS
        preload_config_cycles = get_timestamp() - preload_config_start;
#endif
#endif
        for (uint32_t route_index = 0; route_index < route_count; ++route_index) {
#ifdef TTWV_LWT_2D_PRELOAD_ROUTE_CONFIG
            const auto* route_words = reinterpret_cast<const uint32_t*>(
                writer_config_addr + route_index * ttwv::device_protocol::kLwt2DRouteConfigPageBytes);
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
            const uint32_t output_slot = route_words[ttwv::device_protocol::kLwt2DRouteOutputSlot];
            const Rect output = load_rect(route_words, ttwv::device_protocol::kLwt2DRouteOutputRect);
#ifdef TTWV_CAPTURE_TRANSPORT_METRICS
            uint64_t persistence_cycles = 0;
            uint64_t compute_cycles = 0;
#else
            uint64_t unused_wait_cycles = 0;
            uint64_t unused_persistence_cycles = 0;
#endif
            write_local_output(
                cb_output,
                plane_addrs[output_slot],
                plane_tile_columns[output_slot],
                output,
                output_args,
                capture_snapshots,
                snapshot_addr,
                (global_chunk * route_count + route_index) * snapshot_tiles_per_route,
#ifdef TTWV_CAPTURE_TRANSPORT_METRICS
                compute_cycles,
                persistence_cycles TTWV_PERSISTENCE_VALIDATION_ARGUMENT);
#else
                unused_wait_cycles,
                unused_persistence_cycles TTWV_PERSISTENCE_VALIDATION_ARGUMENT);
#endif
            cb_reserve_back(cb_sync, 1);
            cb_push_back(cb_sync, 1);
#ifdef TTWV_CAPTURE_TRANSPORT_METRICS
            const uint32_t tile_rows =
                (aligned_end(output.y_begin, output.y_length) - aligned_begin(output.y_begin)) / kTileSide;
            const uint32_t tile_columns =
                (aligned_end(output.x_begin, output.x_length) - aligned_begin(output.x_begin)) / kTileSide;
            write_transport_writer_half(
                transport_metric_args,
                transport_metrics_addr,
                transport_metric_pages_per_chunk,
                global_chunk,
                route_index,
                writer_metric_scratch_addr,
                config_cycles,
                persistence_cycles,
                compute_cycles,
                tile_rows * tile_columns,
                TerminalWriteMetrics{},
                0,
                0);
#endif
        }

#ifdef TTWV_CAPTURE_TRANSPORT_METRICS
        const uint64_t terminal_start = get_timestamp();
#endif
        uint32_t band_words[ttwv::device_protocol::kLwt2DBandConfigWordCount];
        load_config_page(
            band_args,
            band_config_addr,
            ttwv::device_protocol::kLwt2DBandConfigPageBytes,
            global_chunk,
            cb_band_config,
            band_words,
            ttwv::device_protocol::kLwt2DBandConfigWordCount);
        const uint32_t final_y_begin = band_words[ttwv::device_protocol::kLwt2DBandFinalYBegin];
        const uint32_t final_y_length = band_words[ttwv::device_protocol::kLwt2DBandFinalYLength];
        const uint32_t final_x_begin = band_words[ttwv::device_protocol::kLwt2DBandFinalXBegin];
        const uint32_t final_x_length = band_words[ttwv::device_protocol::kLwt2DBandFinalXLength];
        constexpr uint32_t band_offsets[4] = {
            ttwv::device_protocol::kLwt2DBandLl,
            ttwv::device_protocol::kLwt2DBandLh,
            ttwv::device_protocol::kLwt2DBandHl,
            ttwv::device_protocol::kLwt2DBandHh,
        };
        TerminalWriteMetrics terminal_metrics{};
#ifdef TTWV_ILWT_2D
        uint32_t parity_slots[4];
        Rect parity_sources[4];
        for (uint32_t parity = 0; parity < 4; ++parity) {
            const uint32_t band_offset = band_offsets[parity];
            parity_slots[parity] =
                band_words[band_offset + ttwv::device_protocol::kLwt2DBandSourceSlot];
            parity_sources[parity] =
                load_rect(band_words, band_offset + ttwv::device_protocol::kLwt2DBandSourceRect);
        }
        write_interleaved_output(
            output_args,
            output_addrs[0],
            output_tile_columns,
            plane_addrs,
            plane_tile_columns,
            parity_slots,
            parity_sources,
            final_y_begin,
            final_y_length,
            final_x_begin,
            final_x_length,
            pad_y,
            pad_x,
            noc_scratch_addr,
            terminal_metrics);
#else
        for (uint32_t band = 0; band < 4; ++band) {
            const uint32_t band_offset = band_offsets[band];
            const uint32_t source_slot = band_words[band_offset + ttwv::device_protocol::kLwt2DBandSourceSlot];
            const Rect source = load_rect(band_words, band_offset + ttwv::device_protocol::kLwt2DBandSourceRect);
            write_band(
                output_args,
                output_addrs[band],
                output_tile_columns,
                plane_addrs[source_slot],
                plane_tile_columns[source_slot],
                source,
                final_y_begin,
                final_y_length,
                final_x_begin,
                final_x_length,
                noc_scratch_addr,
                terminal_metrics);
        }
#endif
#ifdef TTWV_CAPTURE_TRANSPORT_METRICS
        const uint64_t terminal_cycles = get_timestamp() - terminal_start;
#endif
        // Release the reader only after every final band has stopped reading
        // this chunk's workspace.
        cb_reserve_back(cb_sync, 1);
        cb_push_back(cb_sync, 1);
#ifdef TTWV_CAPTURE_TRANSPORT_METRICS
        const uint64_t kernel_cycles = get_timestamp() - chunk_start;
        write_transport_writer_half(
            transport_metric_args,
            transport_metrics_addr,
            transport_metric_pages_per_chunk,
            global_chunk,
            route_count,
            writer_metric_scratch_addr,
            preload_config_cycles,
            0,
            0,
            0,
            terminal_metrics,
            terminal_cycles,
            kernel_cycles);
#elif defined(TTWV_VALIDATE_ROUTE_PERSISTENCE)
        write_persistence_validation_summary(
            transport_metric_args,
            transport_metrics_addr,
            transport_metric_pages_per_chunk,
            global_chunk,
            route_count,
            writer_metric_scratch_addr,
            persistence_validation_metrics);
#endif
    }
}
