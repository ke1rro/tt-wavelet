// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_wavelet/include/lifting/device_2d.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "tt-metalium/buffer.hpp"
#include "tt-metalium/buffer_distribution_spec.hpp"
#include "tt-metalium/circular_buffer_constants.h"
#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/shape.hpp"
#include "tt-metalium/tensor_accessor_args.hpp"
#include "tt-metalium/tile.hpp"

namespace ttwv {
namespace {

constexpr tt::DataFormat kDataFormat = tt::DataFormat::Float32;
constexpr uint32_t kTileBytes = device_protocol::kLwt2DFullTileBytes;
constexpr uint32_t kSource0Cb = tt::CBIndex::c_0;
constexpr uint32_t kSource1Cb = tt::CBIndex::c_1;
constexpr uint32_t kBaseCb = tt::CBIndex::c_2;
constexpr uint32_t kSyncCb = tt::CBIndex::c_3;
constexpr uint32_t kReaderChunkConfigCb = tt::CBIndex::c_4;
constexpr uint32_t kReaderRouteConfigCb = tt::CBIndex::c_5;
constexpr uint32_t kWriterRouteConfigCb = tt::CBIndex::c_6;
constexpr uint32_t kWriterBandConfigCb = tt::CBIndex::c_7;
constexpr uint32_t kNocScratchCb = tt::CBIndex::c_8;
constexpr uint32_t kRouteZeroCb = tt::CBIndex::c_9;
constexpr uint32_t kOutputCb = tt::CBIndex::c_16;
constexpr uint32_t kTileBuffering = 2;

constexpr const char* kReaderKernel = "kernels/dataflow/lwt_2d_reader.cpp";
constexpr const char* kComputeKernel = "kernels/compute/lwt_2d_compute.cpp";
constexpr const char* kWriterKernel = "kernels/dataflow/lwt_2d_writer.cpp";

struct Lwt2DProgram {
    tt::tt_metal::Program program;
    tt::tt_metal::KernelHandle reader{};
    tt::tt_metal::KernelHandle compute{};
    tt::tt_metal::KernelHandle writer{};
};

struct CoreChunkWork {
    tt::tt_metal::CoreCoord core;
    uint32_t chunk_begin{0};
    uint32_t chunk_count{0};
};

[[nodiscard]] uint32_t checked_u32(const size_t value, const char* label) {
    TT_FATAL(value <= std::numeric_limits<uint32_t>::max(), "{} exceeds uint32_t", label);
    return static_cast<uint32_t>(value);
}

[[nodiscard]] std::filesystem::path kernel_path(const std::filesystem::path& root, const char* relative) {
    return root / relative;
}

[[nodiscard]] std::vector<tt::tt_metal::CoreCoord> select_cores(
    tt::tt_metal::distributed::MeshDevice& mesh_device, const uint32_t active_core_count) {
    const auto grid = mesh_device.compute_with_storage_grid_size();
    TT_FATAL(
        active_core_count > 0 && active_core_count <= static_cast<uint32_t>(grid.x * grid.y),
        "2D LWT active core count exceeds the worker grid");
    return tt::tt_metal::grid_to_cores(
        active_core_count, static_cast<uint32_t>(grid.x), static_cast<uint32_t>(grid.y), true);
}

[[nodiscard]] tt::tt_metal::CoreRangeSet core_set(const std::vector<tt::tt_metal::CoreCoord>& cores) {
    std::vector<tt::tt_metal::CoreRange> ranges;
    ranges.reserve(cores.size());
    for (const auto& core : cores) {
        ranges.emplace_back(core);
    }
    return tt::tt_metal::CoreRangeSet(std::move(ranges)).merge_ranges();
}

[[nodiscard]] std::vector<CoreChunkWork> partition_work(
    const std::vector<tt::tt_metal::CoreCoord>& cores, const uint32_t chunk_count) {
    TT_FATAL(!cores.empty() && chunk_count >= cores.size(), "Invalid 2D LWT chunk partition");
    const uint32_t core_count = checked_u32(cores.size(), "2D LWT core count");
    const uint32_t base = chunk_count / core_count;
    const uint32_t extra = chunk_count % core_count;
    std::vector<CoreChunkWork> work;
    work.reserve(cores.size());
    uint32_t begin = 0;
    for (uint32_t core = 0; core < core_count; ++core) {
        const uint32_t count = base + (core < extra ? 1U : 0U);
        work.push_back(CoreChunkWork{
            .core = cores[core],
            .chunk_begin = begin,
            .chunk_count = count,
        });
        begin += count;
    }
    TT_FATAL(begin == chunk_count, "2D LWT chunk partition is incomplete");
    return work;
}

[[nodiscard]] std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> create_dram_pages(
    tt::tt_metal::distributed::MeshDevice& mesh_device, const size_t page_count, const uint32_t page_bytes) {
    TT_FATAL(page_count > 0, "2D LWT DRAM buffer requires at least one page");
    return tt::tt_metal::distributed::MeshBuffer::create(
        tt::tt_metal::distributed::ReplicatedBufferConfig{
            .size = static_cast<uint64_t>(page_count) * page_bytes,
        },
        tt::tt_metal::distributed::DeviceLocalBufferConfig{
            .page_size = page_bytes,
            .buffer_type = tt::tt_metal::BufferType::DRAM,
            .bottom_up = false,
        },
        &mesh_device);
}

[[nodiscard]] std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> create_l1_tile_shards(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const std::vector<tt::tt_metal::CoreCoord>& cores,
    const uint32_t shard_tiles) {
    TT_FATAL(!cores.empty() && shard_tiles > 0, "2D LWT L1 plane requires non-empty shards");
    const size_t capacity_tiles = static_cast<size_t>(shard_tiles) * cores.size();
    const tt::tt_metal::CoreRangeSet owners(cores);
    const tt::tt_metal::BufferDistributionSpec distribution(
        tt::tt_metal::Shape{checked_u32(capacity_tiles, "2D LWT plane tile capacity")},
        tt::tt_metal::Shape{shard_tiles},
        cores);
    const tt::tt_metal::ShardSpecBuffer shard_spec(
        owners,
        {shard_tiles, 1},
        tt::tt_metal::ShardOrientation::ROW_MAJOR,
        {1, 1},
        {checked_u32(capacity_tiles, "2D LWT plane tile capacity"), 1});
    const tt::tt_metal::BufferShardingArgs sharding(
        std::optional<tt::tt_metal::BufferDistributionSpec>{distribution},
        std::optional<tt::tt_metal::ShardSpecBuffer>{shard_spec},
        tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED);
    return tt::tt_metal::distributed::MeshBuffer::create(
        tt::tt_metal::distributed::ReplicatedBufferConfig{
            .size = static_cast<uint64_t>(capacity_tiles) * kTileBytes,
        },
        tt::tt_metal::distributed::DeviceLocalBufferConfig{
            .page_size = kTileBytes,
            .buffer_type = tt::tt_metal::BufferType::L1,
            .sharding_args = sharding,
            .bottom_up = false,
        },
        &mesh_device);
}

void create_cb(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& cores,
    const uint32_t cb,
    const uint32_t pages,
    const uint32_t page_bytes,
    const bool tile) {
    auto config =
        tt::tt_metal::CircularBufferConfig(pages * page_bytes, {{cb, kDataFormat}}).set_page_size(cb, page_bytes);
    if (tile) {
        config = config.set_tile_dims(cb, tt::tt_metal::Tile({32, 32}));
    }
    static_cast<void>(tt::tt_metal::CreateCircularBuffer(program, cores, config));
}

[[nodiscard]] uint32_t route_tile_count(const Lwt2DRoutePlan& route) {
    if (route.output.empty()) {
        return 0;
    }
    const size_t height = plan_2d_detail::aligned_interval_span(route.output.y, kTileHeight2D, "2D route tile height");
    const size_t width = plan_2d_detail::aligned_interval_span(route.output.x, kTileWidth2D, "2D route tile width");
    return checked_u32((height / kTileHeight2D) * (width / kTileWidth2D), "2D route tile count");
}

[[nodiscard]] uint32_t rectangle_tile_count(const IndexRectangle rectangle) {
    if (rectangle.empty()) {
        return 0;
    }
    const size_t height = plan_2d_detail::aligned_interval_span(rectangle.y, kTileHeight2D, "2D rectangle tile height");
    const size_t width = plan_2d_detail::aligned_interval_span(rectangle.x, kTileWidth2D, "2D rectangle tile width");
    return checked_u32((height / kTileHeight2D) * (width / kTileWidth2D), "2D rectangle tile count");
}

[[nodiscard]] std::vector<uint32_t> plane_addresses(const Lwt2DWorkingBuffers& buffers) {
    std::vector<uint32_t> args;
    args.reserve(2 * device_protocol::kLwt2DPlaneCount);
    for (const auto& plane : buffers.planes) {
        args.push_back(static_cast<uint32_t>(plane->get_backing_buffer()->address()));
    }
    for (size_t slot = 0; slot < buffers.planes.size(); ++slot) {
        args.push_back(0);
    }
    return args;
}

template <typename Plan>
void replace_plane_tile_counts_with_widths(std::vector<uint32_t>& args, const Plan& plan) {
    for (size_t slot = 0; slot < device_protocol::kLwt2DPlaneCount; ++slot) {
        args[device_protocol::kLwt2DPlaneCount + slot] = plan.allocated_plane_widths_elements[slot] / kTileWidth2D;
    }
}

[[nodiscard]] std::vector<uint32_t> reader_args(
    const tt::tt_metal::Buffer& input,
    const Lwt2DExecutionPlan& plan,
    const Lwt2DWorkingBuffers& buffers,
    const CoreChunkWork& work) {
    std::vector<uint32_t> args = {
        static_cast<uint32_t>(input.address()),
        checked_u32(plan.input_height, "2D input height"),
        checked_u32(plan.input_width, "2D input width"),
        checked_u32(plan.tiling.input.storage.width / kTileWidth2D, "2D input tile columns"),
        plan.y_plan.preprocess_layout.pad_config.left,
        plan.x_plan.preprocess_layout.pad_config.left,
    };
    std::vector<uint32_t> planes = plane_addresses(buffers);
    replace_plane_tile_counts_with_widths(planes, plan);
    args.insert(args.end(), planes.begin(), planes.end());
    args.push_back(static_cast<uint32_t>(buffers.chunk_config->get_backing_buffer()->address()));
    args.push_back(static_cast<uint32_t>(buffers.route_config->get_backing_buffer()->address()));
    args.push_back(work.chunk_begin);
    args.push_back(work.chunk_count);
    args.push_back(checked_u32(plan.chunks.front().routes.size(), "2D route count"));
    args.push_back(
        buffers.split_metrics ? static_cast<uint32_t>(buffers.split_metrics->get_backing_buffer()->address()) : 0U);
    args.push_back(buffers.split_metrics ? 1U : 0U);
    args.push_back(
        buffers.split_snapshots ? static_cast<uint32_t>(buffers.split_snapshots->get_backing_buffer()->address()) : 0U);
    args.push_back(buffers.split_snapshot_tiles_per_plane);
    args.push_back(
        buffers.transport_metrics ? static_cast<uint32_t>(buffers.transport_metrics->get_backing_buffer()->address())
                                  : 0U);
    args.push_back(checked_u32(plan.chunks.front().routes.size() + 1, "2D transport metric pages per chunk"));
    return args;
}

[[nodiscard]] std::vector<uint32_t> writer_args(
    const Lwt2DExecutionPlan& plan, const Lwt2DWorkingBuffers& buffers, const CoreChunkWork& work) {
    std::vector<uint32_t> args = plane_addresses(buffers);
    replace_plane_tile_counts_with_widths(args, plan);
    args.push_back(static_cast<uint32_t>(buffers.route_config->get_backing_buffer()->address()));
    args.push_back(static_cast<uint32_t>(buffers.band_config->get_backing_buffer()->address()));
    for (const auto& output : buffers.outputs) {
        args.push_back(static_cast<uint32_t>(output->get_backing_buffer()->address()));
    }
    args.push_back(checked_u32(plan.tiling.band.storage.width / kTileWidth2D, "2D band tile columns"));
    args.push_back(work.chunk_begin);
    args.push_back(work.chunk_count);
    args.push_back(checked_u32(plan.chunks.front().routes.size(), "2D route count"));
    args.push_back(buffers.route_snapshots ? 1U : 0U);
    args.push_back(
        buffers.route_snapshots ? static_cast<uint32_t>(buffers.route_snapshots->get_backing_buffer()->address()) : 0U);
    args.push_back(buffers.snapshot_tiles_per_route);
    args.push_back(
        buffers.transport_metrics ? static_cast<uint32_t>(buffers.transport_metrics->get_backing_buffer()->address())
                                  : 0U);
    args.push_back(checked_u32(plan.chunks.front().routes.size() + 1, "2D transport metric pages per chunk"));
    return args;
}

[[nodiscard]] uint32_t encoded_i32(const int64_t value, const char* label) {
    TT_FATAL(
        value >= std::numeric_limits<int32_t>::min() && value <= std::numeric_limits<int32_t>::max(),
        "{} exceeds int32_t",
        label);
    return static_cast<uint32_t>(static_cast<int32_t>(value));
}

[[nodiscard]] std::vector<uint32_t> inverse_reader_args(
    const std::array<const tt::tt_metal::Buffer*, device_protocol::kLwt2DBandCount>& bands,
    const Ilwt2DExecutionPlan& plan,
    const Lwt2DWorkingBuffers& buffers,
    const CoreChunkWork& work) {
    const auto& y_forward = plan.y_plan.forward_trace;
    const auto& x_forward = plan.x_plan.forward_trace;
    const int64_t y_canonical_start =
        static_cast<int64_t>(y_forward.preprocess_layout.pad_config.left + 1) / 2;
    const int64_t x_canonical_start =
        static_cast<int64_t>(x_forward.preprocess_layout.pad_config.left + 1) / 2;
    std::vector<uint32_t> args;
    for (const auto* band : bands) {
        args.push_back(static_cast<uint32_t>(band->address()));
    }
    args.push_back(checked_u32(plan.band_height, "2D ILWT band height"));
    args.push_back(checked_u32(plan.band_width, "2D ILWT band width"));
    args.push_back(checked_u32(plan.tiling.band.storage.width / kTileWidth2D, "2D ILWT band tile columns"));
    args.push_back(encoded_i32(y_canonical_start - y_forward.final_even_shift, "2D ILWT y-even offset"));
    args.push_back(encoded_i32(y_canonical_start - y_forward.final_odd_shift, "2D ILWT y-odd offset"));
    args.push_back(encoded_i32(x_canonical_start - x_forward.final_even_shift, "2D ILWT x-even offset"));
    args.push_back(encoded_i32(x_canonical_start - x_forward.final_odd_shift, "2D ILWT x-odd offset"));
    std::vector<uint32_t> planes = plane_addresses(buffers);
    replace_plane_tile_counts_with_widths(planes, plan);
    args.insert(args.end(), planes.begin(), planes.end());
    args.push_back(static_cast<uint32_t>(buffers.chunk_config->get_backing_buffer()->address()));
    args.push_back(static_cast<uint32_t>(buffers.route_config->get_backing_buffer()->address()));
    args.push_back(work.chunk_begin);
    args.push_back(work.chunk_count);
    args.push_back(checked_u32(plan.chunks.front().routes.size(), "2D ILWT route count"));
    // Keep the production reader's optional diagnostics argument positions
    // stable even though ILWT does not enable those validation variants.
    args.insert(args.end(), 6, 0U);
    return args;
}

[[nodiscard]] std::vector<uint32_t> inverse_writer_args(
    const Ilwt2DExecutionPlan& plan, const Lwt2DWorkingBuffers& buffers, const CoreChunkWork& work) {
    std::vector<uint32_t> args = plane_addresses(buffers);
    replace_plane_tile_counts_with_widths(args, plan);
    args.push_back(static_cast<uint32_t>(buffers.route_config->get_backing_buffer()->address()));
    args.push_back(static_cast<uint32_t>(buffers.band_config->get_backing_buffer()->address()));
    for (uint32_t band = 0; band < device_protocol::kLwt2DBandCount; ++band) {
        args.push_back(static_cast<uint32_t>(buffers.outputs[0]->get_backing_buffer()->address()));
    }
    args.push_back(checked_u32(plan.tiling.input.storage.width / kTileWidth2D, "2D ILWT output tile columns"));
    args.push_back(work.chunk_begin);
    args.push_back(work.chunk_count);
    args.push_back(checked_u32(plan.chunks.front().routes.size(), "2D ILWT route count"));
    args.push_back(0U);  // route snapshots disabled
    args.push_back(0U);
    args.push_back(0U);
    args.push_back(0U);  // transport metrics disabled
    args.push_back(0U);
    args.push_back(plan.y_plan.forward_trace.preprocess_layout.pad_config.left);
    args.push_back(plan.x_plan.forward_trace.preprocess_layout.pad_config.left);
    return args;
}

template <typename Plan>
[[nodiscard]] std::vector<uint32_t> compute_args(const Plan& plan, const CoreChunkWork& work) {
    const size_t route_count = plan.chunks.front().routes.size();
    const size_t packed_words_per_chunk = ceil_div(route_count, static_cast<size_t>(4));
    std::vector<uint32_t> args;
    args.reserve(1 + static_cast<size_t>(work.chunk_count) * packed_words_per_chunk);
    args.push_back(work.chunk_count);
    for (uint32_t local_chunk = 0; local_chunk < work.chunk_count; ++local_chunk) {
        const Lwt2DChunkPlan& chunk = plan.chunks[work.chunk_begin + local_chunk];
        for (size_t route_begin = 0; route_begin < route_count; route_begin += 4) {
            uint32_t packed_counts = 0;
            const size_t route_end = std::min(route_begin + 4, route_count);
            for (size_t route_index = route_begin; route_index < route_end; ++route_index) {
                const uint32_t count = route_tile_count(chunk.routes[route_index]);
                TT_FATAL(
                    count <= std::numeric_limits<uint8_t>::max(), "2D route tile count {} exceeds packed uint8", count);
                packed_counts |= count << (8 * (route_index - route_begin));
            }
            args.push_back(packed_counts);
        }
    }
    return args;
}

[[nodiscard]] Lwt2DProgram create_program(
    const std::filesystem::path& kernel_root,
    const tt::tt_metal::CoreRangeSet& cores,
    const tt::tt_metal::Buffer& input,
    const Lwt2DWorkingBuffers& buffers,
    const char* compute_scheme_header,
    const char* compute_scheme_type,
    const Lwt2DSplitImplementation split_implementation,
    const Lwt2DTransportPolicy transport_policy,
    const bool inverse) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    create_cb(program, cores, kSource0Cb, kTileBuffering, kTileBytes, true);
    create_cb(program, cores, kSource1Cb, kTileBuffering, kTileBytes, true);
    create_cb(program, cores, kBaseCb, kTileBuffering, kTileBytes, true);
    create_cb(program, cores, kOutputCb, kTileBuffering, kTileBytes, true);
    create_cb(program, cores, kSyncCb, 1, kNocAlignmentBytes, false);
    create_cb(program, cores, kReaderChunkConfigCb, 1, device_protocol::kLwt2DChunkConfigPageBytes, false);
    create_cb(program, cores, kReaderRouteConfigCb, 1, device_protocol::kLwt2DRouteConfigPageBytes, false);
    create_cb(program, cores, kWriterRouteConfigCb, 1, device_protocol::kLwt2DRouteConfigPageBytes, false);
    create_cb(program, cores, kWriterBandConfigCb, 1, device_protocol::kLwt2DBandConfigPageBytes, false);
    create_cb(program, cores, kNocScratchCb, device_protocol::kLwt2DSplitScratchTileCount, kTileBytes, true);
    create_cb(program, cores, kRouteZeroCb, 1, kTileBytes, true);
    std::vector<uint32_t> reader_compile_args = {
        kSource0Cb,
        kSource1Cb,
        kBaseCb,
        kSyncCb,
        kReaderChunkConfigCb,
        kReaderRouteConfigCb,
        kNocScratchCb,
        kRouteZeroCb,
    };
    tt::tt_metal::TensorAccessorArgs(input).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(*buffers.chunk_config->get_backing_buffer()).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(*buffers.route_config->get_backing_buffer()).append_to(reader_compile_args);
    if (buffers.split_metrics) {
        tt::tt_metal::TensorAccessorArgs(*buffers.split_metrics->get_backing_buffer()).append_to(reader_compile_args);
    } else {
        tt::tt_metal::TensorAccessorArgs(*buffers.chunk_config->get_backing_buffer()).append_to(reader_compile_args);
    }
    if (buffers.split_snapshots) {
        tt::tt_metal::TensorAccessorArgs(*buffers.split_snapshots->get_backing_buffer()).append_to(reader_compile_args);
    } else {
        tt::tt_metal::TensorAccessorArgs(*buffers.chunk_config->get_backing_buffer()).append_to(reader_compile_args);
    }
    if (buffers.transport_metrics) {
        tt::tt_metal::TensorAccessorArgs(*buffers.transport_metrics->get_backing_buffer())
            .append_to(reader_compile_args);
    } else {
        tt::tt_metal::TensorAccessorArgs(*buffers.chunk_config->get_backing_buffer()).append_to(reader_compile_args);
    }
    std::map<std::string, std::string> reader_defines;
    if (inverse) {
        reader_defines.emplace("TTWV_ILWT_2D", "1");
    } else if (split_implementation == Lwt2DSplitImplementation::kTiled) {
        reader_defines.emplace("TTWV_LWT_2D_TILED_SPLIT", "1");
    }
    if (buffers.split_metrics) {
        reader_defines.emplace("TTWV_CAPTURE_SPLIT_METRICS", "1");
    }
    if (buffers.split_snapshots) {
        reader_defines.emplace("TTWV_CAPTURE_SPLIT_SNAPSHOTS", "1");
    }
    if (transport_policy.route_staging == Lwt2DRouteStagingImplementation::kOptimized) {
        reader_defines.emplace("TTWV_LWT_2D_OPTIMIZED_ROUTE_STAGING", "1");
    }
    if (transport_policy.route_config == Lwt2DRouteConfigImplementation::kPreloaded) {
        reader_defines.emplace("TTWV_LWT_2D_PRELOAD_ROUTE_CONFIG", "1");
    }
    if (transport_policy.exact_transfer == Lwt2DExactTransferImplementation::kL1Copy) {
        reader_defines.emplace("TTWV_LWT_2D_EXACT_L1_COPY", "1");
    }
    if (transport_policy.validate_route_staging) {
        reader_defines.emplace("TTWV_VALIDATE_ROUTE_STAGING", "1");
    }
    if (transport_policy.compute_only_benchmark) {
        reader_defines.emplace("TTWV_LWT_2D_COMPUTE_ONLY_BENCHMARK", "1");
    }
    if (buffers.transport_metrics && !transport_policy.validate_route_staging) {
        reader_defines.emplace("TTWV_CAPTURE_TRANSPORT_METRICS", "1");
    }
    const auto reader = tt::tt_metal::CreateKernel(
        program,
        kernel_path(kernel_root, kReaderKernel),
        cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_args, reader_defines));

    std::vector<uint32_t> writer_compile_args = {
        kOutputCb,
        kSyncCb,
        kWriterRouteConfigCb,
        kWriterBandConfigCb,
        kNocScratchCb,
    };
    tt::tt_metal::TensorAccessorArgs(*buffers.route_config->get_backing_buffer()).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(*buffers.band_config->get_backing_buffer()).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(*buffers.outputs.front()->get_backing_buffer()).append_to(writer_compile_args);
    if (buffers.transport_metrics) {
        tt::tt_metal::TensorAccessorArgs(*buffers.transport_metrics->get_backing_buffer())
            .append_to(writer_compile_args);
    } else {
        tt::tt_metal::TensorAccessorArgs(*buffers.band_config->get_backing_buffer()).append_to(writer_compile_args);
    }
    std::map<std::string, std::string> writer_defines;
    if (inverse) {
        writer_defines.emplace("TTWV_ILWT_2D", "1");
    }
    if (transport_policy.route_persistence == Lwt2DRoutePersistenceImplementation::kFullTile) {
        writer_defines.emplace("TTWV_LWT_2D_FULL_TILE_PERSISTENCE", "1");
    }
    if (transport_policy.terminal_writes == Lwt2DTerminalWriteImplementation::kTiled) {
        writer_defines.emplace("TTWV_LWT_2D_TILED_TERMINAL_WRITES", "1");
    }
    if (transport_policy.route_config == Lwt2DRouteConfigImplementation::kPreloaded) {
        writer_defines.emplace("TTWV_LWT_2D_PRELOAD_ROUTE_CONFIG", "1");
    }
    if (transport_policy.validate_route_staging) {
        writer_defines.emplace("TTWV_VALIDATE_ROUTE_PERSISTENCE", "1");
    }
    if (buffers.transport_metrics && !transport_policy.validate_route_staging) {
        writer_defines.emplace("TTWV_CAPTURE_TRANSPORT_METRICS", "1");
    }
    const auto writer = tt::tt_metal::CreateKernel(
        program,
        kernel_path(kernel_root, kWriterKernel),
        cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_args, writer_defines));

    std::vector<UnpackToDestMode> unpack_modes(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_modes[kSource0Cb] = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[kSource1Cb] = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[kBaseCb] = UnpackToDestMode::UnpackToDestFp32;
    const auto compute = tt::tt_metal::CreateKernel(
        program,
        kernel_path(kernel_root, kComputeKernel),
        cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = unpack_modes,
            .compile_args = {kSource0Cb, kSource1Cb, kBaseCb, kOutputCb},
            .defines =
                {
                    {"TTWV_LWT_2D_SCHEME_HEADER", compute_scheme_header},
                    {"TTWV_LWT_2D_SCHEME_TYPE", compute_scheme_type},
                    {"TTWV_LWT_2D_FUSE_TERMINAL_SCALE",
                     !inverse && transport_policy.scale == Lwt2DScalePolicy::kFused ? "1" : "0"},
                    {"TTWV_INLINE_INVERSE_SCALE", inverse ? "1" : "0"},
                    {"TTWV_ILWT_2D", inverse ? "1" : "0"},
                },
        });
    return Lwt2DProgram{
        .program = std::move(program),
        .reader = reader,
        .compute = compute,
        .writer = writer,
    };
}

}  // namespace

Lwt2DExecutable create_lwt_2d_executable_impl(
    const std::filesystem::path& kernel_root,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const tt::tt_metal::Buffer& input_buffer,
    Lwt2DExecutionPlan plan,
    const char* compute_scheme_header,
    const char* compute_scheme_type,
    const bool capture_route_snapshots,
    const Lwt2DSplitImplementation split_implementation,
    const bool capture_split_metrics,
    const bool capture_split_snapshots,
    const Lwt2DTransportPolicy transport_policy,
    const bool capture_transport_metrics) {
    validate_lwt_2d_tiling_contract(plan.tiling);
    TT_FATAL(
        plan.input_height <= static_cast<size_t>(std::numeric_limits<int32_t>::max() / 2) &&
            plan.input_width <= static_cast<size_t>(std::numeric_limits<int32_t>::max() / 2),
        "2D input dimensions exceed the signed symmetric-index range");
    std::vector<tt::tt_metal::CoreCoord> cores = select_cores(mesh_device, plan.active_core_count);

    std::array<std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>, device_protocol::kLwt2DPlaneCount> planes;
    for (size_t slot = 0; slot < planes.size(); ++slot) {
        const uint32_t tiles =
            checked_u32(plan.allocated_plane_slot_bytes[slot] / kTileBytes, "2D workspace plane tiles");
        planes[slot] = create_l1_tile_shards(mesh_device, cores, tiles);
    }

    const size_t band_tiles =
        checked_shape_area_2d(plan.tiling.band.storage, "2D output band") / device_protocol::kLwt2DFullTileElements;
    std::array<std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>, device_protocol::kLwt2DBandCount> outputs;
    for (auto& output : outputs) {
        output = create_dram_pages(mesh_device, band_tiles, kTileBytes);
    }
    const size_t route_count = plan.chunks.front().routes.size();
    if (transport_policy.route_config == Lwt2DRouteConfigImplementation::kPreloaded) {
        constexpr size_t config_capacity =
            device_protocol::kLwt2DSplitScratchBytes / 2 - device_protocol::kLwt2DTransportMetricPageBytes;
        TT_FATAL(
            route_count * device_protocol::kLwt2DRouteConfigPageBytes <= config_capacity,
            "2D LWT {} route descriptors require {} bytes, exceeding the {}-byte per-RISC preload region",
            route_count,
            route_count * device_protocol::kLwt2DRouteConfigPageBytes,
            config_capacity);
    }
    auto chunk_config = create_dram_pages(mesh_device, plan.chunks.size(), device_protocol::kLwt2DChunkConfigPageBytes);
    auto route_config =
        create_dram_pages(mesh_device, plan.chunks.size() * route_count, device_protocol::kLwt2DRouteConfigPageBytes);
    auto band_config = create_dram_pages(mesh_device, plan.chunks.size(), device_protocol::kLwt2DBandConfigPageBytes);
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> split_metrics;
    if (capture_split_metrics) {
        split_metrics = create_dram_pages(mesh_device, plan.chunks.size(), device_protocol::kLwt2DSplitMetricPageBytes);
    }
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> transport_metrics;
    if (capture_transport_metrics) {
        const size_t metric_pages_per_chunk = route_count + 1;
        transport_metrics = create_dram_pages(
            mesh_device, plan.chunks.size() * metric_pages_per_chunk, device_protocol::kLwt2DTransportMetricPageBytes);
    }
    uint32_t split_snapshot_tiles_per_plane = 0;
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> split_snapshots;
    if (capture_split_snapshots) {
        for (const Lwt2DChunkPlan& chunk : plan.chunks) {
            const std::array<IndexRectangle, 4> initial = {
                chunk.initial.ee,
                chunk.initial.eo,
                chunk.initial.oe,
                chunk.initial.oo,
            };
            for (const IndexRectangle rectangle : initial) {
                split_snapshot_tiles_per_plane =
                    std::max(split_snapshot_tiles_per_plane, rectangle_tile_count(rectangle));
            }
        }
        TT_FATAL(split_snapshot_tiles_per_plane > 0, "2D split snapshot capture found no initial plane tiles");
        split_snapshots =
            create_dram_pages(mesh_device, plan.chunks.size() * 4 * split_snapshot_tiles_per_plane, kTileBytes);
    }
    uint32_t snapshot_tiles_per_route = 0;
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> route_snapshots;
    if (capture_route_snapshots) {
        for (const Lwt2DChunkPlan& chunk : plan.chunks) {
            for (const Lwt2DRoutePlan& route : chunk.routes) {
                snapshot_tiles_per_route = std::max(snapshot_tiles_per_route, route_tile_count(route));
            }
        }
        TT_FATAL(snapshot_tiles_per_route > 0, "2D route snapshot capture found no executable routes");
        route_snapshots =
            create_dram_pages(mesh_device, plan.chunks.size() * route_count * snapshot_tiles_per_route, kTileBytes);
    }

    const uint64_t l1_capacity = mesh_device.l1_size_per_core();
    TT_FATAL(
        plan.allocated_l1_bytes <= l1_capacity,
        "2D LWT allocation requires {} L1 bytes but hardware exposes {}",
        plan.allocated_l1_bytes,
        l1_capacity);
    Lwt2DWorkingBuffers buffers{
        .planes = std::move(planes),
        .outputs = std::move(outputs),
        .chunk_config = std::move(chunk_config),
        .route_config = std::move(route_config),
        .band_config = std::move(band_config),
        .split_metrics = std::move(split_metrics),
        .transport_metrics = std::move(transport_metrics),
        .split_snapshots = std::move(split_snapshots),
        .split_snapshot_tiles_per_plane = split_snapshot_tiles_per_plane,
        .route_snapshots = std::move(route_snapshots),
        .snapshot_tiles_per_route = snapshot_tiles_per_route,
        .cores = std::move(cores),
        .scheduler =
            Lwt2DSchedulerTelemetry{
                .architecture = mesh_device.arch(),
                .logical_input = plan.tiling.input.logical,
                .padded_input = plan.tiling.input.storage,
                .logical_band = plan.tiling.band.logical,
                .padded_band = plan.tiling.band.storage,
                .active_core_count = plan.active_core_count,
                .chunk_count = checked_u32(plan.chunks.size(), "2D chunk count"),
                .chunk_tiles_y = plan.chunk_tiles_y,
                .chunk_tiles_x = plan.chunk_tiles_x,
                .route_count = checked_u32(plan.chunks.front().routes.size(), "2D route count"),
                .executable_route_count = plan.executable_route_count,
                .scale_routes_removed = plan.scale_routes_removed,
                .latency_oriented_planner = plan.latency_oriented_planner,
                .route_domain = plan.route_domain,
                .estimated_latency_cycles = plan.estimated_latency_cycles,
                .max_dependency_overhead = plan.max_dependency_overhead,
                .l1_workspace_bytes = plan.allocated_workspace_bytes,
                .l1_circular_buffer_bytes = plan_2d_detail::kCircularBufferBytes,
                .l1_metadata_bytes = plan_2d_detail::kMetadataBytes,
                .l1_synchronization_bytes = plan_2d_detail::kSynchronizationBytes,
                .l1_total_bytes = plan.allocated_l1_bytes,
                .l1_capacity_bytes = l1_capacity,
                .l1_headroom_bytes = l1_capacity - plan.allocated_l1_bytes,
                .exact_initial_elements = plan.exact_initial_elements,
                .internal_initial_elements = plan.internal_initial_elements,
                .exact_route_elements = plan.exact_route_elements,
                .internal_route_elements = plan.internal_route_elements,
                .exact_final_elements = plan.exact_final_elements,
                .internal_final_elements = plan.internal_final_elements,
            },
    };
    const std::vector<CoreChunkWork> work =
        partition_work(buffers.cores, checked_u32(plan.chunks.size(), "2D chunk count"));
    Lwt2DProgram program = create_program(
        kernel_root,
        core_set(buffers.cores),
        input_buffer,
        buffers,
        compute_scheme_header,
        compute_scheme_type,
        split_implementation,
        transport_policy,
        false);
    for (const CoreChunkWork& core_work : work) {
        tt::tt_metal::SetRuntimeArgs(
            program.program, program.reader, core_work.core, reader_args(input_buffer, plan, buffers, core_work));
        tt::tt_metal::SetRuntimeArgs(program.program, program.compute, core_work.core, compute_args(plan, core_work));
        tt::tt_metal::SetRuntimeArgs(
            program.program, program.writer, core_work.core, writer_args(plan, buffers, core_work));
    }
    tt::tt_metal::distributed::MeshWorkload workload;
    workload.add_program(
        tt::tt_metal::distributed::MeshCoordinateRange(mesh_device.shape()), std::move(program.program));
    return Lwt2DExecutable{
        .plan = std::move(plan),
        .buffers = std::move(buffers),
        .workload = std::move(workload),
    };
}

Ilwt2DExecutable create_ilwt_2d_executable_impl(
    const std::filesystem::path& kernel_root,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const std::array<const tt::tt_metal::Buffer*, device_protocol::kLwt2DBandCount>& band_buffers,
    Ilwt2DExecutionPlan plan,
    const char* inverse_compute_scheme_header,
    const char* inverse_compute_scheme_type) {
    TT_FATAL(!plan.chunks.empty(), "2D ILWT requires at least one planned chunk");
    TT_FATAL(
        plan.output_height <= static_cast<size_t>(std::numeric_limits<int32_t>::max() / 2) &&
            plan.output_width <= static_cast<size_t>(std::numeric_limits<int32_t>::max() / 2),
        "2D ILWT output dimensions exceed device signed-index limits");
    std::vector<tt::tt_metal::CoreCoord> cores = select_cores(mesh_device, plan.active_core_count);

    std::array<std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>, device_protocol::kLwt2DPlaneCount> planes;
    for (size_t slot = 0; slot < planes.size(); ++slot) {
        const uint32_t tiles =
            checked_u32(plan.allocated_plane_slot_bytes[slot] / kTileBytes, "2D ILWT workspace plane tiles");
        planes[slot] = create_l1_tile_shards(mesh_device, cores, tiles);
    }

    const size_t output_tiles =
        checked_shape_area_2d(plan.tiling.input.storage, "2D ILWT output") /
        device_protocol::kLwt2DFullTileElements;
    auto output = create_dram_pages(mesh_device, output_tiles, kTileBytes);
    std::array<std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>, device_protocol::kLwt2DBandCount> outputs;
    outputs.fill(output);

    const size_t route_count = plan.chunks.front().routes.size();
    constexpr size_t config_capacity =
        device_protocol::kLwt2DSplitScratchBytes / 2 - device_protocol::kLwt2DTransportMetricPageBytes;
    TT_FATAL(
        route_count * device_protocol::kLwt2DRouteConfigPageBytes <= config_capacity,
        "2D ILWT route descriptors exceed the per-RISC preload region");
    auto chunk_config = create_dram_pages(mesh_device, plan.chunks.size(), device_protocol::kLwt2DChunkConfigPageBytes);
    auto route_config =
        create_dram_pages(mesh_device, plan.chunks.size() * route_count, device_protocol::kLwt2DRouteConfigPageBytes);
    auto band_config = create_dram_pages(mesh_device, plan.chunks.size(), device_protocol::kLwt2DBandConfigPageBytes);

    const uint64_t l1_capacity = mesh_device.l1_size_per_core();
    TT_FATAL(
        plan.allocated_l1_bytes <= l1_capacity,
        "2D ILWT allocation requires {} L1 bytes but hardware exposes {}",
        plan.allocated_l1_bytes,
        l1_capacity);
    Lwt2DWorkingBuffers buffers{
        .planes = std::move(planes),
        .outputs = std::move(outputs),
        .chunk_config = std::move(chunk_config),
        .route_config = std::move(route_config),
        .band_config = std::move(band_config),
        .cores = std::move(cores),
        .scheduler =
            Lwt2DSchedulerTelemetry{
                .architecture = mesh_device.arch(),
                .logical_input = plan.tiling.input.logical,
                .padded_input = plan.tiling.input.storage,
                .logical_band = plan.tiling.band.logical,
                .padded_band = plan.tiling.band.storage,
                .active_core_count = plan.active_core_count,
                .chunk_count = checked_u32(plan.chunks.size(), "2D ILWT chunk count"),
                .chunk_tiles_y = plan.chunk_tiles_y,
                .chunk_tiles_x = plan.chunk_tiles_x,
                .route_count = checked_u32(route_count, "2D ILWT route count"),
                .executable_route_count = plan.executable_route_count,
                .scale_routes_removed = checked_u32(route_count - plan.executable_route_count, "2D ILWT metadata routes"),
                .latency_oriented_planner = true,
                .max_dependency_overhead = plan.max_dependency_overhead,
                .l1_workspace_bytes = plan.allocated_workspace_bytes,
                .l1_circular_buffer_bytes = plan_2d_detail::kCircularBufferBytes,
                .l1_metadata_bytes = plan_2d_detail::kMetadataBytes,
                .l1_synchronization_bytes = plan_2d_detail::kSynchronizationBytes,
                .l1_total_bytes = plan.allocated_l1_bytes,
                .l1_capacity_bytes = l1_capacity,
                .l1_headroom_bytes = l1_capacity - plan.allocated_l1_bytes,
            },
    };

    const std::vector<CoreChunkWork> work =
        partition_work(buffers.cores, checked_u32(plan.chunks.size(), "2D ILWT chunk count"));
    const Lwt2DTransportPolicy transport_policy{};
    Lwt2DProgram program = create_program(
        kernel_root,
        core_set(buffers.cores),
        *band_buffers[0],
        buffers,
        inverse_compute_scheme_header,
        inverse_compute_scheme_type,
        Lwt2DSplitImplementation::kTiled,
        transport_policy,
        true);
    for (const CoreChunkWork& core_work : work) {
        tt::tt_metal::SetRuntimeArgs(
            program.program,
            program.reader,
            core_work.core,
            inverse_reader_args(band_buffers, plan, buffers, core_work));
        tt::tt_metal::SetRuntimeArgs(program.program, program.compute, core_work.core, compute_args(plan, core_work));
        tt::tt_metal::SetRuntimeArgs(
            program.program, program.writer, core_work.core, inverse_writer_args(plan, buffers, core_work));
    }
    tt::tt_metal::distributed::MeshWorkload workload;
    workload.add_program(
        tt::tt_metal::distributed::MeshCoordinateRange(mesh_device.shape()), std::move(program.program));
    return Ilwt2DExecutable{
        .plan = std::move(plan),
        .buffers = std::move(buffers),
        .workload = std::move(workload),
    };
}

void prepare_lwt_2d(tt::tt_metal::distributed::MeshCommandQueue& command_queue, Lwt2DExecutable& executable) {
    const std::vector<uint32_t> chunks = build_lwt_2d_chunk_config_words(executable.plan);
    const std::vector<uint32_t> routes = build_lwt_2d_route_config_words(executable.plan);
    const std::vector<uint32_t> bands = build_lwt_2d_band_config_words(executable.plan);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(command_queue, executable.buffers.chunk_config, chunks, false);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(command_queue, executable.buffers.route_config, routes, false);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(command_queue, executable.buffers.band_config, bands, false);

    const size_t output_elements = checked_shape_area_2d(executable.plan.tiling.band.storage, "2D output band");
    const std::vector<float> zeros(output_elements, 0.0F);
    for (auto& output : executable.buffers.outputs) {
        tt::tt_metal::distributed::EnqueueWriteMeshBuffer(command_queue, output, zeros, false);
    }
    if (executable.buffers.route_snapshots) {
        const size_t snapshot_elements = executable.plan.chunks.size() * executable.plan.chunks.front().routes.size() *
                                         executable.buffers.snapshot_tiles_per_route *
                                         device_protocol::kLwt2DFullTileElements;
        const std::vector<float> snapshot_zeros(snapshot_elements, 0.0F);
        tt::tt_metal::distributed::EnqueueWriteMeshBuffer(
            command_queue, executable.buffers.route_snapshots, snapshot_zeros, false);
    }
    if (executable.buffers.split_metrics) {
        const std::vector<uint32_t> metric_zeros(
            executable.plan.chunks.size() * device_protocol::kLwt2DSplitMetricWordCount, 0);
        tt::tt_metal::distributed::EnqueueWriteMeshBuffer(
            command_queue, executable.buffers.split_metrics, metric_zeros, false);
    }
    if (executable.buffers.transport_metrics) {
        const size_t pages_per_chunk = executable.plan.chunks.front().routes.size() + 1;
        const std::vector<uint32_t> metric_zeros(
            executable.plan.chunks.size() * pages_per_chunk * device_protocol::kLwt2DTransportMetricWordCount, 0);
        tt::tt_metal::distributed::EnqueueWriteMeshBuffer(
            command_queue, executable.buffers.transport_metrics, metric_zeros, false);
    }
    if (executable.buffers.split_snapshots) {
        const size_t snapshot_elements = executable.plan.chunks.size() * 4 *
                                         executable.buffers.split_snapshot_tiles_per_plane *
                                         device_protocol::kLwt2DFullTileElements;
        const std::vector<float> snapshot_zeros(snapshot_elements, 0.0F);
        tt::tt_metal::distributed::EnqueueWriteMeshBuffer(
            command_queue, executable.buffers.split_snapshots, snapshot_zeros, false);
    }
    tt::tt_metal::distributed::Finish(command_queue);
}

void prepare_ilwt_2d(
    tt::tt_metal::distributed::MeshCommandQueue& command_queue, Ilwt2DExecutable& executable) {
    const std::vector<uint32_t> chunks = build_ilwt_2d_chunk_config_words(executable.plan);
    const std::vector<uint32_t> routes = build_ilwt_2d_route_config_words(executable.plan);
    const std::vector<uint32_t> terminal = build_ilwt_2d_band_config_words(executable.plan);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(command_queue, executable.buffers.chunk_config, chunks, false);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(command_queue, executable.buffers.route_config, routes, false);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(command_queue, executable.buffers.band_config, terminal, false);
    const size_t output_elements = checked_shape_area_2d(executable.plan.tiling.input.storage, "2D ILWT output");
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(
        command_queue, executable.buffers.outputs[0], std::vector<float>(output_elements, 0.0F), false);
    tt::tt_metal::distributed::Finish(command_queue);
}

void execute_ilwt_2d(
    tt::tt_metal::distributed::MeshDevice&,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    Ilwt2DExecutable& executable) {
    tt::tt_metal::distributed::EnqueueMeshWorkload(command_queue, executable.workload, false);
    tt::tt_metal::distributed::Finish(command_queue);
}

void execute_lwt_2d(
    tt::tt_metal::distributed::MeshDevice&,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    Lwt2DExecutable& executable) {
    tt::tt_metal::distributed::EnqueueMeshWorkload(command_queue, executable.workload, false);
    tt::tt_metal::distributed::Finish(command_queue);
}

Lwt2DSplitMetricsSummary read_lwt_2d_split_metrics(
    tt::tt_metal::distributed::MeshCommandQueue& command_queue, Lwt2DExecutable& executable) {
    TT_FATAL(executable.buffers.split_metrics, "2D split metrics were not enabled for this executable");
    std::vector<uint32_t> words;
    tt::tt_metal::distributed::EnqueueReadMeshBuffer(command_queue, words, executable.buffers.split_metrics, true);
    TT_FATAL(
        words.size() == executable.plan.chunks.size() * device_protocol::kLwt2DSplitMetricWordCount,
        "2D split metric buffer has an unexpected size");

    Lwt2DSplitMetricsSummary summary{};
    uint32_t chunk_begin = 0;
    const uint32_t core_count = checked_u32(executable.buffers.cores.size(), "2D split metric core count");
    const uint32_t chunk_count = checked_u32(executable.plan.chunks.size(), "2D split metric chunk count");
    const uint32_t base = chunk_count / core_count;
    const uint32_t extra = chunk_count % core_count;
    for (uint32_t core = 0; core < core_count; ++core) {
        const uint32_t core_chunks = base + (core < extra ? 1U : 0U);
        uint64_t core_cycles = 0;
        uint64_t core_macro_tiles = 0;
        for (uint32_t local_chunk = 0; local_chunk < core_chunks; ++local_chunk) {
            const size_t offset =
                static_cast<size_t>(chunk_begin + local_chunk) * device_protocol::kLwt2DSplitMetricWordCount;
            core_cycles += static_cast<uint64_t>(words[offset + device_protocol::kLwt2DSplitMetricCyclesLow]) |
                           (static_cast<uint64_t>(words[offset + device_protocol::kLwt2DSplitMetricCyclesHigh]) << 32);
            summary.raw_input_bytes += words[offset + device_protocol::kLwt2DSplitMetricInputBytes];
            summary.local_output_bytes += words[offset + device_protocol::kLwt2DSplitMetricLocalOutputBytes];
            summary.noc_read_calls += words[offset + device_protocol::kLwt2DSplitMetricNocReadCalls];
            summary.noc_read_barriers += words[offset + device_protocol::kLwt2DSplitMetricNocReadBarriers];
            summary.interior_macro_tiles += words[offset + device_protocol::kLwt2DSplitMetricInteriorMacroTiles];
            summary.boundary_macro_tiles += words[offset + device_protocol::kLwt2DSplitMetricBoundaryMacroTiles];
            core_macro_tiles += words[offset + device_protocol::kLwt2DSplitMetricInteriorMacroTiles] +
                                words[offset + device_protocol::kLwt2DSplitMetricBoundaryMacroTiles];
        }
        summary.max_core_cycles = std::max(summary.max_core_cycles, core_cycles);
        summary.max_core_macro_tiles = std::max(summary.max_core_macro_tiles, core_macro_tiles);
        chunk_begin += core_chunks;
    }
    TT_FATAL(chunk_begin == chunk_count, "2D split metric chunk partition is incomplete");
    return summary;
}

namespace {

[[nodiscard]] uint64_t metric_u64(const std::vector<uint32_t>& words, const size_t offset) {
    return static_cast<uint64_t>(words[offset]) | (static_cast<uint64_t>(words[offset + 1]) << 32);
}

}  // namespace

Lwt2DTransportMetricsSummary read_lwt_2d_transport_metrics(
    tt::tt_metal::distributed::MeshCommandQueue& command_queue, Lwt2DExecutable& executable) {
    TT_FATAL(executable.buffers.transport_metrics, "2D transport metrics were not enabled for this executable");
    std::vector<uint32_t> words;
    tt::tt_metal::distributed::EnqueueReadMeshBuffer(command_queue, words, executable.buffers.transport_metrics, true);
    const size_t route_count = executable.plan.chunks.front().routes.size();
    const size_t pages_per_chunk = route_count + 1;
    TT_FATAL(
        words.size() ==
            executable.plan.chunks.size() * pages_per_chunk * device_protocol::kLwt2DTransportMetricWordCount,
        "2D transport metric buffer has an unexpected size");

    Lwt2DTransportMetricsSummary summary{};
    summary.routes.reserve(executable.plan.chunks.size() * route_count);
    struct ChunkPhases {
        uint64_t reader_kernel{0};
        uint64_t writer_kernel{0};
        uint64_t route_tiles{0};
        uint64_t config{0};
        uint64_t staging{0};
        uint64_t compute_pipeline{0};
        uint64_t persistence{0};
        uint64_t synchronization_wait{0};
        uint64_t terminal_write{0};
    };
    std::vector<ChunkPhases> chunk_phases(executable.plan.chunks.size());
    for (size_t chunk = 0; chunk < executable.plan.chunks.size(); ++chunk) {
        for (size_t route = 0; route < route_count; ++route) {
            const size_t offset = (chunk * pages_per_chunk + route) * device_protocol::kLwt2DTransportMetricWordCount;
            Lwt2DRouteTransportMetric metric{
                .chunk_index = checked_u32(chunk, "2D transport metric chunk index"),
                .route_index = checked_u32(route, "2D transport metric route index"),
                .axis = words[offset + device_protocol::kLwt2DTransportMetricAxis],
                .step_type = words[offset + device_protocol::kLwt2DTransportMetricStepType],
                .coefficient_count = words[offset + device_protocol::kLwt2DTransportMetricCoefficientCount],
                .output_tiles = words[offset + device_protocol::kLwt2DTransportMetricOutputTiles],
                .reader_config_cycles =
                    metric_u64(words, offset + device_protocol::kLwt2DTransportMetricReaderConfigCyclesLow),
                .staging_cycles = metric_u64(words, offset + device_protocol::kLwt2DTransportMetricStagingCyclesLow),
                .compute_cycles = metric_u64(words, offset + device_protocol::kLwt2DTransportMetricComputeCyclesLow),
                .persistence_cycles =
                    metric_u64(words, offset + device_protocol::kLwt2DTransportMetricPersistenceCyclesLow),
                .synchronization_wait_cycles =
                    metric_u64(words, offset + device_protocol::kLwt2DTransportMetricSyncWaitCyclesLow),
                .writer_config_cycles =
                    metric_u64(words, offset + device_protocol::kLwt2DTransportMetricWriterConfigCyclesLow),
                .exact_source_tiles = words[offset + device_protocol::kLwt2DTransportMetricExactSourceTiles],
                .shifted_source_tiles = words[offset + device_protocol::kLwt2DTransportMetricShiftedSourceTiles],
                .generic_source_tiles = words[offset + device_protocol::kLwt2DTransportMetricGenericSourceTiles],
                .exact_base_tiles = words[offset + device_protocol::kLwt2DTransportMetricExactBaseTiles],
                .shifted_base_tiles = words[offset + device_protocol::kLwt2DTransportMetricShiftedBaseTiles],
                .generic_base_tiles = words[offset + device_protocol::kLwt2DTransportMetricGenericBaseTiles],
                .persistence_tiles = words[offset + device_protocol::kLwt2DTransportMetricPersistenceTiles],
            };
            if (metric.output_tiles == 0) {
                continue;
            }
            summary.max_route_staging_cycles = std::max(summary.max_route_staging_cycles, metric.staging_cycles);
            summary.max_route_compute_cycles = std::max(summary.max_route_compute_cycles, metric.compute_cycles);
            summary.max_route_persistence_cycles =
                std::max(summary.max_route_persistence_cycles, metric.persistence_cycles);
            summary.max_route_synchronization_wait_cycles =
                std::max(summary.max_route_synchronization_wait_cycles, metric.synchronization_wait_cycles);
            summary.total_route_tiles += metric.output_tiles;
            summary.exact_source_tiles += metric.exact_source_tiles;
            summary.shifted_source_tiles += metric.shifted_source_tiles;
            summary.generic_source_tiles += metric.generic_source_tiles;
            summary.exact_base_tiles += metric.exact_base_tiles;
            summary.shifted_base_tiles += metric.shifted_base_tiles;
            summary.generic_base_tiles += metric.generic_base_tiles;
            chunk_phases[chunk].route_tiles += metric.output_tiles;
            chunk_phases[chunk].config += metric.reader_config_cycles + metric.writer_config_cycles;
            chunk_phases[chunk].staging += metric.staging_cycles;
            chunk_phases[chunk].compute_pipeline += metric.compute_cycles;
            chunk_phases[chunk].persistence += metric.persistence_cycles;
            chunk_phases[chunk].synchronization_wait += metric.synchronization_wait_cycles;
            summary.routes.push_back(metric);
        }
        const size_t summary_offset =
            (chunk * pages_per_chunk + route_count) * device_protocol::kLwt2DTransportMetricWordCount;
        chunk_phases[chunk].reader_kernel =
            metric_u64(words, summary_offset + device_protocol::kLwt2DTransportMetricStagingCyclesLow);
        chunk_phases[chunk].writer_kernel =
            metric_u64(words, summary_offset + device_protocol::kLwt2DTransportMetricKernelCyclesLow);
        chunk_phases[chunk].terminal_write =
            metric_u64(words, summary_offset + device_protocol::kLwt2DTransportMetricTerminalWriteCyclesLow);
        chunk_phases[chunk].config +=
            metric_u64(words, summary_offset + device_protocol::kLwt2DTransportMetricReaderConfigCyclesLow) +
            metric_u64(words, summary_offset + device_protocol::kLwt2DTransportMetricWriterConfigCyclesLow);
        summary.max_reader_kernel_cycles =
            std::max(summary.max_reader_kernel_cycles, chunk_phases[chunk].reader_kernel);
        summary.max_writer_kernel_cycles =
            std::max(summary.max_writer_kernel_cycles, chunk_phases[chunk].writer_kernel);
        summary.exact_terminal_tiles +=
            words[summary_offset + device_protocol::kLwt2DTransportMetricExactTerminalTiles];
        summary.fragmented_terminal_tiles +=
            words[summary_offset + device_protocol::kLwt2DTransportMetricFragmentedTerminalTiles];
        summary.validated_staging_tiles +=
            words[summary_offset + device_protocol::kLwt2DTransportMetricValidatedStagingTiles];
        summary.staging_validation_mismatches +=
            words[summary_offset + device_protocol::kLwt2DTransportMetricStagingValidationMismatches];
        summary.validation_exact_mismatches +=
            words[summary_offset + device_protocol::kLwt2DTransportMetricValidationExactMismatches];
        summary.validation_shifted_mismatches +=
            words[summary_offset + device_protocol::kLwt2DTransportMetricValidationShiftedMismatches];
        summary.validation_two_axis_mismatches +=
            words[summary_offset + device_protocol::kLwt2DTransportMetricValidationTwoAxisMismatches];
        summary.validation_partial_mismatches +=
            words[summary_offset + device_protocol::kLwt2DTransportMetricValidationPartialMismatches];
        summary.validation_empty_mismatches +=
            words[summary_offset + device_protocol::kLwt2DTransportMetricValidationEmptyMismatches];
        summary.validated_persistence_tiles +=
            words[summary_offset + device_protocol::kLwt2DTransportMetricValidatedPersistenceTiles];
        summary.persistence_validation_mismatches +=
            words[summary_offset + device_protocol::kLwt2DTransportMetricPersistenceValidationMismatches];
        summary.max_terminal_write_cycles =
            std::max(summary.max_terminal_write_cycles, chunk_phases[chunk].terminal_write);
    }

    const uint32_t core_count = executable.buffers.scheduler.active_core_count;
    const size_t chunks_per_core = chunk_phases.size() / core_count;
    const size_t extra_chunks = chunk_phases.size() % core_count;
    size_t chunk_begin = 0;
    long double total_core_cycles = 0;
    long double total_config = 0;
    long double total_staging = 0;
    long double total_compute_pipeline = 0;
    long double total_persistence = 0;
    long double total_sync = 0;
    long double total_terminal = 0;
    for (uint32_t core = 0; core < core_count; ++core) {
        const size_t core_chunks = chunks_per_core + (core < extra_chunks ? 1U : 0U);
        ChunkPhases core_phases{};
        for (size_t local_chunk = 0; local_chunk < core_chunks; ++local_chunk) {
            const ChunkPhases& chunk = chunk_phases[chunk_begin + local_chunk];
            core_phases.reader_kernel += chunk.reader_kernel;
            core_phases.writer_kernel += chunk.writer_kernel;
            core_phases.route_tiles += chunk.route_tiles;
            core_phases.config += chunk.config;
            core_phases.staging += chunk.staging;
            core_phases.compute_pipeline += chunk.compute_pipeline;
            core_phases.persistence += chunk.persistence;
            core_phases.synchronization_wait += chunk.synchronization_wait;
            core_phases.terminal_write += chunk.terminal_write;
        }
        const uint64_t core_cycles = std::max(core_phases.reader_kernel, core_phases.writer_kernel);
        summary.max_core_cycles = std::max(summary.max_core_cycles, core_cycles);
        summary.max_core_route_tiles = std::max(summary.max_core_route_tiles, core_phases.route_tiles);
        summary.max_core_config_cycles = std::max(summary.max_core_config_cycles, core_phases.config);
        summary.max_core_staging_cycles = std::max(summary.max_core_staging_cycles, core_phases.staging);
        summary.max_core_compute_pipeline_cycles =
            std::max(summary.max_core_compute_pipeline_cycles, core_phases.compute_pipeline);
        summary.max_core_persistence_cycles = std::max(summary.max_core_persistence_cycles, core_phases.persistence);
        summary.max_core_synchronization_wait_cycles =
            std::max(summary.max_core_synchronization_wait_cycles, core_phases.synchronization_wait);
        summary.max_core_terminal_write_cycles =
            std::max(summary.max_core_terminal_write_cycles, core_phases.terminal_write);
        total_core_cycles += core_cycles;
        total_config += core_phases.config;
        total_staging += core_phases.staging;
        total_compute_pipeline += core_phases.compute_pipeline;
        total_persistence += core_phases.persistence;
        total_sync += core_phases.synchronization_wait;
        total_terminal += core_phases.terminal_write;
        chunk_begin += core_chunks;
    }
    TT_FATAL(chunk_begin == chunk_phases.size(), "2D transport metric chunk partition is incomplete");
    summary.mean_active_core_cycles = static_cast<double>(total_core_cycles / core_count);
    summary.mean_active_core_config_cycles = static_cast<double>(total_config / core_count);
    summary.mean_active_core_staging_cycles = static_cast<double>(total_staging / core_count);
    summary.mean_active_core_compute_pipeline_cycles = static_cast<double>(total_compute_pipeline / core_count);
    summary.mean_active_core_persistence_cycles = static_cast<double>(total_persistence / core_count);
    summary.mean_active_core_synchronization_wait_cycles = static_cast<double>(total_sync / core_count);
    summary.mean_active_core_terminal_write_cycles = static_cast<double>(total_terminal / core_count);
    return summary;
}

}  // namespace ttwv
