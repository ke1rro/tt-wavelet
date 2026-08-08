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
#include "tt_wavelet/include/lifting/policy.hpp"

namespace ttwv {
namespace {

constexpr tt::DataFormat kDataFormat = tt::DataFormat::Float32;
constexpr uint32_t kTileBytes = device_protocol::kLwt2DFullTileBytes;
constexpr uint32_t kSource0Cb = tt::CBIndex::c_0;
constexpr uint32_t kSource1Cb = tt::CBIndex::c_1;
constexpr uint32_t kBaseCb = tt::CBIndex::c_2;
constexpr uint32_t kSyncCb = tt::CBIndex::c_3;
constexpr uint32_t kReaderChunkConfigCb = tt::CBIndex::c_4;
constexpr uint32_t kWriterBandConfigCb = tt::CBIndex::c_7;
constexpr uint32_t kNocScratchCb = tt::CBIndex::c_8;
constexpr uint32_t kRouteZeroCb = tt::CBIndex::c_9;
constexpr uint32_t kOutputCb = tt::CBIndex::c_16;
constexpr uint32_t kTileBuffering = 2;
// Wormhole needs 32-byte DRAM-read alignment and Blackhole needs 64.  Using
// the stricter common alignment keeps all metadata CB bases portable.
constexpr uint32_t kConfigNocAlignmentBytes = 64;

constexpr const char* kReaderKernel = "kernels/dataflow/lwt_2d_reader.cpp";
constexpr const char* kComputeKernel = "kernels/compute/lwt_2d_compute.cpp";
constexpr const char* kWriterKernel = "kernels/dataflow/lwt_2d_writer.cpp";

[[nodiscard]] constexpr uint32_t split_scratch_tile_count(const BoundaryMode boundary_mode, const bool inverse) {
    return !inverse && boundary_mode == BoundaryMode::kSymmetric ? device_protocol::kLwt2DSymmetricSplitScratchTileCount
                                                                 : device_protocol::kLwt2DSplitScratchTileCount;
}

struct Lwt2DProgram {
    tt::tt_metal::Program program;
    tt::tt_metal::KernelHandle reader{};
    std::optional<tt::tt_metal::KernelHandle> compute{};
    std::optional<tt::tt_metal::KernelHandle> writer{};
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

[[nodiscard]] uint32_t noc_scratch_tile_count(
    const BoundaryMode boundary_mode, const bool inverse, const size_t route_count) {
    TT_FATAL(
        route_count <= std::numeric_limits<size_t>::max() / (2 * device_protocol::kLwt2DRouteConfigPageBytes),
        "2D route-config scratch size overflows size_t");
    const size_t route_config_bytes = route_count * device_protocol::kLwt2DRouteConfigPageBytes;
    const size_t route_config_tile_count = ceil_div(2 * route_config_bytes, static_cast<size_t>(kTileBytes));
    const size_t tile_count =
        std::max(static_cast<size_t>(split_scratch_tile_count(boundary_mode, inverse)), route_config_tile_count);
    TT_FATAL(
        tile_count <= device_protocol::kLwt2DSplitScratchTileCount,
        "2D route descriptors require {} scratch tiles, exceeding the {}-tile accounted split-scratch budget",
        tile_count,
        device_protocol::kLwt2DSplitScratchTileCount);
    return checked_u32(tile_count, "2D NoC scratch tile count");
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
    const CoreChunkWork& work,
    const uint32_t chunks_per_sample,
    const uint32_t input_tiles_per_sample) {
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
    args.push_back(chunks_per_sample);
    args.push_back(input_tiles_per_sample);
    return args;
}

[[nodiscard]] std::vector<uint32_t> writer_args(
    const Lwt2DExecutionPlan& plan,
    const Lwt2DWorkingBuffers& buffers,
    const CoreChunkWork& work,
    const uint32_t chunks_per_sample,
    const uint32_t output_tiles_per_sample) {
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
    args.push_back(chunks_per_sample);
    args.push_back(output_tiles_per_sample);
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
    const CoreChunkWork& work,
    const uint32_t chunks_per_sample,
    const uint32_t input_tiles_per_sample) {
    const auto& y_forward = plan.y_plan.forward_trace;
    const auto& x_forward = plan.x_plan.forward_trace;
    const int64_t y_canonical_start = static_cast<int64_t>(y_forward.preprocess_layout.pad_config.left + 1) / 2;
    const int64_t x_canonical_start = static_cast<int64_t>(x_forward.preprocess_layout.pad_config.left + 1) / 2;
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
    args.push_back(chunks_per_sample);
    args.push_back(input_tiles_per_sample);
    return args;
}

[[nodiscard]] std::vector<uint32_t> inverse_writer_args(
    const Ilwt2DExecutionPlan& plan,
    const Lwt2DWorkingBuffers& buffers,
    const CoreChunkWork& work,
    const uint32_t chunks_per_sample,
    const uint32_t output_tiles_per_sample) {
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
    args.push_back(plan.y_plan.forward_trace.preprocess_layout.pad_config.left);
    args.push_back(plan.x_plan.forward_trace.preprocess_layout.pad_config.left);
    args.push_back(chunks_per_sample);
    args.push_back(output_tiles_per_sample);
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
        const Lwt2DChunkPlan& chunk = plan.chunks[(work.chunk_begin + local_chunk) % plan.chunks.size()];
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
    const BoundaryMode boundary_mode,
    const bool compact_boundary_code,
    const bool inverse,
    const uint32_t scratch_tile_count) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    const uint32_t scratch_bytes = scratch_tile_count * kTileBytes;
    create_cb(program, cores, kSource0Cb, kTileBuffering, kTileBytes, true);
    create_cb(program, cores, kSource1Cb, kTileBuffering, kTileBytes, true);
    create_cb(program, cores, kBaseCb, kTileBuffering, kTileBytes, true);
    create_cb(program, cores, kOutputCb, kTileBuffering, kTileBytes, true);
    create_cb(program, cores, kSyncCb, 1, kConfigNocAlignmentBytes, false);
    create_cb(program, cores, kReaderChunkConfigCb, 1, device_protocol::kLwt2DChunkConfigPageBytes, false);
    create_cb(program, cores, kWriterBandConfigCb, 1, device_protocol::kLwt2DBandConfigPageBytes, false);
    create_cb(program, cores, kNocScratchCb, scratch_tile_count, kTileBytes, true);
    create_cb(program, cores, kRouteZeroCb, 1, kTileBytes, true);
    std::vector<uint32_t> reader_compile_args = {
        kSource0Cb,
        kSource1Cb,
        kBaseCb,
        kSyncCb,
        kReaderChunkConfigCb,
        kNocScratchCb,
        kRouteZeroCb,
    };
    tt::tt_metal::TensorAccessorArgs(input).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(*buffers.chunk_config->get_backing_buffer()).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(*buffers.route_config->get_backing_buffer()).append_to(reader_compile_args);
    reader_compile_args.push_back(static_cast<uint32_t>(boundary_mode));
    reader_compile_args.push_back(scratch_bytes);
    std::map<std::string, std::string> reader_defines;
    if (inverse) {
        reader_defines.emplace("TTWV_ILWT_2D", "1");
    }
    if (compact_boundary_code) {
        reader_defines.emplace("TTWV_LWT_2D_COMPACT_BOUNDARY_CODE", "1");
    }
    const auto reader = tt::tt_metal::CreateKernel(
        program,
        kernel_path(kernel_root, kReaderKernel),
        cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_args, reader_defines));

    std::vector<uint32_t> writer_compile_args = {
        kOutputCb,
        kSyncCb,
        kWriterBandConfigCb,
        kNocScratchCb,
    };
    tt::tt_metal::TensorAccessorArgs(*buffers.route_config->get_backing_buffer()).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(*buffers.band_config->get_backing_buffer()).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(*buffers.outputs.front()->get_backing_buffer()).append_to(writer_compile_args);
    writer_compile_args.push_back(scratch_bytes);
    std::map<std::string, std::string> writer_defines;
    if (inverse) {
        writer_defines.emplace("TTWV_ILWT_2D", "1");
    }
    const auto writer = tt::tt_metal::CreateKernel(
        program,
        kernel_path(kernel_root, kWriterKernel),
        cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_args, writer_defines));

    std::vector<tt::tt_metal::UnpackToDestMode> unpack_modes(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    unpack_modes[kSource0Cb] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[kSource1Cb] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[kBaseCb] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    std::map<std::string, std::string> compute_defines = {
        {"TTWV_LWT_2D_SCHEME_HEADER", compute_scheme_header},
        {"TTWV_LWT_2D_SCHEME_TYPE", compute_scheme_type},
    };
    if (inverse) {
        compute_defines.emplace("TTWV_ILWT_2D", "1");
    }
    const auto compute = tt::tt_metal::CreateKernel(
        program,
        kernel_path(kernel_root, kComputeKernel),
        cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = unpack_modes,
            .compile_args = {kSource0Cb, kSource1Cb, kBaseCb, kOutputCb},
            .defines = compute_defines,
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
    const uint32_t batch_count,
    const uint32_t core_limit,
    Lwt2DExecutionPlan plan,
    const char* compute_scheme_header,
    const char* compute_scheme_type) {
    validate_lwt_2d_tiling_contract(plan.tiling);
    TT_FATAL(
        plan.input_height <= static_cast<size_t>(std::numeric_limits<int32_t>::max() / 2) &&
            plan.input_width <= static_cast<size_t>(std::numeric_limits<int32_t>::max() / 2),
        "2D input dimensions exceed the signed symmetric-index range");
    const uint32_t chunks_per_sample = checked_u32(plan.chunks.size(), "2D LWT chunks per sample");
    const uint32_t total_work_items =
        checked_u32(static_cast<size_t>(chunks_per_sample) * batch_count, "2D LWT total batch work items");
    const uint32_t effective_core_limit = (core_limit == 0) ? std::numeric_limits<uint32_t>::max() : core_limit;
    std::vector<tt::tt_metal::CoreCoord> cores =
        select_cores(mesh_device, std::min(effective_core_limit, total_work_items));

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
        output = create_dram_pages(mesh_device, static_cast<size_t>(batch_count) * band_tiles, kTileBytes);
    }
    const size_t route_count = plan.chunks.front().routes.size();
    const uint32_t scratch_tile_count =
        noc_scratch_tile_count(plan.y_plan.preprocess_layout.pad_config.mode, false, route_count);
    const size_t config_capacity = static_cast<size_t>(scratch_tile_count) * kTileBytes / 2;
    TT_FATAL(
        route_count * device_protocol::kLwt2DRouteConfigPageBytes <= config_capacity,
        "2D LWT {} route descriptors require {} bytes, exceeding the {}-byte per-RISC preload region",
        route_count,
        route_count * device_protocol::kLwt2DRouteConfigPageBytes,
        config_capacity);
    auto chunk_config = create_dram_pages(mesh_device, plan.chunks.size(), device_protocol::kLwt2DChunkConfigPageBytes);
    auto route_config =
        create_dram_pages(mesh_device, plan.chunks.size() * route_count, device_protocol::kLwt2DRouteConfigPageBytes);
    auto band_config = create_dram_pages(mesh_device, plan.chunks.size(), device_protocol::kLwt2DBandConfigPageBytes);

    const uint64_t l1_capacity = mesh_device.l1_size_per_core();
    TT_FATAL(
        plan.allocated_l1_bytes <= l1_capacity,
        "2D LWT allocation requires {} L1 bytes but hardware exposes {}",
        plan.allocated_l1_bytes,
        l1_capacity);
    const uint32_t active_core_count = checked_u32(cores.size(), "2D LWT active core count");
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
                .boundary_mode = plan.y_plan.preprocess_layout.pad_config.mode,
                .logical_input = plan.tiling.input.logical,
                .padded_input = plan.tiling.input.storage,
                .logical_band = plan.tiling.band.logical,
                .padded_band = plan.tiling.band.storage,
                .available_worker_core_count = static_cast<uint32_t>(
                    mesh_device.compute_with_storage_grid_size().x * mesh_device.compute_with_storage_grid_size().y),
                .active_core_count = active_core_count,
                .batch_count = batch_count,
                .chunks_per_sample = chunks_per_sample,
                .total_work_items = total_work_items,
                .chunk_count = checked_u32(plan.chunks.size(), "2D chunk count"),
                .chunk_tiles_y = plan.chunk_tiles_y,
                .chunk_tiles_x = plan.chunk_tiles_x,
                .route_count = checked_u32(plan.chunks.front().routes.size(), "2D route count"),
                .executable_route_count = plan.executable_route_count,
                .scale_routes_removed = plan.scale_routes_removed,
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
    const std::vector<CoreChunkWork> work = partition_work(buffers.cores, total_work_items);
    const auto [min_work, max_work] = std::minmax_element(
        work.begin(), work.end(), [](const auto& lhs, const auto& rhs) { return lhs.chunk_count < rhs.chunk_count; });
    buffers.scheduler.min_work_items_per_core = min_work->chunk_count;
    buffers.scheduler.max_work_items_per_core = max_work->chunk_count;
    // Wormhole's NCRISC text budget, large route schedules, and antireflect's
    // affine boundary expansion require compact boundary/fallback helpers.
    constexpr size_t kCompactBoundaryRouteThreshold = 52;
    const ArchitecturePolicy architecture_policy = make_architecture_policy(mesh_device.arch());
    const bool compact_boundary_code = architecture_policy.compact_2d_reader ||
                                       route_count >= kCompactBoundaryRouteThreshold ||
                                       plan.y_plan.preprocess_layout.pad_config.mode == BoundaryMode::kAntireflect;
    Lwt2DProgram program = create_program(
        kernel_root,
        core_set(buffers.cores),
        input_buffer,
        buffers,
        compute_scheme_header,
        compute_scheme_type,
        plan.y_plan.preprocess_layout.pad_config.mode,
        compact_boundary_code,
        false,
        scratch_tile_count);
    for (const CoreChunkWork& core_work : work) {
        tt::tt_metal::SetRuntimeArgs(
            program.program,
            program.reader,
            core_work.core,
            reader_args(
                input_buffer,
                plan,
                buffers,
                core_work,
                chunks_per_sample,
                checked_u32(
                    checked_shape_area_2d(plan.tiling.input.storage, "2D input tiles") /
                        device_protocol::kLwt2DFullTileElements,
                    "2D input tiles per sample")));
        if (program.compute) {
            tt::tt_metal::SetRuntimeArgs(
                program.program, *program.compute, core_work.core, compute_args(plan, core_work));
        }
        if (program.writer) {
            tt::tt_metal::SetRuntimeArgs(
                program.program,
                *program.writer,
                core_work.core,
                writer_args(
                    plan,
                    buffers,
                    core_work,
                    chunks_per_sample,
                    checked_u32(band_tiles, "2D output tiles per sample")));
        }
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
    const uint32_t batch_count,
    const uint32_t core_limit,
    Ilwt2DExecutionPlan plan,
    const char* inverse_compute_scheme_header,
    const char* inverse_compute_scheme_type) {
    TT_FATAL(!plan.chunks.empty(), "2D ILWT requires at least one planned chunk");
    TT_FATAL(
        plan.output_height <= static_cast<size_t>(std::numeric_limits<int32_t>::max() / 2) &&
            plan.output_width <= static_cast<size_t>(std::numeric_limits<int32_t>::max() / 2),
        "2D ILWT output dimensions exceed device signed-index limits");
    const uint32_t chunks_per_sample = checked_u32(plan.chunks.size(), "2D ILWT chunks per sample");
    const uint32_t total_work_items =
        checked_u32(static_cast<size_t>(chunks_per_sample) * batch_count, "2D ILWT total batch work items");
    const uint32_t effective_core_limit = (core_limit == 0) ? std::numeric_limits<uint32_t>::max() : core_limit;
    std::vector<tt::tt_metal::CoreCoord> cores =
        select_cores(mesh_device, std::min(effective_core_limit, total_work_items));

    std::array<std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>, device_protocol::kLwt2DPlaneCount> planes;
    for (size_t slot = 0; slot < planes.size(); ++slot) {
        const uint32_t tiles =
            checked_u32(plan.allocated_plane_slot_bytes[slot] / kTileBytes, "2D ILWT workspace plane tiles");
        planes[slot] = create_l1_tile_shards(mesh_device, cores, tiles);
    }

    const size_t output_tiles =
        checked_shape_area_2d(plan.tiling.input.storage, "2D ILWT output") / device_protocol::kLwt2DFullTileElements;
    auto output = create_dram_pages(mesh_device, static_cast<size_t>(batch_count) * output_tiles, kTileBytes);
    std::array<std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>, device_protocol::kLwt2DBandCount> outputs;
    outputs.fill(output);

    const size_t route_count = plan.chunks.front().routes.size();
    const uint32_t scratch_tile_count =
        noc_scratch_tile_count(plan.y_plan.forward_trace.preprocess_layout.pad_config.mode, true, route_count);
    const size_t config_capacity = static_cast<size_t>(scratch_tile_count) * kTileBytes / 2;
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
    const uint32_t active_core_count = checked_u32(cores.size(), "2D ILWT active core count");
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
                .boundary_mode = plan.y_plan.forward_trace.preprocess_layout.pad_config.mode,
                .logical_input = plan.tiling.input.logical,
                .padded_input = plan.tiling.input.storage,
                .logical_band = plan.tiling.band.logical,
                .padded_band = plan.tiling.band.storage,
                .available_worker_core_count = static_cast<uint32_t>(
                    mesh_device.compute_with_storage_grid_size().x * mesh_device.compute_with_storage_grid_size().y),
                .active_core_count = active_core_count,
                .batch_count = batch_count,
                .chunks_per_sample = chunks_per_sample,
                .total_work_items = total_work_items,
                .chunk_count = checked_u32(plan.chunks.size(), "2D ILWT chunk count"),
                .chunk_tiles_y = plan.chunk_tiles_y,
                .chunk_tiles_x = plan.chunk_tiles_x,
                .route_count = checked_u32(route_count, "2D ILWT route count"),
                .executable_route_count = plan.executable_route_count,
                .scale_routes_removed =
                    checked_u32(route_count - plan.executable_route_count, "2D ILWT metadata routes"),
                .estimated_latency_cycles = plan.estimated_latency_cycles,
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

    const std::vector<CoreChunkWork> work = partition_work(buffers.cores, total_work_items);
    const auto [min_work, max_work] = std::minmax_element(
        work.begin(), work.end(), [](const auto& lhs, const auto& rhs) { return lhs.chunk_count < rhs.chunk_count; });
    buffers.scheduler.min_work_items_per_core = min_work->chunk_count;
    buffers.scheduler.max_work_items_per_core = max_work->chunk_count;
    const ArchitecturePolicy architecture_policy = make_architecture_policy(mesh_device.arch());
    Lwt2DProgram program = create_program(
        kernel_root,
        core_set(buffers.cores),
        *band_buffers[0],
        buffers,
        inverse_compute_scheme_header,
        inverse_compute_scheme_type,
        plan.y_plan.forward_trace.preprocess_layout.pad_config.mode,
        architecture_policy.compact_2d_reader,
        true,
        scratch_tile_count);
    TT_FATAL(program.compute && program.writer, "2D ILWT program is missing a route kernel");
    for (const CoreChunkWork& core_work : work) {
        tt::tt_metal::SetRuntimeArgs(
            program.program,
            program.reader,
            core_work.core,
            inverse_reader_args(
                band_buffers,
                plan,
                buffers,
                core_work,
                chunks_per_sample,
                checked_u32(
                    checked_shape_area_2d(plan.tiling.band.storage, "2D ILWT input tiles") /
                        device_protocol::kLwt2DFullTileElements,
                    "2D ILWT input tiles per sample")));
        tt::tt_metal::SetRuntimeArgs(program.program, *program.compute, core_work.core, compute_args(plan, core_work));
        tt::tt_metal::SetRuntimeArgs(
            program.program,
            *program.writer,
            core_work.core,
            inverse_writer_args(
                plan,
                buffers,
                core_work,
                chunks_per_sample,
                checked_u32(output_tiles, "2D ILWT output tiles per sample")));
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

    const size_t output_elements = executable.buffers.scheduler.batch_count *
                                   checked_shape_area_2d(executable.plan.tiling.band.storage, "2D output band");
    const std::vector<float> zeros(output_elements, 0.0F);
    for (auto& output : executable.buffers.outputs) {
        tt::tt_metal::distributed::EnqueueWriteMeshBuffer(command_queue, output, zeros, false);
    }
    tt::tt_metal::distributed::Finish(command_queue);
}

void prepare_ilwt_2d(tt::tt_metal::distributed::MeshCommandQueue& command_queue, Ilwt2DExecutable& executable) {
    const std::vector<uint32_t> chunks = build_ilwt_2d_chunk_config_words(executable.plan);
    const std::vector<uint32_t> routes = build_ilwt_2d_route_config_words(executable.plan);
    const std::vector<uint32_t> terminal = build_ilwt_2d_band_config_words(executable.plan);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(command_queue, executable.buffers.chunk_config, chunks, false);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(command_queue, executable.buffers.route_config, routes, false);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(command_queue, executable.buffers.band_config, terminal, false);
    const size_t output_elements = executable.buffers.scheduler.batch_count *
                                   checked_shape_area_2d(executable.plan.tiling.input.storage, "2D ILWT output");
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(
        command_queue, executable.buffers.outputs[0], std::vector<float>(output_elements, 0.0F), false);
    tt::tt_metal::distributed::Finish(command_queue);
}

void execute_ilwt_2d(
    tt::tt_metal::distributed::MeshDevice&,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    Ilwt2DExecutable& executable) {
    enqueue_ilwt_2d(command_queue, executable);
    tt::tt_metal::distributed::Finish(command_queue);
}

void execute_lwt_2d(
    tt::tt_metal::distributed::MeshDevice&,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    Lwt2DExecutable& executable) {
    enqueue_lwt_2d(command_queue, executable);
    tt::tt_metal::distributed::Finish(command_queue);
}

void enqueue_ilwt_2d(tt::tt_metal::distributed::MeshCommandQueue& command_queue, Ilwt2DExecutable& executable) {
    tt::tt_metal::distributed::EnqueueMeshWorkload(command_queue, executable.workload, false);
}

void enqueue_lwt_2d(tt::tt_metal::distributed::MeshCommandQueue& command_queue, Lwt2DExecutable& executable) {
    tt::tt_metal::distributed::EnqueueMeshWorkload(command_queue, executable.workload, false);
}

}  // namespace ttwv
