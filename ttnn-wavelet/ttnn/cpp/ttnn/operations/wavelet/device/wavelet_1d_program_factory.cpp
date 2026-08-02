// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tt_stl/assert.hpp>
#include <utility>
#include <vector>

#include "tt-metalium/allocator.hpp"
#include "tt-metalium/buffer.hpp"
#include "tt-metalium/circular_buffer_constants.h"
#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/program_descriptors.hpp"
#include "tt-metalium/tensor_accessor_args.hpp"
#include "tt-metalium/tile.hpp"
#include "tt-metalium/workload_descriptor.hpp"
#include "ttnn/operations/wavelet/common/wavelet_host.hpp"
#include "ttnn/operations/wavelet/device/protocol/lwt_config.hpp"
#include "ttnn/operations/wavelet/device/wavelet_1d_program_factory.hpp"
#include "ttnn/operations/wavelet/planner/l1_accounting.hpp"
#include "ttnn/operations/wavelet/planner/inverse_plan.hpp"
#include "ttnn/operations/wavelet/planner/policy.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::prim {

using namespace operations::wavelet;

namespace {

constexpr tt::DataFormat kDataFormat = tt::DataFormat::Float32;

constexpr const char* kLwtReaderKernelPath = "ttnn/cpp/ttnn/operations/wavelet/device/kernels/dataflow/lwt_reader.cpp";
constexpr const char* kLwtWriterKernelPath = "ttnn/cpp/ttnn/operations/wavelet/device/kernels/dataflow/lwt_writer.cpp";
constexpr const char* kLwtComputeKernelPath = "ttnn/cpp/ttnn/operations/wavelet/device/kernels/compute/lwt_compute.cpp";

constexpr uint32_t kSrcTile0Cb = tt::CBIndex::c_0;
constexpr uint32_t kSrcTile1Cb = tt::CBIndex::c_1;
constexpr uint32_t kBaseTileCb = tt::CBIndex::c_2;
constexpr uint32_t kOutputCb = tt::CBIndex::c_16;
constexpr uint32_t kSrcCacheCb = tt::CBIndex::c_3;
constexpr uint32_t kInterleaveCb = tt::CBIndex::c_4;
constexpr uint32_t kSyncCb = tt::CBIndex::c_5;
constexpr uint32_t kReaderConfigCb = tt::CBIndex::c_6;
constexpr uint32_t kWriterConfigCb = tt::CBIndex::c_7;
constexpr uint32_t kWorkspaceCb = tt::CBIndex::c_8;
constexpr uint32_t kTileGroupBuffering = 2;
constexpr uint32_t kDefaultL1SignalBudgetBytes = 768 * 1024;

struct LwtWorkingBuffers {
    std::array<uint32_t, 3> slot_addresses{};
    tt::tt_metal::Buffer* final_even{};
    tt::tt_metal::Buffer* final_odd{};
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> route_config{};
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> chunk_config{};
    std::vector<tt::tt_metal::CoreCoord> cores;

    [[nodiscard]] uint32_t at(const StorageSlot slot) const noexcept {
        return slot_addresses[static_cast<size_t>(slot)];
    }
};

struct IlwtWorkingBuffers {
    std::array<uint32_t, 3> slot_addresses{};
    tt::tt_metal::Buffer* output{};
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> route_config{};
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> chunk_config{};
    std::vector<tt::tt_metal::CoreCoord> cores;

    [[nodiscard]] uint32_t at(const StorageSlot slot) const noexcept {
        return slot_addresses[static_cast<size_t>(slot)];
    }
};

struct UploadedBufferOwner {
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> buffer;
    std::vector<uint32_t> payload;
};

struct CoreChunkWork {
    tt::tt_metal::CoreCoord core;
    uint32_t chunk_begin{0};
    uint32_t chunk_count{0};
};

[[nodiscard]] uint32_t checked_u32(const size_t value, const char* label) {
    TT_FATAL(
        value <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()), "{} {} overflows uint32_t", label, value);
    return static_cast<uint32_t>(value);
}

[[nodiscard]] uint32_t static_workspace_address(tt::tt_metal::distributed::MeshDevice& mesh_device) {
    const size_t alignment = mesh_device.allocator()->get_alignment(tt::tt_metal::BufferType::DRAM);
    size_t address = mesh_device.allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const auto reserve_cb = [&](const size_t bytes) {
        address = round_up(address, alignment);
        TT_FATAL(
            address <= std::numeric_limits<uint32_t>::max() && bytes <= std::numeric_limits<uint32_t>::max() - address,
            "LWT static circular-buffer addresses overflow uint32_t");
        address += bytes;
    };
    reserve_cb(2 * kTileGroupBuffering * device_protocol::kLwtNarrowTileBytes);
    reserve_cb(2 * kTileGroupBuffering * device_protocol::kLwtNarrowTileBytes);
    reserve_cb(3 * kTileGroupBuffering * device_protocol::kLwtNarrowTileBytes);
    reserve_cb(3 * kTileGroupBuffering * device_protocol::kLwtNarrowTileBytes);
    reserve_cb(device_protocol::kLwtCacheStickCount * device_protocol::kStickBytes);
    reserve_cb(device_protocol::kStickBytes);
    reserve_cb(kNocAlignmentBytes);
    reserve_cb(device_protocol::kRouteConfigPageBytes);
    reserve_cb(device_protocol::kRouteConfigPageBytes);
    return checked_u32(round_up(address, alignment), "LWT workspace base address");
}

[[nodiscard]] std::array<uint32_t, 3> workspace_slot_addresses(
    tt::tt_metal::distributed::MeshDevice& mesh_device, const uint32_t workspace_elements) {
    TT_FATAL(workspace_elements > 0, "LWT workspace must contain at least one element");
    TT_FATAL(
        workspace_elements % kStickWidth == 0,
        "LWT workspace length {} is not a multiple of the {}-element stick width",
        workspace_elements,
        kStickWidth);
    const size_t base = static_workspace_address(mesh_device);
    const size_t slot_bytes = static_cast<size_t>(workspace_elements) * sizeof(float);
    TT_FATAL(base <= std::numeric_limits<uint32_t>::max(), "LWT workspace base address overflows uint32_t");
    TT_FATAL(
        slot_bytes <= (std::numeric_limits<uint32_t>::max() - base) / 3,
        "LWT workspace slot addresses overflow uint32_t");
    return {
        checked_u32(base, "LWT workspace base address"),
        checked_u32(base + slot_bytes, "LWT workspace slot B address"),
        checked_u32(base + 2 * slot_bytes, "LWT workspace slot C address"),
    };
}

[[nodiscard]] uint32_t available_static_l1_bytes(tt::tt_metal::distributed::MeshDevice& mesh_device) {
    const uint64_t base = mesh_device.allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const uint64_t frontier = mesh_device.lowest_occupied_compute_l1_address().value_or(mesh_device.l1_size_per_core());
    TT_FATAL(
        frontier >= base, "LWT allocator reports occupied L1 frontier {} below unreserved base {}", frontier, base);
    return checked_u32(frontier - base, "LWT available static L1 bytes");
}

[[nodiscard]] uint32_t output_group_count(const size_t output_length) {
    return checked_u32(
        ceil_div(output_length, static_cast<size_t>(device_protocol::kLwtGroupOutputElements)), "LWT group count");
}

[[nodiscard]] uint32_t l1_signal_budget_bytes(
    tt::tt_metal::distributed::MeshDevice& mesh_device, const uint32_t architecture_scratch_bytes) {
    const uint64_t fixed_bytes = l1_detail::kSourceTileCircularBuffersBytes + l1_detail::kBaseTileCircularBufferBytes +
                                 l1_detail::kOutputTileCircularBufferBytes + l1_detail::kInterleaveOutputBytes +
                                 l1_detail::kCacheBytes + l1_detail::kSynchronizationBytes + l1_detail::kMetadataBytes +
                                 architecture_scratch_bytes;
    const uint64_t available_bytes = available_static_l1_bytes(mesh_device);
    TT_FATAL(
        available_bytes >= fixed_bytes + 3 * device_protocol::kStickBytes,
        "LWT requires at least {} bytes of free per-core L1 after external L1 tensor allocation, but only {} remain",
        fixed_bytes + 3 * device_protocol::kStickBytes,
        available_bytes);
    return checked_u32(
        std::min<uint64_t>(kDefaultL1SignalBudgetBytes, available_bytes - fixed_bytes), "LWT signal workspace budget");
}

[[nodiscard]] std::optional<WorkspaceLayout> workspace_layout_override() { return std::nullopt; }

[[nodiscard]] bool prefer_tile_native_workspace(const LwtExecutionPlan& plan) {
    // Tile-native persistence makes an aligned base a three-page transfer and
    // makes every output write three pages instead of 96 half-sticks.  A
    // shifted base still needs a tile/row remap, so keep row-major storage for
    // schemes where fewer than half of predict/update routes can use the page
    // path.  The override above keeps this policy directly benchmarkable.
    uint32_t predict_update_count = 0;
    uint32_t aligned_base_count = 0;
    TT_FATAL(!plan.chunks.empty(), "LWT workspace selection requires at least one chunk");
    for (const auto& route : plan.chunks.front().routes) {
        if (!is_predict_update_step(route.type)) {
            continue;
        }
        ++predict_update_count;
        aligned_base_count += route.base_offset_elements == 0 ? 1U : 0U;
    }
    return predict_update_count > 0 && 2U * aligned_base_count >= predict_update_count;
}

[[nodiscard]] uint32_t core_limit(tt::tt_metal::distributed::MeshDevice& mesh_device) {
    const auto grid = mesh_device.compute_with_storage_grid_size();
    const uint32_t hardware_cores = static_cast<uint32_t>(grid.x * grid.y);
    TT_FATAL(hardware_cores > 0, "LWT requires at least one hardware worker core");
    return hardware_cores;
}

[[nodiscard]] std::vector<tt::tt_metal::CoreCoord> select_cores(
    tt::tt_metal::distributed::MeshDevice& mesh_device, const uint32_t active_core_count) {
    const auto grid = mesh_device.compute_with_storage_grid_size();
    return tt::tt_metal::grid_to_cores(
        active_core_count, static_cast<uint32_t>(grid.x), static_cast<uint32_t>(grid.y), true);
}

[[nodiscard]] tt::tt_metal::CoreRangeSet core_range_set(const std::vector<tt::tt_metal::CoreCoord>& cores) {
    std::vector<tt::tt_metal::CoreRange> ranges;
    ranges.reserve(cores.size());
    for (const auto& core : cores) {
        ranges.emplace_back(core);
    }
    return tt::tt_metal::CoreRangeSet(std::move(ranges)).merge_ranges();
}

[[nodiscard]] std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> create_dram_buffer(
    tt::tt_metal::distributed::MeshDevice& mesh_device, const size_t page_count, const uint32_t page_bytes) {
    const size_t physical_page_count = std::max(page_count, size_t{1});
    const tt::tt_metal::distributed::DeviceLocalBufferConfig local_config{
        .page_size = page_bytes,
        .buffer_type = tt::tt_metal::BufferType::DRAM,
        .bottom_up = false,
    };
    const tt::tt_metal::distributed::ReplicatedBufferConfig replicated_config{
        .size = static_cast<uint64_t>(physical_page_count * page_bytes),
    };
    return tt::tt_metal::distributed::MeshBuffer::create(replicated_config, local_config, &mesh_device);
}

void add_circular_buffer(
    tt::tt_metal::ProgramDescriptor& descriptor,
    const tt::tt_metal::CoreRangeSet& cores,
    const uint32_t cb_index,
    const uint32_t entry_count,
    const uint32_t page_bytes) {
    descriptor.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = entry_count * page_bytes,
        .core_ranges = cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_index),
            .data_format = kDataFormat,
            .page_size = page_bytes,
        }}},
    });
}

void add_narrow_tile_circular_buffer(
    tt::tt_metal::ProgramDescriptor& descriptor,
    const tt::tt_metal::CoreRangeSet& cores,
    const uint32_t cb_index,
    const uint32_t entry_count) {
    descriptor.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = entry_count * device_protocol::kLwtNarrowTileBytes,
        .core_ranges = cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_index),
            .data_format = kDataFormat,
            .page_size = device_protocol::kLwtNarrowTileBytes,
            .tile = tt::tt_metal::TileDescriptor{32, 16, false},
        }}},
    });
}

[[nodiscard]] std::vector<CoreChunkWork> partition_chunk_work(
    const std::vector<tt::tt_metal::CoreCoord>& cores, const uint32_t chunk_count) {
    TT_FATAL(!cores.empty(), "LWT chunk partition requires cores");
    TT_FATAL(chunk_count >= cores.size(), "LWT active core count exceeds chunk count");

    const uint32_t active_core_count = checked_u32(cores.size(), "LWT core count");
    const uint32_t base_chunks = chunk_count / active_core_count;
    const uint32_t extra_chunks = chunk_count % active_core_count;
    uint32_t chunk_begin = 0;
    std::vector<CoreChunkWork> work;
    work.reserve(cores.size());
    for (uint32_t core_index = 0; core_index < active_core_count; ++core_index) {
        const uint32_t count = base_chunks + (core_index < extra_chunks ? 1U : 0U);
        work.push_back(CoreChunkWork{
            .core = cores[core_index],
            .chunk_begin = chunk_begin,
            .chunk_count = count,
        });
        chunk_begin += count;
    }
    TT_FATAL(chunk_begin == chunk_count, "LWT chunk partition is incomplete");
    return work;
}

[[nodiscard]] uint32_t resolve_output_address(const LwtWorkingBuffers& buffers, const RouteOutputRef output) {
    switch (output.storage) {
        case RouteOutputStorage::kWorkspaceSlot: return buffers.at(output.slot);
        case RouteOutputStorage::kFinalEvenDram: return 0;
        case RouteOutputStorage::kFinalOddDram: return 1;
    }
    TT_THROW("Unsupported LWT output storage");
}

[[nodiscard]] uint32_t resolve_workspace_address(const LwtWorkingBuffers& buffers, const StreamRef stream) {
    return buffers.at(stream.slot);
}

[[nodiscard]] std::vector<uint32_t> build_chunk_config_words(const LwtExecutionPlan& plan) {
    std::vector<uint32_t> words(std::max(plan.chunks.size(), size_t{1}) * device_protocol::kLwtChunkConfigWordCount, 0);
    for (size_t chunk_index = 0; chunk_index < plan.chunks.size(); ++chunk_index) {
        const auto& chunk = plan.chunks[chunk_index];
        const size_t offset = chunk_index * device_protocol::kLwtChunkConfigWordCount;
        words[offset + device_protocol::kLwtInitialEvenBegin] =
            checked_u32(chunk.initial_even.begin, "initial even begin");
        words[offset + device_protocol::kLwtInitialEvenLength] =
            checked_u32(chunk.initial_even.length(), "initial even length");
        words[offset + device_protocol::kLwtInitialOddBegin] =
            checked_u32(chunk.initial_odd.begin, "initial odd begin");
        words[offset + device_protocol::kLwtInitialOddLength] =
            checked_u32(chunk.initial_odd.length(), "initial odd length");
    }
    return words;
}

[[nodiscard]] std::vector<uint32_t> build_route_config_words(
    const LwtExecutionPlan& plan, const LwtWorkingBuffers& buffers) {
    TT_FATAL(!plan.chunks.empty(), "LWT plan has no chunks");
    const size_t route_count = plan.chunks.front().routes.size();
    std::vector<uint32_t> words(
        std::max(plan.chunks.size() * route_count, size_t{1}) * device_protocol::kRouteConfigWordCount, 0);

    for (size_t chunk_index = 0; chunk_index < plan.chunks.size(); ++chunk_index) {
        const auto& chunk = plan.chunks[chunk_index];
        TT_FATAL(chunk.routes.size() == route_count, "LWT chunks have inconsistent route counts");
        for (size_t route_index = 0; route_index < route_count; ++route_index) {
            const auto& route = chunk.routes[route_index];
            const uint32_t output_offset = checked_u32(route.output_offset_elements, "LWT output offset");
            const size_t word_offset =
                (chunk_index * route_count + route_index) * device_protocol::kRouteConfigWordCount;
            words[word_offset + device_protocol::kRouteType] = static_cast<uint32_t>(route.type);
            words[word_offset + device_protocol::kRouteSourceAddr] = resolve_workspace_address(buffers, route.source);
            words[word_offset + device_protocol::kRouteSourceLength] =
                checked_u32(route.source_storage_length, "LWT source storage end");
            words[word_offset + device_protocol::kRouteBaseAddr] = resolve_workspace_address(buffers, route.base);
            words[word_offset + device_protocol::kRouteBaseLength] =
                checked_u32(route.base_storage_length, "LWT base storage end");
            words[word_offset + device_protocol::kRouteOutputAddr] = resolve_output_address(buffers, route.output);
            words[word_offset + device_protocol::kRouteOutputLength] =
                checked_u32(route.output_length, "LWT output length");
            words[word_offset + device_protocol::kRouteSourceOffset] =
                checked_u32(route.source_offset_elements, "LWT source offset");
            words[word_offset + device_protocol::kRouteBaseOffset] =
                checked_u32(route.base_offset_elements, "LWT base offset");
            words[word_offset + device_protocol::kRouteSourceLeftPad] = route.source_left_pad_elements;
            words[word_offset + device_protocol::kRouteOutputOffset] = output_offset;
            words[word_offset + device_protocol::kRouteGroupCount] = output_group_count(route.output_length);
            uint32_t route_flags = 0;
            if (route.output.storage == RouteOutputStorage::kFinalEvenDram) {
                route_flags = device_protocol::kRouteFlagFinalDram | device_protocol::kRouteFlagFinalEven;
            } else if (route.output.storage == RouteOutputStorage::kFinalOddDram) {
                route_flags = device_protocol::kRouteFlagFinalDram | device_protocol::kRouteFlagFinalOdd;
            }
            words[word_offset + device_protocol::kRouteFlags] = route_flags;
        }
    }
    return words;
}

[[nodiscard]] tt::tt_metal::KernelDescriptor::RTArgList reader_runtime_args(
    const LwtExecutionPlan& plan,
    const LwtWorkingBuffers& buffers,
    const tt::tt_metal::Buffer& input_buffer,
    const CoreChunkWork& work) {
    tt::tt_metal::KernelDescriptor::RTArgList args;
    args.reserve(10);
    args.push_back(const_cast<tt::tt_metal::Buffer*>(&input_buffer));
    args.push_back(checked_u32(plan.full_plan.preprocess_layout.input.length, "LWT input length"));
    args.push_back(plan.full_plan.preprocess_layout.pad_config.left);
    args.push_back(buffers.at(StorageSlot::kA));
    args.push_back(buffers.at(StorageSlot::kB));
    args.push_back(buffers.chunk_config->get_backing_buffer());
    args.push_back(buffers.route_config->get_backing_buffer());
    args.push_back(work.chunk_begin);
    args.push_back(work.chunk_count);
    args.push_back(checked_u32(plan.chunks.front().routes.size(), "LWT route count"));
    return args;
}

[[nodiscard]] tt::tt_metal::KernelDescriptor::RTArgList writer_runtime_args(
    const LwtExecutionPlan& plan, const LwtWorkingBuffers& buffers, const CoreChunkWork& work) {
    tt::tt_metal::KernelDescriptor::RTArgList args;
    args.reserve(6);
    args.push_back(buffers.route_config->get_backing_buffer());
    args.push_back(work.chunk_begin);
    args.push_back(work.chunk_count);
    args.push_back(checked_u32(plan.chunks.front().routes.size(), "LWT route count"));
    args.push_back(buffers.final_even);
    args.push_back(buffers.final_odd);
    return args;
}

[[nodiscard]] std::vector<uint32_t> compute_runtime_args(const LwtExecutionPlan& plan, const CoreChunkWork& work) {
    const size_t route_count = plan.chunks.front().routes.size();
    std::vector<uint32_t> args;
    args.reserve(1 + static_cast<size_t>(work.chunk_count) * route_count);
    args.push_back(work.chunk_count);
    for (uint32_t local_chunk = 0; local_chunk < work.chunk_count; ++local_chunk) {
        const auto& chunk = plan.chunks[work.chunk_begin + local_chunk];
        for (const auto& route : chunk.routes) {
            args.push_back(output_group_count(route.output_length));
        }
    }
    return args;
}

[[nodiscard]] tt::tt_metal::ProgramDescriptor create_forward_program_descriptor(
    const tt::tt_metal::CoreRangeSet& cores,
    const tt::tt_metal::Buffer& input_buffer,
    const LwtWorkingBuffers& buffers,
    const LwtExecutionPlan& plan,
    const WorkspaceLayout workspace_layout,
    const BoundaryMode boundary_mode,
    const char* compute_scheme_header,
    const char* compute_scheme_type,
    const std::vector<CoreChunkWork>& work) {
    tt::tt_metal::ProgramDescriptor descriptor;
    add_narrow_tile_circular_buffer(descriptor, cores, kSrcTile0Cb, 2 * kTileGroupBuffering);
    add_narrow_tile_circular_buffer(descriptor, cores, kSrcTile1Cb, 2 * kTileGroupBuffering);
    add_narrow_tile_circular_buffer(descriptor, cores, kBaseTileCb, 3 * kTileGroupBuffering);
    add_narrow_tile_circular_buffer(descriptor, cores, kOutputCb, 3 * kTileGroupBuffering);
    add_circular_buffer(
        descriptor, cores, kSrcCacheCb, device_protocol::kLwtCacheStickCount, device_protocol::kStickBytes);
    add_circular_buffer(descriptor, cores, kInterleaveCb, 1, device_protocol::kStickBytes);
    add_circular_buffer(descriptor, cores, kSyncCb, 1, kNocAlignmentBytes);
    add_circular_buffer(descriptor, cores, kReaderConfigCb, 1, device_protocol::kRouteConfigPageBytes);
    add_circular_buffer(descriptor, cores, kWriterConfigCb, 1, device_protocol::kRouteConfigPageBytes);
    add_circular_buffer(
        descriptor,
        cores,
        kWorkspaceCb,
        checked_u32(3 * static_cast<size_t>(plan.workspace_elements) / kStickWidth, "LWT workspace stick count"),
        device_protocol::kStickBytes);

    const auto& config_buffer = *buffers.route_config->get_backing_buffer();

    std::vector<uint32_t> reader_compile_args = {
        kReaderConfigCb,
        kSrcTile0Cb,
        kSrcTile1Cb,
        kBaseTileCb,
        kSrcCacheCb,
        kSyncCb,
        static_cast<uint32_t>(workspace_layout == WorkspaceLayout::kTileNative),
        0U,
        static_cast<uint32_t>(boundary_mode),
        input_buffer.page_size(),
    };
    tt::tt_metal::TensorAccessorArgs(config_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_compile_args);

    std::vector<uint32_t> writer_compile_args = {
        kWriterConfigCb,
        kOutputCb,
        kSyncCb,
        1U,
        static_cast<uint32_t>(workspace_layout == WorkspaceLayout::kTileNative),
        0U,
        kInterleaveCb,
        buffers.final_even->page_size(),
    };
    tt::tt_metal::TensorAccessorArgs(config_buffer).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(*buffers.final_even).append_to(writer_compile_args);

    const std::vector<uint32_t> compute_compile_args = {kSrcTile0Cb, kSrcTile1Cb, kBaseTileCb, kOutputCb};
    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    unpack_to_dest_mode[kSrcTile0Cb] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest_mode[kSrcTile1Cb] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest_mode[kBaseTileCb] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;

    tt::tt_metal::KernelDescriptor reader_descriptor;
    reader_descriptor.kernel_source = kLwtReaderKernelPath;
    reader_descriptor.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_descriptor.core_ranges = cores;
    reader_descriptor.compile_time_args = std::move(reader_compile_args);
    reader_descriptor.config = tt::tt_metal::ReaderConfigDescriptor{};

    tt::tt_metal::KernelDescriptor writer_descriptor;
    writer_descriptor.kernel_source = kLwtWriterKernelPath;
    writer_descriptor.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer_descriptor.core_ranges = cores;
    writer_descriptor.compile_time_args = std::move(writer_compile_args);
    writer_descriptor.config = tt::tt_metal::WriterConfigDescriptor{};

    tt::tt_metal::KernelDescriptor compute_descriptor;
    compute_descriptor.kernel_source = kLwtComputeKernelPath;
    compute_descriptor.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    compute_descriptor.core_ranges = cores;
    compute_descriptor.compile_time_args = compute_compile_args;
    compute_descriptor.defines = {
        {"LWT_SCHEME_HEADER", compute_scheme_header},
        {"LWT_SCHEME_TYPE", compute_scheme_type},
        {"LWT_INLINE_TERMINAL_SCALE", "1"},
    };
    compute_descriptor.config = tt::tt_metal::ComputeConfigDescriptor{
        .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
        .fp32_dest_acc_en = true,
        .unpack_to_dest_mode = unpack_to_dest_mode,
    };

    for (const auto& core_work : work) {
        reader_descriptor.emplace_runtime_args(
            core_work.core, reader_runtime_args(plan, buffers, input_buffer, core_work));
        tt::tt_metal::KernelDescriptor::RTArgList compute_args;
        compute_args.append(compute_runtime_args(plan, core_work));
        compute_descriptor.emplace_runtime_args(core_work.core, compute_args);
        writer_descriptor.emplace_runtime_args(core_work.core, writer_runtime_args(plan, buffers, core_work));
    }

    descriptor.kernels.push_back(std::move(reader_descriptor));
    descriptor.kernels.push_back(std::move(compute_descriptor));
    descriptor.kernels.push_back(std::move(writer_descriptor));
    return descriptor;
}

[[nodiscard]] uint32_t resolve_workspace_address(const IlwtWorkingBuffers& buffers, const StreamRef stream) {
    return buffers.at(stream.slot);
}

[[nodiscard]] std::vector<uint32_t> build_inverse_chunk_config_words(
    const IlwtExecutionPlan& plan, const IlwtWorkingBuffers& buffers) {
    std::vector<uint32_t> words(std::max(plan.chunks.size(), size_t{1}) * device_protocol::kLwtChunkConfigWordCount, 0);
    for (size_t chunk_index = 0; chunk_index < plan.chunks.size(); ++chunk_index) {
        const auto& chunk = plan.chunks[chunk_index];
        const size_t offset = chunk_index * device_protocol::kLwtChunkConfigWordCount;
        words[offset + device_protocol::kIlwtApproximationBegin] =
            checked_u32(chunk.canonical_approximation.begin, "ILWT approximation begin");
        words[offset + device_protocol::kIlwtApproximationLength] =
            checked_u32(chunk.canonical_approximation.length(), "ILWT approximation length");
        words[offset + device_protocol::kIlwtDetailBegin] =
            checked_u32(chunk.canonical_detail.begin, "ILWT detail begin");
        words[offset + device_protocol::kIlwtDetailLength] =
            checked_u32(chunk.canonical_detail.length(), "ILWT detail length");
        words[offset + device_protocol::kIlwtFinalEvenAddr] = resolve_workspace_address(buffers, chunk.final_even);
        words[offset + device_protocol::kIlwtFinalEvenStorageLength] =
            checked_u32(chunk.final_even_storage_length, "ILWT final even storage length");
        words[offset + device_protocol::kIlwtFinalEvenOffset] =
            checked_u32(chunk.final_even_offset_elements, "ILWT final even offset");
        words[offset + device_protocol::kIlwtFinalEvenBegin] =
            checked_u32(chunk.reconstructed_even.begin, "ILWT final even begin");
        words[offset + device_protocol::kIlwtFinalOddAddr] = resolve_workspace_address(buffers, chunk.final_odd);
        words[offset + device_protocol::kIlwtFinalOddStorageLength] =
            checked_u32(chunk.final_odd_storage_length, "ILWT final odd storage length");
        words[offset + device_protocol::kIlwtFinalOddOffset] =
            checked_u32(chunk.final_odd_offset_elements, "ILWT final odd offset");
        words[offset + device_protocol::kIlwtFinalOddBegin] =
            checked_u32(chunk.reconstructed_odd.begin, "ILWT final odd begin");
        words[offset + device_protocol::kIlwtOutputBegin] = checked_u32(chunk.output_signal.begin, "ILWT output begin");
        words[offset + device_protocol::kIlwtOutputLength] =
            checked_u32(chunk.output_signal.length(), "ILWT output length");
    }
    return words;
}

[[nodiscard]] std::vector<uint32_t> build_inverse_route_config_words(
    const IlwtExecutionPlan& plan, const IlwtWorkingBuffers& buffers) {
    TT_FATAL(!plan.chunks.empty(), "ILWT plan has no chunks");
    const size_t route_count = plan.chunks.front().routes.size();
    std::vector<uint32_t> words(
        std::max(plan.chunks.size() * route_count, size_t{1}) * device_protocol::kRouteConfigWordCount, 0);
    for (size_t chunk_index = 0; chunk_index < plan.chunks.size(); ++chunk_index) {
        const auto& chunk = plan.chunks[chunk_index];
        TT_FATAL(chunk.routes.size() == route_count, "ILWT chunks have inconsistent route counts");
        for (size_t route_index = 0; route_index < route_count; ++route_index) {
            const auto& route = chunk.routes[route_index];
            TT_FATAL(
                route.output.storage == RouteOutputStorage::kWorkspaceSlot,
                "ILWT intermediate route must target a local workspace slot");
            const size_t word_offset =
                (chunk_index * route_count + route_index) * device_protocol::kRouteConfigWordCount;
            words[word_offset + device_protocol::kRouteType] = static_cast<uint32_t>(route.type);
            words[word_offset + device_protocol::kRouteSourceAddr] = resolve_workspace_address(buffers, route.source);
            words[word_offset + device_protocol::kRouteSourceLength] =
                checked_u32(route.source_storage_length, "ILWT source storage length");
            words[word_offset + device_protocol::kRouteBaseAddr] = resolve_workspace_address(buffers, route.base);
            words[word_offset + device_protocol::kRouteBaseLength] =
                checked_u32(route.base_storage_length, "ILWT base storage length");
            words[word_offset + device_protocol::kRouteOutputAddr] =
                resolve_workspace_address(buffers, StreamRef{.slot = route.output.slot});
            words[word_offset + device_protocol::kRouteOutputLength] =
                checked_u32(route.output_length, "ILWT output length");
            words[word_offset + device_protocol::kRouteSourceOffset] =
                checked_u32(route.source_offset_elements, "ILWT source offset");
            words[word_offset + device_protocol::kRouteBaseOffset] =
                checked_u32(route.base_offset_elements, "ILWT base offset");
            words[word_offset + device_protocol::kRouteSourceLeftPad] = route.source_left_pad_elements;
            words[word_offset + device_protocol::kRouteOutputOffset] = 0;
            words[word_offset + device_protocol::kRouteGroupCount] = output_group_count(route.output_length);
            words[word_offset + device_protocol::kRouteFlags] =
                plan.final_interleave_direct && route_index + 1 == route_count
                    ? device_protocol::kRouteFlagIlwtFinalInterleave
                    : 0U;
        }
    }
    return words;
}

[[nodiscard]] tt::tt_metal::KernelDescriptor::RTArgList inverse_reader_runtime_args(
    const IlwtExecutionPlan& plan,
    const IlwtWorkingBuffers& buffers,
    const tt::tt_metal::Buffer& approximation_buffer,
    const tt::tt_metal::Buffer& detail_buffer,
    const CoreChunkWork& work) {
    tt::tt_metal::KernelDescriptor::RTArgList args;
    args.reserve(10);
    args.push_back(const_cast<tt::tt_metal::Buffer*>(&approximation_buffer));
    args.push_back(const_cast<tt::tt_metal::Buffer*>(&detail_buffer));
    args.push_back(checked_u32(plan.full_plan.coefficient_length, "ILWT coefficient length"));
    args.push_back(buffers.at(StorageSlot::kA));
    args.push_back(buffers.at(StorageSlot::kB));
    args.push_back(buffers.chunk_config->get_backing_buffer());
    args.push_back(buffers.route_config->get_backing_buffer());
    args.push_back(work.chunk_begin);
    args.push_back(work.chunk_count);
    args.push_back(checked_u32(plan.chunks.front().routes.size(), "ILWT route count"));
    return args;
}

[[nodiscard]] tt::tt_metal::KernelDescriptor::RTArgList inverse_writer_runtime_args(
    const IlwtExecutionPlan& plan, const IlwtWorkingBuffers& buffers, const CoreChunkWork& work) {
    tt::tt_metal::KernelDescriptor::RTArgList args;
    args.reserve(7);
    args.push_back(buffers.route_config->get_backing_buffer());
    args.push_back(work.chunk_begin);
    args.push_back(work.chunk_count);
    args.push_back(checked_u32(plan.chunks.front().routes.size(), "ILWT route count"));
    args.push_back(buffers.chunk_config->get_backing_buffer());
    args.push_back(buffers.output);
    args.push_back(plan.full_plan.forward_trace.preprocess_layout.pad_config.left);
    return args;
}

[[nodiscard]] std::vector<uint32_t> inverse_compute_runtime_args(
    const IlwtExecutionPlan& plan, const CoreChunkWork& work) {
    const size_t route_count = plan.chunks.front().routes.size();
    std::vector<uint32_t> args;
    args.reserve(1 + static_cast<size_t>(work.chunk_count) * route_count);
    args.push_back(work.chunk_count);
    for (uint32_t local_chunk = 0; local_chunk < work.chunk_count; ++local_chunk) {
        const auto& chunk = plan.chunks[work.chunk_begin + local_chunk];
        for (const auto& route : chunk.routes) {
            args.push_back(output_group_count(route.output_length));
        }
    }
    return args;
}

[[nodiscard]] tt::tt_metal::ProgramDescriptor create_inverse_program_descriptor(
    const tt::tt_metal::CoreRangeSet& cores,
    const tt::tt_metal::Buffer& approximation_buffer,
    const tt::tt_metal::Buffer& detail_buffer,
    const IlwtWorkingBuffers& buffers,
    const IlwtExecutionPlan& plan,
    const WorkspaceLayout workspace_layout,
    const char* compute_scheme_header,
    const char* compute_scheme_type,
    const std::vector<CoreChunkWork>& work) {
    tt::tt_metal::ProgramDescriptor descriptor;
    add_narrow_tile_circular_buffer(descriptor, cores, kSrcTile0Cb, 2 * kTileGroupBuffering);
    add_narrow_tile_circular_buffer(descriptor, cores, kSrcTile1Cb, 2 * kTileGroupBuffering);
    add_narrow_tile_circular_buffer(descriptor, cores, kBaseTileCb, 3 * kTileGroupBuffering);
    add_narrow_tile_circular_buffer(descriptor, cores, kOutputCb, 3 * kTileGroupBuffering);
    add_circular_buffer(
        descriptor, cores, kSrcCacheCb, device_protocol::kLwtCacheStickCount, device_protocol::kStickBytes);
    add_circular_buffer(descriptor, cores, kInterleaveCb, 1, device_protocol::kStickBytes);
    add_circular_buffer(descriptor, cores, kSyncCb, 1, kNocAlignmentBytes);
    add_circular_buffer(descriptor, cores, kReaderConfigCb, 1, device_protocol::kRouteConfigPageBytes);
    add_circular_buffer(descriptor, cores, kWriterConfigCb, 1, device_protocol::kRouteConfigPageBytes);
    add_circular_buffer(
        descriptor,
        cores,
        kWorkspaceCb,
        checked_u32(3 * static_cast<size_t>(plan.workspace_elements) / kStickWidth, "ILWT workspace stick count"),
        device_protocol::kStickBytes);

    const auto& config_buffer = *buffers.route_config->get_backing_buffer();
    std::vector<uint32_t> reader_compile_args = {
        kReaderConfigCb,
        kSrcTile0Cb,
        kSrcTile1Cb,
        kBaseTileCb,
        kSrcCacheCb,
        kSyncCb,
        static_cast<uint32_t>(workspace_layout == WorkspaceLayout::kTileNative),
        1U,
        // Inverse reads canonical coefficients, not an extended original
        // signal. Keep its unused shared-reader boundary specialization fixed
        // so every ILWT mode reuses the same device binary.
        static_cast<uint32_t>(BoundaryMode::kSymmetric),
        approximation_buffer.page_size(),
    };
    tt::tt_metal::TensorAccessorArgs(config_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(approximation_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(detail_buffer).append_to(reader_compile_args);

    std::vector<uint32_t> writer_compile_args = {
        kWriterConfigCb,
        kOutputCb,
        kSyncCb,
        1U,
        static_cast<uint32_t>(workspace_layout == WorkspaceLayout::kTileNative),
        1U,
        kInterleaveCb,
        buffers.output->page_size(),
    };
    tt::tt_metal::TensorAccessorArgs(config_buffer).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(*buffers.output).append_to(writer_compile_args);

    const std::vector<uint32_t> compute_compile_args = {kSrcTile0Cb, kSrcTile1Cb, kBaseTileCb, kOutputCb};
    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    unpack_to_dest_mode[kSrcTile0Cb] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest_mode[kSrcTile1Cb] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest_mode[kBaseTileCb] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;

    tt::tt_metal::KernelDescriptor reader_descriptor;
    reader_descriptor.kernel_source = kLwtReaderKernelPath;
    reader_descriptor.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_descriptor.core_ranges = cores;
    reader_descriptor.compile_time_args = std::move(reader_compile_args);
    reader_descriptor.config = tt::tt_metal::ReaderConfigDescriptor{};

    tt::tt_metal::KernelDescriptor writer_descriptor;
    writer_descriptor.kernel_source = kLwtWriterKernelPath;
    writer_descriptor.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer_descriptor.core_ranges = cores;
    writer_descriptor.compile_time_args = std::move(writer_compile_args);
    writer_descriptor.config = tt::tt_metal::WriterConfigDescriptor{};

    tt::tt_metal::KernelDescriptor compute_descriptor;
    compute_descriptor.kernel_source = kLwtComputeKernelPath;
    compute_descriptor.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    compute_descriptor.core_ranges = cores;
    compute_descriptor.compile_time_args = compute_compile_args;
    compute_descriptor.defines = {
        {"ILWT_SCHEME_HEADER", compute_scheme_header},
        {"ILWT_SCHEME_TYPE", compute_scheme_type},
        {"ILWT_INLINE_INVERSE_SCALE", "1"},
    };
    compute_descriptor.config = tt::tt_metal::ComputeConfigDescriptor{
        .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
        .fp32_dest_acc_en = true,
        .unpack_to_dest_mode = unpack_to_dest_mode,
    };

    for (const auto& core_work : work) {
        reader_descriptor.emplace_runtime_args(
            core_work.core, inverse_reader_runtime_args(plan, buffers, approximation_buffer, detail_buffer, core_work));
        tt::tt_metal::KernelDescriptor::RTArgList compute_args;
        compute_args.append(inverse_compute_runtime_args(plan, core_work));
        compute_descriptor.emplace_runtime_args(core_work.core, compute_args);
        writer_descriptor.emplace_runtime_args(core_work.core, inverse_writer_runtime_args(plan, buffers, core_work));
    }

    descriptor.kernels.push_back(std::move(reader_descriptor));
    descriptor.kernels.push_back(std::move(compute_descriptor));
    descriptor.kernels.push_back(std::move(writer_descriptor));
    return descriptor;
}

}  // namespace

namespace {

void validate_output_memory_config(const MemoryConfig& memory_config, const char* operation_name) {
    TT_FATAL(
        memory_config.memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED &&
            memory_config.buffer_type() == tt::tt_metal::BufferType::DRAM && !memory_config.is_sharded(),
        "{} supports only DRAM-interleaved outputs in its first TTNN version",
        operation_name);
}

void validate_input_memory_config(const MemoryConfig& memory_config, const char* tensor_name) {
    const bool supported_buffer = memory_config.buffer_type() == tt::tt_metal::BufferType::DRAM ||
                                  memory_config.buffer_type() == tt::tt_metal::BufferType::L1;
    TT_FATAL(
        memory_config.memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED && supported_buffer &&
            !memory_config.is_sharded(),
        "{} must use INTERLEAVED memory with DRAM or L1 storage; sharded inputs are unsupported",
        tensor_name);
}

void validate_rank_one_tensor(const Tensor& tensor, const char* tensor_name) {
    TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "{} must be a device tensor", tensor_name);
    TT_FATAL(
        tensor.is_allocated() && tensor.buffer() != nullptr, "{} must have an allocated device buffer", tensor_name);
    TT_FATAL(tensor.device() != nullptr, "{} has no device", tensor_name);
    TT_FATAL(tensor.device()->num_devices() == 1, "{} must be placed on exactly one physical device", tensor_name);
    TT_FATAL(tensor.dtype() == DataType::FLOAT32, "{} must have FLOAT32 dtype", tensor_name);
    TT_FATAL(tensor.layout() == Layout::ROW_MAJOR, "{} must use ROW_MAJOR layout", tensor_name);
    TT_FATAL(tensor.logical_shape().rank() == 1, "{} must have exact rank 1", tensor_name);
    TT_FATAL(tensor.logical_shape()[0] > 0, "{} must be non-empty", tensor_name);
    TT_FATAL(
        tensor.logical_shape()[0] <= std::numeric_limits<uint32_t>::max(),
        "{} length {} exceeds the device uint32 range",
        tensor_name,
        tensor.logical_shape()[0]);
    validate_input_memory_config(tensor.memory_config(), tensor_name);

    const uint64_t required_bytes = static_cast<uint64_t>(tensor.logical_shape()[0]) * sizeof(float);
    TT_FATAL(
        tensor.buffer()->size() >= required_bytes,
        "{} physical buffer has {} bytes but the logical signal requires at least {} bytes",
        tensor_name,
        tensor.buffer()->size(),
        required_bytes);
    static_cast<void>(make_architecture_policy(tensor.device()->arch()));
}

[[nodiscard]] tt::tt_metal::TensorSpec rank_one_output_spec(const uint32_t length, const MemoryConfig& memory_config) {
    return tt::tt_metal::TensorSpec(
        Shape({length}),
        tt::tt_metal::TensorLayout(
            DataType::FLOAT32,
            tt::tt_metal::PageConfig(Layout::ROW_MAJOR),
            memory_config,
            tt::tt_metal::Alignment{32}));
}

void validate_preallocated_output(
    const Tensor& output,
    const tt::tt_metal::TensorSpec& expected_spec,
    const tt::tt_metal::distributed::MeshDevice* expected_device,
    const char* output_name) {
    validate_rank_one_tensor(output, output_name);
    validate_output_memory_config(output.memory_config(), output_name);
    TT_FATAL(output.device() == expected_device, "{} must be on the same device as the inputs", output_name);
    TT_FATAL(
        output.tensor_spec() == expected_spec,
        "{} tensor spec does not match the wavelet output specification",
        output_name);
}

[[nodiscard]] std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> upload_metadata(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const size_t page_count,
    const uint32_t page_bytes,
    std::vector<uint32_t> payload,
    tt::tt_metal::WorkloadDescriptor& workload) {
    auto buffer = create_dram_buffer(mesh_device, page_count, page_bytes);
    const size_t physical_words = static_cast<size_t>(buffer->get_backing_buffer()->size()) / sizeof(uint32_t);
    TT_FATAL(
        payload.size() <= physical_words,
        "Wavelet metadata payload has {} words but its device buffer holds only {}",
        payload.size(),
        physical_words);
    payload.resize(physical_words, 0);

    auto owner = std::make_shared<UploadedBufferOwner>(UploadedBufferOwner{
        .buffer = buffer,
        .payload = std::move(payload),
    });
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(
        mesh_device.mesh_command_queue(), owner->buffer, owner->payload, false);
    workload.buffers.push_back({owner, buffer->get_backing_buffer()});
    return buffer;
}

void append_programs(
    tt::tt_metal::WorkloadDescriptor& workload,
    tt::tt_metal::ProgramDescriptor descriptor,
    const MeshCoordinateRangeSet& tensor_coords) {
    const auto ranges = tensor_coords.ranges();
    TT_FATAL(!ranges.empty(), "Wavelet workload has no mesh coordinate range");
    for (size_t index = 0; index + 1 < ranges.size(); ++index) {
        workload.programs.push_back({ranges[index], descriptor});
    }
    workload.programs.push_back({ranges.back(), std::move(descriptor)});
}

template <typename Scheme>
[[nodiscard]] LwtExecutionPlan make_forward_execution_plan(
    tt::tt_metal::distributed::MeshDevice& mesh_device, const size_t input_length, const BoundaryMode boundary_mode) {
    const SignalBuffer input{
        .dram_address = 0,
        .length = input_length,
        .stick_width = kStickWidth,
        .element_size_bytes = sizeof(float),
    };
    LiftingForwardPlan full_plan = make_forward_lifting_plan<Scheme>(input, 0, 0, boundary_mode);
    TT_FATAL(
        full_plan.preprocess_layout.padded_length() <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
        "LWT padded input length exceeds the device signed-index range");

    const uint32_t max_cores = core_limit(mesh_device);
    const ArchitecturePolicy architecture_policy = make_architecture_policy(mesh_device.arch());
    const uint32_t signal_budget_bytes = l1_signal_budget_bytes(mesh_device, architecture_policy.l1_scratch_bytes);
    const std::optional<WorkspaceLayout> workspace_override = workspace_layout_override();
    const WorkspaceLayout initial_layout = workspace_override.value_or(WorkspaceLayout::kRowMajor);
    LwtExecutionPlan plan =
        make_lwt_execution_plan(std::move(full_plan), max_cores, signal_budget_bytes, initial_layout);
    if (!workspace_override.has_value() && prefer_tile_native_workspace(plan)) {
        plan = make_lwt_execution_plan(
            std::move(plan.full_plan), max_cores, signal_budget_bytes, WorkspaceLayout::kTileNative);
    }
    static_cast<void>(make_l1_accounting(
        plan.workspace_elements,
        plan.max_workspace_elements,
        architecture_policy.l1_scratch_bytes,
        available_static_l1_bytes(mesh_device)));
    return plan;
}

template <typename Scheme>
[[nodiscard]] IlwtExecutionPlan make_inverse_execution_plan(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const uint32_t original_length,
    const size_t coefficient_length,
    const BoundaryMode boundary_mode) {
    const std::optional<WorkspaceLayout> workspace_override = workspace_layout_override();
    const ArchitecturePolicy architecture_policy = make_architecture_policy(mesh_device.arch(), workspace_override);
    TT_FATAL(architecture_policy.inverse_scale_inline, "ILWT must preserve inline FP32 inverse scaling");
    IlwtExecutionPlan plan = make_ilwt_execution_plan(
        make_inverse_lifting_plan<Scheme>(original_length, coefficient_length, boundary_mode),
        core_limit(mesh_device),
        l1_signal_budget_bytes(mesh_device, architecture_policy.l1_scratch_bytes),
        architecture_policy.ilwt_layout,
        architecture_policy.final_interleave_direct);
    static_cast<void>(make_l1_accounting(
        plan.workspace_elements,
        plan.max_workspace_elements,
        architecture_policy.l1_scratch_bytes,
        available_static_l1_bytes(mesh_device)));
    return plan;
}

template <typename Scheme>
[[nodiscard]] tt::tt_metal::WorkloadDescriptor build_forward_workload(
    const Lwt1DParams& operation_attributes,
    const Lwt1DInputs& tensor_args,
    std::tuple<Tensor, Tensor>& tensor_return_value,
    const MeshCoordinateRangeSet& tensor_coords) {
    auto& mesh_device = *tensor_args.input.device();
    const auto& input_buffer = *tensor_args.input.buffer();
    LwtExecutionPlan plan = make_forward_execution_plan<Scheme>(
        mesh_device, tensor_args.input.logical_shape()[0], operation_attributes.boundary_mode);

    tt::tt_metal::WorkloadDescriptor workload;
    std::vector<tt::tt_metal::CoreCoord> cores = select_cores(mesh_device, plan.active_core_count);

    const size_t route_count = plan.chunks.front().routes.size();
    auto route_config =
        create_dram_buffer(mesh_device, plan.chunks.size() * route_count, device_protocol::kRouteConfigPageBytes);
    auto chunk_config = create_dram_buffer(mesh_device, plan.chunks.size(), device_protocol::kLwtChunkConfigPageBytes);
    LwtWorkingBuffers buffers{
        .slot_addresses = workspace_slot_addresses(mesh_device, plan.workspace_elements),
        .final_even = std::get<0>(tensor_return_value).buffer(),
        .final_odd = std::get<1>(tensor_return_value).buffer(),
        .route_config = route_config,
        .chunk_config = chunk_config,
        .cores = std::move(cores),
    };

    constexpr int32_t canonical_start = static_cast<int32_t>(Scheme::tap_size / 2);
    const int32_t final_even_delta = plan.full_plan.final_even_shift - canonical_start;
    const int32_t final_odd_delta = plan.full_plan.final_odd_shift - canonical_start;
    TT_FATAL(
        final_even_delta <= 0 && static_cast<int64_t>(plan.full_plan.final_even_length) + final_even_delta >=
                                     static_cast<int64_t>(plan.full_plan.output_length),
        "LWT approximation stream does not cover the canonical coefficient interval");
    TT_FATAL(
        final_odd_delta <= 0 && static_cast<int64_t>(plan.full_plan.final_odd_length) + final_odd_delta >=
                                    static_cast<int64_t>(plan.full_plan.output_length),
        "LWT detail stream does not cover the canonical coefficient interval");
    const std::vector<uint32_t> chunk_words = build_chunk_config_words(plan);
    const std::vector<uint32_t> route_words = build_route_config_words(plan, buffers);
    buffers.chunk_config = upload_metadata(
        mesh_device, plan.chunks.size(), device_protocol::kLwtChunkConfigPageBytes, chunk_words, workload);
    buffers.route_config = upload_metadata(
        mesh_device, plan.chunks.size() * route_count, device_protocol::kRouteConfigPageBytes, route_words, workload);

    const std::vector<CoreChunkWork> work =
        partition_chunk_work(buffers.cores, checked_u32(plan.chunks.size(), "LWT chunk count"));
    auto descriptor = create_forward_program_descriptor(
        core_range_set(buffers.cores),
        input_buffer,
        buffers,
        plan,
        plan.workspace_layout,
        operation_attributes.boundary_mode,
        Scheme::compute_scheme_header,
        Scheme::compute_scheme_type,
        work);
    append_programs(workload, std::move(descriptor), tensor_coords);
    return workload;
}

template <typename Scheme>
[[nodiscard]] tt::tt_metal::WorkloadDescriptor build_inverse_workload(
    const Ilwt1DParams& operation_attributes,
    const Ilwt1DInputs& tensor_args,
    Tensor& tensor_return_value,
    const MeshCoordinateRangeSet& tensor_coords) {
    auto& mesh_device = *tensor_args.approximation.device();
    const auto& approximation_buffer = *tensor_args.approximation.buffer();
    const auto& detail_buffer = *tensor_args.detail.buffer();
    IlwtExecutionPlan plan = make_inverse_execution_plan<Scheme>(
        mesh_device,
        operation_attributes.original_length,
        tensor_args.approximation.logical_shape()[0],
        operation_attributes.boundary_mode);

    tt::tt_metal::WorkloadDescriptor workload;
    std::vector<tt::tt_metal::CoreCoord> cores = select_cores(mesh_device, plan.active_core_count);

    const size_t route_count = plan.chunks.front().routes.size();
    auto route_config =
        create_dram_buffer(mesh_device, plan.chunks.size() * route_count, device_protocol::kRouteConfigPageBytes);
    auto chunk_config = create_dram_buffer(mesh_device, plan.chunks.size(), device_protocol::kLwtChunkConfigPageBytes);
    IlwtWorkingBuffers buffers{
        .slot_addresses = workspace_slot_addresses(mesh_device, plan.workspace_elements),
        .output = tensor_return_value.buffer(),
        .route_config = route_config,
        .chunk_config = chunk_config,
        .cores = std::move(cores),
    };

    const std::vector<uint32_t> chunk_words = build_inverse_chunk_config_words(plan, buffers);
    const std::vector<uint32_t> route_words = build_inverse_route_config_words(plan, buffers);
    buffers.chunk_config = upload_metadata(
        mesh_device, plan.chunks.size(), device_protocol::kLwtChunkConfigPageBytes, chunk_words, workload);
    buffers.route_config = upload_metadata(
        mesh_device, plan.chunks.size() * route_count, device_protocol::kRouteConfigPageBytes, route_words, workload);

    const std::vector<CoreChunkWork> work =
        partition_chunk_work(buffers.cores, checked_u32(plan.chunks.size(), "ILWT chunk count"));
    using InverseScheme = typename Scheme::inverse;
    auto descriptor = create_inverse_program_descriptor(
        core_range_set(buffers.cores),
        approximation_buffer,
        detail_buffer,
        buffers,
        plan,
        plan.workspace_layout,
        InverseScheme::compute_scheme_header,
        InverseScheme::compute_scheme_type,
        work);
    append_programs(workload, std::move(descriptor), tensor_coords);
    return workload;
}

void validate_forward_inputs(const Lwt1DParams& operation_attributes, const Lwt1DInputs& tensor_args) {
    validate_rank_one_tensor(tensor_args.input, "LWT input");
    validate_output_memory_config(operation_attributes.output_memory_config, "LWT");
    TT_FATAL(operation_attributes.scheme_id != SchemeId::kUnknown, "LWT received an invalid wavelet scheme identifier");
    TT_FATAL(
        is_supported_lwt_boundary_mode(operation_attributes.boundary_mode),
        "LWT received an unsupported boundary mode");
    TT_FATAL(
        !boundary_mode_requires_multiple_samples(operation_attributes.boundary_mode) ||
            tensor_args.input.logical_shape()[0] > 1,
        "LWT reflect and antireflect modes require an input length greater than one");

    const auto expected_specs = Lwt1DDeviceOperation::compute_output_specs(operation_attributes, tensor_args);
    if (tensor_args.preallocated_outputs.has_value()) {
        validate_preallocated_output(
            std::get<0>(*tensor_args.preallocated_outputs),
            std::get<0>(expected_specs),
            tensor_args.input.device(),
            "LWT approximation output");
        validate_preallocated_output(
            std::get<1>(*tensor_args.preallocated_outputs),
            std::get<1>(expected_specs),
            tensor_args.input.device(),
            "LWT detail output");
        TT_FATAL(
            std::get<0>(*tensor_args.preallocated_outputs).buffer() !=
                std::get<1>(*tensor_args.preallocated_outputs).buffer(),
            "LWT approximation and detail outputs must not alias");
    }
}

void validate_inverse_inputs(const Ilwt1DParams& operation_attributes, const Ilwt1DInputs& tensor_args) {
    validate_rank_one_tensor(tensor_args.approximation, "ILWT approximation input");
    validate_rank_one_tensor(tensor_args.detail, "ILWT detail input");
    validate_output_memory_config(operation_attributes.output_memory_config, "ILWT");
    TT_FATAL(
        tensor_args.approximation.device() == tensor_args.detail.device(),
        "ILWT approximation and detail inputs must be on the same device");
    TT_FATAL(
        tensor_args.approximation.logical_shape() == tensor_args.detail.logical_shape(),
        "ILWT approximation and detail inputs must have identical shapes");
    TT_FATAL(
        tensor_args.approximation.buffer() != tensor_args.detail.buffer(),
        "ILWT approximation and detail inputs must not alias");
    TT_FATAL(operation_attributes.original_length > 0, "ILWT original_length must be greater than zero");
    TT_FATAL(
        operation_attributes.scheme_id != SchemeId::kUnknown, "ILWT received an invalid wavelet scheme identifier");
    TT_FATAL(
        is_supported_lwt_boundary_mode(operation_attributes.boundary_mode),
        "ILWT received an unsupported boundary mode");
    TT_FATAL(
        !boundary_mode_requires_multiple_samples(operation_attributes.boundary_mode) ||
            operation_attributes.original_length > 1,
        "ILWT reflect and antireflect modes require original_length greater than one");

    dispatch_scheme(operation_attributes.scheme_id, [&]<typename Scheme>() {
        static_cast<void>(make_inverse_lifting_plan<Scheme>(
            operation_attributes.original_length,
            tensor_args.approximation.logical_shape()[0],
            operation_attributes.boundary_mode));
    });

    if (tensor_args.preallocated_output.has_value()) {
        validate_preallocated_output(
            *tensor_args.preallocated_output,
            Ilwt1DDeviceOperation::compute_output_specs(operation_attributes, tensor_args),
            tensor_args.approximation.device(),
            "ILWT output");
        TT_FATAL(
            tensor_args.preallocated_output->buffer() != tensor_args.approximation.buffer() &&
                tensor_args.preallocated_output->buffer() != tensor_args.detail.buffer(),
            "ILWT output must not alias an input");
    }
}

}  // namespace

tt::tt_metal::WorkloadDescriptor Lwt1DDeviceOperation::ProgramFactory::create_workload_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const MeshCoordinateRangeSet& tensor_coords) {
    return dispatch_scheme(operation_attributes.scheme_id, [&]<typename Scheme>() {
        return build_forward_workload<Scheme>(operation_attributes, tensor_args, tensor_return_value, tensor_coords);
    });
}

void Lwt1DDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_forward_inputs(operation_attributes, tensor_args);
}

void Lwt1DDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_forward_inputs(operation_attributes, tensor_args);
}

Lwt1DDeviceOperation::spec_return_value_t Lwt1DDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& info = scheme_info(operation_attributes.scheme_id);
    const uint64_t input_length = tensor_args.input.logical_shape()[0];
    const uint64_t coefficient_length = (input_length + info.tap_size - 1) / 2;
    TT_FATAL(
        coefficient_length <= std::numeric_limits<uint32_t>::max(),
        "LWT coefficient length {} exceeds the device uint32 range",
        coefficient_length);
    auto spec =
        rank_one_output_spec(static_cast<uint32_t>(coefficient_length), operation_attributes.output_memory_config);
    return {spec, spec};
}

Lwt1DDeviceOperation::tensor_return_value_t Lwt1DDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_outputs.has_value()) {
        return *tensor_args.preallocated_outputs;
    }
    auto specs = compute_output_specs(operation_attributes, tensor_args);
    return {
        create_device_tensor(std::get<0>(specs), tensor_args.input.device()),
        create_device_tensor(std::get<1>(specs), tensor_args.input.device()),
    };
}

tt::tt_metal::WorkloadDescriptor Ilwt1DDeviceOperation::ProgramFactory::create_workload_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const MeshCoordinateRangeSet& tensor_coords) {
    return dispatch_scheme(operation_attributes.scheme_id, [&]<typename Scheme>() {
        return build_inverse_workload<Scheme>(operation_attributes, tensor_args, tensor_return_value, tensor_coords);
    });
}

void Ilwt1DDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inverse_inputs(operation_attributes, tensor_args);
}

void Ilwt1DDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inverse_inputs(operation_attributes, tensor_args);
}

Ilwt1DDeviceOperation::spec_return_value_t Ilwt1DDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t&) {
    return rank_one_output_spec(operation_attributes.original_length, operation_attributes.output_memory_config);
}

Ilwt1DDeviceOperation::tensor_return_value_t Ilwt1DDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return *tensor_args.preallocated_output;
    }
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.approximation.device());
}

std::tuple<Tensor, Tensor> lwt(
    const Tensor& input,
    const SchemeId scheme_id,
    const BoundaryMode boundary_mode,
    const MemoryConfig& output_memory_config,
    const std::optional<std::tuple<Tensor, Tensor>>& preallocated_outputs) {
    return device_operation::launch<Lwt1DDeviceOperation>(
        Lwt1DParams{
            .scheme_id = scheme_id,
            .boundary_mode = boundary_mode,
            .output_memory_config = output_memory_config,
        },
        Lwt1DInputs{
            .input = input,
            .preallocated_outputs = preallocated_outputs,
        });
}

Tensor ilwt(
    const Tensor& approximation,
    const Tensor& detail,
    const SchemeId scheme_id,
    const BoundaryMode boundary_mode,
    const uint32_t original_length,
    const MemoryConfig& output_memory_config,
    const std::optional<Tensor>& preallocated_output) {
    return device_operation::launch<Ilwt1DDeviceOperation>(
        Ilwt1DParams{
            .scheme_id = scheme_id,
            .boundary_mode = boundary_mode,
            .original_length = original_length,
            .output_memory_config = output_memory_config,
        },
        Ilwt1DInputs{
            .approximation = approximation,
            .detail = detail,
            .preallocated_output = preallocated_output,
        });
}

}  // namespace ttnn::prim
