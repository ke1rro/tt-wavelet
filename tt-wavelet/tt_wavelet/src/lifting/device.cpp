#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <tt_stl/assert.hpp>
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
#include "tt_wavelet/include/device_protocol/lwt_config.hpp"
#include "tt_wavelet/include/lifting/device.hpp"
#include "tt_wavelet/include/lifting/l1_accounting.hpp"
#include "tt_wavelet/include/lifting/policy.hpp"

namespace ttwv {

namespace {

constexpr tt::DataFormat kDataFormat = tt::DataFormat::Float32;

constexpr const char* kLwtReaderKernelPath = "kernels/dataflow/lwt_reader.cpp";
constexpr const char* kLwtWriterKernelPath = "kernels/dataflow/lwt_writer.cpp";
constexpr const char* kLwtComputeKernelPath = "kernels/compute/lwt_compute.cpp";

constexpr uint32_t kSrcTile0Cb = tt::CBIndex::c_0;
constexpr uint32_t kSrcTile1Cb = tt::CBIndex::c_1;
constexpr uint32_t kBaseTileCb = tt::CBIndex::c_2;
constexpr uint32_t kOutputCb = tt::CBIndex::c_16;
constexpr uint32_t kSrcCacheCb = tt::CBIndex::c_3;
constexpr uint32_t kInterleaveCb = tt::CBIndex::c_4;
constexpr uint32_t kSyncCb = tt::CBIndex::c_5;
constexpr uint32_t kReaderConfigCb = tt::CBIndex::c_6;
constexpr uint32_t kWriterConfigCb = tt::CBIndex::c_7;
constexpr uint32_t kTileGroupBuffering = 2;
constexpr uint32_t kDefaultL1SignalBudgetBytes = 768 * 1024;
constexpr const char* kL1SignalBudgetEnv = "TT_WAVELET_L1_SIGNAL_BUDGET_BYTES";
constexpr const char* kWorkspaceLayoutEnv = "TT_WAVELET_LWT_WORKSPACE_LAYOUT";

struct LwtProgram {
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

[[nodiscard]] std::filesystem::path kernel_path(const std::filesystem::path& root, const char* relative_path) {
    return root / relative_path;
}

[[nodiscard]] uint32_t checked_u32(const size_t value, const char* label) {
    TT_FATAL(
        value <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()), "{} {} overflows uint32_t", label, value);
    return static_cast<uint32_t>(value);
}

[[nodiscard]] uint32_t output_group_count(const size_t output_length) {
    return checked_u32(
        ceil_div(output_length, static_cast<size_t>(device_protocol::kLwtGroupOutputElements)), "LWT group count");
}

[[nodiscard]] uint32_t parse_positive_env(const char* name, const uint32_t fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        return fallback;
    }

    char* end = nullptr;
    errno = 0;
    const unsigned long value = std::strtoul(raw, &end, 10);
    TT_FATAL(
        errno == 0 && end != raw && *end == '\0' && value > 0 &&
            value <= static_cast<unsigned long>(std::numeric_limits<uint32_t>::max()),
        "{} must be a positive uint32 value, got '{}'",
        name,
        raw);
    return static_cast<uint32_t>(value);
}

[[nodiscard]] uint32_t l1_signal_budget_bytes() {
    return parse_positive_env(kL1SignalBudgetEnv, kDefaultL1SignalBudgetBytes);
}

[[nodiscard]] std::optional<WorkspaceLayout> workspace_layout_override() {
    const char* raw = std::getenv(kWorkspaceLayoutEnv);
    if (raw == nullptr || raw[0] == '\0' || std::strcmp(raw, "auto") == 0) {
        return std::nullopt;
    }
    if (std::strcmp(raw, "row-major") == 0) {
        return WorkspaceLayout::kRowMajor;
    }
    TT_FATAL(
        std::strcmp(raw, "tile-native") == 0,
        "{} must be 'auto', 'row-major', or 'tile-native', got '{}'",
        kWorkspaceLayoutEnv,
        raw);
    return WorkspaceLayout::kTileNative;
}

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
    return std::min(hardware_cores, parse_positive_env("TT_WAVELET_LWT_MAX_CORES", hardware_cores));
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

[[nodiscard]] std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> create_dram_signal_buffer(
    tt::tt_metal::distributed::MeshDevice& mesh_device, const SignalBuffer& desc) {
    const uint32_t page_bytes = desc.aligned_stick_bytes(kNocAlignmentBytes);
    return create_dram_buffer(mesh_device, desc.stick_count(), page_bytes);
}

[[nodiscard]] std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> create_workspace_buffer(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const std::vector<tt::tt_metal::CoreCoord>& cores,
    const uint32_t workspace_elements) {
    TT_FATAL(!cores.empty(), "LWT workspace requires at least one owner core");
    TT_FATAL(workspace_elements > 0, "LWT workspace must contain at least one element");
    TT_FATAL(
        workspace_elements % kStickWidth == 0,
        "LWT workspace length {} is not a multiple of the {}-element stick width",
        workspace_elements,
        kStickWidth);

    const uint32_t shard_sticks = workspace_elements / kStickWidth;
    const size_t capacity_sticks = static_cast<size_t>(shard_sticks) * cores.size();
    const size_t physical_nbytes = capacity_sticks * device_protocol::kStickBytes;
    TT_FATAL(
        physical_nbytes <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
        "LWT workspace slot size {} exceeds device buffer limits",
        physical_nbytes);

    const tt::tt_metal::CoreRangeSet core_set(cores);
    const tt::tt_metal::BufferDistributionSpec distribution(
        tt::tt_metal::Shape{static_cast<uint32_t>(capacity_sticks)}, tt::tt_metal::Shape{shard_sticks}, cores);
    const tt::tt_metal::ShardSpecBuffer shard_spec(
        core_set,
        {shard_sticks, kStickWidth},
        tt::tt_metal::ShardOrientation::ROW_MAJOR,
        {1, kStickWidth},
        {static_cast<uint32_t>(capacity_sticks), 1});
    const tt::tt_metal::BufferShardingArgs sharding_args(
        std::optional<tt::tt_metal::BufferDistributionSpec>{distribution},
        std::optional<tt::tt_metal::ShardSpecBuffer>{shard_spec},
        tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED);
    const tt::tt_metal::distributed::DeviceLocalBufferConfig local_config{
        .page_size = device_protocol::kStickBytes,
        .buffer_type = tt::tt_metal::BufferType::L1,
        .sharding_args = sharding_args,
        .bottom_up = false,
    };
    const tt::tt_metal::distributed::ReplicatedBufferConfig replicated_config{
        .size = static_cast<uint64_t>(physical_nbytes),
    };
    return tt::tt_metal::distributed::MeshBuffer::create(replicated_config, local_config, &mesh_device);
}

void create_circular_buffer(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& cores,
    const uint32_t cb_index,
    const uint32_t entry_count,
    const uint32_t page_bytes) {
    const auto config = tt::tt_metal::CircularBufferConfig(entry_count * page_bytes, {{cb_index, kDataFormat}})
                            .set_page_size(cb_index, page_bytes);
    static_cast<void>(tt::tt_metal::CreateCircularBuffer(program, cores, config));
}

void create_narrow_tile_circular_buffer(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& cores,
    const uint32_t cb_index,
    const uint32_t entry_count,
    const tt::tt_metal::Tile& tile) {
    const uint32_t page_bytes = tile.get_tile_size(kDataFormat);
    const auto config = tt::tt_metal::CircularBufferConfig(entry_count * page_bytes, {{cb_index, kDataFormat}})
                            .set_page_size(cb_index, page_bytes)
                            .set_tile_dims(cb_index, tile);
    static_cast<void>(tt::tt_metal::CreateCircularBuffer(program, cores, config));
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
        work.push_back(
            CoreChunkWork{
                .core = cores[core_index],
                .chunk_begin = chunk_begin,
                .chunk_count = count,
            });
        chunk_begin += count;
    }
    TT_FATAL(chunk_begin == chunk_count, "LWT chunk partition is incomplete");
    return work;
}

template <typename Plan>
void add_l1_telemetry(
    LiftingSchedulerTelemetry& telemetry,
    const Plan& plan,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const ArchitecturePolicy& policy) {
    const L1Accounting accounting = make_l1_accounting(
        plan.workspace_elements,
        plan.max_workspace_elements,
        policy.l1_scratch_bytes,
        mesh_device.l1_size_per_core());
    telemetry.architecture = policy.architecture;
    telemetry.workspace_layout = plan.workspace_layout;
    telemetry.max_workspace_elements = plan.max_workspace_elements;
    telemetry.l1_slots_bytes = accounting.slots_bytes;
    telemetry.l1_circular_buffers_bytes = accounting.circular_buffers_bytes;
    telemetry.l1_cache_bytes = accounting.cache_bytes;
    telemetry.l1_output_bytes = accounting.output_bytes;
    telemetry.l1_synchronization_bytes = accounting.synchronization_bytes;
    telemetry.l1_metadata_bytes = accounting.metadata_bytes;
    telemetry.l1_alignment_bytes = accounting.alignment_bytes;
    telemetry.l1_padding_bytes = accounting.padding_bytes;
    telemetry.l1_architecture_scratch_bytes = accounting.architecture_scratch_bytes;
    telemetry.l1_total_bytes = accounting.total_bytes;
    telemetry.l1_capacity_bytes = accounting.capacity_bytes;
    telemetry.l1_headroom_bytes = accounting.headroom_bytes;
}

[[nodiscard]] uint32_t resolve_output_address(const LwtWorkingBuffers& buffers, const RouteOutputRef output) {
    switch (output.storage) {
        case RouteOutputStorage::kWorkspaceSlot:
            return static_cast<uint32_t>(buffers.at(output.slot)->get_backing_buffer()->address());
        case RouteOutputStorage::kFinalEvenDram:
            return static_cast<uint32_t>(buffers.final_even->get_backing_buffer()->address());
        case RouteOutputStorage::kFinalOddDram:
            return static_cast<uint32_t>(buffers.final_odd->get_backing_buffer()->address());
    }
    TT_THROW("Unsupported LWT output storage");
}

[[nodiscard]] uint32_t resolve_workspace_address(const LwtWorkingBuffers& buffers, const StreamRef stream) {
    return static_cast<uint32_t>(buffers.at(stream.slot)->get_backing_buffer()->address());
}

[[nodiscard]] std::vector<uint32_t> build_chunk_config_words(const LwtExecutionPlan& plan) {
    std::vector<uint32_t> words(
        std::max(plan.chunks.size(), size_t{1}) * device_protocol::kLwtChunkConfigWordCount, 0);
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
            words[word_offset + device_protocol::kRouteFlags] =
                route.output.storage == RouteOutputStorage::kWorkspaceSlot ? 0U : device_protocol::kRouteFlagFinalDram;
        }
    }
    return words;
}

[[nodiscard]] std::vector<uint32_t> reader_runtime_args(
    const LwtExecutionPlan& plan,
    const LwtWorkingBuffers& buffers,
    const tt::tt_metal::Buffer& input_buffer,
    const CoreChunkWork& work) {
    return {
        static_cast<uint32_t>(input_buffer.address()),
        checked_u32(plan.full_plan.preprocess_layout.input.length, "LWT input length"),
        plan.full_plan.preprocess_layout.pad_config.left,
        static_cast<uint32_t>(buffers.at(StorageSlot::kA)->get_backing_buffer()->address()),
        static_cast<uint32_t>(buffers.at(StorageSlot::kB)->get_backing_buffer()->address()),
        static_cast<uint32_t>(buffers.chunk_config->get_backing_buffer()->address()),
        static_cast<uint32_t>(buffers.route_config->get_backing_buffer()->address()),
        work.chunk_begin,
        work.chunk_count,
        checked_u32(plan.chunks.front().routes.size(), "LWT route count"),
    };
}

[[nodiscard]] std::vector<uint32_t> writer_runtime_args(
    const LwtExecutionPlan& plan, const LwtWorkingBuffers& buffers, const CoreChunkWork& work) {
    return {
        static_cast<uint32_t>(buffers.route_config->get_backing_buffer()->address()),
        work.chunk_begin,
        work.chunk_count,
        checked_u32(plan.chunks.front().routes.size(), "LWT route count"),
    };
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

[[nodiscard]] LwtProgram create_program(
    const std::filesystem::path& kernel_root,
    const tt::tt_metal::CoreRangeSet& cores,
    const tt::tt_metal::Buffer& input_buffer,
    const LwtWorkingBuffers& buffers,
    const WorkspaceLayout workspace_layout,
    const BoundaryMode boundary_mode,
    const char* compute_scheme_header,
    const char* compute_scheme_type) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    const tt::tt_metal::Tile narrow_tile({32, 16});
    create_narrow_tile_circular_buffer(program, cores, kSrcTile0Cb, 2 * kTileGroupBuffering, narrow_tile);
    create_narrow_tile_circular_buffer(program, cores, kSrcTile1Cb, 2 * kTileGroupBuffering, narrow_tile);
    create_narrow_tile_circular_buffer(program, cores, kBaseTileCb, 3 * kTileGroupBuffering, narrow_tile);
    create_narrow_tile_circular_buffer(program, cores, kOutputCb, 3 * kTileGroupBuffering, narrow_tile);
    create_circular_buffer(
        program, cores, kSrcCacheCb, device_protocol::kLwtCacheStickCount, device_protocol::kStickBytes);
    create_circular_buffer(program, cores, kInterleaveCb, 1, device_protocol::kStickBytes);
    create_circular_buffer(program, cores, kSyncCb, 1, kNocAlignmentBytes);
    create_circular_buffer(program, cores, kReaderConfigCb, 1, device_protocol::kRouteConfigPageBytes);
    create_circular_buffer(program, cores, kWriterConfigCb, 1, device_protocol::kRouteConfigPageBytes);

    const auto& config_buffer = *buffers.route_config->get_backing_buffer();
    const auto& final_buffer = *buffers.final_even->get_backing_buffer();

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
    };
    tt::tt_metal::TensorAccessorArgs(config_buffer).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(final_buffer).append_to(writer_compile_args);

    const std::vector<uint32_t> compute_compile_args = {kSrcTile0Cb, kSrcTile1Cb, kBaseTileCb, kOutputCb};
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_to_dest_mode[kSrcTile0Cb] = UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest_mode[kSrcTile1Cb] = UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest_mode[kBaseTileCb] = UnpackToDestMode::UnpackToDestFp32;

    const auto reader = tt::tt_metal::CreateKernel(
        program,
        kernel_path(kernel_root, kLwtReaderKernelPath),
        cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_args));
    const auto writer = tt::tt_metal::CreateKernel(
        program,
        kernel_path(kernel_root, kLwtWriterKernelPath),
        cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_args));
    const auto compute = tt::tt_metal::CreateKernel(
        program,
        kernel_path(kernel_root, kLwtComputeKernelPath),
        cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .compile_args = compute_compile_args,
            .defines =
                {
                    {"TTWV_LWT_SCHEME_HEADER", compute_scheme_header},
                    {"TTWV_LWT_SCHEME_TYPE", compute_scheme_type},
                    {"TTWV_INLINE_TERMINAL_SCALE", "1"},
                },
        });
    return LwtProgram{.program = std::move(program), .reader = reader, .compute = compute, .writer = writer};
}

void set_runtime_args(
    const LwtProgram& program,
    const tt::tt_metal::Buffer& input_buffer,
    const LwtExecutionPlan& plan,
    const LwtWorkingBuffers& buffers,
    const std::vector<CoreChunkWork>& work) {
    for (const auto& core_work : work) {
        tt::tt_metal::SetRuntimeArgs(
            program.program,
            program.reader,
            core_work.core,
            reader_runtime_args(plan, buffers, input_buffer, core_work));
        tt::tt_metal::SetRuntimeArgs(
            program.program, program.compute, core_work.core, compute_runtime_args(plan, core_work));
        tt::tt_metal::SetRuntimeArgs(
            program.program, program.writer, core_work.core, writer_runtime_args(plan, buffers, core_work));
    }
}

[[nodiscard]] uint32_t resolve_workspace_address(const IlwtWorkingBuffers& buffers, const StreamRef stream) {
    return static_cast<uint32_t>(buffers.at(stream.slot)->get_backing_buffer()->address());
}

[[nodiscard]] std::vector<uint32_t> build_inverse_chunk_config_words(
    const IlwtExecutionPlan& plan, const IlwtWorkingBuffers& buffers) {
    std::vector<uint32_t> words(
        std::max(plan.chunks.size(), size_t{1}) * device_protocol::kLwtChunkConfigWordCount, 0);
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

[[nodiscard]] std::vector<uint32_t> inverse_reader_runtime_args(
    const IlwtExecutionPlan& plan,
    const IlwtWorkingBuffers& buffers,
    const tt::tt_metal::Buffer& approximation_buffer,
    const tt::tt_metal::Buffer& detail_buffer,
    const CoreChunkWork& work) {
    return {
        static_cast<uint32_t>(approximation_buffer.address()),
        static_cast<uint32_t>(detail_buffer.address()),
        checked_u32(plan.full_plan.coefficient_length, "ILWT coefficient length"),
        static_cast<uint32_t>(buffers.at(StorageSlot::kA)->get_backing_buffer()->address()),
        static_cast<uint32_t>(buffers.at(StorageSlot::kB)->get_backing_buffer()->address()),
        static_cast<uint32_t>(buffers.chunk_config->get_backing_buffer()->address()),
        static_cast<uint32_t>(buffers.route_config->get_backing_buffer()->address()),
        work.chunk_begin,
        work.chunk_count,
        checked_u32(plan.chunks.front().routes.size(), "ILWT route count"),
    };
}

[[nodiscard]] std::vector<uint32_t> inverse_writer_runtime_args(
    const IlwtExecutionPlan& plan, const IlwtWorkingBuffers& buffers, const CoreChunkWork& work) {
    return {
        static_cast<uint32_t>(buffers.route_config->get_backing_buffer()->address()),
        work.chunk_begin,
        work.chunk_count,
        checked_u32(plan.chunks.front().routes.size(), "ILWT route count"),
        static_cast<uint32_t>(buffers.chunk_config->get_backing_buffer()->address()),
        static_cast<uint32_t>(buffers.output->get_backing_buffer()->address()),
        plan.full_plan.forward_trace.preprocess_layout.pad_config.left,
    };
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

[[nodiscard]] LwtProgram create_inverse_program(
    const std::filesystem::path& kernel_root,
    const tt::tt_metal::CoreRangeSet& cores,
    const tt::tt_metal::Buffer& approximation_buffer,
    const tt::tt_metal::Buffer& detail_buffer,
    const IlwtWorkingBuffers& buffers,
    const WorkspaceLayout workspace_layout,
    const char* compute_scheme_header,
    const char* compute_scheme_type) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    const tt::tt_metal::Tile narrow_tile({32, 16});
    create_narrow_tile_circular_buffer(program, cores, kSrcTile0Cb, 2 * kTileGroupBuffering, narrow_tile);
    create_narrow_tile_circular_buffer(program, cores, kSrcTile1Cb, 2 * kTileGroupBuffering, narrow_tile);
    create_narrow_tile_circular_buffer(program, cores, kBaseTileCb, 3 * kTileGroupBuffering, narrow_tile);
    create_narrow_tile_circular_buffer(program, cores, kOutputCb, 3 * kTileGroupBuffering, narrow_tile);
    create_circular_buffer(
        program, cores, kSrcCacheCb, device_protocol::kLwtCacheStickCount, device_protocol::kStickBytes);
    create_circular_buffer(program, cores, kInterleaveCb, 1, device_protocol::kStickBytes);
    create_circular_buffer(program, cores, kSyncCb, 1, kNocAlignmentBytes);
    create_circular_buffer(program, cores, kReaderConfigCb, 1, device_protocol::kRouteConfigPageBytes);
    create_circular_buffer(program, cores, kWriterConfigCb, 1, device_protocol::kRouteConfigPageBytes);

    const auto& config_buffer = *buffers.route_config->get_backing_buffer();
    const auto& output_buffer = *buffers.output->get_backing_buffer();
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
    };
    tt::tt_metal::TensorAccessorArgs(config_buffer).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_compile_args);

    const std::vector<uint32_t> compute_compile_args = {kSrcTile0Cb, kSrcTile1Cb, kBaseTileCb, kOutputCb};
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_to_dest_mode[kSrcTile0Cb] = UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest_mode[kSrcTile1Cb] = UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest_mode[kBaseTileCb] = UnpackToDestMode::UnpackToDestFp32;

    const auto reader = tt::tt_metal::CreateKernel(
        program,
        kernel_path(kernel_root, kLwtReaderKernelPath),
        cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_args));
    const auto writer = tt::tt_metal::CreateKernel(
        program,
        kernel_path(kernel_root, kLwtWriterKernelPath),
        cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_args));
    const auto compute = tt::tt_metal::CreateKernel(
        program,
        kernel_path(kernel_root, kLwtComputeKernelPath),
        cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .compile_args = compute_compile_args,
            .defines =
                {
                    {"TTWV_LWT_SCHEME_HEADER", compute_scheme_header},
                    {"TTWV_LWT_SCHEME_TYPE", compute_scheme_type},
                    {"TTWV_INLINE_TERMINAL_SCALE", "0"},
                    {"TTWV_INLINE_INVERSE_SCALE", "1"},
                },
        });
    return LwtProgram{.program = std::move(program), .reader = reader, .compute = compute, .writer = writer};
}

void set_inverse_runtime_args(
    const LwtProgram& program,
    const tt::tt_metal::Buffer& approximation_buffer,
    const tt::tt_metal::Buffer& detail_buffer,
    const IlwtExecutionPlan& plan,
    const IlwtWorkingBuffers& buffers,
    const std::vector<CoreChunkWork>& work) {
    for (const auto& core_work : work) {
        tt::tt_metal::SetRuntimeArgs(
            program.program,
            program.reader,
            core_work.core,
            inverse_reader_runtime_args(plan, buffers, approximation_buffer, detail_buffer, core_work));
        tt::tt_metal::SetRuntimeArgs(
            program.program, program.compute, core_work.core, inverse_compute_runtime_args(plan, core_work));
        tt::tt_metal::SetRuntimeArgs(
            program.program, program.writer, core_work.core, inverse_writer_runtime_args(plan, buffers, core_work));
    }
}

void enqueue_program(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    tt::tt_metal::Program&& program) {
    tt::tt_metal::distributed::MeshWorkload workload;
    workload.add_program(tt::tt_metal::distributed::MeshCoordinateRange(mesh_device.shape()), std::move(program));
    tt::tt_metal::distributed::EnqueueMeshWorkload(command_queue, workload, false);
}

}  // namespace

LwtExecutable create_lwt_executable_impl(
    const std::filesystem::path& kernel_root,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const tt::tt_metal::Buffer& input_buffer,
    LiftingForwardPlan full_plan,
    const char* compute_scheme_header,
    const char* compute_scheme_type) {
    const BoundaryMode boundary_mode = full_plan.preprocess_layout.pad_config.mode;
    TT_FATAL(is_supported_lwt_boundary_mode(boundary_mode), "LWT received an unsupported boundary mode");
    TT_FATAL(
        !boundary_mode_requires_multiple_samples(boundary_mode) || full_plan.preprocess_layout.input.length > 1,
        "reflect and antireflect boundary modes require an input length greater than one");
    TT_FATAL(
        full_plan.preprocess_layout.padded_length() <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
        "LWT padded input length exceeds device signed-index range");

    const uint32_t max_cores = core_limit(mesh_device);
    const uint32_t signal_budget_bytes = l1_signal_budget_bytes();
    const ArchitecturePolicy architecture_policy = make_architecture_policy(mesh_device.arch());
    const std::optional<WorkspaceLayout> workspace_override = workspace_layout_override();
    const WorkspaceLayout initial_workspace_layout = workspace_override.value_or(WorkspaceLayout::kRowMajor);
    LwtExecutionPlan plan =
        make_lwt_execution_plan(std::move(full_plan), max_cores, signal_budget_bytes, initial_workspace_layout);
    if (!workspace_override.has_value() && prefer_tile_native_workspace(plan)) {
        plan = make_lwt_execution_plan(
            std::move(plan.full_plan), max_cores, signal_budget_bytes, WorkspaceLayout::kTileNative);
    }
    std::vector<tt::tt_metal::CoreCoord> cores = select_cores(mesh_device, plan.active_core_count);
    std::array<std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>, 3> slots;
    for (auto& slot : slots) {
        slot = create_workspace_buffer(mesh_device, cores, plan.workspace_elements);
    }

    SignalBuffer final_even_desc = plan.full_plan.preprocess_layout.output.even;
    final_even_desc.length = plan.full_plan.final_even_length;
    SignalBuffer final_odd_desc = plan.full_plan.preprocess_layout.output.odd;
    final_odd_desc.length = plan.full_plan.final_odd_length;
    auto final_even = create_dram_signal_buffer(mesh_device, final_even_desc);
    auto final_odd = create_dram_signal_buffer(mesh_device, final_odd_desc);
    const size_t route_count = plan.chunks.front().routes.size();
    auto route_config =
        create_dram_buffer(mesh_device, plan.chunks.size() * route_count, device_protocol::kRouteConfigPageBytes);
    auto chunk_config = create_dram_buffer(mesh_device, plan.chunks.size(), device_protocol::kLwtChunkConfigPageBytes);

    const size_t max_final_length = std::max(plan.full_plan.final_even_length, plan.full_plan.final_odd_length);
    LwtWorkingBuffers buffers{
        .slots = std::move(slots),
        .final_even = std::move(final_even),
        .final_odd = std::move(final_odd),
        .route_config = std::move(route_config),
        .chunk_config = std::move(chunk_config),
        .cores = std::move(cores),
        .scheduler =
            LiftingSchedulerTelemetry{
                .signal_length = plan.full_plan.preprocess_layout.input.length,
                .max_group_count = output_group_count(max_final_length),
                .active_core_count = plan.active_core_count,
                .chunk_count = checked_u32(plan.chunks.size(), "LWT chunk count"),
                .groups_per_chunk = plan.groups_per_chunk,
                .workspace_elements = plan.workspace_elements,
                .max_dependency_overhead = plan.max_dependency_overhead,
                .terminal_scale_inline = true,
                .workspace_layout = plan.workspace_layout,
            },
    };
    add_l1_telemetry(buffers.scheduler, plan, mesh_device, architecture_policy);

    const std::vector<CoreChunkWork> work =
        partition_chunk_work(buffers.cores, checked_u32(plan.chunks.size(), "LWT chunk count"));
    LwtProgram program = create_program(
        kernel_root,
        core_range_set(buffers.cores),
        input_buffer,
        buffers,
        plan.workspace_layout,
        boundary_mode,
        compute_scheme_header,
        compute_scheme_type);
    set_runtime_args(program, input_buffer, plan, buffers, work);
    return LwtExecutable{
        .plan = std::move(plan),
        .buffers = std::move(buffers),
        .lifting = std::move(program.program),
    };
}

void prepare_lwt(
    tt::tt_metal::distributed::MeshCommandQueue& command_queue, LwtExecutable& executable) {
    const std::vector<uint32_t> chunk_words = build_chunk_config_words(executable.plan);
    const std::vector<uint32_t> route_words = build_route_config_words(executable.plan, executable.buffers);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(
        command_queue, executable.buffers.chunk_config, chunk_words, false);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(
        command_queue, executable.buffers.route_config, route_words, false);
    tt::tt_metal::distributed::Finish(command_queue);
}

void execute_lwt(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    LwtExecutable& executable) {
    enqueue_program(mesh_device, command_queue, std::move(executable.lifting));
    tt::tt_metal::distributed::Finish(command_queue);
}

IlwtExecutable create_ilwt_executable_impl(
    const std::filesystem::path& kernel_root,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const tt::tt_metal::Buffer& approximation_buffer,
    const tt::tt_metal::Buffer& detail_buffer,
    LiftingInversePlan full_plan,
    const char* inverse_compute_scheme_header,
    const char* inverse_compute_scheme_type) {
    TT_FATAL(
        is_supported_lwt_boundary_mode(full_plan.forward_trace.preprocess_layout.pad_config.mode),
        "ILWT received an unsupported boundary mode");
    TT_FATAL(
        !boundary_mode_requires_multiple_samples(full_plan.forward_trace.preprocess_layout.pad_config.mode) ||
            full_plan.original_length > 1,
        "reflect and antireflect boundary modes require an original length greater than one");
    TT_FATAL(
        full_plan.original_length <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
        "ILWT output length exceeds device limits");

    const uint32_t max_cores = core_limit(mesh_device);
    const uint32_t signal_budget_bytes = l1_signal_budget_bytes();
    const std::optional<WorkspaceLayout> workspace_override = workspace_layout_override();
    const ArchitecturePolicy architecture_policy =
        make_architecture_policy(mesh_device.arch(), workspace_override);
    TT_FATAL(architecture_policy.inverse_scale_inline, "ILWT policy must preserve inline FP32 inverse scaling");
    IlwtExecutionPlan plan = make_ilwt_execution_plan(
        std::move(full_plan),
        max_cores,
        signal_budget_bytes,
        architecture_policy.ilwt_layout,
        architecture_policy.final_interleave_direct);

    std::vector<tt::tt_metal::CoreCoord> cores = select_cores(mesh_device, plan.active_core_count);
    std::array<std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>, 3> slots;
    for (auto& slot : slots) {
        slot = create_workspace_buffer(mesh_device, cores, plan.workspace_elements);
    }

    const SignalBuffer output_desc{
        .dram_address = 0,
        .length = plan.full_plan.original_length,
        .stick_width = kStickWidth,
        .element_size_bytes = sizeof(float),
    };
    auto output = create_dram_signal_buffer(mesh_device, output_desc);
    const size_t route_count = plan.chunks.front().routes.size();
    auto route_config =
        create_dram_buffer(mesh_device, plan.chunks.size() * route_count, device_protocol::kRouteConfigPageBytes);
    auto chunk_config = create_dram_buffer(mesh_device, plan.chunks.size(), device_protocol::kLwtChunkConfigPageBytes);

    IlwtWorkingBuffers buffers{
        .slots = std::move(slots),
        .output = std::move(output),
        .route_config = std::move(route_config),
        .chunk_config = std::move(chunk_config),
        .cores = std::move(cores),
        .scheduler =
            LiftingSchedulerTelemetry{
                .signal_length = plan.full_plan.original_length,
                .max_group_count = checked_u32(
                    ceil_div(plan.full_plan.original_length, size_t{device_protocol::kIlwtGroupOutputElements}),
                    "ILWT output group count"),
                .active_core_count = plan.active_core_count,
                .chunk_count = checked_u32(plan.chunks.size(), "ILWT chunk count"),
                .groups_per_chunk = plan.output_groups_per_chunk,
                .workspace_elements = plan.workspace_elements,
                .max_dependency_overhead = plan.max_dependency_overhead,
                .terminal_scale_inline = false,
                .inverse_scale_inline = architecture_policy.inverse_scale_inline,
                .inverse_final_interleave_direct = plan.final_interleave_direct,
                .workspace_layout = plan.workspace_layout,
            },
    };
    add_l1_telemetry(buffers.scheduler, plan, mesh_device, architecture_policy);

    const std::vector<CoreChunkWork> work =
        partition_chunk_work(buffers.cores, checked_u32(plan.chunks.size(), "ILWT chunk count"));
    LwtProgram program = create_inverse_program(
        kernel_root,
        core_range_set(buffers.cores),
        approximation_buffer,
        detail_buffer,
        buffers,
        plan.workspace_layout,
        inverse_compute_scheme_header,
        inverse_compute_scheme_type);
    set_inverse_runtime_args(program, approximation_buffer, detail_buffer, plan, buffers, work);
    return IlwtExecutable{
        .plan = std::move(plan),
        .buffers = std::move(buffers),
        .lifting = std::move(program.program),
    };
}

void prepare_ilwt(
    tt::tt_metal::distributed::MeshCommandQueue& command_queue, IlwtExecutable& executable) {
    const std::vector<uint32_t> chunk_words = build_inverse_chunk_config_words(executable.plan, executable.buffers);
    const std::vector<uint32_t> route_words = build_inverse_route_config_words(executable.plan, executable.buffers);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(
        command_queue, executable.buffers.chunk_config, chunk_words, false);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(
        command_queue, executable.buffers.route_config, route_words, false);
    tt::tt_metal::distributed::Finish(command_queue);
}

void execute_ilwt(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    IlwtExecutable& executable) {
    enqueue_program(mesh_device, command_queue, std::move(executable.lifting));
    tt::tt_metal::distributed::Finish(command_queue);
}

}  // namespace ttwv
