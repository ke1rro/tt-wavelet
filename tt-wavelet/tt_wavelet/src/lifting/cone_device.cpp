#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <tt_stl/assert.hpp>
#include <utility>
#include <vector>

#include "tt-metalium/buffer.hpp"
#include "tt-metalium/circular_buffer_constants.h"
#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/tensor_accessor_args.hpp"
#include "tt_wavelet/include/device_protocol/lwt_config.hpp"
#include "tt_wavelet/include/lifting/device.hpp"

namespace ttwv {

namespace {

constexpr tt::DataFormat kDataFormat = tt::DataFormat::Float32;

constexpr const char* kConeReaderKernelPath = "kernels/dataflow/lwt_cone_reader.cpp";
constexpr const char* kConeWriterKernelPath = "kernels/dataflow/lwt_cone_writer.cpp";
constexpr const char* kConeComputeKernelPath = "kernels/compute/lwt_cone_compute.cpp";

constexpr uint32_t kSrcTile0Cb = tt::CBIndex::c_0;
constexpr uint32_t kSrcTile1Cb = tt::CBIndex::c_1;
constexpr uint32_t kBaseTileCb = tt::CBIndex::c_2;
constexpr uint32_t kOutputCb = tt::CBIndex::c_16;
constexpr uint32_t kSrcCacheCb = tt::CBIndex::c_3;
constexpr uint32_t kSyncCb = tt::CBIndex::c_5;
constexpr uint32_t kReaderConfigCb = tt::CBIndex::c_6;
constexpr uint32_t kWriterConfigCb = tt::CBIndex::c_7;
constexpr uint32_t kWorkspaceACb = tt::CBIndex::c_8;
constexpr uint32_t kWorkspaceBCb = tt::CBIndex::c_9;
constexpr uint32_t kWorkspaceScratchCb = tt::CBIndex::c_10;

constexpr uint32_t kTileGroupBuffering = 2;
constexpr uint32_t kDefaultL1SignalBudgetBytes = 768 * 1024;
constexpr const char* kL1SignalBudgetEnv = "TT_WAVELET_L1_SIGNAL_BUDGET_BYTES";

struct ConeProgram {
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
        ceil_div(output_length, static_cast<size_t>(device_protocol::kLwtGroupOutputElements)), "cone group count");
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

[[nodiscard]] uint32_t core_limit(tt::tt_metal::distributed::MeshDevice& mesh_device) {
    const auto grid = mesh_device.compute_with_storage_grid_size();
    const uint32_t hardware_cores = static_cast<uint32_t>(grid.x * grid.y);
    TT_FATAL(hardware_cores > 0, "ConeStreamed requires at least one hardware worker core");
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

[[nodiscard]] std::vector<CoreChunkWork> partition_chunk_work(
    const std::vector<tt::tt_metal::CoreCoord>& cores, const uint32_t chunk_count) {
    TT_FATAL(!cores.empty(), "Cone chunk partition requires cores");
    TT_FATAL(chunk_count >= cores.size(), "Cone active core count exceeds chunk count");

    const uint32_t active_core_count = checked_u32(cores.size(), "cone core count");
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
    TT_FATAL(chunk_begin == chunk_count, "Cone chunk partition is incomplete");
    return work;
}

[[nodiscard]] uint32_t resolve_output_address(const ConeWorkingBuffers& buffers, const RouteOutputRef output) {
    switch (output.storage) {
        case RouteOutputStorage::kResidentSlot: return static_cast<uint32_t>(output.slot);
        case RouteOutputStorage::kFinalEvenDram: return static_cast<uint32_t>(buffers.final_even->address());
        case RouteOutputStorage::kFinalOddDram: return static_cast<uint32_t>(buffers.final_odd->address());
    }
    TT_THROW("Unsupported ConeStreamed output storage");
}

[[nodiscard]] std::vector<uint32_t> build_chunk_config_words(const ConeExecutionPlan& plan) {
    std::vector<uint32_t> words(
        std::max(plan.chunks.size(), size_t{1}) * device_protocol::kConeChunkConfigWordCount, 0);
    for (size_t chunk_index = 0; chunk_index < plan.chunks.size(); ++chunk_index) {
        const auto& chunk = plan.chunks[chunk_index];
        const size_t offset = chunk_index * device_protocol::kConeChunkConfigWordCount;
        words[offset + device_protocol::kConeInitialEvenBegin] =
            checked_u32(chunk.initial_even.begin, "initial even begin");
        words[offset + device_protocol::kConeInitialEvenLength] =
            checked_u32(chunk.initial_even.length(), "initial even length");
        words[offset + device_protocol::kConeInitialOddBegin] =
            checked_u32(chunk.initial_odd.begin, "initial odd begin");
        words[offset + device_protocol::kConeInitialOddLength] =
            checked_u32(chunk.initial_odd.length(), "initial odd length");
    }
    return words;
}

[[nodiscard]] std::vector<uint32_t> build_route_config_words(
    const ConeExecutionPlan& plan, const ConeWorkingBuffers& buffers) {
    TT_FATAL(!plan.chunks.empty(), "ConeStreamed plan has no chunks");
    const size_t route_count = plan.chunks.front().routes.size();
    std::vector<uint32_t> words(
        std::max(plan.chunks.size() * route_count, size_t{1}) * device_protocol::kRouteConfigWordCount, 0);

    for (size_t chunk_index = 0; chunk_index < plan.chunks.size(); ++chunk_index) {
        const auto& chunk = plan.chunks[chunk_index];
        TT_FATAL(chunk.routes.size() == route_count, "Cone chunks have inconsistent route counts");
        for (size_t route_index = 0; route_index < route_count; ++route_index) {
            const auto& route = chunk.routes[route_index];
            const uint32_t output_offset = checked_u32(route.output_offset_elements, "cone output offset");
            const size_t word_offset =
                (chunk_index * route_count + route_index) * device_protocol::kRouteConfigWordCount;
            words[word_offset + device_protocol::kRouteType] = static_cast<uint32_t>(route.type);
            words[word_offset + device_protocol::kRouteSourceAddr] = static_cast<uint32_t>(route.source.slot);
            words[word_offset + device_protocol::kRouteSourceLength] =
                checked_u32(route.source_storage_length, "cone source storage end");
            words[word_offset + device_protocol::kRouteBaseAddr] = static_cast<uint32_t>(route.base.slot);
            words[word_offset + device_protocol::kRouteBaseLength] =
                checked_u32(route.base_storage_length, "cone base storage end");
            words[word_offset + device_protocol::kRouteOutputAddr] = resolve_output_address(buffers, route.output);
            words[word_offset + device_protocol::kRouteOutputLength] =
                checked_u32(route.output_length, "cone output length");
            words[word_offset + device_protocol::kRouteSourceOffset] =
                checked_u32(route.source_offset_elements, "cone source offset");
            words[word_offset + device_protocol::kRouteBaseOffset] =
                checked_u32(route.base_offset_elements, "cone base offset");
            words[word_offset + device_protocol::kRouteSourceLeftPad] = route.source_left_pad_elements;
            words[word_offset + device_protocol::kRouteOutputOffset] = output_offset;
            words[word_offset + device_protocol::kRouteGroupCount] = output_group_count(route.output_length);
        }
    }
    return words;
}

[[nodiscard]] std::vector<uint32_t> reader_runtime_args(
    const ConeExecutionPlan& plan,
    const ConeWorkingBuffers& buffers,
    const tt::tt_metal::Buffer& input_buffer,
    const CoreChunkWork& work) {
    return {
        static_cast<uint32_t>(input_buffer.address()),
        checked_u32(plan.full_plan.preprocess_layout.input.length, "cone input length"),
        plan.full_plan.preprocess_layout.pad_config.left,
        static_cast<uint32_t>(buffers.chunk_config->address()),
        static_cast<uint32_t>(buffers.route_config->address()),
        work.chunk_begin,
        work.chunk_count,
        checked_u32(plan.chunks.front().routes.size(), "cone route count"),
    };
}

[[nodiscard]] std::vector<uint32_t> writer_runtime_args(
    const ConeExecutionPlan& plan, const ConeWorkingBuffers& buffers, const CoreChunkWork& work) {
    return {
        static_cast<uint32_t>(buffers.route_config->address()),
        work.chunk_begin,
        work.chunk_count,
        checked_u32(plan.chunks.front().routes.size(), "cone route count"),
    };
}

[[nodiscard]] std::vector<uint32_t> compute_runtime_args(const ConeExecutionPlan& plan, const CoreChunkWork& work) {
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

[[nodiscard]] ConeProgram create_program(
    const std::filesystem::path& kernel_root,
    const tt::tt_metal::CoreRangeSet& cores,
    const tt::tt_metal::Buffer& input_buffer,
    const ConeWorkingBuffers& buffers,
    const char* compute_scheme_header,
    const char* compute_scheme_type) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    const uint32_t tile_bytes = tt::tile_size(kDataFormat);
    create_circular_buffer(program, cores, kSrcTile0Cb, kTileGroupBuffering, tile_bytes);
    create_circular_buffer(program, cores, kSrcTile1Cb, kTileGroupBuffering, tile_bytes);
    create_circular_buffer(program, cores, kBaseTileCb, 2 * kTileGroupBuffering, tile_bytes);
    create_circular_buffer(program, cores, kOutputCb, 2 * kTileGroupBuffering, tile_bytes);
    create_circular_buffer(
        program, cores, kSrcCacheCb, device_protocol::kLwtCacheStickCount, device_protocol::kStickBytes);
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
    };
    tt::tt_metal::TensorAccessorArgs(config_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_compile_args);

    std::vector<uint32_t> writer_compile_args = {kWriterConfigCb, kOutputCb, kSyncCb};
    tt::tt_metal::TensorAccessorArgs(config_buffer).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(final_buffer).append_to(writer_compile_args);

    const std::vector<uint32_t> compute_compile_args = {kSrcTile0Cb, kSrcTile1Cb, kBaseTileCb, kOutputCb};
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_to_dest_mode[kSrcTile0Cb] = UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest_mode[kSrcTile1Cb] = UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest_mode[kBaseTileCb] = UnpackToDestMode::UnpackToDestFp32;

    const auto reader = tt::tt_metal::CreateKernel(
        program,
        kernel_path(kernel_root, kConeReaderKernelPath),
        cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_args));
    const auto writer = tt::tt_metal::CreateKernel(
        program,
        kernel_path(kernel_root, kConeWriterKernelPath),
        cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_args));
    const auto compute = tt::tt_metal::CreateKernel(
        program,
        kernel_path(kernel_root, kConeComputeKernelPath),
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
                },
        });
    return ConeProgram{.program = std::move(program), .reader = reader, .compute = compute, .writer = writer};
}

void set_runtime_args(
    const ConeProgram& program,
    const tt::tt_metal::Buffer& input_buffer,
    const ConeExecutionPlan& plan,
    const ConeWorkingBuffers& buffers,
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

void enqueue_program(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    tt::tt_metal::Program&& program) {
    tt::tt_metal::distributed::MeshWorkload workload;
    workload.add_program(tt::tt_metal::distributed::MeshCoordinateRange(mesh_device.shape()), std::move(program));
    tt::tt_metal::distributed::EnqueueMeshWorkload(command_queue, workload, false);
}

}  // namespace

ConeStreamedLwtExecutable create_cone_streamed_lwt_executable_impl(
    const std::filesystem::path& kernel_root,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const tt::tt_metal::Buffer& input_buffer,
    LiftingForwardPlan full_plan,
    const char* compute_scheme_header,
    const char* compute_scheme_type) {
    TT_FATAL(
        full_plan.preprocess_layout.pad_config.mode == BoundaryMode::kSymmetric,
        "ConeStreamed currently supports symmetric boundary mode only");
    TT_FATAL(
        full_plan.preprocess_layout.padded_length() <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
        "ConeStreamed padded input length exceeds device signed-index range");

    ConeExecutionPlan plan =
        make_cone_execution_plan(std::move(full_plan), core_limit(mesh_device), l1_signal_budget_bytes());
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
    auto chunk_config = create_dram_buffer(mesh_device, plan.chunks.size(), device_protocol::kConeChunkConfigPageBytes);

    const size_t max_final_length = std::max(plan.full_plan.final_even_length, plan.full_plan.final_odd_length);
    ConeWorkingBuffers buffers{
        .slots = std::move(slots),
        .final_even = std::move(final_even),
        .final_odd = std::move(final_odd),
        .route_config = std::move(route_config),
        .chunk_config = std::move(chunk_config),
        .cores = std::move(cores),
        .scheduler =
            LiftingSchedulerTelemetry{
                .memory_mode = LwtMemoryMode::kConeStreamed,
                .max_group_count = output_group_count(max_final_length),
                .groups_per_shard = 0,
                .active_core_count = plan.active_core_count,
                .shard_elements = plan.workspace_elements,
                .zero_work_cores_per_route = {},
                .chunk_count = checked_u32(plan.chunks.size(), "cone chunk count"),
                .groups_per_chunk = plan.groups_per_chunk,
                .workspace_elements = plan.workspace_elements,
                .max_dependency_overhead = plan.max_dependency_overhead,
            },
    };

    const std::vector<CoreChunkWork> work =
        partition_chunk_work(buffers.cores, checked_u32(plan.chunks.size(), "cone chunk count"));
    ConeProgram program = create_program(
        kernel_root, core_range_set(buffers.cores), input_buffer, buffers, compute_scheme_header, compute_scheme_type);
    set_runtime_args(program, input_buffer, plan, buffers, work);
    return ConeStreamedLwtExecutable{
        .plan = std::move(plan),
        .buffers = std::move(buffers),
        .lifting = std::move(program.program),
    };
}

void prepare_cone_streamed_lwt(
    tt::tt_metal::distributed::MeshCommandQueue& command_queue, ConeStreamedLwtExecutable& executable) {
    const std::vector<uint32_t> chunk_words = build_chunk_config_words(executable.plan);
    const std::vector<uint32_t> route_words = build_route_config_words(executable.plan, executable.buffers);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(
        command_queue, executable.buffers.chunk_config, chunk_words, false);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(
        command_queue, executable.buffers.route_config, route_words, false);
    tt::tt_metal::distributed::Finish(command_queue);
}

void execute_cone_streamed_lwt(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    ConeStreamedLwtExecutable& executable) {
    enqueue_program(mesh_device, command_queue, std::move(executable.lifting));
    tt::tt_metal::distributed::Finish(command_queue);
}

}  // namespace ttwv
