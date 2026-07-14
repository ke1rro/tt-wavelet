#include "tt_wavelet/include/lifting/device.hpp"

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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
#include "tt_wavelet/include/device_protocol/lwt_config.hpp"

namespace ttwv {

namespace {

constexpr tt::DataFormat kDataFormat = tt::DataFormat::Float32;

constexpr const char* kLwtReaderKernelPath = "kernels/dataflow/lwt_reader.cpp";
constexpr const char* kLwtWriterKernelPath = "kernels/dataflow/lwt_writer.cpp";
constexpr const char* kLwtComputeKernelPath = "kernels/compute/lwt_compute.cpp";

constexpr uint32_t kLwtSrcTile0Cb = tt::CBIndex::c_0;
constexpr uint32_t kLwtSrcTile1Cb = tt::CBIndex::c_1;
constexpr uint32_t kLwtBaseTileCb = tt::CBIndex::c_2;
constexpr uint32_t kLwtOutputCb = tt::CBIndex::c_16;
constexpr uint32_t kLwtSrcCacheCb = tt::CBIndex::c_3;
constexpr uint32_t kLwtBaseCacheCb = tt::CBIndex::c_4;
constexpr uint32_t kLwtSyncCb = tt::CBIndex::c_5;
constexpr uint32_t kLwtReaderConfigCb = tt::CBIndex::c_6;
constexpr uint32_t kLwtWriterConfigCb = tt::CBIndex::c_7;

constexpr uint32_t kTileGroupBuffering = 2;
// Keep the three resident signal slots below half of Wormhole Tensix L1. The
// remaining space is needed by circular buffers, compiled kernels, and the
// dispatch/runtime allocations that share the same SRAM.
constexpr uint32_t kDefaultL1SignalBudgetBytes = 768 * 1024;
constexpr const char* kL1SignalBudgetEnv = "TT_WAVELET_L1_SIGNAL_BUDGET_BYTES";

struct LwtProgram {
    tt::tt_metal::Program program;
    tt::tt_metal::KernelHandle reader{};
    tt::tt_metal::KernelHandle compute{};
    tt::tt_metal::KernelHandle writer{};
};

struct ResidentShardPlan {
    std::vector<tt::tt_metal::CoreCoord> cores;
    uint32_t max_group_count{0};
    uint32_t groups_per_shard{0};
    uint32_t shard_elements{0};
};

struct RouteGroupRange {
    uint32_t begin{0};
    uint32_t count{0};
};

struct LwtCoreWork {
    tt::tt_metal::CoreCoord core;
    std::vector<RouteGroupRange> routes;
};

[[nodiscard]] std::filesystem::path kernel_path(const std::filesystem::path& kernel_root, const char* relative_path) {
    return kernel_root / relative_path;
}

[[nodiscard]] std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> create_dram_signal_mesh_buffer(
    tt::tt_metal::distributed::MeshDevice& mesh_device, const SignalBuffer& buffer) {
    const uint32_t page_size = buffer.aligned_stick_bytes(kNocAlignmentBytes);
    const size_t physical_nbytes = std::max(buffer.physical_nbytes(kNocAlignmentBytes), static_cast<size_t>(page_size));

    const tt::tt_metal::distributed::DeviceLocalBufferConfig local_config{
        .page_size = page_size,
        .buffer_type = tt::tt_metal::BufferType::DRAM,
        .bottom_up = false,
    };
    const tt::tt_metal::distributed::ReplicatedBufferConfig replicated_config{
        .size = static_cast<uint64_t>(physical_nbytes),
    };
    return tt::tt_metal::distributed::MeshBuffer::create(replicated_config, local_config, &mesh_device);
}

[[nodiscard]] std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> create_resident_signal_mesh_buffer(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const std::vector<tt::tt_metal::CoreCoord>& cores,
    const uint32_t shard_elements) {
    TT_FATAL(!cores.empty(), "Resident LWT storage requires at least one core");
    TT_FATAL(shard_elements > 0, "Resident LWT shard size must be non-zero");
    TT_FATAL(
        shard_elements % kStickWidth == 0,
        "Resident LWT shard size {} is not a multiple of the {}-element stick width",
        shard_elements,
        kStickWidth);

    const uint32_t shard_sticks = shard_elements / kStickWidth;
    const size_t capacity_sticks = static_cast<size_t>(shard_sticks) * cores.size();
    const size_t physical_nbytes = capacity_sticks * device_protocol::kStickBytes;
    TT_FATAL(
        physical_nbytes <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
        "Resident LWT slot size {} exceeds device buffer limits",
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

[[nodiscard]] std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> create_route_config_mesh_buffer(
    tt::tt_metal::distributed::MeshDevice& mesh_device, const size_t route_count) {
    const size_t page_count = std::max(route_count, size_t{1});
    const tt::tt_metal::distributed::DeviceLocalBufferConfig local_config{
        .page_size = device_protocol::kRouteConfigPageBytes,
        .buffer_type = tt::tt_metal::BufferType::DRAM,
    };
    const tt::tt_metal::distributed::ReplicatedBufferConfig replicated_config{
        .size = static_cast<uint64_t>(page_count * device_protocol::kRouteConfigPageBytes),
    };
    return tt::tt_metal::distributed::MeshBuffer::create(replicated_config, local_config, &mesh_device);
}

void create_circular_buffer(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& cores,
    const uint32_t cb_index,
    const uint32_t entry_count,
    const uint32_t page_size) {
    const auto config = tt::tt_metal::CircularBufferConfig(entry_count * page_size, {{cb_index, kDataFormat}})
                            .set_page_size(cb_index, page_size);
    static_cast<void>(tt::tt_metal::CreateCircularBuffer(program, cores, config));
}

[[nodiscard]] uint32_t checked_length(const size_t value, const char* label) {
    TT_FATAL(
        value <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()), "{} {} overflows uint32_t", label, value);
    return static_cast<uint32_t>(value);
}

[[nodiscard]] uint32_t checked_i32_arg(const int32_t value) { return static_cast<uint32_t>(value); }

[[nodiscard]] uint32_t lwt_output_group_count(const size_t output_length) {
    return checked_length(
        ceil_div(output_length, static_cast<size_t>(device_protocol::kLwtGroupOutputElements)), "output groups");
}

[[nodiscard]] bool is_executable_route(const LiftingStepRoute& route) noexcept {
    return is_predict_update_step(route.type) || is_scale_step(route.type);
}

[[nodiscard]] std::vector<uint32_t> build_route_group_counts(const LiftingForwardPlan& plan) {
    std::vector<uint32_t> group_counts;
    group_counts.reserve(plan.routes.size());

    for (const auto& route : plan.routes) {
        if (!is_executable_route(route)) {
            continue;
        }
        group_counts.push_back(lwt_output_group_count(route.output_length));
    }

    return group_counts;
}

[[nodiscard]] uint32_t max_route_group_count(const std::vector<uint32_t>& route_group_counts) {
    return route_group_counts.empty() ? 0U : *std::max_element(route_group_counts.begin(), route_group_counts.end());
}

[[nodiscard]] uint32_t env_lwt_max_cores() {
    const char* raw = std::getenv("TT_WAVELET_LWT_MAX_CORES");
    if (raw == nullptr || raw[0] == '\0') {
        return std::numeric_limits<uint32_t>::max();
    }

    char* end = nullptr;
    errno = 0;
    const unsigned long value = std::strtoul(raw, &end, 10);
    TT_FATAL(
        errno == 0 && end != raw && *end == '\0' && value > 0 &&
            value <= static_cast<unsigned long>(std::numeric_limits<uint32_t>::max()),
        "TT_WAVELET_LWT_MAX_CORES must be a positive uint32 value, got '{}'",
        raw);
    return static_cast<uint32_t>(value);
}

[[nodiscard]] uint32_t env_l1_signal_budget_bytes() {
    const char* raw = std::getenv(kL1SignalBudgetEnv);
    if (raw == nullptr || raw[0] == '\0') {
        return kDefaultL1SignalBudgetBytes;
    }

    char* end = nullptr;
    errno = 0;
    const unsigned long value = std::strtoul(raw, &end, 10);
    TT_FATAL(
        errno == 0 && end != raw && *end == '\0' && value > 0 &&
            value <= static_cast<unsigned long>(std::numeric_limits<uint32_t>::max()),
        "{} must be a positive uint32 value, got '{}'",
        kL1SignalBudgetEnv,
        raw);
    return static_cast<uint32_t>(value);
}

[[nodiscard]] std::vector<tt::tt_metal::CoreCoord> select_auto_cores(
    tt::tt_metal::distributed::MeshDevice& mesh_device, const uint32_t active_core_count);

[[nodiscard]] ResidentShardPlan make_resident_shard_plan(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const LiftingForwardPlan& plan,
    const std::vector<uint32_t>& route_group_counts) {
    const auto grid = mesh_device.compute_with_storage_grid_size();
    const uint32_t hardware_core_count = static_cast<uint32_t>(grid.x * grid.y);
    TT_FATAL(hardware_core_count > 0, "LWT path requires at least one worker core");

    const size_t initial_max_length =
        std::max(plan.preprocess_layout.output.even.length, plan.preprocess_layout.output.odd.length);
    const uint32_t initial_group_count = lwt_output_group_count(initial_max_length);
    const uint32_t max_group_count = std::max({max_route_group_count(route_group_counts), initial_group_count, 1U});
    const uint32_t core_limit = std::max(1U, std::min(hardware_core_count, env_lwt_max_cores()));
    const uint32_t groups_per_shard = static_cast<uint32_t>(ceil_div(max_group_count, core_limit));
    const uint32_t active_core_count = static_cast<uint32_t>(ceil_div(max_group_count, groups_per_shard));
    const uint32_t shard_elements = groups_per_shard * device_protocol::kLwtGroupOutputElements;
    const uint64_t signal_bytes_per_core = uint64_t{3} * shard_elements * sizeof(float);
    const uint32_t signal_budget_bytes = env_l1_signal_budget_bytes();

    TT_FATAL(
        signal_bytes_per_core <= signal_budget_bytes,
        "ResidentSharded LWT requires {} signal bytes/core for three {}-element slots, exceeding the {}-byte {}. "
        "ConeStreamed fallback is not implemented yet.",
        signal_bytes_per_core,
        shard_elements,
        signal_budget_bytes,
        kL1SignalBudgetEnv);

    return ResidentShardPlan{
        .cores = select_auto_cores(mesh_device, active_core_count),
        .max_group_count = max_group_count,
        .groups_per_shard = groups_per_shard,
        .shard_elements = shard_elements,
    };
}

[[nodiscard]] tt::tt_metal::CoreRangeSet core_range_set_from_cores(const std::vector<tt::tt_metal::CoreCoord>& cores) {
    std::vector<tt::tt_metal::CoreRange> ranges;
    ranges.reserve(cores.size());
    for (const auto& core : cores) {
        ranges.emplace_back(core);
    }
    return tt::tt_metal::CoreRangeSet(std::move(ranges)).merge_ranges();
}

[[nodiscard]] std::vector<tt::tt_metal::CoreCoord> select_auto_cores(
    tt::tt_metal::distributed::MeshDevice& mesh_device, const uint32_t active_core_count) {
    const auto grid = mesh_device.compute_with_storage_grid_size();
    const uint32_t hardware_core_count = static_cast<uint32_t>(grid.x * grid.y);
    TT_FATAL(active_core_count > 0, "LWT active core count must be non-zero");
    TT_FATAL(
        active_core_count <= hardware_core_count,
        "LWT active core count {} exceeds hardware worker core count {}",
        active_core_count,
        hardware_core_count);
    return tt::tt_metal::grid_to_cores(
        active_core_count, static_cast<uint32_t>(grid.x), static_cast<uint32_t>(grid.y), true);
}

[[nodiscard]] tt::tt_metal::CoreRangeSet build_active_core_range_set(
    tt::tt_metal::distributed::MeshDevice& mesh_device, const std::vector<tt::tt_metal::CoreCoord>& cores) {
    const auto grid = mesh_device.compute_with_storage_grid_size();
    const uint32_t hardware_core_count = static_cast<uint32_t>(grid.x * grid.y);
    if (cores.size() == hardware_core_count) {
        return tt::tt_metal::CoreRangeSet(
            tt::tt_metal::CoreRange({0, 0}, {static_cast<size_t>(grid.x - 1), static_cast<size_t>(grid.y - 1)}));
    }
    return core_range_set_from_cores(cores);
}

[[nodiscard]] std::vector<LwtCoreWork> build_core_work(
    const std::vector<tt::tt_metal::CoreCoord>& cores,
    const std::vector<uint32_t>& route_group_counts,
    const uint32_t groups_per_shard) {
    const uint32_t active_core_count = checked_length(cores.size(), "active core count");
    TT_FATAL(groups_per_shard > 0, "LWT groups per shard must be non-zero");
    std::vector<LwtCoreWork> work;
    work.reserve(cores.size());

    for (uint32_t core_index = 0; core_index < active_core_count; ++core_index) {
        LwtCoreWork core_work{.core = cores.at(core_index)};
        core_work.routes.reserve(route_group_counts.size());
        const uint32_t shard_group_begin = core_index * groups_per_shard;
        for (const uint32_t route_group_count : route_group_counts) {
            const uint32_t count = shard_group_begin < route_group_count
                                       ? std::min(groups_per_shard, route_group_count - shard_group_begin)
                                       : 0U;
            core_work.routes.push_back(RouteGroupRange{.begin = shard_group_begin, .count = count});
        }
        work.push_back(std::move(core_work));
    }

    return work;
}

[[nodiscard]] const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& resolve_mesh_buffer(
    const LiftingWorkingBuffers& buffers, const StreamRef stream) {
    return buffers.at(stream.slot);
}

[[nodiscard]] const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& resolve_output_mesh_buffer(
    const LiftingWorkingBuffers& buffers, const RouteOutputRef output) {
    switch (output.storage) {
        case RouteOutputStorage::kResidentSlot: return buffers.at(output.slot);
        case RouteOutputStorage::kFinalEvenDram: return buffers.final_even;
        case RouteOutputStorage::kFinalOddDram: return buffers.final_odd;
    }
    TT_THROW("Unsupported lifting route output storage");
}

void enqueue_program(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    tt::tt_metal::Program&& program) {
    tt::tt_metal::distributed::MeshWorkload workload;
    const auto device_range = tt::tt_metal::distributed::MeshCoordinateRange(mesh_device.shape());
    workload.add_program(device_range, std::move(program));
    tt::tt_metal::distributed::EnqueueMeshWorkload(command_queue, workload, false);
}

[[nodiscard]] std::vector<uint32_t> build_route_config_words(
    const LiftingForwardPlan& plan, const LiftingWorkingBuffers& buffers) {
    const size_t executable_route_count =
        std::count_if(plan.routes.begin(), plan.routes.end(), [](const LiftingStepRoute& route) {
            return is_executable_route(route);
        });
    std::vector<uint32_t> words(
        std::max(executable_route_count, size_t{1}) * device_protocol::kRouteConfigWordCount, 0);

    size_t executable_index = 0;
    for (const auto& route : plan.routes) {
        if (!is_executable_route(route)) {
            continue;
        }

        const RouteOutputStorage expected_output_storage =
            route.type == StepType::kScaleEven
                ? RouteOutputStorage::kFinalEvenDram
                : (route.type == StepType::kScaleOdd ? RouteOutputStorage::kFinalOddDram
                                                     : RouteOutputStorage::kResidentSlot);
        TT_FATAL(
            route.output.storage == expected_output_storage,
            "Lifting route type {} has inconsistent output storage {}",
            static_cast<uint32_t>(route.type),
            static_cast<uint32_t>(route.output.storage));

        const auto& src_buffer = resolve_mesh_buffer(buffers, route.source);
        const auto& base_buffer = resolve_mesh_buffer(buffers, route.base);
        const auto& output_buffer = resolve_output_mesh_buffer(buffers, route.output);

        const size_t off = executable_index * device_protocol::kRouteConfigWordCount;
        words[off + device_protocol::kRouteType] = static_cast<uint32_t>(route.type);
        words[off + device_protocol::kRouteSourceAddr] = static_cast<uint32_t>(src_buffer->address());
        words[off + device_protocol::kRouteSourceLength] = checked_length(route.source_length, "source length");
        words[off + device_protocol::kRouteBaseAddr] = static_cast<uint32_t>(base_buffer->address());
        words[off + device_protocol::kRouteBaseLength] = checked_length(route.base_length, "base length");
        words[off + device_protocol::kRouteOutputAddr] = static_cast<uint32_t>(output_buffer->address());
        words[off + device_protocol::kRouteOutputLength] = checked_length(route.output_length, "output length");
        words[off + device_protocol::kRouteSourceOffset] = checked_length(route.source_offset, "source offset");
        words[off + device_protocol::kRouteBaseOffset] = checked_length(route.base_offset, "base offset");
        words[off + device_protocol::kRouteSourceLeftPad] = route.source_left_pad;
        ++executable_index;
    }

    return words;
}

[[nodiscard]] std::vector<uint32_t> build_compute_runtime_args(const LwtCoreWork& work) {
    std::vector<uint32_t> args;
    args.reserve(work.routes.size() + 1);
    args.push_back(checked_length(work.routes.size(), "executable route count"));

    for (const auto& route : work.routes) {
        args.push_back(route.count);
    }

    return args;
}

[[nodiscard]] std::vector<uint32_t> build_reader_runtime_args(
    const uint32_t route_config_addr, const LwtCoreWork& work) {
    std::vector<uint32_t> args;
    args.reserve(2 + 2 * work.routes.size());
    args.push_back(route_config_addr);
    args.push_back(checked_length(work.routes.size(), "executable route count"));
    for (const auto& route : work.routes) {
        args.push_back(route.begin);
        args.push_back(route.count);
    }
    return args;
}

[[nodiscard]] std::vector<uint32_t> build_writer_runtime_args(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const uint32_t route_config_addr,
    const std::vector<tt::tt_metal::CoreCoord>& cores,
    const uint32_t core_index,
    const LwtCoreWork& work) {
    std::vector<uint32_t> args = build_reader_runtime_args(route_config_addr, work);
    args.push_back(core_index);
    args.push_back(checked_length(cores.size(), "active core count"));
    for (const auto& core : cores) {
        const auto noc_core = mesh_device.worker_core_from_logical_core(core);
        args.push_back(static_cast<uint32_t>(noc_core.x));
        args.push_back(static_cast<uint32_t>(noc_core.y));
    }
    return args;
}

[[nodiscard]] LwtProgram create_lwt_program(
    const std::filesystem::path& kernel_root,
    const tt::tt_metal::CoreRangeSet& cores,
    const LiftingForwardPlan& plan,
    const LiftingWorkingBuffers& buffers,
    const char* compute_scheme_header,
    const char* compute_scheme_type) {
    TT_FATAL(plan.preprocess_layout.output.even.stick_width == kStickWidth, "LWT path expects 32-element sticks");
    TT_FATAL(plan.preprocess_layout.output.even.element_size_bytes == sizeof(float), "LWT path is fp32-only");
    TT_FATAL(buffers.route_config != nullptr, "LWT route config buffer was not allocated");
    TT_FATAL(
        buffers.final_even != nullptr && buffers.final_odd != nullptr, "LWT final DRAM buffers were not allocated");

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    const uint32_t tile_size = tt::tile_size(kDataFormat);
    create_circular_buffer(program, cores, kLwtSrcTile0Cb, kTileGroupBuffering, tile_size);
    create_circular_buffer(program, cores, kLwtSrcTile1Cb, kTileGroupBuffering, tile_size);
    create_circular_buffer(program, cores, kLwtBaseTileCb, 2 * kTileGroupBuffering, tile_size);
    create_circular_buffer(program, cores, kLwtOutputCb, 2 * kTileGroupBuffering, tile_size);
    create_circular_buffer(
        program, cores, kLwtSrcCacheCb, device_protocol::kLwtCacheStickCount, device_protocol::kStickBytes);
    create_circular_buffer(
        program, cores, kLwtBaseCacheCb, device_protocol::kLwtCacheStickCount, device_protocol::kStickBytes);
    create_circular_buffer(program, cores, kLwtSyncCb, 1, kNocAlignmentBytes);
    create_circular_buffer(program, cores, kLwtReaderConfigCb, 1, device_protocol::kRouteConfigPageBytes);
    create_circular_buffer(program, cores, kLwtWriterConfigCb, 1, device_protocol::kRouteConfigPageBytes);

    const uint32_t route_barrier_semaphore = tt::tt_metal::CreateSemaphore(program, cores, 0);

    const auto& route_config_buffer = *(buffers.route_config->get_backing_buffer());
    const auto& row_major_buffer = *(buffers.at(StorageSlot::kA)->get_backing_buffer());
    const auto& final_dram_buffer = *(buffers.final_even->get_backing_buffer());

    std::vector<uint32_t> reader_ct_args = {
        kLwtReaderConfigCb,
        kLwtSrcTile0Cb,
        kLwtSrcTile1Cb,
        kLwtBaseTileCb,
        kLwtSrcCacheCb,
        kLwtBaseCacheCb,
        kLwtSyncCb,
    };
    tt::tt_metal::TensorAccessorArgs(route_config_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(row_major_buffer).append_to(reader_ct_args);

    std::vector<uint32_t> writer_ct_args = {
        kLwtWriterConfigCb,
        kLwtOutputCb,
        kLwtSyncCb,
        route_barrier_semaphore,
    };
    tt::tt_metal::TensorAccessorArgs(route_config_buffer).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(row_major_buffer).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(final_dram_buffer).append_to(writer_ct_args);

    const std::vector<uint32_t> compute_ct_args = {
        kLwtSrcTile0Cb,
        kLwtSrcTile1Cb,
        kLwtBaseTileCb,
        kLwtOutputCb,
    };

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_to_dest_mode[kLwtSrcTile0Cb] = UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest_mode[kLwtSrcTile1Cb] = UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest_mode[kLwtBaseTileCb] = UnpackToDestMode::UnpackToDestFp32;

    const auto reader = tt::tt_metal::CreateKernel(
        program,
        kernel_path(kernel_root, kLwtReaderKernelPath),
        cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));
    const auto writer = tt::tt_metal::CreateKernel(
        program,
        kernel_path(kernel_root, kLwtWriterKernelPath),
        cores,
        tt::tt_metal::WriterDataMovementConfig(writer_ct_args));
    const auto compute = tt::tt_metal::CreateKernel(
        program,
        kernel_path(kernel_root, kLwtComputeKernelPath),
        cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .compile_args = compute_ct_args,
            .defines =
                {
                    {"TTWV_LWT_SCHEME_HEADER", compute_scheme_header},
                    {"TTWV_LWT_SCHEME_TYPE", compute_scheme_type},
                },
        });

    return LwtProgram{
        .program = std::move(program),
        .reader = reader,
        .compute = compute,
        .writer = writer,
    };
}

void set_lwt_runtime_args(
    const LwtProgram& program_bundle,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const std::vector<tt::tt_metal::CoreCoord>& cores,
    const std::vector<LwtCoreWork>& work,
    const LiftingWorkingBuffers& buffers) {
    const uint32_t route_config_addr = static_cast<uint32_t>(buffers.route_config->address());

    for (uint32_t core_index = 0; core_index < work.size(); ++core_index) {
        const auto& core_work = work.at(core_index);
        const std::vector<uint32_t> reader_rt = build_reader_runtime_args(route_config_addr, core_work);
        const std::vector<uint32_t> compute_rt = build_compute_runtime_args(core_work);
        const std::vector<uint32_t> writer_rt =
            build_writer_runtime_args(mesh_device, route_config_addr, cores, core_index, core_work);

        tt::tt_metal::SetRuntimeArgs(program_bundle.program, program_bundle.reader, core_work.core, reader_rt);
        tt::tt_metal::SetRuntimeArgs(program_bundle.program, program_bundle.compute, core_work.core, compute_rt);
        tt::tt_metal::SetRuntimeArgs(program_bundle.program, program_bundle.writer, core_work.core, writer_rt);
    }
}

}  // namespace

LiftingWorkingBuffers create_lifting_working_buffers(
    tt::tt_metal::distributed::MeshDevice& mesh_device, const LiftingForwardPlan& provisional_plan) {
    const std::vector<uint32_t> route_group_counts = build_route_group_counts(provisional_plan);
    ResidentShardPlan shard_plan = make_resident_shard_plan(mesh_device, provisional_plan, route_group_counts);

    std::array<std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>, 3> slots;
    for (auto& slot : slots) {
        slot = create_resident_signal_mesh_buffer(mesh_device, shard_plan.cores, shard_plan.shard_elements);
    }

    SignalBuffer final_even_desc = provisional_plan.preprocess_layout.output.even;
    final_even_desc.length = provisional_plan.final_even_length;
    SignalBuffer final_odd_desc = provisional_plan.preprocess_layout.output.odd;
    final_odd_desc.length = provisional_plan.final_odd_length;

    auto final_even = create_dram_signal_mesh_buffer(mesh_device, final_even_desc);
    auto final_odd = create_dram_signal_mesh_buffer(mesh_device, final_odd_desc);
    auto route_config = create_route_config_mesh_buffer(mesh_device, route_group_counts.size());
    const uint32_t active_core_count = checked_length(shard_plan.cores.size(), "resident core count");
    std::vector<uint32_t> zero_work_cores_per_route;
    zero_work_cores_per_route.reserve(route_group_counts.size());
    for (const uint32_t group_count : route_group_counts) {
        const uint32_t working_cores = static_cast<uint32_t>(ceil_div(group_count, shard_plan.groups_per_shard));
        zero_work_cores_per_route.push_back(active_core_count - working_cores);
    }

    return LiftingWorkingBuffers{
        .slots = std::move(slots),
        .final_even = std::move(final_even),
        .final_odd = std::move(final_odd),
        .route_config = std::move(route_config),
        .cores = std::move(shard_plan.cores),
        .scheduler =
            LiftingSchedulerTelemetry{
                .max_group_count = shard_plan.max_group_count,
                .groups_per_shard = shard_plan.groups_per_shard,
                .active_core_count = active_core_count,
                .shard_elements = shard_plan.shard_elements,
                .zero_work_cores_per_route = std::move(zero_work_cores_per_route),
            },
    };
}

tt::tt_metal::Program create_lwt_device_program(
    const std::filesystem::path& kernel_root,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const LiftingForwardPlan& plan,
    const LiftingWorkingBuffers& buffers,
    const char* compute_scheme_header,
    const char* compute_scheme_type) {
    const std::vector<uint32_t> route_group_counts = build_route_group_counts(plan);
    const std::vector<tt::tt_metal::CoreCoord>& cores = buffers.cores;
    TT_FATAL(!cores.empty(), "Resident LWT working buffers have no owner cores");
    TT_FATAL(
        buffers.scheduler.groups_per_shard > 0 &&
            buffers.scheduler.shard_elements ==
                buffers.scheduler.groups_per_shard * device_protocol::kLwtGroupOutputElements,
        "Resident LWT scheduler geometry is inconsistent");

    const auto active_cores = build_active_core_range_set(mesh_device, cores);
    const std::vector<LwtCoreWork> work =
        build_core_work(cores, route_group_counts, buffers.scheduler.groups_per_shard);
    auto program_bundle =
        create_lwt_program(kernel_root, active_cores, plan, buffers, compute_scheme_header, compute_scheme_type);
    set_lwt_runtime_args(program_bundle, mesh_device, cores, work, buffers);
    return std::move(program_bundle.program);
}

void prepare_resident_lwt(
    tt::tt_metal::distributed::MeshCommandQueue& mesh_command_queue, ResidentLwtExecutable& executable) {
    const std::vector<uint32_t> route_config_words = build_route_config_words(executable.plan, executable.buffers);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(
        mesh_command_queue, executable.buffers.route_config, route_config_words, false);
    tt::tt_metal::distributed::Finish(mesh_command_queue);
}

void execute_resident_lwt(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    ResidentLwtExecutable& executable) {
    enqueue_program(mesh_device, command_queue, std::move(executable.preprocess.program));
    enqueue_program(mesh_device, command_queue, std::move(executable.lifting));
    tt::tt_metal::distributed::Finish(command_queue);
}

}  // namespace ttwv
