#include "tt_wavelet/include/lifting/device.hpp"

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
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

namespace ttwv {

namespace {

constexpr uint32_t kNocAlignmentBytes = 32;
constexpr uint32_t kCircularBufferAlignmentBytes = 32;
constexpr tt::DataFormat kDataFormat = tt::DataFormat::Float32;

constexpr const char* kFusedLwtReaderKernelPath = "kernels/dataflow/lwt_fused_reader.cpp";
constexpr const char* kFusedLwtWriterKernelPath = "kernels/dataflow/lwt_fused_writer.cpp";

constexpr uint32_t kLwtSrcTile0Cb = tt::CBIndex::c_0;
constexpr uint32_t kLwtSrcTile1Cb = tt::CBIndex::c_1;
constexpr uint32_t kLwtBaseTileCb = tt::CBIndex::c_2;
constexpr uint32_t kLwtOutputCb = tt::CBIndex::c_16;
constexpr uint32_t kLwtSrcCacheCb = tt::CBIndex::c_3;
constexpr uint32_t kLwtBaseCacheCb = tt::CBIndex::c_4;
constexpr uint32_t kLwtSyncCb = tt::CBIndex::c_5;
constexpr uint32_t kLwtReaderConfigCb = tt::CBIndex::c_6;
constexpr uint32_t kLwtWriterConfigCb = tt::CBIndex::c_7;
constexpr uint32_t kLwtSyncPageSize = 32;

struct FusedLwtProgram {
    tt::tt_metal::Program program;
    tt::tt_metal::KernelHandle reader{};
    tt::tt_metal::KernelHandle compute{};
    tt::tt_metal::KernelHandle writer{};
};

struct RouteGroupRange {
    uint32_t begin{0};
    uint32_t count{0};
};

struct FusedLwtCoreWork {
    tt::tt_metal::CoreCoord core;
    std::vector<RouteGroupRange> routes;
};

[[nodiscard]] std::filesystem::path kernel_path(const std::filesystem::path& kernel_root, const char* relative_path) {
    return kernel_root / relative_path;
}

[[nodiscard]] std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> create_signal_mesh_buffer(
    tt::tt_metal::distributed::MeshDevice& mesh_device, const SignalBuffer& buffer) {
    const uint32_t page_size = buffer.aligned_stick_bytes(kNocAlignmentBytes);
    const size_t physical_nbytes = std::max(buffer.physical_nbytes(kNocAlignmentBytes), static_cast<size_t>(page_size));

    const tt::tt_metal::distributed::DeviceLocalBufferConfig local_config{
        .page_size = page_size,
        .buffer_type = tt::tt_metal::BufferType::DRAM,
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
        .page_size = device_protocol::fused_route_config_page_bytes,
        .buffer_type = tt::tt_metal::BufferType::DRAM,
    };
    const tt::tt_metal::distributed::ReplicatedBufferConfig replicated_config{
        .size = static_cast<uint64_t>(page_count * device_protocol::fused_route_config_page_bytes),
    };
    return tt::tt_metal::distributed::MeshBuffer::create(replicated_config, local_config, &mesh_device);
}

void create_circular_buffer(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& cores,
    const uint32_t cb_index,
    const uint32_t entry_count,
    const uint32_t page_size,
    const tt::DataFormat data_format = kDataFormat) {
    const auto config = tt::tt_metal::CircularBufferConfig(entry_count * page_size, {{cb_index, data_format}})
                            .set_page_size(cb_index, page_size);
    static_cast<void>(tt::tt_metal::CreateCircularBuffer(program, cores, config));
}

[[nodiscard]] uint32_t checked_length(const size_t value, const char* label) {
    TT_FATAL(
        value <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()), "{} {} overflows uint32_t", label, value);
    return static_cast<uint32_t>(value);
}

[[nodiscard]] uint32_t lwt_output_group_count(const size_t output_length) {
    return checked_length(ceil_div(output_length, static_cast<size_t>(kLwtGroupOutputElements)), "output groups");
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

[[nodiscard]] uint32_t choose_auto_core_count(
    tt::tt_metal::distributed::MeshDevice& mesh_device, const std::vector<uint32_t>& route_group_counts) {
    const auto grid = mesh_device.compute_with_storage_grid_size();
    const uint32_t hardware_core_count = static_cast<uint32_t>(grid.x * grid.y);
    TT_FATAL(hardware_core_count > 0, "LWT path requires at least one worker core");

    const uint32_t useful_core_count = std::max(max_route_group_count(route_group_counts), 1U);
    return std::max(1U, std::min({useful_core_count, hardware_core_count, env_lwt_max_cores()}));
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

[[nodiscard]] RouteGroupRange split_route_groups(
    const uint32_t output_group_count, const uint32_t active_core_count, const uint32_t core_index) {
    const uint32_t base = output_group_count / active_core_count;
    const uint32_t remainder = output_group_count % active_core_count;
    const uint32_t count = base + (core_index < remainder ? 1U : 0U);
    const uint32_t begin = core_index * base + std::min(core_index, remainder);
    return RouteGroupRange{.begin = begin, .count = count};
}

[[nodiscard]] std::vector<FusedLwtCoreWork> build_core_work(
    const std::vector<tt::tt_metal::CoreCoord>& cores, const std::vector<uint32_t>& route_group_counts) {
    const uint32_t active_core_count = checked_length(cores.size(), "active core count");
    std::vector<FusedLwtCoreWork> work;
    work.reserve(cores.size());

    for (uint32_t core_index = 0; core_index < active_core_count; ++core_index) {
        FusedLwtCoreWork core_work{.core = cores.at(core_index)};
        core_work.routes.reserve(route_group_counts.size());
        for (const uint32_t route_group_count : route_group_counts) {
            core_work.routes.push_back(split_route_groups(route_group_count, active_core_count, core_index));
        }
        work.push_back(std::move(core_work));
    }

    return work;
}

[[nodiscard]] const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& resolve_mesh_buffer(
    const LiftingWorkingBuffers& buffers, const StreamRef stream) {
    return stream.family == LogicalStream::kEven ? buffers.even.at(stream.slot) : buffers.odd.at(stream.slot);
}

[[nodiscard]] const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& resolve_output_mesh_buffer(
    const LiftingWorkingBuffers& buffers, const LiftingStepRoute& route) {
    return resolve_mesh_buffer(buffers, route.output);
}

void run_program(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    tt::tt_metal::Program&& program) {
    tt::tt_metal::distributed::MeshWorkload workload;
    const auto device_range = tt::tt_metal::distributed::MeshCoordinateRange(mesh_device.shape());
    workload.add_program(device_range, std::move(program));
    tt::tt_metal::distributed::EnqueueMeshWorkload(command_queue, workload, false);
    tt::tt_metal::distributed::Finish(command_queue);
}

[[nodiscard]] std::vector<uint32_t> build_route_config_words(const LiftingPreprocessDeviceProgram& bundle) {
    const size_t executable_route_count =
        std::count_if(bundle.plan.routes.begin(), bundle.plan.routes.end(), [](const LiftingStepRoute& route) {
            return is_executable_route(route);
        });
    std::vector<uint32_t> words(
        std::max(executable_route_count, size_t{1}) * device_protocol::fused_route_config_word_count, 0);

    size_t executable_index = 0;
    for (const auto& route : bundle.plan.routes) {
        if (!is_executable_route(route)) {
            continue;
        }

        const auto& src_buffer = resolve_mesh_buffer(bundle.buffers, route.source);
        const auto& base_buffer = resolve_mesh_buffer(bundle.buffers, route.base);
        const auto& output_buffer = resolve_output_mesh_buffer(bundle.buffers, route);

        const size_t off = executable_index * device_protocol::fused_route_config_word_count;
        words[off + device_protocol::kRouteType] = static_cast<uint32_t>(route.type);
        words[off + device_protocol::kRouteSourceAddr] = static_cast<uint32_t>(src_buffer->address());
        words[off + device_protocol::kRouteSourceLength] = checked_length(route.source_length, "source length");
        words[off + device_protocol::kRouteBaseAddr] = static_cast<uint32_t>(base_buffer->address());
        words[off + device_protocol::kRouteBaseLength] = checked_length(route.base_length, "base length");
        words[off + device_protocol::kRouteOutputAddr] = static_cast<uint32_t>(output_buffer->address());
        words[off + device_protocol::kRouteOutputLength] = checked_length(route.output_length, "output length");
        words[off + device_protocol::kRouteOutputGroupCount] = lwt_output_group_count(route.output_length);
        words[off + device_protocol::kRouteSourceOffset] = checked_length(route.source_offset, "source offset");
        words[off + device_protocol::kRouteBaseOffset] = checked_length(route.base_offset, "base offset");
        words[off + device_protocol::kRouteSourceLeftPad] = route.source_left_pad;
        ++executable_index;
    }

    return words;
}

[[nodiscard]] std::vector<uint32_t> build_compute_runtime_args(const FusedLwtCoreWork& work) {
    std::vector<uint32_t> args;
    args.reserve(work.routes.size() + 1);
    args.push_back(checked_length(work.routes.size(), "executable route count"));

    for (const auto& route : work.routes) {
        args.push_back(route.count);
    }

    return args;
}

[[nodiscard]] std::vector<uint32_t> build_reader_runtime_args(
    const uint32_t route_config_addr, const FusedLwtCoreWork& work) {
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
    const FusedLwtCoreWork& work) {
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

[[nodiscard]] FusedLwtProgram create_fused_lwt_program(
    const std::filesystem::path& kernel_root,
    const tt::tt_metal::CoreRangeSet& cores,
    const LiftingPreprocessDeviceProgram& bundle,
    const char* compute_kernel_path) {
    TT_FATAL(bundle.plan.preprocess_layout.output.even.stick_width == 32, "LWT path expects 32-element sticks");
    TT_FATAL(bundle.plan.preprocess_layout.output.even.element_size_bytes == sizeof(float), "LWT path is fp32-only");
    TT_FATAL(bundle.buffers.route_config != nullptr, "Fused LWT route config buffer was not allocated");

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    const uint32_t tile_size = tt::tile_size(kDataFormat);
    const uint32_t stick_nbytes =
        bundle.plan.preprocess_layout.output.even.aligned_stick_bytes(kCircularBufferAlignmentBytes);
    const uint32_t stick_width = bundle.plan.preprocess_layout.output.even.stick_width;

    create_circular_buffer(program, cores, kLwtSrcTile0Cb, 1, tile_size);
    create_circular_buffer(program, cores, kLwtSrcTile1Cb, 1, tile_size);
    create_circular_buffer(program, cores, kLwtBaseTileCb, 2, tile_size);
    create_circular_buffer(program, cores, kLwtOutputCb, 2, tile_size);
    create_circular_buffer(program, cores, kLwtSrcCacheCb, 1, stick_nbytes);
    create_circular_buffer(program, cores, kLwtBaseCacheCb, 1, stick_nbytes);
    create_circular_buffer(program, cores, kLwtSyncCb, 1, kLwtSyncPageSize);
    create_circular_buffer(
        program, cores, kLwtReaderConfigCb, 1, device_protocol::fused_route_config_page_bytes, kDataFormat);
    create_circular_buffer(
        program, cores, kLwtWriterConfigCb, 1, device_protocol::fused_route_config_page_bytes, kDataFormat);

    const uint32_t route_barrier_semaphore = tt::tt_metal::CreateSemaphore(program, cores, 0);

    const auto& route_config_buffer = *(bundle.buffers.route_config->get_backing_buffer());
    const auto& row_major_buffer = *(bundle.buffers.even.ping->get_backing_buffer());

    std::vector<uint32_t> reader_ct_args = {
        kLwtReaderConfigCb,
        device_protocol::fused_route_config_page_bytes,
        kLwtSrcTile0Cb,
        kLwtSrcTile1Cb,
        kLwtBaseTileCb,
        stick_nbytes,
        kLwtSrcCacheCb,
        kLwtBaseCacheCb,
        stick_width,
        kLwtSyncCb,
    };
    tt::tt_metal::TensorAccessorArgs(route_config_buffer).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(row_major_buffer).append_to(reader_ct_args);

    std::vector<uint32_t> writer_ct_args = {
        kLwtWriterConfigCb,
        device_protocol::fused_route_config_page_bytes,
        kLwtOutputCb,
        stick_nbytes,
        kLwtSyncCb,
        route_barrier_semaphore,
    };
    tt::tt_metal::TensorAccessorArgs(route_config_buffer).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(row_major_buffer).append_to(writer_ct_args);

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
        kernel_path(kernel_root, kFusedLwtReaderKernelPath),
        cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));
    const auto writer = tt::tt_metal::CreateKernel(
        program,
        kernel_path(kernel_root, kFusedLwtWriterKernelPath),
        cores,
        tt::tt_metal::WriterDataMovementConfig(writer_ct_args));
    const auto compute = tt::tt_metal::CreateKernel(
        program,
        kernel_path(kernel_root, compute_kernel_path),
        cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .compile_args = compute_ct_args,
        });

    return FusedLwtProgram{
        .program = std::move(program),
        .reader = reader,
        .compute = compute,
        .writer = writer,
    };
}

void set_fused_lwt_runtime_args(
    const FusedLwtProgram& program_bundle,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const std::vector<tt::tt_metal::CoreCoord>& cores,
    const std::vector<FusedLwtCoreWork>& work,
    const LiftingPreprocessDeviceProgram& bundle) {
    const uint32_t route_config_addr = static_cast<uint32_t>(bundle.buffers.route_config->address());

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

[[nodiscard]] bool debug_step_readback_enabled() {
    const char* raw = std::getenv("TT_WAVELET_DEBUG_STEP_READBACK");
    return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
}

void debug_print_signal(const char* label, const std::vector<float>& values, const size_t logical_length) {
    std::cout << label << " (" << logical_length << "): [";
    for (size_t i = 0; i < logical_length; ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << std::scientific << std::setprecision(8) << static_cast<double>(values[i]);
    }
    std::cout << std::defaultfloat << "]\n";
}

}  // namespace

SignalBuffer with_address(const SignalBuffer& buffer, const uint64_t dram_address) {
    SignalBuffer out = buffer;
    out.dram_address = dram_address;
    return out;
}

LiftingWorkingBuffers create_lifting_working_buffers(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const PadSplit1DLayout& provisional_layout,
    const size_t route_count) {
    auto even_ping = create_signal_mesh_buffer(mesh_device, provisional_layout.output.even);
    auto even_pong = create_signal_mesh_buffer(mesh_device, provisional_layout.output.even);
    auto odd_ping = create_signal_mesh_buffer(mesh_device, provisional_layout.output.odd);
    auto odd_pong = create_signal_mesh_buffer(mesh_device, provisional_layout.output.odd);
    auto route_config = create_route_config_mesh_buffer(mesh_device, route_count);

    return LiftingWorkingBuffers{
        .even = MeshBufferPair{.ping = std::move(even_ping), .pong = std::move(even_pong)},
        .odd = MeshBufferPair{.ping = std::move(odd_ping), .pong = std::move(odd_pong)},
        .route_config = std::move(route_config),
    };
}

void run_preprocess(
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    LiftingPreprocessDeviceProgram& bundle) {
    run_program(mesh_device, command_queue, std::move(bundle.preprocess.program));

    if (debug_step_readback_enabled()) {
        std::vector<float> even_output;
        auto even_buffer = bundle.buffers.even.ping;
        tt::tt_metal::distributed::EnqueueReadMeshBuffer(command_queue, even_output, even_buffer, true);
        debug_print_signal("preprocess even", even_output, bundle.plan.preprocess_layout.output.even.length);

        std::vector<float> odd_output;
        auto odd_buffer = bundle.buffers.odd.ping;
        tt::tt_metal::distributed::EnqueueReadMeshBuffer(command_queue, odd_output, odd_buffer, true);
        debug_print_signal("preprocess odd", odd_output, bundle.plan.preprocess_layout.output.odd.length);
    }
}

LiftingActiveStreams lwt_static(
    const std::filesystem::path& kernel_root,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    const tt::tt_metal::CoreCoord& core,
    const LiftingPreprocessDeviceProgram& bundle,
    const char* compute_kernel_path) {
    const std::vector<uint32_t> route_config_words = build_route_config_words(bundle);
    auto route_config_buffer = bundle.buffers.route_config;
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(command_queue, route_config_buffer, route_config_words, false);

    const std::vector<uint32_t> route_group_counts = build_route_group_counts(bundle.plan);
    const std::vector<tt::tt_metal::CoreCoord> cores = {core};
    const auto active_cores = core_range_set_from_cores(cores);
    const std::vector<FusedLwtCoreWork> work = build_core_work(cores, route_group_counts);

    auto program_bundle = create_fused_lwt_program(kernel_root, active_cores, bundle, compute_kernel_path);
    set_fused_lwt_runtime_args(program_bundle, mesh_device, cores, work, bundle);
    run_program(mesh_device, command_queue, std::move(program_bundle.program));

    return bundle.plan.final_active;
}

LiftingActiveStreams lwt_static(
    const std::filesystem::path& kernel_root,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    const LiftingPreprocessDeviceProgram& bundle,
    const char* compute_kernel_path) {
    const std::vector<uint32_t> route_config_words = build_route_config_words(bundle);
    auto route_config_buffer = bundle.buffers.route_config;
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(command_queue, route_config_buffer, route_config_words, false);

    const std::vector<uint32_t> route_group_counts = build_route_group_counts(bundle.plan);
    const uint32_t active_core_count = choose_auto_core_count(mesh_device, route_group_counts);
    const std::vector<tt::tt_metal::CoreCoord> cores = select_auto_cores(mesh_device, active_core_count);
    const auto active_cores = build_active_core_range_set(mesh_device, cores);
    const std::vector<FusedLwtCoreWork> work = build_core_work(cores, route_group_counts);

    auto program_bundle = create_fused_lwt_program(kernel_root, active_cores, bundle, compute_kernel_path);
    set_fused_lwt_runtime_args(program_bundle, mesh_device, cores, work, bundle);
    run_program(mesh_device, command_queue, std::move(program_bundle.program));

    return bundle.plan.final_active;
}

}  // namespace ttwv
