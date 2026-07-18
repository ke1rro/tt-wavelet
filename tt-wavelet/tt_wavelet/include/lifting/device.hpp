#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <tt_stl/assert.hpp>
#include <utility>
#include <vector>

#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/distributed.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt_wavelet/include/lifting/cone_plan.hpp"
#include "tt_wavelet/include/lifting/inverse_plan.hpp"
#include "tt_wavelet/include/lifting/plan.hpp"
#include "tt_wavelet/include/lifting/static_scheme.hpp"
#include "tt_wavelet/include/pad_split/device.hpp"

namespace ttwv {

enum class LwtMemoryMode : uint8_t {
    kResidentSharded,
    kConeStreamed,
};

struct LiftingSchedulerTelemetry {
    LwtMemoryMode memory_mode{LwtMemoryMode::kResidentSharded};
    uint32_t max_group_count{0};
    uint32_t groups_per_shard{0};
    uint32_t active_core_count{0};
    uint32_t shard_elements{0};
    std::vector<uint32_t> zero_work_cores_per_route;
    uint32_t chunk_count{0};
    uint32_t groups_per_chunk{0};
    uint32_t workspace_elements{0};
    double max_dependency_overhead{0.0};
    bool terminal_scale_fused{false};
    bool inverse_scale_fused{false};
    bool inverse_final_interleave_fused{false};
    bool tile_native_workspace{false};
};

struct LiftingWorkingBuffers {
    std::array<std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>, 3> slots{};
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> final_even{};
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> final_odd{};
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> route_config{};
    std::vector<tt::tt_metal::CoreCoord> cores;
    LiftingSchedulerTelemetry scheduler{};

    [[nodiscard]] const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& at(
        const StorageSlot slot) const noexcept {
        return slots[static_cast<size_t>(slot)];
    }
};

struct ResidentLwtExecutable {
    LiftingForwardPlan plan{};
    LiftingWorkingBuffers buffers{};
    PadSplit1DDeviceProgram preprocess{};
    tt::tt_metal::Program lifting{};
};

struct ConeWorkingBuffers {
    std::array<std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>, 3> slots{};
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> final_even{};
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> final_odd{};
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> route_config{};
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> chunk_config{};
    std::vector<tt::tt_metal::CoreCoord> cores;
    LiftingSchedulerTelemetry scheduler{};

    [[nodiscard]] const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& at(
        const StorageSlot slot) const noexcept {
        return slots[static_cast<size_t>(slot)];
    }
};

struct ConeStreamedLwtExecutable {
    ConeExecutionPlan plan{};
    ConeWorkingBuffers buffers{};
    tt::tt_metal::Program lifting{};
};

struct ConeIlwtWorkingBuffers {
    std::array<std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>, 3> slots{};
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> output{};
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> route_config{};
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> chunk_config{};
    std::vector<tt::tt_metal::CoreCoord> cores;
    LiftingSchedulerTelemetry scheduler{};

    [[nodiscard]] const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& at(
        const StorageSlot slot) const noexcept {
        return slots[static_cast<size_t>(slot)];
    }
};

struct ConeStreamedIlwtExecutable {
    InverseConeExecutionPlan plan{};
    ConeIlwtWorkingBuffers buffers{};
    tt::tt_metal::Program lifting{};
};

[[nodiscard]] LiftingWorkingBuffers create_lifting_working_buffers(
    tt::tt_metal::distributed::MeshDevice& mesh_device, const LiftingForwardPlan& provisional_plan);

[[nodiscard]] tt::tt_metal::Program create_lwt_device_program(
    const std::filesystem::path& kernel_root,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const LiftingForwardPlan& plan,
    const LiftingWorkingBuffers& buffers,
    const char* compute_scheme_header,
    const char* compute_scheme_type);

template <typename Scheme>
[[nodiscard]] ResidentLwtExecutable create_resident_lwt_executable(
    const std::filesystem::path& kernel_root,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const tt::tt_metal::Buffer& input_buffer,
    const SignalBuffer& input_desc) {
    TT_FATAL(input_desc.length > 0, "Input signal must be non-empty");
    TT_FATAL(input_desc.element_size_bytes == sizeof(float), "Lifting preprocess currently supports fp32 only");

    SignalBuffer planned_input = input_desc;
    planned_input.dram_address = input_buffer.address();
    const LiftingForwardPlan provisional_plan = make_forward_lifting_plan<Scheme>(planned_input, 0, 0);
    LiftingWorkingBuffers buffers = create_lifting_working_buffers(mesh_device, provisional_plan);
    const LiftingForwardPlan plan = make_forward_lifting_plan<Scheme>(
        planned_input,
        buffers.at(StorageSlot::kA)->get_backing_buffer()->address(),
        buffers.at(StorageSlot::kB)->get_backing_buffer()->address());

    PadSplit1DDeviceProgram preprocess = create_pad_split_1d_program(
        kernel_root,
        mesh_device,
        input_buffer,
        *(buffers.at(StorageSlot::kA)->get_backing_buffer()),
        *(buffers.at(StorageSlot::kB)->get_backing_buffer()),
        plan.preprocess_layout);
    tt::tt_metal::Program lifting = create_lwt_device_program(
        kernel_root, mesh_device, plan, buffers, Scheme::compute_scheme_header, Scheme::compute_scheme_type);

    return ResidentLwtExecutable{
        .plan = plan,
        .buffers = std::move(buffers),
        .preprocess = std::move(preprocess),
        .lifting = std::move(lifting),
    };
}

void prepare_resident_lwt(
    tt::tt_metal::distributed::MeshCommandQueue& mesh_command_queue, ResidentLwtExecutable& executable);

void execute_resident_lwt(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    ResidentLwtExecutable& executable);

[[nodiscard]] ConeStreamedLwtExecutable create_cone_streamed_lwt_executable_impl(
    const std::filesystem::path& kernel_root,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const tt::tt_metal::Buffer& input_buffer,
    LiftingForwardPlan full_plan,
    const char* compute_scheme_header,
    const char* compute_scheme_type);

template <typename Scheme>
[[nodiscard]] ConeStreamedLwtExecutable create_cone_streamed_lwt_executable(
    const std::filesystem::path& kernel_root,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const tt::tt_metal::Buffer& input_buffer,
    const SignalBuffer& input_desc) {
    TT_FATAL(input_desc.length > 0, "Input signal must be non-empty");
    TT_FATAL(input_desc.element_size_bytes == sizeof(float), "ConeStreamed LWT currently supports fp32 only");

    SignalBuffer planned_input = input_desc;
    planned_input.dram_address = input_buffer.address();
    return create_cone_streamed_lwt_executable_impl(
        kernel_root,
        mesh_device,
        input_buffer,
        make_forward_lifting_plan<Scheme>(planned_input, 0, 0),
        Scheme::compute_scheme_header,
        Scheme::compute_scheme_type);
}

void prepare_cone_streamed_lwt(
    tt::tt_metal::distributed::MeshCommandQueue& command_queue, ConeStreamedLwtExecutable& executable);

void execute_cone_streamed_lwt(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    ConeStreamedLwtExecutable& executable);

[[nodiscard]] ConeStreamedIlwtExecutable create_cone_streamed_ilwt_executable_impl(
    const std::filesystem::path& kernel_root,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const tt::tt_metal::Buffer& approximation_buffer,
    const tt::tt_metal::Buffer& detail_buffer,
    LiftingInversePlan full_plan,
    const char* inverse_compute_scheme_header,
    const char* inverse_compute_scheme_type);

template <typename Scheme>
[[nodiscard]] ConeStreamedIlwtExecutable create_cone_streamed_ilwt_executable(
    const std::filesystem::path& kernel_root,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const tt::tt_metal::Buffer& approximation_buffer,
    const tt::tt_metal::Buffer& detail_buffer,
    const size_t coefficient_length,
    const size_t original_length) {
    using InverseScheme = typename Scheme::inverse;
    return create_cone_streamed_ilwt_executable_impl(
        kernel_root,
        mesh_device,
        approximation_buffer,
        detail_buffer,
        make_inverse_lifting_plan<Scheme>(original_length, coefficient_length),
        InverseScheme::compute_scheme_header,
        InverseScheme::compute_scheme_type);
}

void prepare_cone_streamed_ilwt(
    tt::tt_metal::distributed::MeshCommandQueue& command_queue, ConeStreamedIlwtExecutable& executable);

void execute_cone_streamed_ilwt(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    ConeStreamedIlwtExecutable& executable);

}  // namespace ttwv
