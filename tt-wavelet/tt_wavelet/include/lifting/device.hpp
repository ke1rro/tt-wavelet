#pragma once

#include <filesystem>
#include <memory>
#include <tt_stl/assert.hpp>
#include <utility>

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt_wavelet/include/lifting/plan.hpp"
#include "tt_wavelet/include/lifting/static_scheme.hpp"
#include "tt_wavelet/include/pad_split/device.hpp"

namespace ttwv {

struct MeshBufferPair {
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> ping;
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> pong;

    [[nodiscard]] const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& at(
        const StreamSlot slot) const noexcept {
        return slot == StreamSlot::kPing ? ping : pong;
    }
};

struct LiftingWorkingBuffers {
    MeshBufferPair even{};
    MeshBufferPair odd{};
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> route_config{};
};

struct LiftingPreprocessDeviceProgram {
    LiftingForwardPlan plan{};
    LiftingWorkingBuffers buffers{};
    PadSplit1DDeviceProgram preprocess{};
};

[[nodiscard]] LiftingWorkingBuffers create_lifting_working_buffers(
    tt::tt_metal::distributed::MeshDevice& mesh_device, const PadSplit1DLayout& provisional_layout, size_t route_count);

[[nodiscard]] SignalBuffer with_address(const SignalBuffer& buffer, uint64_t dram_address);

[[nodiscard]] LiftingActiveStreams lwt_static(
    const std::filesystem::path& kernel_root,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    const tt::tt_metal::CoreCoord& core,
    const LiftingPreprocessDeviceProgram& bundle,
    const char* compute_kernel_path);

template <typename Scheme>
[[nodiscard]] LiftingPreprocessDeviceProgram create_lifting_preprocess_program(
    const std::filesystem::path& kernel_root,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const tt::tt_metal::CoreCoord& core,
    const tt::tt_metal::Buffer& input_buffer,
    const SignalBuffer& input_desc) {
    TT_FATAL(input_desc.length > 0, "Input signal must be non-empty");
    TT_FATAL(input_desc.element_size_bytes == sizeof(float), "Lifting preprocess currently supports fp32 only");

    const SignalBuffer planned_input = with_address(input_desc, input_buffer.address());
    const uint32_t wavelet_pad = static_cast<uint32_t>(Scheme::tap_size - 1);
    const PadSplit1DLayout provisional_layout = make_pad_split_1d_layout(
        planned_input, 0, 0, Pad1DConfig{.mode = BoundaryMode::kSymmetric, .left = wavelet_pad, .right = wavelet_pad});

    LiftingWorkingBuffers buffers =
        create_lifting_working_buffers(mesh_device, provisional_layout, executable_step_count<Scheme>());
    const LiftingForwardPlan plan = make_forward_lifting_plan<Scheme>(
        planned_input,
        buffers.even.ping->get_backing_buffer()->address(),
        buffers.even.pong->get_backing_buffer()->address(),
        buffers.odd.ping->get_backing_buffer()->address(),
        buffers.odd.pong->get_backing_buffer()->address());

    PadSplit1DDeviceProgram preprocess = create_pad_split_1d_program(
        kernel_root,
        core,
        input_buffer,
        *(buffers.even.ping->get_backing_buffer()),
        *(buffers.odd.ping->get_backing_buffer()),
        plan.preprocess_layout);

    return LiftingPreprocessDeviceProgram{
        .plan = plan,
        .buffers = std::move(buffers),
        .preprocess = std::move(preprocess),
    };
}

void run_preprocess(
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    LiftingPreprocessDeviceProgram& bundle);

template <typename Scheme>
[[nodiscard]] LiftingActiveStreams lwt(
    const std::filesystem::path& kernel_root,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    const tt::tt_metal::CoreCoord& core,
    const LiftingPreprocessDeviceProgram& bundle) {
    return lwt_static(kernel_root, mesh_device, command_queue, core, bundle, Scheme::compute_kernel_path);
}

}  // namespace ttwv
