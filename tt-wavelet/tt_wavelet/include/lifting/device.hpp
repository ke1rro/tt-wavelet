#pragma once

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <tt_stl/assert.hpp>
#include <utility>

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt_wavelet/include/lifting/cone_plan.hpp"
#include "tt_wavelet/include/lifting/plan.hpp"
#include "tt_wavelet/include/lifting/static_scheme.hpp"
#include "tt_wavelet/include/pad_split/device.hpp"

namespace ttwv {

constexpr uint32_t kChunkedSfpuMaxStaticSteps = 15;

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
    const tt::tt_metal::Buffer* input_buffer{};
};

[[nodiscard]] LiftingWorkingBuffers create_lifting_working_buffers(
    tt::tt_metal::distributed::MeshDevice& mesh_device, const PadSplit1DLayout& provisional_layout, size_t route_count);

[[nodiscard]] LiftingActiveStreams lwt_static(
    const std::filesystem::path& kernel_root,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    const LiftingPreprocessDeviceProgram& bundle,
    const char* compute_scheme_header,
    const char* compute_scheme_type);

[[nodiscard]] LiftingActiveStreams lwt_chunked_sfpu_static(
    const std::filesystem::path& kernel_root,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    const LiftingPreprocessDeviceProgram& bundle,
    const ConePlan& cone_plan,
    const char* compute_scheme_header,
    const char* compute_scheme_type);

template <typename Scheme>
[[nodiscard]] LiftingPreprocessDeviceProgram create_lifting_preprocess_program(
    const std::filesystem::path& kernel_root,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const tt::tt_metal::Buffer& input_buffer,
    const SignalBuffer& input_desc) {
    TT_FATAL(input_desc.length > 0, "Input signal must be non-empty");
    TT_FATAL(input_desc.element_size_bytes == sizeof(float), "Lifting preprocess currently supports fp32 only");

    SignalBuffer planned_input = input_desc;
    planned_input.dram_address = input_buffer.address();
    const uint32_t wavelet_pad = static_cast<uint32_t>(Scheme::tap_size - 1);
    const PadSplit1DLayout provisional_layout = make_pad_split_1d_layout(
        planned_input, 0, 0, Pad1DConfig{.mode = BoundaryMode::kSymmetric, .left = wavelet_pad, .right = wavelet_pad});

    LiftingWorkingBuffers buffers =
        create_lifting_working_buffers(mesh_device, provisional_layout, executable_step_count<Scheme>());
    const LiftingForwardPlan plan = make_forward_lifting_plan<Scheme>(
        planned_input,
        buffers.even.ping->get_backing_buffer()->address(),
        buffers.odd.ping->get_backing_buffer()->address());

    PadSplit1DDeviceProgram preprocess = create_pad_split_1d_program(
        kernel_root,
        mesh_device,
        input_buffer,
        *(buffers.even.ping->get_backing_buffer()),
        *(buffers.odd.ping->get_backing_buffer()),
        plan.preprocess_layout);

    return LiftingPreprocessDeviceProgram{
        .plan = plan,
        .buffers = std::move(buffers),
        .preprocess = std::move(preprocess),
        .input_buffer = &input_buffer,
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
    const LiftingPreprocessDeviceProgram& bundle) {
    const char* backend = std::getenv("TT_WAVELET_LWT_BACKEND");
    const bool use_route_backend = backend != nullptr && std::strcmp(backend, "route") == 0;
    const bool force_chunk_backend =
        backend != nullptr && (std::strcmp(backend, "chunk") == 0 || std::strcmp(backend, "chunked") == 0);
    const bool auto_chunk_backend = backend != nullptr && std::strcmp(backend, "auto") == 0;

    if (!use_route_backend && (force_chunk_backend || auto_chunk_backend)) {
        if constexpr (Scheme::num_steps > kChunkedSfpuMaxStaticSteps) {
            if (!force_chunk_backend) {
                return lwt_static(
                    kernel_root,
                    mesh_device,
                    command_queue,
                    bundle,
                    Scheme::compute_scheme_header,
                    Scheme::compute_scheme_type);
            }
        }

        const char* raw_chunk_groups = std::getenv("TT_WAVELET_LWT_CONE_CHUNK_GROUPS");
        if (raw_chunk_groups == nullptr || raw_chunk_groups[0] == '\0') {
            raw_chunk_groups = "1";
        }

        char* end = nullptr;
        errno = 0;
        const unsigned long value = std::strtoul(raw_chunk_groups, &end, 10);
        TT_FATAL(
            errno == 0 && end != raw_chunk_groups && *end == '\0' && value > 0 &&
                value <= static_cast<unsigned long>(std::numeric_limits<uint32_t>::max()),
            "TT_WAVELET_LWT_CONE_CHUNK_GROUPS must be a positive uint32 value, got '{}'",
            raw_chunk_groups);
        const ConePlan cone_plan = make_cone_plan<Scheme>(bundle.plan, static_cast<uint32_t>(value));
        return lwt_chunked_sfpu_static(
            kernel_root,
            mesh_device,
            command_queue,
            bundle,
            cone_plan,
            Scheme::compute_scheme_header,
            Scheme::compute_scheme_type);
    }

    return lwt_static(
        kernel_root, mesh_device, command_queue, bundle, Scheme::compute_scheme_header, Scheme::compute_scheme_type);
}

}  // namespace ttwv
