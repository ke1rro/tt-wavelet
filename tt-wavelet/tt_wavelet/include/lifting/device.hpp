#pragma once

#include <filesystem>
#include <memory>

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt_wavelet/include/lifting/plan.hpp"
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

struct TileMeshBufferPair {
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
    TileMeshBufferPair even_tile{};
    TileMeshBufferPair odd_tile{};
};

struct LiftingPreprocessDeviceProgram {
    RuntimeLiftingScheme scheme{};
    LiftingForwardPlan plan{};
    LiftingWorkingBuffers buffers{};
    PadSplit1DDeviceProgram preprocess{};
};

[[nodiscard]] LiftingPreprocessDeviceProgram create_lifting_preprocess_program(
    const std::filesystem::path& kernel_root,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const tt::tt_metal::CoreCoord& core,
    const tt::tt_metal::Buffer& input_buffer,
    const SignalBuffer& input_desc,
    const RuntimeLiftingScheme& scheme);

void run_preprocess(
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    LiftingPreprocessDeviceProgram& bundle);

[[nodiscard]] LiftingActiveStreams execute_forward_lifting(
    const std::filesystem::path& kernel_root,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    const tt::tt_metal::CoreCoord& core,
    const LiftingPreprocessDeviceProgram& bundle);

}  // namespace ttwv
