#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "tt_wavelet/include/boundary.hpp"
#include "tt_wavelet/include/device_step_desc.hpp"
#include "tt_wavelet/include/pad_split.hpp"
#include "tt_wavelet/include/signal.hpp"
#include "tt_wavelet/include/lifting/lifting_step.hpp"

namespace ttwv {

struct RuntimeLiftingStep {
    StepType type{StepType::Predict};
    std::vector<float> coefficients{};
};

struct RuntimeLiftingScheme {
    BoundaryMode mode{BoundaryMode::Symmetric};
    int tap_size{0};
    std::vector<RuntimeLiftingStep> steps{};
};

enum class StreamSlot : uint8_t {
    Ping = 0,
    Pong = 1,
};

enum class LogicalStream : uint8_t {
    Even = 0,
    Odd = 1,
};

struct StreamRef {
    LogicalStream family{LogicalStream::Even};
    StreamSlot slot{StreamSlot::Ping};
};

struct SignalBufferPair {
    SignalBuffer ping{};
    SignalBuffer pong{};

    [[nodiscard]] constexpr const SignalBuffer& at(const StreamSlot slot) const noexcept {
        return slot == StreamSlot::Ping ? ping : pong;
    }
};

struct LiftingStepRoute {
    StepType type{StepType::Predict};
    StreamRef source{};
    StreamRef base{};
    StreamRef output{};
};

struct LiftingActiveStreams {
    StreamRef even{.family = LogicalStream::Even, .slot = StreamSlot::Ping};
    StreamRef odd{.family = LogicalStream::Odd, .slot = StreamSlot::Ping};
};

struct LiftingForwardPlan {
    RuntimeLiftingScheme scheme{};
    PadSplit1DLayout preprocess_layout{};
    SignalBufferPair even_buffers{};
    SignalBufferPair odd_buffers{};
    std::vector<device::DeviceStepDesc> packed_steps{};
    std::vector<LiftingStepRoute> routes{};
    LiftingActiveStreams final_active{};
    size_t output_length{0};

    [[nodiscard]] constexpr const SignalBuffer& resolve_stream_buffer(const StreamRef stream) const noexcept {
        return stream.family == LogicalStream::Even ? even_buffers.at(stream.slot) : odd_buffers.at(stream.slot);
    }

    [[nodiscard]] constexpr const SignalBuffer& even_active_buffer() const noexcept {
        return resolve_stream_buffer(final_active.even);
    }

    [[nodiscard]] constexpr const SignalBuffer& odd_active_buffer() const noexcept {
        return resolve_stream_buffer(final_active.odd);
    }
};

[[nodiscard]] RuntimeLiftingScheme load_runtime_lifting_scheme(
    const std::filesystem::path& path,
    BoundaryMode mode = BoundaryMode::Symmetric);

[[nodiscard]] device::DeviceStepDesc pack_device_step_desc(const RuntimeLiftingStep& step);

[[nodiscard]] std::vector<device::DeviceStepDesc> pack_device_step_descs(const std::vector<RuntimeLiftingStep>& steps);

[[nodiscard]] LiftingForwardPlan make_forward_lifting_plan(
    const SignalBuffer& input,
    const RuntimeLiftingScheme& scheme,
    uint64_t even_ping_addr,
    uint64_t even_pong_addr,
    uint64_t odd_ping_addr,
    uint64_t odd_pong_addr);

}  // namespace ttwv
