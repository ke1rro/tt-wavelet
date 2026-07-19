#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "tt_wavelet/include/common/boundary.hpp"
#include "tt_wavelet/include/common/signal.hpp"
#include "tt_wavelet/include/device_protocol/step_desc.hpp"
#include "tt_wavelet/include/lifting/step.hpp"
#include "tt_wavelet/include/pad_split/layout.hpp"

namespace ttwv {

struct RuntimeLiftingStep {
    StepType type{StepType::kPredict};
    int shift{0};
    std::vector<float> coefficients;
};

struct RuntimeLiftingScheme {
    BoundaryMode mode{BoundaryMode::kSymmetric};
    int tap_size{0};
    int delay_even{0};
    int delay_odd{0};
    std::vector<RuntimeLiftingStep> steps;
};

enum class StreamSlot : uint8_t {
    kPing = 0,
    kPong = 1,
};

enum class LogicalStream : uint8_t {
    kEven = 0,
    kOdd = 1,
};

struct StreamRef {
    LogicalStream family{LogicalStream::kEven};
    StreamSlot slot{StreamSlot::kPing};
};

struct SignalBufferPair {
    SignalBuffer ping{};
    SignalBuffer pong{};

    [[nodiscard]] constexpr const SignalBuffer& at(const StreamSlot slot) const noexcept {
        return slot == StreamSlot::kPing ? ping : pong;
    }
};

struct LiftingStepRoute {
    StepType type{StepType::kPredict};
    StreamRef source{};
    StreamRef base{};
    StreamRef output{};
    size_t source_length{0};
    size_t base_length{0};
    size_t source_offset{0};
    size_t base_offset{0};
    size_t output_length{0};
};

struct LiftingActiveStreams {
    StreamRef even{.family = LogicalStream::kEven, .slot = StreamSlot::kPing};
    StreamRef odd{.family = LogicalStream::kOdd, .slot = StreamSlot::kPing};
};

struct LiftingForwardPlan {
    RuntimeLiftingScheme scheme{};
    PadSplit1DLayout preprocess_layout{};
    SignalBufferPair even_buffers{};
    SignalBufferPair odd_buffers{};
    std::vector<device_protocol::DeviceStepDesc> packed_steps;
    std::vector<LiftingStepRoute> routes;
    LiftingActiveStreams final_active{};
    int final_even_shift{0};
    int final_odd_shift{0};
    size_t final_even_length{0};
    size_t final_odd_length{0};
    size_t output_length{0};

    [[nodiscard]] constexpr const SignalBuffer& resolve_stream_buffer(const StreamRef stream) const noexcept {
        return stream.family == LogicalStream::kEven ? even_buffers.at(stream.slot) : odd_buffers.at(stream.slot);
    }

    [[nodiscard]] constexpr const SignalBuffer& even_active_buffer() const noexcept {
        return resolve_stream_buffer(final_active.even);
    }

    [[nodiscard]] constexpr const SignalBuffer& odd_active_buffer() const noexcept {
        return resolve_stream_buffer(final_active.odd);
    }
};

[[nodiscard]] RuntimeLiftingScheme load_runtime_lifting_scheme(
    const std::filesystem::path& path, BoundaryMode mode = BoundaryMode::kSymmetric);

[[nodiscard]] device_protocol::DeviceStepDesc pack_device_step_desc(const RuntimeLiftingStep& step);

[[nodiscard]] std::vector<device_protocol::DeviceStepDesc> pack_device_step_descs(
    const std::vector<RuntimeLiftingStep>& steps);

[[nodiscard]] LiftingForwardPlan make_forward_lifting_plan(
    const SignalBuffer& input,
    const RuntimeLiftingScheme& scheme,
    uint64_t even_ping_addr,
    uint64_t even_pong_addr,
    uint64_t odd_ping_addr,
    uint64_t odd_pong_addr);

}  // namespace ttwv
