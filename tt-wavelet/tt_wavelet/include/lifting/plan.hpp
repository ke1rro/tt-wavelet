#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tt_stl/assert.hpp>
#include <tuple>
#include <vector>

#include "tt_wavelet/include/common/boundary.hpp"
#include "tt_wavelet/include/common/signal.hpp"
#include "tt_wavelet/include/device_protocol/lwt_config.hpp"
#include "tt_wavelet/include/lifting/static_scheme.hpp"
#include "tt_wavelet/include/lifting/step.hpp"
#include "tt_wavelet/include/pad_split/layout.hpp"

namespace ttwv {

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
    uint32_t source_left_pad{0};
    size_t output_length{0};
};

struct LiftingActiveStreams {
    StreamRef even{.family = LogicalStream::kEven, .slot = StreamSlot::kPing};
    StreamRef odd{.family = LogicalStream::kOdd, .slot = StreamSlot::kPing};
};

struct LiftingForwardPlan {
    PadSplit1DLayout preprocess_layout{};
    SignalBufferPair even_buffers{};
    SignalBufferPair odd_buffers{};
    std::vector<LiftingStepRoute> routes;
    LiftingActiveStreams final_active{};
    size_t final_even_length{0};
    size_t final_odd_length{0};
    int final_even_shift{0};
    int final_odd_shift{0};
    size_t output_length{0};

    [[nodiscard]] constexpr const SignalBuffer& resolve_stream_buffer(const StreamRef stream) const noexcept {
        return stream.family == LogicalStream::kEven ? even_buffers.at(stream.slot) : odd_buffers.at(stream.slot);
    }
};

namespace detail {

[[nodiscard]] constexpr StreamSlot toggle_slot(const StreamSlot slot) noexcept {
    return slot == StreamSlot::kPing ? StreamSlot::kPong : StreamSlot::kPing;
}

[[nodiscard]] constexpr StreamRef with_toggled_slot(const StreamRef stream) noexcept {
    return StreamRef{.family = stream.family, .slot = toggle_slot(stream.slot)};
}

[[nodiscard]] constexpr SignalBuffer clone_signal_buffer_with_address(
    const SignalBuffer& buffer, const uint64_t dram_address) noexcept {
    SignalBuffer out = buffer;
    out.dram_address = dram_address;
    return out;
}

struct StreamState {
    int shift{0};
    size_t length{0};
};

[[nodiscard]] inline std::tuple<int, size_t, size_t, size_t> compute_step_geometry(
    const StreamState& source, const int kernel_shift, const size_t k, const StreamState& base) {
    const int conv_shift = source.shift + kernel_shift + static_cast<int>(std::min(source.length, k)) - 1;
    const size_t conv_length = source.length >= k ? source.length - k + 1 : 0;

    const int out_shift = std::max(base.shift, conv_shift);
    const int out_end =
        std::min(base.shift + static_cast<int>(base.length), conv_shift + static_cast<int>(conv_length));
    const size_t out_length = out_end > out_shift ? static_cast<size_t>(out_end - out_shift) : 0;

    const size_t source_offset = static_cast<size_t>(out_shift - conv_shift);
    const size_t base_offset = static_cast<size_t>(out_shift - base.shift);

    return {out_shift, out_length, source_offset, base_offset};
}

template <typename Scheme, size_t Index>
void append_forward_route(
    std::vector<LiftingStepRoute>& routes,
    LiftingActiveStreams& active,
    StreamState& even_state,
    StreamState& odd_state) {
    using Step = SchemeStep<Scheme, Index>;

    if constexpr (Step::type == StepType::kPredict) {
        static_assert(Step::k > 0, "Predict steps must have at least one coefficient");
        static_assert(
            Step::k <= device_protocol::step_coeff_capacity, "Predict step exceeds device coefficient capacity");
        const auto [out_shift, out_length, src_off, base_off] =
            compute_step_geometry(even_state, Step::shift, Step::k, odd_state);
        const StreamRef output = with_toggled_slot(active.odd);
        routes.push_back(
            LiftingStepRoute{
                .type = Step::type,
                .source = active.even,
                .base = active.odd,
                .output = output,
                .source_length = even_state.length,
                .base_length = odd_state.length,
                .source_offset = src_off,
                .base_offset = base_off,
                .source_left_pad = device_protocol::step_coeff_capacity - Step::k,
                .output_length = out_length,
            });
        odd_state = StreamState{.shift = out_shift, .length = out_length};
        active.odd = output;
    } else if constexpr (Step::type == StepType::kUpdate) {
        static_assert(Step::k > 0, "Update steps must have at least one coefficient");
        static_assert(
            Step::k <= device_protocol::step_coeff_capacity, "Update step exceeds device coefficient capacity");
        const auto [out_shift, out_length, src_off, base_off] =
            compute_step_geometry(odd_state, Step::shift, Step::k, even_state);
        const StreamRef output = with_toggled_slot(active.even);
        routes.push_back(
            LiftingStepRoute{
                .type = Step::type,
                .source = active.odd,
                .base = active.even,
                .output = output,
                .source_length = odd_state.length,
                .base_length = even_state.length,
                .source_offset = src_off,
                .base_offset = base_off,
                .source_left_pad = device_protocol::step_coeff_capacity - Step::k,
                .output_length = out_length,
            });
        even_state = StreamState{.shift = out_shift, .length = out_length};
        active.even = output;
    } else if constexpr (Step::type == StepType::kScaleOdd) {
        static_assert(Step::k == 1, "Scale odd steps must have exactly one coefficient");
        const StreamRef output = with_toggled_slot(active.odd);
        routes.push_back(
            LiftingStepRoute{
                .type = Step::type,
                .source = active.odd,
                .base = active.odd,
                .output = output,
                .source_length = odd_state.length,
                .base_length = odd_state.length,
                .source_offset = 0,
                .base_offset = 0,
                .source_left_pad = 0,
                .output_length = odd_state.length,
            });
        active.odd = output;
    } else if constexpr (Step::type == StepType::kScaleEven) {
        static_assert(Step::k == 1, "Scale even steps must have exactly one coefficient");
        const StreamRef output = with_toggled_slot(active.even);
        routes.push_back(
            LiftingStepRoute{
                .type = Step::type,
                .source = active.even,
                .base = active.even,
                .output = output,
                .source_length = even_state.length,
                .base_length = even_state.length,
                .source_offset = 0,
                .base_offset = 0,
                .source_left_pad = 0,
                .output_length = even_state.length,
            });
        active.even = output;
    } else {
        static_assert(Step::type == StepType::kSwap, "Unsupported static lifting step type");
        static_assert(Step::k == 0, "Swap steps must not have coefficients");
        routes.push_back(
            LiftingStepRoute{
                .type = Step::type,
                .source = active.even,
                .base = active.odd,
                .output = active.even,
                .source_length = even_state.length,
                .base_length = odd_state.length,
                .source_offset = 0,
                .base_offset = 0,
                .source_left_pad = 0,
                .output_length = 0,
            });
        std::swap(active.even, active.odd);
        std::swap(even_state, odd_state);
    }
}

template <typename Scheme, size_t Index = 0>
void append_forward_routes(
    std::vector<LiftingStepRoute>& routes,
    LiftingActiveStreams& active,
    StreamState& even_state,
    StreamState& odd_state) {
    if constexpr (Index < Scheme::num_steps) {
        append_forward_route<Scheme, Index>(routes, active, even_state, odd_state);
        append_forward_routes<Scheme, Index + 1>(routes, active, even_state, odd_state);
    }
}

}  // namespace detail

template <typename Scheme>
[[nodiscard]] LiftingForwardPlan make_forward_lifting_plan(
    const SignalBuffer& input,
    uint64_t even_ping_addr,
    uint64_t even_pong_addr,
    uint64_t odd_ping_addr,
    uint64_t odd_pong_addr) {
    static_assert(Scheme::tap_size > 0, "Static lifting schemes must have a positive tap size");
    static_assert(Scheme::num_steps > 0, "Static lifting schemes must have at least one step");
    TT_FATAL(input.element_size_bytes == sizeof(float), "Forward lifting plan currently supports fp32 only");
    TT_FATAL(
        input.length <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
        "Input length {} exceeds uint32_t runtime limits",
        input.length);

    const uint32_t wavelet_pad = static_cast<uint32_t>(Scheme::tap_size - 1);
    const PadSplit1DLayout preprocess_layout = make_pad_split_1d_layout(
        input,
        even_ping_addr,
        odd_ping_addr,
        Pad1DConfig{.mode = BoundaryMode::kSymmetric, .left = wavelet_pad, .right = wavelet_pad});

    const SignalBuffer even_ping = preprocess_layout.output.even;
    const SignalBuffer odd_ping = preprocess_layout.output.odd;
    const SignalBuffer even_pong = detail::clone_signal_buffer_with_address(even_ping, even_pong_addr);
    const SignalBuffer odd_pong = detail::clone_signal_buffer_with_address(odd_ping, odd_pong_addr);

    detail::StreamState even_state{.shift = Scheme::delay_even, .length = even_ping.length};
    detail::StreamState odd_state{.shift = Scheme::delay_odd, .length = odd_ping.length};

    std::vector<LiftingStepRoute> routes;
    routes.reserve(Scheme::num_steps);

    LiftingActiveStreams active{};
    detail::append_forward_routes<Scheme>(routes, active, even_state, odd_state);

    return LiftingForwardPlan{
        .preprocess_layout = preprocess_layout,
        .even_buffers = SignalBufferPair{.ping = even_ping, .pong = even_pong},
        .odd_buffers = SignalBufferPair{.ping = odd_ping, .pong = odd_pong},
        .routes = std::move(routes),
        .final_active = active,
        .final_even_length = even_state.length,
        .final_odd_length = odd_state.length,
        .final_even_shift = even_state.shift,
        .final_odd_shift = odd_state.shift,
        .output_length = (input.length + static_cast<size_t>(Scheme::tap_size) - 1) / size_t{2},
    };
}

}  // namespace ttwv
