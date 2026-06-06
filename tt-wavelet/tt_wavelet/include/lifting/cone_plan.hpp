#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "tt_wavelet/include/common/constants.hpp"
#include "tt_wavelet/include/device_protocol/lwt_config.hpp"
#include "tt_wavelet/include/lifting/plan.hpp"
#include "tt_wavelet/include/lifting/static_scheme.hpp"
#include "tt_wavelet/include/lifting/step.hpp"

namespace ttwv {

struct ConeInterval {
    int32_t lo{1};
    int32_t hi{0};
};

struct ConeLayer {
    ConeInterval even{};
    ConeInterval odd{};
};

struct ConeStepPlan {
    StepType type{StepType::kSwap};
    LogicalStream source_family{LogicalStream::kEven};
    LogicalStream base_family{LogicalStream::kOdd};
    LogicalStream target_family{LogicalStream::kOdd};
    ConeInterval source{};
    ConeInterval base{};
    ConeInterval target{};
    ConeInterval materialized_target{};
    int32_t source_span_lo{0};
    int32_t source_span_hi{0};
    int32_t target_group_begin{0};
    uint32_t target_group_count{0};
};

struct ConeChunkPlan {
    uint32_t chunk_index{0};
    uint32_t owned_group_begin{0};
    uint32_t owned_group_count{0};
    ConeInterval final_even_owned{};
    ConeInterval final_odd_owned{};
    std::vector<ConeLayer> layers;
    std::vector<ConeStepPlan> steps;
    uint32_t max_even_sticks{0};
    uint32_t max_odd_sticks{0};
    uint32_t local_scratch_bytes{0};
    uint32_t final_even_scale_bits{0x3F800000U};
    uint32_t final_odd_scale_bits{0x3F800000U};
};

struct ConePlan {
    uint32_t chunk_group_count{1};
    uint32_t output_group_count{0};
    uint32_t output_length{0};
    std::vector<ConeChunkPlan> chunks;
};

namespace detail {

[[nodiscard]] constexpr ConeInterval empty_cone_interval() noexcept { return ConeInterval{}; }

[[nodiscard]] constexpr bool is_empty(const ConeInterval interval) noexcept { return interval.hi < interval.lo; }

[[nodiscard]] constexpr ConeInterval make_cone_interval(const int32_t lo, const uint32_t length) noexcept {
    return length == 0 ? empty_cone_interval() : ConeInterval{.lo = lo, .hi = lo + static_cast<int32_t>(length) - 1};
}

[[nodiscard]] constexpr uint32_t interval_length(const ConeInterval interval) noexcept {
    return is_empty(interval) ? 0U : static_cast<uint32_t>(interval.hi - interval.lo + 1);
}

[[nodiscard]] constexpr ConeInterval intersect(const ConeInterval lhs, const ConeInterval rhs) noexcept {
    if (is_empty(lhs) || is_empty(rhs)) {
        return empty_cone_interval();
    }
    const ConeInterval out{.lo = std::max(lhs.lo, rhs.lo), .hi = std::min(lhs.hi, rhs.hi)};
    return is_empty(out) ? empty_cone_interval() : out;
}

[[nodiscard]] constexpr ConeInterval unite(const ConeInterval lhs, const ConeInterval rhs) noexcept {
    if (is_empty(lhs)) {
        return rhs;
    }
    if (is_empty(rhs)) {
        return lhs;
    }
    return ConeInterval{.lo = std::min(lhs.lo, rhs.lo), .hi = std::max(lhs.hi, rhs.hi)};
}

[[nodiscard]] constexpr ConeInterval shift(const ConeInterval interval, const int32_t lo, const int32_t hi) noexcept {
    return is_empty(interval) ? interval : ConeInterval{.lo = interval.lo + lo, .hi = interval.hi + hi};
}

[[nodiscard]] constexpr int32_t floor_div(const int32_t value, const int32_t divisor) noexcept {
    const int32_t quotient = value / divisor;
    const int32_t remainder = value % divisor;
    return (remainder != 0 && ((remainder < 0) != (divisor < 0))) ? quotient - 1 : quotient;
}

[[nodiscard]] constexpr int32_t ceil_div_i32(const int32_t value, const int32_t divisor) noexcept {
    return -floor_div(-value, divisor);
}

[[nodiscard]] constexpr ConeInterval quantize_to_lwt_groups(const ConeInterval interval) noexcept {
    if (is_empty(interval)) {
        return interval;
    }
    constexpr int32_t group = static_cast<int32_t>(device_protocol::kLwtGroupOutputElements);
    const int32_t group_lo = floor_div(interval.lo, group);
    const int32_t group_hi = ceil_div_i32(interval.hi + 1, group) - 1;
    return ConeInterval{.lo = group_lo * group, .hi = (group_hi + 1) * group - 1};
}

[[nodiscard]] constexpr uint32_t stick_count_for_interval(const ConeInterval interval) noexcept {
    const uint32_t elements = interval_length(interval);
    return (elements + kStickWidth - 1) / kStickWidth;
}

template <typename Step>
[[nodiscard]] constexpr int32_t source_span_lo() noexcept {
    return -Step::shift - static_cast<int32_t>(Step::k) + 1;
}

template <typename Step>
[[nodiscard]] constexpr int32_t source_span_hi() noexcept {
    return -Step::shift;
}

template <typename Step>
[[nodiscard]] constexpr ConeInterval source_dependency(const ConeInterval target) noexcept {
    return shift(target, source_span_lo<Step>(), source_span_hi<Step>());
}

template <typename Scheme, size_t Index>
void fill_backward_layer(std::vector<ConeLayer>& layers) {
    if constexpr (Index > 0) {
        using Step = SchemeStep<Scheme, Index - 1>;
        const ConeLayer after = layers[Index];
        ConeLayer before = after;

        if constexpr (Step::type == StepType::kSwap) {
            before.even = after.odd;
            before.odd = after.even;
        } else if constexpr (Step::type == StepType::kPredict) {
            const ConeInterval target = quantize_to_lwt_groups(after.odd);
            before.even = unite(after.even, source_dependency<Step>(target));
            before.odd = target;
        } else if constexpr (Step::type == StepType::kUpdate) {
            const ConeInterval target = quantize_to_lwt_groups(after.even);
            before.even = target;
            before.odd = unite(after.odd, source_dependency<Step>(target));
        } else {
            static_assert(
                Step::type == StepType::kScaleEven || Step::type == StepType::kScaleOdd,
                "Unsupported lifting step type in dependency cone");
        }

        layers[Index - 1] = before;
        fill_backward_layer<Scheme, Index - 1>(layers);
    }
}

template <typename Step>
[[nodiscard]] ConeStepPlan make_predict_step_plan(const ConeLayer& before, const ConeLayer& after) {
    const ConeInterval target = after.odd;
    const ConeInterval materialized_target = quantize_to_lwt_groups(target);
    constexpr int32_t group = static_cast<int32_t>(device_protocol::kLwtGroupOutputElements);
    const int32_t group_begin = is_empty(materialized_target) ? 0 : floor_div(materialized_target.lo, group);
    return ConeStepPlan{
        .type = StepType::kPredict,
        .source_family = LogicalStream::kEven,
        .base_family = LogicalStream::kOdd,
        .target_family = LogicalStream::kOdd,
        .source = intersect(before.even, source_dependency<Step>(materialized_target)),
        .base = intersect(before.odd, materialized_target),
        .target = target,
        .materialized_target = materialized_target,
        .source_span_lo = source_span_lo<Step>(),
        .source_span_hi = source_span_hi<Step>(),
        .target_group_begin = group_begin,
        .target_group_count = static_cast<uint32_t>(
            is_empty(materialized_target)
                ? 0
                : interval_length(materialized_target) / device_protocol::kLwtGroupOutputElements),
    };
}

template <typename Step>
[[nodiscard]] ConeStepPlan make_update_step_plan(const ConeLayer& before, const ConeLayer& after) {
    const ConeInterval target = after.even;
    const ConeInterval materialized_target = quantize_to_lwt_groups(target);
    constexpr int32_t group = static_cast<int32_t>(device_protocol::kLwtGroupOutputElements);
    const int32_t group_begin = is_empty(materialized_target) ? 0 : floor_div(materialized_target.lo, group);
    return ConeStepPlan{
        .type = StepType::kUpdate,
        .source_family = LogicalStream::kOdd,
        .base_family = LogicalStream::kEven,
        .target_family = LogicalStream::kEven,
        .source = intersect(before.odd, source_dependency<Step>(materialized_target)),
        .base = intersect(before.even, materialized_target),
        .target = target,
        .materialized_target = materialized_target,
        .source_span_lo = source_span_lo<Step>(),
        .source_span_hi = source_span_hi<Step>(),
        .target_group_begin = group_begin,
        .target_group_count = static_cast<uint32_t>(
            is_empty(materialized_target)
                ? 0
                : interval_length(materialized_target) / device_protocol::kLwtGroupOutputElements),
    };
}

template <typename Scheme, size_t Index = 0>
void append_cone_steps(const std::vector<ConeLayer>& layers, std::vector<ConeStepPlan>& steps) {
    if constexpr (Index < Scheme::num_steps) {
        using Step = SchemeStep<Scheme, Index>;
        if constexpr (Step::type == StepType::kPredict) {
            steps.push_back(make_predict_step_plan<Step>(layers[Index], layers[Index + 1]));
        } else if constexpr (Step::type == StepType::kUpdate) {
            steps.push_back(make_update_step_plan<Step>(layers[Index], layers[Index + 1]));
        }
        append_cone_steps<Scheme, Index + 1>(layers, steps);
    }
}

template <typename Scheme, size_t Index = 0>
[[nodiscard]] constexpr uint32_t final_even_scale_bits(const uint32_t scale = 0x3F800000U) noexcept {
    if constexpr (Index >= Scheme::num_steps) {
        return scale;
    } else {
        using Step = SchemeStep<Scheme, Index>;
        if constexpr (Step::type == StepType::kScaleEven) {
            static_assert(Step::k == 1, "ScaleEven steps must have exactly one coefficient");
            return final_even_scale_bits<Scheme, Index + 1>(Step::coeff_bits[0]);
        } else {
            return final_even_scale_bits<Scheme, Index + 1>(scale);
        }
    }
}

template <typename Scheme, size_t Index = 0>
[[nodiscard]] constexpr uint32_t final_odd_scale_bits(const uint32_t scale = 0x3F800000U) noexcept {
    if constexpr (Index >= Scheme::num_steps) {
        return scale;
    } else {
        using Step = SchemeStep<Scheme, Index>;
        if constexpr (Step::type == StepType::kScaleOdd) {
            static_assert(Step::k == 1, "ScaleOdd steps must have exactly one coefficient");
            return final_odd_scale_bits<Scheme, Index + 1>(Step::coeff_bits[0]);
        } else {
            return final_odd_scale_bits<Scheme, Index + 1>(scale);
        }
    }
}

}  // namespace detail

template <typename Scheme>
[[nodiscard]] ConeChunkPlan make_cone_chunk_plan(
    const LiftingForwardPlan& plan,
    const uint32_t chunk_index,
    const uint32_t owned_group_begin,
    const uint32_t owned_group_count) {
    constexpr int32_t canonical_start = static_cast<int32_t>(Scheme::tap_size / 2);
    const uint32_t owned_output_begin = owned_group_begin * device_protocol::kLwtGroupOutputElements;
    TT_FATAL(plan.output_length <= std::numeric_limits<uint32_t>::max(), "Dependency-cone output length overflows u32");
    const uint32_t output_length = static_cast<uint32_t>(plan.output_length);
    const uint32_t owned_output_end =
        std::min(output_length, owned_output_begin + owned_group_count * device_protocol::kLwtGroupOutputElements);

    ConeChunkPlan chunk{
        .chunk_index = chunk_index,
        .owned_group_begin = owned_group_begin,
        .owned_group_count = owned_group_count,
        .final_even_owned = detail::intersect(
            ConeInterval{
                .lo = canonical_start + static_cast<int32_t>(owned_output_begin),
                .hi = canonical_start + static_cast<int32_t>(owned_output_end) - 1},
            detail::make_cone_interval(plan.final_even_shift, static_cast<uint32_t>(plan.final_even_length))),
        .final_odd_owned = detail::intersect(
            ConeInterval{
                .lo = canonical_start + static_cast<int32_t>(owned_output_begin),
                .hi = canonical_start + static_cast<int32_t>(owned_output_end) - 1},
            detail::make_cone_interval(plan.final_odd_shift, static_cast<uint32_t>(plan.final_odd_length))),
        .layers = std::vector<ConeLayer>(Scheme::num_steps + 1),
        .steps = {},
        .final_even_scale_bits = detail::final_even_scale_bits<Scheme>(),
        .final_odd_scale_bits = detail::final_odd_scale_bits<Scheme>(),
    };

    chunk.layers[Scheme::num_steps] = ConeLayer{
        .even = detail::quantize_to_lwt_groups(chunk.final_even_owned),
        .odd = detail::quantize_to_lwt_groups(chunk.final_odd_owned)};
    detail::fill_backward_layer<Scheme, Scheme::num_steps>(chunk.layers);
    detail::append_cone_steps<Scheme>(chunk.layers, chunk.steps);

    for (const ConeLayer& layer : chunk.layers) {
        chunk.max_even_sticks = std::max(chunk.max_even_sticks, detail::stick_count_for_interval(layer.even));
        chunk.max_odd_sticks = std::max(chunk.max_odd_sticks, detail::stick_count_for_interval(layer.odd));
    }

    constexpr uint32_t ping_pong_stream_count = 2;
    chunk.local_scratch_bytes =
        ping_pong_stream_count * (chunk.max_even_sticks + chunk.max_odd_sticks) * device_protocol::kStickBytes;
    return chunk;
}

template <typename Scheme>
[[nodiscard]] ConePlan make_cone_plan(const LiftingForwardPlan& plan, const uint32_t chunk_group_count) {
    TT_FATAL(chunk_group_count > 0, "Dependency-cone chunk_group_count must be positive");
    TT_FATAL(plan.output_length <= std::numeric_limits<uint32_t>::max(), "Dependency-cone output length overflows u32");

    const uint32_t output_length = static_cast<uint32_t>(plan.output_length);
    const uint32_t output_group_count = static_cast<uint32_t>(
        (plan.output_length + device_protocol::kLwtGroupOutputElements - 1) / device_protocol::kLwtGroupOutputElements);
    const uint32_t chunk_count = (output_group_count + chunk_group_count - 1) / chunk_group_count;

    ConePlan cone{
        .chunk_group_count = chunk_group_count,
        .output_group_count = output_group_count,
        .output_length = output_length,
    };
    cone.chunks.reserve(chunk_count);

    for (uint32_t chunk_index = 0; chunk_index < chunk_count; ++chunk_index) {
        const uint32_t group_begin = chunk_index * chunk_group_count;
        const uint32_t group_count = std::min(chunk_group_count, output_group_count - group_begin);
        cone.chunks.push_back(make_cone_chunk_plan<Scheme>(plan, chunk_index, group_begin, group_count));
    }

    return cone;
}

}  // namespace ttwv
