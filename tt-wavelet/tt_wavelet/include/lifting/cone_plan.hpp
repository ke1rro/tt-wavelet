#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tt_stl/assert.hpp>
#include <utility>
#include <vector>

#include "tt_wavelet/include/common/signal.hpp"
#include "tt_wavelet/include/device_protocol/lwt_config.hpp"
#include "tt_wavelet/include/lifting/plan.hpp"

namespace ttwv {

struct IndexInterval {
    size_t begin{0};
    size_t end{0};

    [[nodiscard]] constexpr bool empty() const noexcept { return begin == end; }
    [[nodiscard]] constexpr size_t length() const noexcept { return end - begin; }
};

struct ConeDescriptor {
    int32_t even_left{0};
    int32_t even_right{0};
    int32_t odd_left{0};
    int32_t odd_right{0};
};

struct ConeStepRoute {
    StepType type{StepType::kPredict};
    StreamRef source{};
    StreamRef base{};
    RouteOutputRef output{};
    size_t source_storage_length{0};
    size_t base_storage_length{0};
    size_t source_offset_elements{0};
    size_t base_offset_elements{0};
    uint32_t source_left_pad_elements{0};
    size_t output_length{0};
    size_t output_offset_elements{0};
};

struct ConeChunkPlan {
    IndexInterval final_even{};
    IndexInterval final_odd{};
    IndexInterval initial_even{};
    IndexInterval initial_odd{};
    ConeDescriptor descriptor{};
    std::vector<ConeStepRoute> routes;
    size_t max_workspace_elements{0};
    double dependency_overhead{0.0};
};

enum class TerminalScaleFusionMode : uint8_t {
    kAuto,
    kDisabled,
    kEnabled,
};

struct ConeExecutionPlan {
    LiftingForwardPlan full_plan{};
    std::vector<ConeChunkPlan> chunks;
    uint32_t groups_per_chunk{0};
    uint32_t workspace_elements{0};
    uint32_t active_core_count{0};
    double max_dependency_overhead{0.0};
    bool terminal_scale_fused{false};
};

namespace cone_detail {

struct RequiredStreams {
    IndexInterval even{};
    IndexInterval odd{};
};

struct StoredStream {
    StorageSlot slot{StorageSlot::kA};
    IndexInterval storage{};
};

struct TerminalScaleFusion {
    size_t predict_update_route_index{0};
    StepType scale_type{StepType::kScaleEven};
    RouteOutputStorage final_storage{RouteOutputStorage::kFinalEvenDram};
};

[[nodiscard]] constexpr bool contains(const IndexInterval outer, const IndexInterval inner) noexcept {
    return inner.empty() || (outer.begin <= inner.begin && inner.end <= outer.end);
}

[[nodiscard]] constexpr IndexInterval hull(const IndexInterval lhs, const IndexInterval rhs) noexcept {
    if (lhs.empty()) {
        return rhs;
    }
    if (rhs.empty()) {
        return lhs;
    }
    return IndexInterval{.begin = std::min(lhs.begin, rhs.begin), .end = std::max(lhs.end, rhs.end)};
}

[[nodiscard]] inline IndexInterval translated(
    const IndexInterval interval, const size_t left_offset, const size_t right_expansion = 0) {
    if (interval.empty()) {
        return interval;
    }
    TT_FATAL(
        interval.begin <= std::numeric_limits<size_t>::max() - left_offset &&
            interval.end <= std::numeric_limits<size_t>::max() - left_offset - right_expansion,
        "Cone interval translation overflows size_t");
    return IndexInterval{
        .begin = interval.begin + left_offset,
        .end = interval.end + left_offset + right_expansion,
    };
}

inline void validate_interval(const IndexInterval interval, const size_t stream_length, const char* label) {
    TT_FATAL(interval.begin <= interval.end, "{} cone interval is inverted", label);
    TT_FATAL(
        interval.end <= stream_length,
        "{} cone interval [{}, {}) exceeds stream length {}",
        label,
        interval.begin,
        interval.end,
        stream_length);
}

[[nodiscard]] inline uint32_t coefficient_count(const LiftingStepRoute& route) {
    TT_FATAL(
        route.source_left_pad <= device_protocol::kStepCoeffCapacity,
        "Lifting route left pad {} exceeds coefficient capacity {}",
        route.source_left_pad,
        device_protocol::kStepCoeffCapacity);
    const uint32_t count = device_protocol::kStepCoeffCapacity - route.source_left_pad;
    TT_FATAL(count > 0, "Predict/update cone route has no coefficients");
    return count;
}

[[nodiscard]] inline TerminalScaleFusion terminal_scale_fusion(const LiftingForwardPlan& plan) {
    TT_FATAL(plan.routes.size() >= 3, "ConeStreamed terminal-scale fusion requires a predict/update and two scales");

    size_t predict_update_route_index = plan.routes.size();
    for (size_t reverse_index = plan.routes.size() - 2; reverse_index > 0; --reverse_index) {
        const size_t route_index = reverse_index - 1;
        if (is_predict_update_step(plan.routes[route_index].type)) {
            predict_update_route_index = route_index;
            break;
        }
    }
    TT_FATAL(
        predict_update_route_index < plan.routes.size(),
        "ConeStreamed terminal-scale fusion could not find a predict/update route");

    bool final_even = plan.routes[predict_update_route_index].type == StepType::kUpdate;
    for (size_t route_index = predict_update_route_index + 1; route_index + 2 < plan.routes.size(); ++route_index) {
        TT_FATAL(
            plan.routes[route_index].type == StepType::kSwap,
            "Only metadata swaps may follow the final predict/update before terminal scales");
        final_even = !final_even;
    }

    const StepType scale_type = final_even ? StepType::kScaleEven : StepType::kScaleOdd;
    const auto final_storage =
        final_even ? RouteOutputStorage::kFinalEvenDram : RouteOutputStorage::kFinalOddDram;
    const bool scale_exists =
        std::any_of(plan.routes.end() - 2, plan.routes.end(), [scale_type](const LiftingStepRoute& route) {
            return route.type == scale_type;
        });
    TT_FATAL(scale_exists, "ConeStreamed fused terminal scale is missing from the forward plan");
    return TerminalScaleFusion{
        .predict_update_route_index = predict_update_route_index,
        .scale_type = scale_type,
        .final_storage = final_storage,
    };
}

[[nodiscard]] inline std::vector<RequiredStreams> backpropagate_requirements(
    const LiftingForwardPlan& plan, const IndexInterval final_even, const IndexInterval final_odd) {
    std::vector<RequiredStreams> required(plan.routes.size() + 1);
    required.back() = RequiredStreams{.even = final_even, .odd = final_odd};

    size_t even_length = plan.final_even_length;
    size_t odd_length = plan.final_odd_length;
    validate_interval(final_even, even_length, "final even");
    validate_interval(final_odd, odd_length, "final odd");

    for (size_t reverse_index = plan.routes.size(); reverse_index > 0; --reverse_index) {
        const size_t route_index = reverse_index - 1;
        const auto& route = plan.routes[route_index];
        const RequiredStreams after = required[route_index + 1];
        validate_interval(after.even, even_length, "required even after route");
        validate_interval(after.odd, odd_length, "required odd after route");

        RequiredStreams before{};
        switch (route.type) {
            case StepType::kPredict: {
                TT_FATAL(
                    even_length == route.source_length && odd_length == route.output_length,
                    "Predict route length state is inconsistent with the forward plan");
                const uint32_t k = coefficient_count(route);
                before.even = hull(after.even, translated(after.odd, route.source_offset, static_cast<size_t>(k - 1)));
                before.odd = translated(after.odd, route.base_offset);
                even_length = route.source_length;
                odd_length = route.base_length;
                break;
            }
            case StepType::kUpdate: {
                TT_FATAL(
                    odd_length == route.source_length && even_length == route.output_length,
                    "Update route length state is inconsistent with the forward plan");
                const uint32_t k = coefficient_count(route);
                before.even = translated(after.even, route.base_offset);
                before.odd = hull(after.odd, translated(after.even, route.source_offset, static_cast<size_t>(k - 1)));
                even_length = route.base_length;
                odd_length = route.source_length;
                break;
            }
            case StepType::kScaleEven:
                TT_FATAL(even_length == route.output_length, "Scale-even output length is inconsistent");
                before.even = translated(after.even, route.source_offset);
                before.odd = after.odd;
                even_length = route.source_length;
                break;
            case StepType::kScaleOdd:
                TT_FATAL(odd_length == route.output_length, "Scale-odd output length is inconsistent");
                before.even = after.even;
                before.odd = translated(after.odd, route.source_offset);
                odd_length = route.source_length;
                break;
            case StepType::kSwap:
                TT_FATAL(
                    even_length == route.base_length && odd_length == route.source_length,
                    "Swap route length state is inconsistent with the forward plan");
                before.even = after.odd;
                before.odd = after.even;
                even_length = route.source_length;
                odd_length = route.base_length;
                break;
        }

        validate_interval(before.even, even_length, "required even before route");
        validate_interval(before.odd, odd_length, "required odd before route");
        required[route_index] = before;
    }

    TT_FATAL(
        even_length == plan.preprocess_layout.output.even.length &&
            odd_length == plan.preprocess_layout.output.odd.length,
        "Backward cone did not reach the initial split-stream lengths");
    return required;
}

[[nodiscard]] inline size_t local_offset(const IndexInterval storage, const IndexInterval required) {
    TT_FATAL(contains(storage, required), "Cone route requires data outside its resident local storage");
    return required.empty() ? 0 : required.begin - storage.begin;
}

[[nodiscard]] inline int32_t checked_descriptor_extent(const int64_t value, const char* label) {
    TT_FATAL(
        value >= std::numeric_limits<int32_t>::min() && value <= std::numeric_limits<int32_t>::max(),
        "{} cone extent {} overflows int32_t",
        label,
        value);
    return static_cast<int32_t>(value);
}

[[nodiscard]] inline ConeChunkPlan build_chunk(
    const LiftingForwardPlan& plan,
    const IndexInterval final_even,
    const IndexInterval final_odd,
    const bool fuse_terminal_scale) {
    const std::vector<RequiredStreams> required = backpropagate_requirements(plan, final_even, final_odd);
    const TerminalScaleFusion fusion = terminal_scale_fusion(plan);
    StoredStream active_even{.slot = StorageSlot::kA, .storage = required.front().even};
    StoredStream active_odd{.slot = StorageSlot::kB, .storage = required.front().odd};
    StorageSlot free_slot = StorageSlot::kScratch;
    size_t max_workspace_elements = std::max(required.front().even.length(), required.front().odd.length());

    std::vector<ConeStepRoute> routes;
    routes.reserve(plan.routes.size());
    for (size_t route_index = 0; route_index < plan.routes.size(); ++route_index) {
        const auto& full_route = plan.routes[route_index];
        const RequiredStreams& after = required[route_index + 1];

        if (full_route.type == StepType::kSwap) {
            std::swap(active_even, active_odd);
            continue;
        }

        if (full_route.type == StepType::kPredict || full_route.type == StepType::kUpdate) {
            const bool predict = full_route.type == StepType::kPredict;
            const bool fuse_route = fuse_terminal_scale && route_index == fusion.predict_update_route_index;
            const uint32_t k = coefficient_count(full_route);
            const IndexInterval output = predict ? after.odd : after.even;
            const IndexInterval source_required =
                translated(output, full_route.source_offset, static_cast<size_t>(k - 1));
            const IndexInterval base_required = translated(output, full_route.base_offset);
            const StoredStream& source = predict ? active_even : active_odd;
            const StoredStream& base = predict ? active_odd : active_even;
            const RouteOutputRef output_ref =
                fuse_route ? RouteOutputRef{.storage = fusion.final_storage, .slot = free_slot}
                           : detail::resident_output(free_slot);

            routes.push_back(ConeStepRoute{
                .type = full_route.type,
                .source = StreamRef{.slot = source.slot},
                .base = StreamRef{.slot = base.slot},
                .output = output_ref,
                .source_storage_length = source.storage.length(),
                .base_storage_length = base.storage.length(),
                .source_offset_elements = local_offset(source.storage, source_required),
                .base_offset_elements = local_offset(base.storage, base_required),
                .source_left_pad_elements = full_route.source_left_pad,
                .output_length = output.length(),
                .output_offset_elements = fuse_route && !output.empty() ? output.begin : 0,
            });

            const StoredStream replacement{.slot = free_slot, .storage = output};
            max_workspace_elements = std::max(max_workspace_elements, output.length());
            if (predict) {
                free_slot = active_odd.slot;
                active_odd = replacement;
            } else {
                free_slot = active_even.slot;
                active_even = replacement;
            }
            continue;
        }

        const bool scale_even = full_route.type == StepType::kScaleEven;
        TT_FATAL(scale_even || full_route.type == StepType::kScaleOdd, "Unsupported cone route type");
        if (fuse_terminal_scale && full_route.type == fusion.scale_type) {
            continue;
        }
        const StoredStream& source = scale_even ? active_even : active_odd;
        const IndexInterval output = scale_even ? after.even : after.odd;
        const auto final_storage = scale_even ? RouteOutputStorage::kFinalEvenDram : RouteOutputStorage::kFinalOddDram;
        routes.push_back(ConeStepRoute{
            .type = full_route.type,
            .source = StreamRef{.slot = source.slot},
            .base = StreamRef{.slot = source.slot},
            .output = RouteOutputRef{.storage = final_storage, .slot = source.slot},
            .source_storage_length = source.storage.length(),
            .base_storage_length = source.storage.length(),
            .source_offset_elements = local_offset(source.storage, output),
            .base_offset_elements = local_offset(source.storage, output),
            .source_left_pad_elements = 0,
            .output_length = output.length(),
            .output_offset_elements = output.empty() ? 0 : output.begin,
        });
    }

    const IndexInterval initial_even = required.front().even;
    const IndexInterval initial_odd = required.front().odd;
    const size_t final_begin = std::min(final_even.begin, final_odd.begin);
    const size_t final_end = std::max(final_even.end, final_odd.end);
    const size_t final_elements = final_even.length() + final_odd.length();
    const size_t dependency_elements = initial_even.length() + initial_odd.length();
    const double dependency_overhead =
        final_elements == 0 ? 0.0
                            : static_cast<double>(dependency_elements - std::min(dependency_elements, final_elements)) /
                                  static_cast<double>(final_elements);

    return ConeChunkPlan{
        .final_even = final_even,
        .final_odd = final_odd,
        .initial_even = initial_even,
        .initial_odd = initial_odd,
        .descriptor =
            ConeDescriptor{
                .even_left = checked_descriptor_extent(
                    static_cast<int64_t>(final_begin) - static_cast<int64_t>(initial_even.begin), "even-left"),
                .even_right = checked_descriptor_extent(
                    static_cast<int64_t>(initial_even.end) - static_cast<int64_t>(final_end), "even-right"),
                .odd_left = checked_descriptor_extent(
                    static_cast<int64_t>(final_begin) - static_cast<int64_t>(initial_odd.begin), "odd-left"),
                .odd_right = checked_descriptor_extent(
                    static_cast<int64_t>(initial_odd.end) - static_cast<int64_t>(final_end), "odd-right"),
            },
        .routes = std::move(routes),
        .max_workspace_elements = max_workspace_elements,
        .dependency_overhead = dependency_overhead,
    };
}

[[nodiscard]] constexpr IndexInterval clipped_chunk_interval(
    const size_t begin, const size_t end, const size_t stream_length) noexcept {
    const size_t clipped_begin = std::min(begin, stream_length);
    return IndexInterval{.begin = clipped_begin, .end = std::min(end, stream_length)};
}

[[nodiscard]] inline std::vector<ConeChunkPlan> build_chunks(
    const LiftingForwardPlan& plan,
    const uint32_t requested_chunk_count,
    const bool fuse_terminal_scale) {
    TT_FATAL(requested_chunk_count > 0, "Cone chunk count must be non-zero");
    const size_t max_final_length = std::max(plan.final_even_length, plan.final_odd_length);
    const size_t final_group_count =
        std::max(ceil_div(max_final_length, static_cast<size_t>(device_protocol::kLwtGroupOutputElements)), size_t{1});
    const size_t chunk_count = std::min(static_cast<size_t>(requested_chunk_count), final_group_count);
    const size_t base_groups = final_group_count / chunk_count;
    const size_t extra_groups = final_group_count % chunk_count;

    std::vector<ConeChunkPlan> chunks;
    chunks.reserve(chunk_count);
    size_t group_begin = 0;
    for (size_t chunk_index = 0; chunk_index < chunk_count; ++chunk_index) {
        const size_t group_count = base_groups + (chunk_index < extra_groups ? 1 : 0);
        const size_t begin = group_begin * device_protocol::kLwtGroupOutputElements;
        const size_t end =
            std::min((group_begin + group_count) * device_protocol::kLwtGroupOutputElements, max_final_length);
        chunks.push_back(build_chunk(
            plan,
            clipped_chunk_interval(begin, end, plan.final_even_length),
            clipped_chunk_interval(begin, end, plan.final_odd_length),
            fuse_terminal_scale));
        group_begin += group_count;
    }
    TT_FATAL(group_begin == final_group_count, "Cone chunks do not cover every final output group");
    return chunks;
}

}  // namespace cone_detail

[[nodiscard]] inline ConeExecutionPlan make_cone_execution_plan(
    LiftingForwardPlan full_plan,
    const uint32_t core_limit,
    const uint32_t l1_signal_budget_bytes,
    const TerminalScaleFusionMode fusion_mode = TerminalScaleFusionMode::kAuto) {
    TT_FATAL(core_limit > 0, "ConeStreamed requires at least one worker core");
    TT_FATAL(l1_signal_budget_bytes >= 3 * device_protocol::kStickBytes, "ConeStreamed L1 budget is too small");

    const size_t max_final_length = std::max(full_plan.final_even_length, full_plan.final_odd_length);
    const uint32_t final_group_count = static_cast<uint32_t>(
        std::max(ceil_div(max_final_length, static_cast<size_t>(device_protocol::kLwtGroupOutputElements)), size_t{1}));
    const bool terminal_scale_fused =
        fusion_mode == TerminalScaleFusionMode::kEnabled ||
        (fusion_mode == TerminalScaleFusionMode::kAuto && final_group_count > core_limit);
    uint32_t chunk_count = std::min(final_group_count, core_limit);
    std::vector<ConeChunkPlan> chunks;
    uint32_t workspace_elements = 0;

    const auto build_candidate = [&](const uint32_t candidate_chunk_count) {
        auto candidate_chunks =
            cone_detail::build_chunks(full_plan, candidate_chunk_count, terminal_scale_fused);
        size_t max_workspace_elements = 0;
        for (const auto& chunk : candidate_chunks) {
            max_workspace_elements = std::max(max_workspace_elements, chunk.max_workspace_elements);
        }
        const size_t aligned_workspace = round_up(max_workspace_elements, static_cast<size_t>(kStickWidth));
        TT_FATAL(
            aligned_workspace <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
            "Cone workspace length {} overflows uint32_t",
            aligned_workspace);
        return std::pair{std::move(candidate_chunks), static_cast<uint32_t>(aligned_workspace)};
    };

    for (;;) {
        auto candidate = build_candidate(chunk_count);
        chunks = std::move(candidate.first);
        workspace_elements = candidate.second;
        const uint64_t workspace_bytes_per_core = uint64_t{3} * workspace_elements * sizeof(float);
        if (workspace_bytes_per_core <= l1_signal_budget_bytes) {
            break;
        }
        TT_FATAL(
            chunk_count < final_group_count,
            "One-group ConeStreamed workspace requires {} bytes/core, exceeding the {}-byte L1 signal budget",
            workspace_bytes_per_core,
            l1_signal_budget_bytes);
        chunk_count =
            static_cast<uint32_t>(std::min(static_cast<uint64_t>(final_group_count), uint64_t{2} * chunk_count));
    }

    const uint32_t active_core_count = static_cast<uint32_t>(std::min(chunks.size(), static_cast<size_t>(core_limit)));
    const uint32_t groups_per_chunk =
        static_cast<uint32_t>(ceil_div(static_cast<size_t>(final_group_count), chunks.size()));
    double max_dependency_overhead = 0.0;
    for (const auto& chunk : chunks) {
        max_dependency_overhead = std::max(max_dependency_overhead, chunk.dependency_overhead);
    }

    return ConeExecutionPlan{
        .full_plan = std::move(full_plan),
        .chunks = std::move(chunks),
        .groups_per_chunk = groups_per_chunk,
        .workspace_elements = workspace_elements,
        .active_core_count = active_core_count,
        .max_dependency_overhead = max_dependency_overhead,
        .terminal_scale_fused = terminal_scale_fused,
    };
}

}  // namespace ttwv
