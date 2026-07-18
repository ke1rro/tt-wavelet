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
#include "tt_wavelet/include/lifting/cone_plan.hpp"
#include "tt_wavelet/include/lifting/plan.hpp"

namespace ttwv {

struct LiftingInversePlan {
    LiftingForwardPlan forward_trace{};
    size_t original_length{0};
    size_t coefficient_length{0};
};

struct InverseConeChunkPlan {
    IndexInterval output_signal{};
    IndexInterval reconstructed_even{};
    IndexInterval reconstructed_odd{};
    IndexInterval canonical_approximation{};
    IndexInterval canonical_detail{};
    std::vector<ConeStepRoute> routes;
    StreamRef final_even{};
    StreamRef final_odd{};
    size_t final_even_storage_length{0};
    size_t final_odd_storage_length{0};
    size_t final_even_offset_elements{0};
    size_t final_odd_offset_elements{0};
    size_t max_workspace_elements{0};
    double dependency_overhead{0.0};
};

struct InverseConeExecutionPlan {
    LiftingInversePlan full_plan{};
    std::vector<InverseConeChunkPlan> chunks;
    uint32_t output_groups_per_chunk{0};
    uint32_t workspace_elements{0};
    uint32_t active_core_count{0};
    double max_dependency_overhead{0.0};
    bool inverse_scale_fused{false};
    bool final_interleave_fused{false};
    ConeWorkspaceLayout workspace_layout{ConeWorkspaceLayout::kRowMajor};
};

namespace inverse_detail {

using RequiredStreams = cone_detail::RequiredStreams;

struct StoredStream {
    StorageSlot slot{StorageSlot::kA};
    IndexInterval storage{};
};

inline void validate_inverse_scale_fusion(const LiftingForwardPlan& plan) {
    TT_FATAL(plan.routes.size() >= 3, "Inverse scale fusion requires predict/update routes and two terminal scales");
    bool found_even = false;
    bool found_odd = false;
    for (size_t route_index = plan.routes.size() - 2; route_index < plan.routes.size(); ++route_index) {
        const StepType type = plan.routes[route_index].type;
        TT_FATAL(is_scale_step(type), "Inverse scale fusion requires the final two forward routes to be scales");
        found_even = found_even || type == StepType::kScaleEven;
        found_odd = found_odd || type == StepType::kScaleOdd;
    }
    TT_FATAL(found_even && found_odd, "Inverse scale fusion requires one terminal scale for each split stream");
}

[[nodiscard]] inline IndexInterval subtract_offset(
    const IndexInterval interval, const size_t offset, const char* label) {
    if (interval.empty()) {
        return interval;
    }
    TT_FATAL(interval.begin >= offset, "{} interval begins before route base offset", label);
    return IndexInterval{.begin = interval.begin - offset, .end = interval.end - offset};
}

[[nodiscard]] inline IndexInterval canonical_interval(
    const IndexInterval internal,
    const int stream_shift,
    const int canonical_start,
    const size_t coefficient_length,
    const char* label) {
    if (internal.empty()) {
        return internal;
    }
    const int64_t begin = static_cast<int64_t>(internal.begin) + static_cast<int64_t>(stream_shift) - canonical_start;
    const int64_t end = static_cast<int64_t>(internal.end) + static_cast<int64_t>(stream_shift) - canonical_start;
    TT_FATAL(
        begin >= 0 && end >= begin && static_cast<uint64_t>(end) <= coefficient_length,
        "Required {} internal interval [{}, {}) maps outside canonical coefficient range [0, {}): [{}, {})",
        label,
        internal.begin,
        internal.end,
        coefficient_length,
        begin,
        end);
    return IndexInterval{.begin = static_cast<size_t>(begin), .end = static_cast<size_t>(end)};
}

[[nodiscard]] inline std::vector<RequiredStreams> propagate_requirements(
    const LiftingForwardPlan& plan, const IndexInterval target_even, const IndexInterval target_odd) {
    std::vector<RequiredStreams> required(plan.routes.size() + 1);
    required.front() = RequiredStreams{.even = target_even, .odd = target_odd};

    size_t even_length = plan.preprocess_layout.output.even.length;
    size_t odd_length = plan.preprocess_layout.output.odd.length;
    cone_detail::validate_interval(target_even, even_length, "inverse target even");
    cone_detail::validate_interval(target_odd, odd_length, "inverse target odd");

    for (size_t route_index = 0; route_index < plan.routes.size(); ++route_index) {
        const auto& route = plan.routes[route_index];
        const RequiredStreams before = required[route_index];
        cone_detail::validate_interval(before.even, even_length, "inverse required even before route");
        cone_detail::validate_interval(before.odd, odd_length, "inverse required odd before route");

        RequiredStreams after{};
        switch (route.type) {
            case StepType::kPredict: {
                TT_FATAL(
                    even_length == route.source_length && odd_length == route.base_length,
                    "Inverse predict length state is inconsistent with the forward trace");
                const IndexInterval reconstructable{
                    .begin = route.base_offset, .end = route.base_offset + route.output_length};
                TT_FATAL(
                    cone_detail::contains(reconstructable, before.odd),
                    "Inverse predict requires discarded odd values [{}, {}) outside [{}, {})",
                    before.odd.begin,
                    before.odd.end,
                    reconstructable.begin,
                    reconstructable.end);
                const IndexInterval target = subtract_offset(before.odd, route.base_offset, "inverse predict");
                const uint32_t k = cone_detail::coefficient_count(route);
                after.even = cone_detail::hull(
                    before.even, cone_detail::translated(target, route.source_offset, static_cast<size_t>(k - 1)));
                after.odd = target;
                even_length = route.source_length;
                odd_length = route.output_length;
                break;
            }
            case StepType::kUpdate: {
                TT_FATAL(
                    odd_length == route.source_length && even_length == route.base_length,
                    "Inverse update length state is inconsistent with the forward trace");
                const IndexInterval reconstructable{
                    .begin = route.base_offset, .end = route.base_offset + route.output_length};
                TT_FATAL(
                    cone_detail::contains(reconstructable, before.even),
                    "Inverse update requires discarded even values [{}, {}) outside [{}, {})",
                    before.even.begin,
                    before.even.end,
                    reconstructable.begin,
                    reconstructable.end);
                const IndexInterval target = subtract_offset(before.even, route.base_offset, "inverse update");
                const uint32_t k = cone_detail::coefficient_count(route);
                after.even = target;
                after.odd = cone_detail::hull(
                    before.odd, cone_detail::translated(target, route.source_offset, static_cast<size_t>(k - 1)));
                even_length = route.output_length;
                odd_length = route.source_length;
                break;
            }
            case StepType::kScaleEven:
                TT_FATAL(even_length == route.source_length, "Inverse scale-even input length is inconsistent");
                after = before;
                even_length = route.output_length;
                break;
            case StepType::kScaleOdd:
                TT_FATAL(odd_length == route.source_length, "Inverse scale-odd input length is inconsistent");
                after = before;
                odd_length = route.output_length;
                break;
            case StepType::kSwap:
                TT_FATAL(
                    even_length == route.source_length && odd_length == route.base_length,
                    "Inverse swap length state is inconsistent with the forward trace");
                after = RequiredStreams{.even = before.odd, .odd = before.even};
                std::swap(even_length, odd_length);
                break;
        }

        cone_detail::validate_interval(after.even, even_length, "inverse required even after route");
        cone_detail::validate_interval(after.odd, odd_length, "inverse required odd after route");
        required[route_index + 1] = after;
    }

    TT_FATAL(
        even_length == plan.final_even_length && odd_length == plan.final_odd_length,
        "Inverse requirement propagation did not reach the final stream lengths");
    return required;
}

[[nodiscard]] inline InverseConeChunkPlan build_chunk(
    const LiftingInversePlan& inverse_plan, const IndexInterval output_signal, const bool inverse_scale_fused) {
    const auto& plan = inverse_plan.forward_trace;
    const size_t pad = plan.preprocess_layout.pad_config.left;
    TT_FATAL(
        output_signal.begin <= output_signal.end && output_signal.end <= inverse_plan.original_length,
        "Inverse output chunk [{}, {}) exceeds original signal length {}",
        output_signal.begin,
        output_signal.end,
        inverse_plan.original_length);

    const size_t padded_begin = output_signal.begin + pad;
    const size_t padded_end = output_signal.end + pad;
    const IndexInterval target_even{.begin = (padded_begin + 1) / 2, .end = (padded_end + 1) / 2};
    const IndexInterval target_odd{.begin = padded_begin / 2, .end = padded_end / 2};
    const std::vector<RequiredStreams> required = propagate_requirements(plan, target_even, target_odd);

    const int canonical_start = static_cast<int>(plan.preprocess_layout.pad_config.left + 1) / 2;
    // tap_size / 2 equals (left_pad + 1) / 2 because left_pad = tap_size - 1.
    const IndexInterval canonical_approximation = canonical_interval(
        required.back().even, plan.final_even_shift, canonical_start, inverse_plan.coefficient_length, "approximation");
    const IndexInterval canonical_detail = canonical_interval(
        required.back().odd, plan.final_odd_shift, canonical_start, inverse_plan.coefficient_length, "detail");
    TT_FATAL(
        canonical_approximation.length() == required.back().even.length() &&
            canonical_detail.length() == required.back().odd.length(),
        "Canonical inverse intervals changed required stream lengths");

    StoredStream active_even{.slot = StorageSlot::kA, .storage = required.back().even};
    StoredStream active_odd{.slot = StorageSlot::kB, .storage = required.back().odd};
    StorageSlot free_slot = StorageSlot::kScratch;
    size_t max_workspace_elements = std::max(active_even.storage.length(), active_odd.storage.length());
    std::vector<ConeStepRoute> routes;
    routes.reserve(plan.routes.size());

    if (inverse_scale_fused) {
        validate_inverse_scale_fusion(plan);
    }

    for (size_t reverse_index = plan.routes.size(); reverse_index > 0; --reverse_index) {
        const size_t route_index = reverse_index - 1;
        const auto& forward_route = plan.routes[route_index];
        const RequiredStreams& before = required[route_index];

        if (forward_route.type == StepType::kSwap) {
            std::swap(active_even, active_odd);
            continue;
        }

        // The inverse scheme starts with the reciprocal of the two terminal
        // forward scales.  When enabled, compute applies both reciprocal
        // scales to the first predict/update inputs before the stencil, so no
        // scaled stream needs a separate resident slot.
        if (inverse_scale_fused && route_index + 2 >= plan.routes.size()) {
            TT_FATAL(is_scale_step(forward_route.type), "Only terminal inverse scales may be fused");
            continue;
        }

        if (is_predict_update_step(forward_route.type)) {
            const bool predict = forward_route.type == StepType::kPredict;
            const IndexInterval output = predict ? before.odd : before.even;
            const IndexInterval target =
                subtract_offset(output, forward_route.base_offset, "inverse predict/update route");
            const uint32_t k = cone_detail::coefficient_count(forward_route);
            const IndexInterval source_required =
                cone_detail::translated(target, forward_route.source_offset, static_cast<size_t>(k - 1));
            const IndexInterval base_required = target;
            const StoredStream& source = predict ? active_even : active_odd;
            const StoredStream& base = predict ? active_odd : active_even;

            routes.push_back(
                ConeStepRoute{
                    .type = forward_route.type,
                    .source = StreamRef{.slot = source.slot},
                    .base = StreamRef{.slot = base.slot},
                    .output = detail::resident_output(free_slot),
                    .source_storage_length = source.storage.length(),
                    .base_storage_length = base.storage.length(),
                    .source_offset_elements = cone_detail::local_offset(source.storage, source_required),
                    .base_offset_elements = cone_detail::local_offset(base.storage, base_required),
                    .source_left_pad_elements = forward_route.source_left_pad,
                    .output_length = output.length(),
                    .output_offset_elements = 0,
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

        const bool scale_even = forward_route.type == StepType::kScaleEven;
        TT_FATAL(scale_even || forward_route.type == StepType::kScaleOdd, "Unsupported inverse route type");
        const IndexInterval output = scale_even ? before.even : before.odd;
        const StoredStream& source = scale_even ? active_even : active_odd;
        routes.push_back(
            ConeStepRoute{
                .type = forward_route.type,
                .source = StreamRef{.slot = source.slot},
                .base = StreamRef{.slot = source.slot},
                .output = detail::resident_output(free_slot),
                .source_storage_length = source.storage.length(),
                .base_storage_length = source.storage.length(),
                .source_offset_elements = cone_detail::local_offset(source.storage, output),
                .base_offset_elements = cone_detail::local_offset(source.storage, output),
                .source_left_pad_elements = 0,
                .output_length = output.length(),
                .output_offset_elements = 0,
            });

        const StoredStream replacement{.slot = free_slot, .storage = output};
        max_workspace_elements = std::max(max_workspace_elements, output.length());
        if (scale_even) {
            free_slot = active_even.slot;
            active_even = replacement;
        } else {
            free_slot = active_odd.slot;
            active_odd = replacement;
        }
    }

    TT_FATAL(
        cone_detail::contains(active_even.storage, target_even) &&
            cone_detail::contains(active_odd.storage, target_odd),
        "Inverse routes did not reconstruct the requested initial split intervals");
    const size_t output_elements = output_signal.length();
    const size_t dependency_elements = canonical_approximation.length() + canonical_detail.length();
    const double dependency_overhead =
        output_elements == 0
            ? 0.0
            : static_cast<double>(dependency_elements - std::min(dependency_elements, output_elements)) /
                  static_cast<double>(output_elements);

    return InverseConeChunkPlan{
        .output_signal = output_signal,
        .reconstructed_even = target_even,
        .reconstructed_odd = target_odd,
        .canonical_approximation = canonical_approximation,
        .canonical_detail = canonical_detail,
        .routes = std::move(routes),
        .final_even = StreamRef{.slot = active_even.slot},
        .final_odd = StreamRef{.slot = active_odd.slot},
        .final_even_storage_length = active_even.storage.length(),
        .final_odd_storage_length = active_odd.storage.length(),
        .final_even_offset_elements = cone_detail::local_offset(active_even.storage, target_even),
        .final_odd_offset_elements = cone_detail::local_offset(active_odd.storage, target_odd),
        .max_workspace_elements = max_workspace_elements,
        .dependency_overhead = dependency_overhead,
    };
}

[[nodiscard]] inline std::vector<InverseConeChunkPlan> build_chunks(
    const LiftingInversePlan& plan, const uint32_t requested_chunk_count, const bool inverse_scale_fused) {
    TT_FATAL(requested_chunk_count > 0, "Inverse cone chunk count must be non-zero");
    constexpr size_t output_group_elements = 2 * device_protocol::kLwtGroupOutputElements;
    const size_t output_group_count = std::max(ceil_div(plan.original_length, output_group_elements), size_t{1});
    const size_t chunk_count = std::min(static_cast<size_t>(requested_chunk_count), output_group_count);
    const size_t base_groups = output_group_count / chunk_count;
    const size_t extra_groups = output_group_count % chunk_count;

    std::vector<InverseConeChunkPlan> chunks;
    chunks.reserve(chunk_count);
    size_t group_begin = 0;
    for (size_t chunk_index = 0; chunk_index < chunk_count; ++chunk_index) {
        const size_t group_count = base_groups + (chunk_index < extra_groups ? 1 : 0);
        const size_t begin = group_begin * output_group_elements;
        const size_t end = std::min((group_begin + group_count) * output_group_elements, plan.original_length);
        chunks.push_back(build_chunk(plan, IndexInterval{.begin = begin, .end = end}, inverse_scale_fused));
        group_begin += group_count;
    }
    TT_FATAL(group_begin == output_group_count, "Inverse cone chunks do not cover every output group");
    return chunks;
}

}  // namespace inverse_detail

template <typename Scheme>
[[nodiscard]] LiftingInversePlan make_inverse_lifting_plan(
    const size_t original_length,
    const size_t coefficient_length,
    const BoundaryMode boundary_mode = BoundaryMode::kSymmetric) {
    static_assert(Scheme::tap_size > 0, "Static lifting schemes must have a positive tap size");
    TT_FATAL(original_length > 0, "Inverse lifting requires a non-empty output signal");
    const SignalBuffer original{
        .dram_address = 0,
        .length = original_length,
        .stick_width = kStickWidth,
        .element_size_bytes = sizeof(float),
    };
    LiftingForwardPlan trace = make_forward_lifting_plan<Scheme>(original, 0, 0, boundary_mode);
    TT_FATAL(
        coefficient_length == trace.output_length,
        "ILWT coefficient length {} does not match expected length {} for original length {}",
        coefficient_length,
        trace.output_length,
        original_length);
    return LiftingInversePlan{
        .forward_trace = std::move(trace),
        .original_length = original_length,
        .coefficient_length = coefficient_length,
    };
}

[[nodiscard]] inline InverseConeExecutionPlan make_inverse_cone_execution_plan(
    LiftingInversePlan full_plan,
    const uint32_t core_limit,
    const uint32_t l1_signal_budget_bytes,
    const ConeWorkspaceLayout workspace_layout = ConeWorkspaceLayout::kRowMajor,
    const bool inverse_scale_fused = true,
    const bool final_interleave_fused = true) {
    TT_FATAL(core_limit > 0, "ConeStreamed ILWT requires at least one worker core");
    TT_FATAL(l1_signal_budget_bytes >= 3 * device_protocol::kStickBytes, "ConeStreamed ILWT L1 budget is too small");

    constexpr size_t output_group_elements = 2 * device_protocol::kLwtGroupOutputElements;
    const uint32_t final_group_count =
        static_cast<uint32_t>(std::max(ceil_div(full_plan.original_length, output_group_elements), size_t{1}));
    uint32_t chunk_count = std::min(final_group_count, core_limit);
    std::vector<InverseConeChunkPlan> chunks;
    uint32_t workspace_elements = 0;

    const auto build_candidate = [&](const uint32_t candidate_chunk_count) {
        auto candidate_chunks = inverse_detail::build_chunks(full_plan, candidate_chunk_count, inverse_scale_fused);
        size_t max_workspace_elements = 0;
        for (const auto& chunk : candidate_chunks) {
            max_workspace_elements = std::max(max_workspace_elements, chunk.max_workspace_elements);
        }
        const size_t workspace_alignment = workspace_layout == ConeWorkspaceLayout::kTileNative
                                               ? static_cast<size_t>(device_protocol::kLwtGroupOutputElements)
                                               : static_cast<size_t>(kStickWidth);
        const size_t aligned_workspace = round_up(max_workspace_elements, workspace_alignment);
        TT_FATAL(
            aligned_workspace > 0 && aligned_workspace <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
            "Inverse cone workspace length {} is invalid",
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
            "One-group ConeStreamed ILWT workspace requires {} bytes/core, exceeding the {}-byte L1 budget",
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
        TT_FATAL(!chunk.routes.empty(), "ConeStreamed ILWT requires at least one inverse predict/update route");
        if (final_interleave_fused) {
            const auto& final_route = chunk.routes.back();
            TT_FATAL(
                is_predict_update_step(final_route.type),
                "Fused ILWT interleave requires the final inverse route to be predict/update");
            const bool updates_even = final_route.type == StepType::kUpdate;
            const StreamRef final_stream = updates_even ? chunk.final_even : chunk.final_odd;
            const size_t final_offset =
                updates_even ? chunk.final_even_offset_elements : chunk.final_odd_offset_elements;
            const size_t reconstructed_length =
                updates_even ? chunk.reconstructed_even.length() : chunk.reconstructed_odd.length();
            TT_FATAL(
                final_stream.slot == final_route.output.slot && final_offset == 0 &&
                    reconstructed_length == final_route.output_length,
                "Fused ILWT interleave requires the final route to produce the complete reconstructed split stream");
        }
    }

    return InverseConeExecutionPlan{
        .full_plan = std::move(full_plan),
        .chunks = std::move(chunks),
        .output_groups_per_chunk = groups_per_chunk,
        .workspace_elements = workspace_elements,
        .active_core_count = active_core_count,
        .max_dependency_overhead = max_dependency_overhead,
        .inverse_scale_fused = inverse_scale_fused,
        .final_interleave_fused = final_interleave_fused,
        .workspace_layout = workspace_layout,
    };
}

}  // namespace ttwv
