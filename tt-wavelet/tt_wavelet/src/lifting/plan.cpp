#include "tt_wavelet/include/lifting/plan.hpp"

#include <bit>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <tt_stl/assert.hpp>

#include "tt_wavelet/include/lifting/wavelets/wavelet_registry.hpp"

namespace ttwv {

namespace {

[[nodiscard]] constexpr StreamSlot toggle_slot(const StreamSlot slot) noexcept {
    return slot == StreamSlot::kPing ? StreamSlot::kPong : StreamSlot::kPing;
}

[[nodiscard]] constexpr StreamRef with_toggled_slot(const StreamRef stream) noexcept {
    return StreamRef{.family = stream.family, .slot = toggle_slot(stream.slot)};
}

[[nodiscard]] constexpr device_protocol::DeviceStepType to_device_step_type(const StepType type) noexcept {
    switch (type) {
        case StepType::kPredict: return device_protocol::DeviceStepType::kPredict;
        case StepType::kUpdate: return device_protocol::DeviceStepType::kUpdate;
        case StepType::kScaleEven: return device_protocol::DeviceStepType::kScaleEven;
        case StepType::kScaleOdd: return device_protocol::DeviceStepType::kScaleOdd;
        case StepType::kSwap: return device_protocol::DeviceStepType::kSwap;
    }

    return device_protocol::DeviceStepType::kPredict;
}

[[nodiscard]] StepType parse_step_type(const std::string_view raw_type) {
    if (raw_type == "predict") {
        return StepType::kPredict;
    }
    if (raw_type == "update") {
        return StepType::kUpdate;
    }
    if (raw_type == "scale-even") {
        return StepType::kScaleEven;
    }
    if (raw_type == "scale-odd") {
        return StepType::kScaleOdd;
    }
    if (raw_type == "swap") {
        return StepType::kSwap;
    }

    TT_THROW("Unsupported lifting step type: {}", raw_type);
}

[[nodiscard]] float parse_coefficient(const nlohmann::json& coeff) {
    if (coeff.is_number()) {
        return coeff.get<float>();
    }

    if (coeff.is_object()) {
        const auto numerator = coeff.at("numerator").get<double>();
        const auto denominator = coeff.at("denominator").get<double>();
        TT_FATAL(denominator != 0.0, "Coefficient denominator must be non-zero");
        return static_cast<float>(numerator / denominator);
    }

    TT_THROW("Unsupported coefficient encoding in lifting scheme JSON");
}

void validate_runtime_step(const RuntimeLiftingStep& step) {
    TT_FATAL(
        step.coefficients.size() <= device_protocol::step_coeff_capacity,
        "Step coefficient count {} exceeds device capacity {}",
        step.coefficients.size(),
        device_protocol::step_coeff_capacity);

    if (step.type == StepType::kScaleEven || step.type == StepType::kScaleOdd) {
        TT_FATAL(step.coefficients.size() == 1, "Scale steps must have exactly one coefficient");
    }

    if (step.type == StepType::kSwap) {
        TT_FATAL(step.coefficients.empty(), "Swap steps must not carry coefficients");
    }
}

void validate_runtime_scheme(const RuntimeLiftingScheme& scheme) {
    TT_FATAL(scheme.mode == BoundaryMode::kSymmetric, "Only symmetric mode is currently supported");
    TT_FATAL(scheme.tap_size > 0, "tap_size must be positive");
    TT_FATAL(!scheme.steps.empty(), "Lifting scheme must contain at least one step");

    for (const auto& step : scheme.steps) {
        validate_runtime_step(step);
    }
}

[[nodiscard]] SignalBuffer clone_signal_buffer_with_address(const SignalBuffer& buffer, const uint64_t dram_address) {
    SignalBuffer out = buffer;
    out.dram_address = dram_address;
    return out;
}

}  // namespace

RuntimeLiftingScheme load_runtime_lifting_scheme(const std::filesystem::path& path, const BoundaryMode mode) {
    std::ifstream handle(path);
    TT_FATAL(handle.good(), "Failed to open lifting scheme file: {}", path.string());

    const nlohmann::json obj = nlohmann::json::parse(handle);
    const std::string scheme_name = path.stem().string();
    const SchemeInfo* registry_info = find_scheme(scheme_name);
    TT_FATAL(registry_info != nullptr, "Scheme '{}' not found in generated wavelet registry", scheme_name);

    RuntimeLiftingScheme scheme{
        .name = scheme_name,
        .wavelet_id = registry_info != nullptr ? registry_info->id : -1,
        .mode = mode,
        .tap_size = obj.at("tap_size").get<int>(),
        .delay_even = obj.value("delay", nlohmann::json::object()).value("even", 0),
        .delay_odd = obj.value("delay", nlohmann::json::object()).value("odd", 0),
        .steps = {},
    };

    for (const auto& raw_step : obj.at("steps")) {
        RuntimeLiftingStep step{
            .type = parse_step_type(raw_step.at("type").get<std::string>()),
            .shift = raw_step.value("shift", 0),
            .coefficients = {},
        };

        for (const auto& coeff : raw_step.value("coefficients", nlohmann::json::array())) {
            step.coefficients.push_back(parse_coefficient(coeff));
        }

        scheme.steps.push_back(std::move(step));
    }

    validate_runtime_scheme(scheme);
    return scheme;
}

device_protocol::DeviceStepDesc pack_device_step_desc(const RuntimeLiftingStep& step) {
    validate_runtime_step(step);

    device_protocol::DeviceStepDesc desc{};
    desc.type = static_cast<uint32_t>(to_device_step_type(step.type));
    desc.k = static_cast<uint32_t>(step.coefficients.size());

    for (size_t i = 0; i < step.coefficients.size(); ++i) {
        desc.coeffs_packed[i] = std::bit_cast<uint32_t>(step.coefficients[i]);
    }

    return desc;
}

std::vector<device_protocol::DeviceStepDesc> pack_device_step_descs(const std::vector<RuntimeLiftingStep>& steps) {
    std::vector<device_protocol::DeviceStepDesc> packed;
    packed.reserve(steps.size());
    for (const auto& step : steps) {
        packed.push_back(pack_device_step_desc(step));
    }
    return packed;
}

LiftingForwardPlan make_forward_lifting_plan(
    const SignalBuffer& input,
    const RuntimeLiftingScheme& scheme,
    const uint64_t even_ping_addr,
    const uint64_t even_pong_addr,
    const uint64_t odd_ping_addr,
    const uint64_t odd_pong_addr) {
    validate_runtime_scheme(scheme);
    TT_FATAL(input.element_size_bytes == sizeof(float), "Forward lifting plan currently supports fp32 only");
    TT_FATAL(
        input.length <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
        "Input length {} exceeds uint32_t runtime limits",
        input.length);

    const uint32_t wavelet_pad = static_cast<uint32_t>(scheme.tap_size - 1);
    const PadSplit1DLayout preprocess_layout = make_pad_split_1d_layout(
        input,
        even_ping_addr,
        odd_ping_addr,
        Pad1DConfig{.mode = scheme.mode, .left = wavelet_pad, .right = wavelet_pad});

    const SignalBuffer even_ping = preprocess_layout.output.even;
    const SignalBuffer odd_ping = preprocess_layout.output.odd;
    const SignalBuffer even_pong = clone_signal_buffer_with_address(even_ping, even_pong_addr);
    const SignalBuffer odd_pong = clone_signal_buffer_with_address(odd_ping, odd_pong_addr);

    // ShiftedArray shift tracking: (shift, length) per logical stream.
    struct StreamState {
        int shift{0};
        size_t length{0};
    };

    auto compute_step_geometry = [](const StreamState& source,
                                    int kernel_shift,
                                    size_t k,
                                    const StreamState& base) -> std::tuple<int, size_t, size_t, size_t> {
        // ShiftedArray mapping for one predict/update step:
        //
        //   conv_shift = source.shift + kernel_shift + min(source.length, k) - 1
        //   conv_len   = max(source.length - k + 1, 0)
        //
        // The dense output written by TT is the overlap of the base stream and the
        // convolution result:
        //
        //   out_shift  = max(base.shift, conv_shift)
        //   out_end    = min(base.end, conv_shift + conv_len)
        //   out_len    = max(out_end - out_shift, 0)
        //
        // Reader-side physical indices are dense coordinates inside those logical
        // spans:
        //
        //   source_offset = out_shift - conv_shift
        //   base_offset   = out_shift - base.shift
        //
        // For physical output index p in [0, out_len), compute produces:
        //
        //   out[p] = base[base_offset + p]
        //          + sum_{j=0}^{k-1} h[j] * source[source_offset + p + k - 1 - j]
        //
        // That is exactly the valid-convolution overlap used by ShiftedArray.
        const int conv_shift = source.shift + kernel_shift + static_cast<int>(std::min(source.length, k)) - 1;
        const size_t conv_length = source.length >= k ? source.length - k + 1 : 0;

        // Overlap with base (ShiftedArray.__add__)
        const int out_shift = std::max(base.shift, conv_shift);
        const int out_end =
            std::min(base.shift + static_cast<int>(base.length), conv_shift + static_cast<int>(conv_length));
        const size_t out_length = out_end > out_shift ? static_cast<size_t>(out_end - out_shift) : 0;

        const size_t source_offset = static_cast<size_t>(out_shift - conv_shift);
        const size_t base_offset = static_cast<size_t>(out_shift - base.shift);

        return {out_shift, out_length, source_offset, base_offset};
    };

    StreamState even_state{.shift = scheme.delay_even, .length = even_ping.length};
    StreamState odd_state{.shift = scheme.delay_odd, .length = odd_ping.length};

    std::vector<LiftingStepRoute> routes;
    routes.reserve(scheme.steps.size());

    LiftingActiveStreams active{};
    for (const auto& step : scheme.steps) {
        if (step.type == StepType::kPredict) {
            const auto [out_shift, out_length, src_off, base_off] =
                compute_step_geometry(even_state, step.shift, step.coefficients.size(), odd_state);
            const StreamRef output = with_toggled_slot(active.odd);
            routes.push_back(
                LiftingStepRoute{
                    .type = step.type,
                    .source = active.even,
                    .base = active.odd,
                    .output = output,
                    .source_length = even_state.length,
                    .base_length = odd_state.length,
                    .source_offset = src_off,
                    .base_offset = base_off,
                    .output_length = out_length,
                });
            odd_state = StreamState{.shift = out_shift, .length = out_length};
            active.odd = output;
            continue;
        }

        if (step.type == StepType::kUpdate) {
            const auto [out_shift, out_length, src_off, base_off] =
                compute_step_geometry(odd_state, step.shift, step.coefficients.size(), even_state);
            const StreamRef output = with_toggled_slot(active.even);
            routes.push_back(
                LiftingStepRoute{
                    .type = step.type,
                    .source = active.odd,
                    .base = active.even,
                    .output = output,
                    .source_length = odd_state.length,
                    .base_length = even_state.length,
                    .source_offset = src_off,
                    .base_offset = base_off,
                    .output_length = out_length,
                });
            even_state = StreamState{.shift = out_shift, .length = out_length};
            active.even = output;
            continue;
        }

        if (step.type == StepType::kScaleOdd) {
            const StreamRef output = with_toggled_slot(active.odd);
            routes.push_back(
                LiftingStepRoute{
                    .type = step.type,
                    .source = active.odd,
                    .base = active.odd,
                    .output = output,
                    .source_length = odd_state.length,
                    .base_length = odd_state.length,
                    .source_offset = 0,
                    .base_offset = 0,
                    .output_length = odd_state.length,
                });
            active.odd = output;
            continue;
        }

        if (step.type == StepType::kScaleEven) {
            const StreamRef output = with_toggled_slot(active.even);
            routes.push_back(
                LiftingStepRoute{
                    .type = step.type,
                    .source = active.even,
                    .base = active.even,
                    .output = output,
                    .source_length = even_state.length,
                    .base_length = even_state.length,
                    .source_offset = 0,
                    .base_offset = 0,
                    .output_length = even_state.length,
                });
            active.even = output;
            continue;
        }

        routes.push_back(
            LiftingStepRoute{
                .type = step.type,
                .source = active.even,
                .base = active.odd,
                .output = active.even,
                .source_length = even_state.length,
                .base_length = odd_state.length,
                .source_offset = 0,
                .base_offset = 0,
                .output_length = 0,
            });
        std::swap(active.even, active.odd);
        std::swap(even_state, odd_state);
    }

    return LiftingForwardPlan{
        .scheme = scheme,
        .preprocess_layout = preprocess_layout,
        .even_buffers = SignalBufferPair{.ping = even_ping, .pong = even_pong},
        .odd_buffers = SignalBufferPair{.ping = odd_ping, .pong = odd_pong},
        .routes = std::move(routes),
        .final_active = active,
        .final_even_shift = even_state.shift,
        .final_odd_shift = odd_state.shift,
        .final_even_length = even_state.length,
        .final_odd_length = odd_state.length,
        .output_length = ceil_div(input.length + static_cast<size_t>(scheme.tap_size) - 1, size_t{2}),
    };
}

}  // namespace ttwv
