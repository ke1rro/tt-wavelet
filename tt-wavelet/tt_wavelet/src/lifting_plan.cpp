#include "tt_wavelet/include/lifting/lifting_plan.hpp"

#include <bit>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <string_view>

#include <tt_stl/assert.hpp>

namespace ttwv {

namespace {

[[nodiscard]] constexpr StreamSlot toggle_slot(const StreamSlot slot) noexcept {
    return slot == StreamSlot::Ping ? StreamSlot::Pong : StreamSlot::Ping;
}

[[nodiscard]] constexpr StreamRef with_toggled_slot(const StreamRef stream) noexcept {
    return StreamRef{.family = stream.family, .slot = toggle_slot(stream.slot)};
}

[[nodiscard]] constexpr device::DeviceStepType to_device_step_type(const StepType type) noexcept {
    switch (type) {
        case StepType::Predict: return device::DeviceStepType::Predict;
        case StepType::Update: return device::DeviceStepType::Update;
        case StepType::ScaleEven: return device::DeviceStepType::ScaleEven;
        case StepType::ScaleOdd: return device::DeviceStepType::ScaleOdd;
        case StepType::Swap: return device::DeviceStepType::Swap;
    }

    return device::DeviceStepType::Predict;
}

[[nodiscard]] StepType parse_step_type(const std::string_view raw_type) {
    if (raw_type == "predict") {
        return StepType::Predict;
    }
    if (raw_type == "update") {
        return StepType::Update;
    }
    if (raw_type == "scale-even") {
        return StepType::ScaleEven;
    }
    if (raw_type == "scale-odd") {
        return StepType::ScaleOdd;
    }
    if (raw_type == "swap") {
        return StepType::Swap;
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
        step.coefficients.size() <= device::step_coeff_capacity,
        "Step coefficient count {} exceeds device capacity {}",
        step.coefficients.size(),
        device::step_coeff_capacity);

    if (step.type == StepType::ScaleEven || step.type == StepType::ScaleOdd) {
        TT_FATAL(step.coefficients.size() == 1, "Scale steps must have exactly one coefficient");
    }

    if (step.type == StepType::Swap) {
        TT_FATAL(step.coefficients.empty(), "Swap steps must not carry coefficients");
    }
}

void validate_runtime_scheme(const RuntimeLiftingScheme& scheme) {
    TT_FATAL(scheme.mode == BoundaryMode::Symmetric, "Only symmetric mode is currently supported");
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

    RuntimeLiftingScheme scheme{
        .mode = mode,
        .tap_size = obj.at("tap_size").get<int>(),
        .steps = {},
    };

    for (const auto& raw_step : obj.at("steps")) {
        RuntimeLiftingStep step{
            .type = parse_step_type(raw_step.at("type").get<std::string>()),
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

device::DeviceStepDesc pack_device_step_desc(const RuntimeLiftingStep& step) {
    validate_runtime_step(step);

    device::DeviceStepDesc desc{};
    desc.type = static_cast<uint32_t>(to_device_step_type(step.type));
    desc.k = static_cast<uint32_t>(step.coefficients.size());

    for (size_t i = 0; i < step.coefficients.size(); ++i) {
        desc.coeffs_packed[i] = std::bit_cast<uint32_t>(step.coefficients[i]);
    }

    return desc;
}

std::vector<device::DeviceStepDesc> pack_device_step_descs(const std::vector<RuntimeLiftingStep>& steps) {
    std::vector<device::DeviceStepDesc> packed;
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
        input, even_ping_addr, odd_ping_addr, Pad1DConfig{.mode = scheme.mode, .left = wavelet_pad, .right = wavelet_pad});

    const SignalBuffer even_ping = preprocess_layout.output.even;
    const SignalBuffer odd_ping = preprocess_layout.output.odd;
    const SignalBuffer even_pong = clone_signal_buffer_with_address(even_ping, even_pong_addr);
    const SignalBuffer odd_pong = clone_signal_buffer_with_address(odd_ping, odd_pong_addr);

    std::vector<LiftingStepRoute> routes;
    routes.reserve(scheme.steps.size());

    LiftingActiveStreams active{};
    for (const auto& step : scheme.steps) {
        if (step.type == StepType::Predict) {
            const StreamRef output = with_toggled_slot(active.odd);
            routes.push_back(LiftingStepRoute{
                .type = step.type,
                .source = active.even,
                .base = active.odd,
                .output = output,
            });
            active.odd = output;
            continue;
        }

        if (step.type == StepType::Update) {
            const StreamRef output = with_toggled_slot(active.even);
            routes.push_back(LiftingStepRoute{
                .type = step.type,
                .source = active.odd,
                .base = active.even,
                .output = output,
            });
            active.even = output;
            continue;
        }

        if (step.type == StepType::ScaleOdd) {
            const StreamRef output = with_toggled_slot(active.odd);
            routes.push_back(LiftingStepRoute{
                .type = step.type,
                .source = active.odd,
                .base = active.odd,
                .output = output,
            });
            active.odd = output;
            continue;
        }

        if (step.type == StepType::ScaleEven) {
            const StreamRef output = with_toggled_slot(active.even);
            routes.push_back(LiftingStepRoute{
                .type = step.type,
                .source = active.even,
                .base = active.even,
                .output = output,
            });
            active.even = output;
            continue;
        }

        routes.push_back(LiftingStepRoute{
            .type = step.type,
            .source = active.even,
            .base = active.odd,
            .output = active.even,
        });
        std::swap(active.even, active.odd);
    }

    return LiftingForwardPlan{
        .scheme = scheme,
        .preprocess_layout = preprocess_layout,
        .even_buffers = SignalBufferPair{.ping = even_ping, .pong = even_pong},
        .odd_buffers = SignalBufferPair{.ping = odd_ping, .pong = odd_pong},
        .packed_steps = pack_device_step_descs(scheme.steps),
        .routes = std::move(routes),
        .final_active = active,
        .output_length = ceil_div(input.length + static_cast<size_t>(scheme.tap_size) - 1, size_t{2}),
    };
}

}  // namespace ttwv
