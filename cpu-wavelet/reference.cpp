#include "cpu-wavelet/reference.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#include "cpu-wavelet/pad_split.hpp"

namespace ttwv::debug {

namespace {

constexpr float kAbsoluteTolerance = 1e-2F;
constexpr float kRelativeTolerance = 1e-3F;

[[nodiscard]] bool within_tolerance(const float device_value, const float reference_value) {
    const float abs_diff = std::fabs(device_value - reference_value);
    const float tolerance = kAbsoluteTolerance + kRelativeTolerance * std::fabs(reference_value);
    return abs_diff <= tolerance;
}

[[nodiscard]] std::vector<float> apply_stencil_reference(
    const std::span<const float> source, const std::vector<float>& coefficients, const size_t output_length) {
    std::vector<float> output(output_length, 0.0F);
    const int64_t k = static_cast<int64_t>(coefficients.size());

    for (size_t i = 0; i < output_length; ++i) {
        float sum = 0.0F;
        for (size_t j = 0; j < coefficients.size(); ++j) {
            const int64_t logical_index = static_cast<int64_t>(i) + k - 1 - static_cast<int64_t>(j);
            if (logical_index >= 0 && logical_index < static_cast<int64_t>(source.size())) {
                sum += coefficients[j] * source[static_cast<size_t>(logical_index)];
            }
        }
        output[i] = sum;
    }

    return output;
}

struct HostShiftedSignal {
    std::vector<float> data;
    int shift{0};

    [[nodiscard]] int end() const {
        return shift + static_cast<int>(data.size());
    }
};

[[nodiscard]] HostShiftedSignal apply_shifted_stencil_reference(
    const HostShiftedSignal& source, const int kernel_shift, const std::vector<float>& coefficients) {
    const size_t conv_length =
        source.data.size() >= coefficients.size() ? source.data.size() - coefficients.size() + 1 : 0;
    const int conv_shift =
        source.shift + kernel_shift + static_cast<int>(std::min(source.data.size(), coefficients.size())) - 1;
    return HostShiftedSignal{
        .data = apply_stencil_reference(source.data, coefficients, conv_length),
        .shift = conv_shift,
    };
}

[[nodiscard]] HostShiftedSignal add_shifted_overlap_reference(
    const HostShiftedSignal& base, const HostShiftedSignal& conv) {
    const int output_shift = std::max(base.shift, conv.shift);
    const int output_end = std::min(base.end(), conv.end());
    if (output_end <= output_shift) {
        return HostShiftedSignal{.data = {}, .shift = output_shift};
    }

    const size_t output_length = static_cast<size_t>(output_end - output_shift);
    const size_t base_offset = static_cast<size_t>(output_shift - base.shift);
    const size_t conv_offset = static_cast<size_t>(output_shift - conv.shift);

    std::vector<float> output(output_length, 0.0F);
    for (size_t i = 0; i < output_length; ++i) {
        output[i] = base.data[base_offset + i] + conv.data[conv_offset + i];
    }

    return HostShiftedSignal{
        .data = std::move(output),
        .shift = output_shift,
    };
}

}  // namespace

bool matches_reference(
    const std::vector<float>& device_values, const std::vector<float>& reference, const size_t logical_length) {
    bool pass = true;
    for (size_t i = 0; i < logical_length; ++i) {
        if (!within_tolerance(device_values[i], reference[i])) {
            std::cerr << "Mismatch at index " << i << ": device=" << device_values[i]
                      << " reference=" << reference[i] << '\n';
            pass = false;
        }
    }
    return pass;
}

HostLiftingReference materialize_reference_forward_lifting(
    const std::span<const float> input, const LiftingPreprocessDeviceProgram& bundle) {
    const auto split = ttwv::cpu::materialize_reference_pad_split(input, bundle.plan.preprocess_layout);

    HostShiftedSignal even{
        .data = std::vector<float>(
            split.even.begin(),
            split.even.begin() + static_cast<std::ptrdiff_t>(bundle.plan.preprocess_layout.output.even.length)),
        .shift = bundle.scheme.delay_even,
    };
    HostShiftedSignal odd{
        .data = std::vector<float>(
            split.odd.begin(),
            split.odd.begin() + static_cast<std::ptrdiff_t>(bundle.plan.preprocess_layout.output.odd.length)),
        .shift = bundle.scheme.delay_odd,
    };

    for (const auto& step : bundle.scheme.steps) {
        switch (step.type) {
            case StepType::kPredict:
                odd = add_shifted_overlap_reference(odd, apply_shifted_stencil_reference(even, step.shift, step.coefficients));
                break;
            case StepType::kUpdate:
                even =
                    add_shifted_overlap_reference(even, apply_shifted_stencil_reference(odd, step.shift, step.coefficients));
                break;
            case StepType::kScaleEven:
                for (auto& value : even.data) {
                    value *= step.coefficients[0];
                }
                break;
            case StepType::kScaleOdd:
                for (auto& value : odd.data) {
                    value *= step.coefficients[0];
                }
                break;
            case StepType::kSwap:
                std::swap(even, odd);
                break;
        }
    }

    return HostLiftingReference{
        .even = std::move(even.data),
        .odd = std::move(odd.data),
        .even_shift = even.shift,
        .odd_shift = odd.shift,
    };
}

std::vector<float> canonicalize_forward_output(
    const std::vector<float>& values,
    const int signal_shift,
    const int direct_shift,
    const size_t output_length,
    const char* label) {
    const int crop_shift = signal_shift - direct_shift;
    if (crop_shift > 0) {
        throw std::runtime_error(std::string(label) + " has positive crop shift " + std::to_string(crop_shift));
    }

    const size_t crop_offset = static_cast<size_t>(-crop_shift);
    if (crop_offset + output_length > values.size()) {
        throw std::runtime_error(
            std::string(label) + " canonical slice exceeds available data: offset=" + std::to_string(crop_offset) +
            " output_length=" + std::to_string(output_length) + " values.size()=" + std::to_string(values.size()));
    }

    return std::vector<float>(
        values.begin() + static_cast<std::ptrdiff_t>(crop_offset),
        values.begin() + static_cast<std::ptrdiff_t>(crop_offset + output_length));
}

}  // namespace ttwv::debug
