#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <tt_stl/assert.hpp>
#include <utility>
#include <vector>

#include "tt_wavelet/include/common/boundary.hpp"
#include "tt_wavelet/include/common/fp32_arithmetic.hpp"
#include "tt_wavelet/include/common/signal_extension.hpp"
#include "tt_wavelet/include/lifting/plan.hpp"
#include "tt_wavelet/include/lifting/static_scheme.hpp"

namespace ttwv {

struct Lwt1DReferenceOutput {
    std::vector<float> approximation;
    std::vector<float> detail;
};

struct Lwt2DReferenceOutput {
    size_t band_height{0};
    size_t band_width{0};
    std::vector<float> ll;
    std::vector<float> lh;
    std::vector<float> hl;
    std::vector<float> hh;
};

namespace reference_2d_detail {

struct Stream {
    std::vector<float> values;
    int shift{0};
};

struct SpanSourceReader {
    std::span<const float> input;
    [[nodiscard]] float operator()(const uint32_t index) const { return input[index]; }
};

[[nodiscard]] inline float padded_value(
    const std::span<const float> input, const int64_t index, const BoundaryMode mode) {
    TT_FATAL(!input.empty(), "FP32 reference boundary extension requires a non-empty signal");
    const uint32_t length = static_cast<uint32_t>(input.size());
    return evaluate_extended_index(
        make_extended_index(mode, index, length),
        length,
        SpanSourceReader{.input = input});
}

template <typename Step>
[[nodiscard]] std::vector<float> predict_update(
    const Stream& source, const Stream& base, int& output_shift, const Fp32Arithmetic arithmetic) {
    static_assert(is_predict_update_step(Step::type));
    const auto [shift, output_length, source_offset, base_offset] = detail::compute_step_geometry(
        detail::StreamState{.shift = source.shift, .length = source.values.size()},
        Step::shift,
        Step::k,
        detail::StreamState{.shift = base.shift, .length = base.values.size()});
    output_shift = shift;
    std::vector<float> output(output_length);
    for (size_t output_index = 0; output_index < output_length; ++output_index) {
        float accumulator = base.values[base_offset + output_index];
        // The generated lifting coefficients follow valid-convolution order:
        // h[0] multiplies the newest sample in the K-wide source window.
        for (size_t coefficient = 0; coefficient < Step::k; ++coefficient) {
            const float value = source.values[source_offset + output_index + Step::k - 1 - coefficient];
            accumulator = fp32_fma(std::bit_cast<float>(Step::coeff_bits[coefficient]), value, accumulator, arithmetic);
        }
        output[output_index] = accumulator;
    }
    return output;
}

template <typename Scheme, size_t StepIndex = 0>
void apply_steps(Stream& even, Stream& odd, const Fp32Arithmetic arithmetic) {
    if constexpr (StepIndex < Scheme::num_steps) {
        using Step = SchemeStep<Scheme, StepIndex>;
        if constexpr (Step::type == StepType::kPredict) {
            int output_shift = 0;
            std::vector<float> output = predict_update<Step>(even, odd, output_shift, arithmetic);
            odd = Stream{.values = std::move(output), .shift = output_shift};
        } else if constexpr (Step::type == StepType::kUpdate) {
            int output_shift = 0;
            std::vector<float> output = predict_update<Step>(odd, even, output_shift, arithmetic);
            even = Stream{.values = std::move(output), .shift = output_shift};
        } else if constexpr (Step::type == StepType::kScaleEven) {
            static_assert(Step::k == 1);
            const float scale = std::bit_cast<float>(Step::coeff_bits[0]);
            for (float& value : even.values) {
                value = fp32_multiply(value, scale, arithmetic);
            }
        } else if constexpr (Step::type == StepType::kScaleOdd) {
            static_assert(Step::k == 1);
            const float scale = std::bit_cast<float>(Step::coeff_bits[0]);
            for (float& value : odd.values) {
                value = fp32_multiply(value, scale, arithmetic);
            }
        } else {
            static_assert(Step::type == StepType::kSwap);
            std::swap(even, odd);
        }
        apply_steps<Scheme, StepIndex + 1>(even, odd, arithmetic);
    }
}

[[nodiscard]] inline std::vector<float> canonicalize(
    const Stream& stream, const size_t output_length, const int canonical_start) {
    std::vector<float> output(output_length, 0.0F);
    const int64_t source_offset = static_cast<int64_t>(canonical_start) - static_cast<int64_t>(stream.shift);
    const size_t source_begin = source_offset > 0 ? static_cast<size_t>(source_offset) : size_t{0};
    const size_t output_begin = source_offset < 0 ? static_cast<size_t>(-source_offset) : size_t{0};
    if (source_begin >= stream.values.size() || output_begin >= output.size()) {
        return output;
    }
    const size_t count = std::min(stream.values.size() - source_begin, output.size() - output_begin);
    std::copy_n(
        stream.values.begin() + static_cast<std::ptrdiff_t>(source_begin),
        count,
        output.begin() + static_cast<std::ptrdiff_t>(output_begin));
    return output;
}

}  // namespace reference_2d_detail

/**
 * Scalar forward reference using the same FP32 coefficient bit patterns,
 * lifting-step order, valid-convolution geometry, and canonical crop as the
 * device implementation.
 */
template <typename Scheme>
[[nodiscard]] Lwt1DReferenceOutput lwt_1d_fp32_reference(
    const std::span<const float> input,
    const BoundaryMode boundary_mode = BoundaryMode::kSymmetric,
    const Fp32Arithmetic arithmetic = Fp32Arithmetic::kIeee) {
    TT_FATAL(!input.empty(), "FP32 LWT reference input must be non-empty");
    TT_FATAL(
        !boundary_mode_requires_multiple_samples(boundary_mode) || input.size() > 1,
        "Reflect and antireflect extension require at least two samples");
    constexpr size_t padding = Scheme::tap_size - 1;
    TT_FATAL(
        input.size() <= std::numeric_limits<size_t>::max() - 2 * padding,
        "FP32 LWT reference padded length overflows size_t");
    const size_t padded_length = input.size() + 2 * padding;

    reference_2d_detail::Stream even{
        .values = {},
        .shift = Scheme::delay_even,
    };
    reference_2d_detail::Stream odd{
        .values = {},
        .shift = Scheme::delay_odd,
    };
    even.values.reserve(ceil_div(padded_length, size_t{2}));
    odd.values.reserve(padded_length / 2);
    for (size_t padded_index = 0; padded_index < padded_length; ++padded_index) {
        const int64_t input_index = static_cast<int64_t>(padded_index) - static_cast<int64_t>(padding);
        const float value = reference_2d_detail::padded_value(input, input_index, boundary_mode);
        ((padded_index & 1U) == 0 ? even.values : odd.values).push_back(value);
    }

    reference_2d_detail::apply_steps<Scheme>(even, odd, arithmetic);
    const size_t output_length = (input.size() + Scheme::tap_size - 1) / 2;
    constexpr int canonical_start = static_cast<int>(Scheme::tap_size / 2);
    return Lwt1DReferenceOutput{
        .approximation = reference_2d_detail::canonicalize(even, output_length, canonical_start),
        .detail = reference_2d_detail::canonicalize(odd, output_length, canonical_start),
    };
}

/**
 * Vertical-first separable 2D FP32 reference.
 *
 * Band labels use the first letter for the vertical result:
 * LL=(vertical low, horizontal low), LH=(vertical low, horizontal high),
 * HL=(vertical high, horizontal low), HH=(vertical high, horizontal high).
 */
template <typename Scheme>
[[nodiscard]] Lwt2DReferenceOutput lwt_2d_fp32_reference(
    const std::span<const float> input,
    const size_t height,
    const size_t width,
    const BoundaryMode boundary_mode = BoundaryMode::kSymmetric,
    const Fp32Arithmetic arithmetic = Fp32Arithmetic::kIeee) {
    TT_FATAL(height > 0 && width > 0, "2D FP32 LWT reference dimensions must be positive");
    TT_FATAL(
        height <= std::numeric_limits<size_t>::max() / width && input.size() == height * width,
        "2D FP32 LWT reference input size {} does not match shape {}x{}",
        input.size(),
        height,
        width);

    constexpr size_t output_expansion = Scheme::tap_size - 1;
    TT_FATAL(
        height <= std::numeric_limits<size_t>::max() - output_expansion &&
            width <= std::numeric_limits<size_t>::max() - output_expansion,
        "2D FP32 LWT output shape overflows size_t");
    const size_t band_height = (height + output_expansion) / 2;
    const size_t band_width = (width + output_expansion) / 2;
    TT_FATAL(
        band_height <= std::numeric_limits<size_t>::max() / width,
        "2D FP32 LWT vertical workspace size overflows size_t");
    TT_FATAL(
        band_width == 0 || band_height <= std::numeric_limits<size_t>::max() / band_width,
        "2D FP32 LWT band size overflows size_t");
    const size_t vertical_elements = band_height * width;
    const size_t band_elements = band_height * band_width;
    std::vector<float> vertical_low(vertical_elements);
    std::vector<float> vertical_high(vertical_elements);
    std::vector<float> column(height);
    for (size_t x = 0; x < width; ++x) {
        for (size_t y = 0; y < height; ++y) {
            column[y] = input[y * width + x];
        }
        Lwt1DReferenceOutput transformed = lwt_1d_fp32_reference<Scheme>(column, boundary_mode, arithmetic);
        for (size_t y = 0; y < band_height; ++y) {
            vertical_low[y * width + x] = transformed.approximation[y];
            vertical_high[y * width + x] = transformed.detail[y];
        }
    }

    Lwt2DReferenceOutput output{
        .band_height = band_height,
        .band_width = band_width,
        .ll = std::vector<float>(band_elements),
        .lh = std::vector<float>(band_elements),
        .hl = std::vector<float>(band_elements),
        .hh = std::vector<float>(band_elements),
    };
    std::vector<float> row(width);
    for (size_t y = 0; y < band_height; ++y) {
        std::copy_n(vertical_low.begin() + static_cast<std::ptrdiff_t>(y * width), width, row.begin());
        Lwt1DReferenceOutput low = lwt_1d_fp32_reference<Scheme>(row, boundary_mode, arithmetic);
        std::copy_n(
            low.approximation.begin(), band_width, output.ll.begin() + static_cast<std::ptrdiff_t>(y * band_width));
        std::copy_n(low.detail.begin(), band_width, output.lh.begin() + static_cast<std::ptrdiff_t>(y * band_width));

        std::copy_n(vertical_high.begin() + static_cast<std::ptrdiff_t>(y * width), width, row.begin());
        Lwt1DReferenceOutput high = lwt_1d_fp32_reference<Scheme>(row, boundary_mode, arithmetic);
        std::copy_n(
            high.approximation.begin(), band_width, output.hl.begin() + static_cast<std::ptrdiff_t>(y * band_width));
        std::copy_n(high.detail.begin(), band_width, output.hh.begin() + static_cast<std::ptrdiff_t>(y * band_width));
    }
    return output;
}

}  // namespace ttwv
