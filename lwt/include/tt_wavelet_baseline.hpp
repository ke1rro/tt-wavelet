#pragma once

#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>

#include "tt_wavelet/wavelet_registry.hpp"

namespace ttwv::baseline {

inline float get_padded(const std::vector<float>& arr, int idx) {
    int N = arr.size();
    if (N == 0) return 0.0f;
    while (idx < 0 || idx >= N) {
        if (idx < 0) {
            idx = -idx - 1;
        } else if (idx >= N) {
            idx = 2 * N - 1 - idx;
        }
    }
    return arr[idx];
}

template <typename Step>
void apply_predict(const Step& step, const std::vector<float>& even, std::vector<float>& odd) {
    int N = odd.size();
    for (int i = 0; i < N; ++i) {
        float sum = 0.0f;
        for (std::size_t j = 0; j < Step::n_coeffs; ++j) {
            int e_idx = i - step.shift - j;
            sum += step.coefficients[j] * get_padded(even, e_idx);
        }
        odd[i] += sum;
    }
}

template <typename Step>
void apply_update(const Step& step, std::vector<float>& even, const std::vector<float>& odd) {
    int N = even.size();
    for (int i = 0; i < N; ++i) {
        float sum = 0.0f;
        for (std::size_t j = 0; j < Step::n_coeffs; ++j) {
            int o_idx = i - step.shift - j;
            sum += step.coefficients[j] * get_padded(odd, o_idx);
        }
        even[i] += sum;
    }
}

template <typename Step>
void apply_scale_odd(const Step& step, std::vector<float>& odd) {
    for (float& val : odd) {
        val *= step.coefficients[0];
    }
}

template <typename Step>
void apply_scale_even(const Step& step, std::vector<float>& even) {
    for (float& val : even) {
        val *= step.coefficients[0];
    }
}

template <typename Step>
void apply_swap(const Step& step, std::vector<float>& even, std::vector<float>& odd) {
    std::swap(even, odd);
}

template <typename Tag>
std::vector<float> lwt_1d(const std::vector<float>& signal) {
    int N = signal.size();
    if (N <= 1) return signal;

    int half_N = (N + 1) / 2;
    std::vector<float> even, odd;
    even.reserve(half_N);
    odd.reserve(N / 2);

    for (int i = 0; i < N; ++i) {
        if (i % 2 == 0) {
            even.push_back(signal[i]);
        } else {
            odd.push_back(signal[i]);
        }
    }

    const auto& scheme = get_scheme<Tag>();

    auto visitor = [&even, &odd](auto&& step, auto) {
        using StepT = std::decay_t<decltype(step)>;
        if constexpr (StepT::type == StepType::Predict) {
            apply_predict(step, even, odd);
        } else if constexpr (StepT::type == StepType::Update) {
            apply_update(step, even, odd);
        } else if constexpr (StepT::type == StepType::ScaleEven) {
            apply_scale_even(step, even);
        } else if constexpr (StepT::type == StepType::ScaleOdd) {
            apply_scale_odd(step, odd);
        } else if constexpr (StepT::type == StepType::Swap) {
            apply_swap(step, even, odd);
        }
    };

    scheme.for_each_step_indexed(visitor);

    std::vector<float> result;
    result.reserve(N);
    result.insert(result.end(), even.begin(), even.end());
    result.insert(result.end(), odd.begin(), odd.end());
    // lowpass then highpass

    return result;
}

} // namespace ttwv::baseline
