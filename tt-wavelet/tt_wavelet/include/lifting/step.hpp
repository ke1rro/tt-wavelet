#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace ttwv {

enum class StepType : uint8_t {
    kPredict = 0,
    kUpdate = 1,
    kScaleEven = 2,
    kScaleOdd = 3,
    kSwap = 4,
};

template <StepType Type, std::size_t N>
struct LiftingStep {
    static constexpr StepType type = Type;
    static constexpr std::size_t n_coeffs = N;

    std::array<float, N> coefficients;
    int shift;
};

template <std::size_t N>
using PredictStep = LiftingStep<StepType::kPredict, N>;

template <std::size_t N>
using UpdateStep = LiftingStep<StepType::kUpdate, N>;

template <std::size_t N>
using ScaleEvenStep = LiftingStep<StepType::kScaleEven, N>;

template <std::size_t N>
using ScaleOddStep = LiftingStep<StepType::kScaleOdd, N>;

using SwapStep = LiftingStep<StepType::kSwap, 0>;

template <typename Step>
inline constexpr bool is_update_step = (Step::type == StepType::kUpdate || Step::type == StepType::kScaleEven);

template <typename Step>
inline constexpr bool is_predict_step = (Step::type == StepType::kPredict || Step::type == StepType::kScaleOdd);

template <typename Step>
inline constexpr bool is_swap_step = (Step::type == StepType::kSwap);

template <typename Step>
inline constexpr bool is_scale_step = (Step::type == StepType::kScaleEven || Step::type == StepType::kScaleOdd);

}  // namespace ttwv
