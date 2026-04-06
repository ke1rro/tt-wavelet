#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace ttwv {

enum class StepType : uint8_t {
    Predict = 0,
    Update = 1,
    ScaleEven = 2,
    ScaleOdd = 3,
    Swap = 4,
};

template <StepType Type, std::size_t N>
struct LiftingStep {
    static constexpr StepType type = Type;
    static constexpr std::size_t n_coeffs = N;

    std::array<float, N> coefficients;
    int shift;
};

template <std::size_t N>
using PredictStep = LiftingStep<StepType::Predict, N>;

template <std::size_t N>
using UpdateStep = LiftingStep<StepType::Update, N>;

template <std::size_t N>
using ScaleEvenStep = LiftingStep<StepType::ScaleEven, N>;

template <std::size_t N>
using ScaleOddStep = LiftingStep<StepType::ScaleOdd, N>;

using SwapStep = LiftingStep<StepType::Swap, 0>;

template <typename Step>
inline constexpr bool is_update_step = (Step::type == StepType::Update || Step::type == StepType::ScaleEven);

template <typename Step>
inline constexpr bool is_predict_step = (Step::type == StepType::Predict || Step::type == StepType::ScaleOdd);

template <typename Step>
inline constexpr bool is_swap_step = (Step::type == StepType::Swap);

template <typename Step>
inline constexpr bool is_scale_step = (Step::type == StepType::ScaleEven || Step::type == StepType::ScaleOdd);

}  // namespace ttwv
