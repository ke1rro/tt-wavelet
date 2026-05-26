#pragma once

#include <cstdint>
#include <initializer_list>
#include <string_view>
#include <utility>
#include <vector>

namespace ttwv {

enum class LiftingStepType : uint8_t {
    kPredict,
    kUpdate,
    kScaleEven,
    kScaleOdd,
    kSwap,
};

class LiftingStep {
public:
    LiftingStep(LiftingStepType type, std::vector<float>&& coeffs, int32_t shift) :
        type_(type), coeffs_(std::move(coeffs)), shift_(shift) {}
    LiftingStep(LiftingStepType type, const std::vector<float>& coeffs, int32_t shift) :
        type_(type), coeffs_(coeffs), shift_(shift) {}
    LiftingStep(LiftingStepType type, std::initializer_list<float> coeffs, int32_t shift) :
        type_(type), coeffs_(coeffs), shift_(shift) {}

    [[nodiscard]] LiftingStepType type() const noexcept { return type_; }
    [[nodiscard]] const std::vector<float>& coeffs() const noexcept { return coeffs_; }
    [[nodiscard]] int32_t shift() const noexcept { return shift_; }

private:
    LiftingStepType type_;
    std::vector<float> coeffs_;
    int32_t shift_;
};

class LiftingScheme {
public:
    LiftingScheme(int32_t delay_even, int32_t delay_odd, uint32_t tap_size, std::vector<LiftingStep>&& steps) :
        steps_(std::move(steps)), delay_even_(delay_even), delay_odd_(delay_odd), tap_size_(tap_size) {}
    LiftingScheme(int32_t delay_even, int32_t delay_odd, uint32_t tap_size, const std::vector<LiftingStep>& steps) :
        steps_(steps), delay_even_(delay_even), delay_odd_(delay_odd), tap_size_(tap_size) {}
    LiftingScheme(int32_t delay_even, int32_t delay_odd, uint32_t tap_size, std::initializer_list<LiftingStep> steps) :
        steps_(steps), delay_even_(delay_even), delay_odd_(delay_odd), tap_size_(tap_size) {}

    [[nodiscard]] const std::vector<LiftingStep>& steps() const noexcept { return steps_; }
    [[nodiscard]] int32_t delay_even() const noexcept { return delay_even_; }
    [[nodiscard]] int32_t delay_odd() const noexcept { return delay_odd_; }
    [[nodiscard]] uint32_t tap_size() const noexcept { return tap_size_; }

private:
    std::vector<LiftingStep> steps_;
    int32_t delay_even_;
    int32_t delay_odd_;
    uint32_t tap_size_;
};

}  // namespace ttwv
