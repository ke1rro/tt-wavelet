#pragma once

namespace ttwv {

enum LiftingStepType : uint8_t {
    kPredict,
    kUpdate,
    kScaleEven,
    kScaleOdd,
    kSwap,
};

class LiftingStep {
public:
    LiftingStep(LiftingStepType type, std::vector<float>&& coeffs, int32_t shift) :
        _type(type), _coeffs(std::move(coeffs)), _shift(shift) {}
    LiftingStep(LiftingStepType type, const std::vector<float>& coeffs, int32_t shift) :
        _type(type), _coeffs(coeffs), _shift(shift) {}
    LiftingStep(LiftingStepType type, std::initializer_list<float> coeffs, int32_t shift) :
        _type(type), _coeffs(coeffs), _shift(shift) {}
    ~LiftingStep() = default;

    [[nodiscard]] LiftingStepType type() const noexcept { return _type; }
    [[nodiscard]] const std::vector<float>& coeffs() const& noexcept { return _coeffs; }
    [[nodiscard]] int32_t shift() const noexcept { return _shift; }

private:
    LiftingStepType _type;
    std::vector<float> _coeffs;
    int32_t _shift;
};

class LiftingScheme {
public:
    LiftingScheme(int32_t delay_even, int32_t delay_odd, uint32_t tap_size, std::vector<LiftingStep>&& steps) :
        _steps(std::move(steps)), delay_even(delay_even), delay_odd(delay_odd), tap_size(tap_size) {}
    LiftingScheme(int32_t delay_even, int32_t delay_odd, uint32_t tap_size, const std::vector<LiftingStep>& steps) :
        _steps(steps), delay_even(delay_even), delay_odd(delay_odd), tap_size(tap_size) {}
    LiftingScheme(int32_t delay_even, int32_t delay_odd, uint32_t tap_size, std::initializer_list<LiftingStep> steps) :
        _steps(steps), delay_even(delay_even), delay_odd(delay_odd), tap_size(tap_size) {}
    LiftingScheme(int32_t delay_even, int32_t delay_odd, uint32_t tap_size, LiftingStep&&... steps) :
        _steps(std::forward<LiftingStep>(steps)...), delay_even(delay_even), delay_odd(delay_odd), tap_size(tap_size) {}

    ~LiftingScheme() = default;

    [[nodiscard]] const std::vector<LiftingStep>& steps() const& noexcept { return _steps; }

private:
    std::vector<LiftingStep> _steps;
    int32_t delay_even;
    int32_t delay_odd;
    uint32_t tap_size;
};

}  // namespace ttwv
