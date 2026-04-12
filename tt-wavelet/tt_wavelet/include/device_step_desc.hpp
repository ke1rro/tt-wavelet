#pragma once

#include <cstdint>

namespace ttwv::device {

enum class DeviceStepType : uint32_t {
    Predict = 0,
    Update = 1,
    ScaleEven = 2,
    ScaleOdd = 3,
    Swap = 4,
};

constexpr uint32_t step_coeff_capacity = 17;
constexpr uint32_t step_desc_header_words = 2;
constexpr uint32_t step_desc_word_count = step_desc_header_words + step_coeff_capacity;
constexpr uint32_t step_type_arg_idx = 0;
constexpr uint32_t step_k_arg_idx = 1;
constexpr uint32_t step_coeffs_arg_idx = 2;

struct DeviceStepDesc {
    uint32_t type;
    uint32_t k;
    uint32_t coeffs_packed[step_coeff_capacity];
};

static_assert(sizeof(DeviceStepDesc) == step_desc_word_count * sizeof(uint32_t));

}  // namespace ttwv::device
