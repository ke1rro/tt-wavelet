#pragma once

#include <cstdint>

#include "tt_wavelet/include/device_protocol/lwt_config.hpp"

namespace ttwv::device_protocol {

enum class DeviceStepType : uint32_t {
    kPredict = 0,
    kUpdate = 1,
    kScaleEven = 2,
    kScaleOdd = 3,
    kSwap = 4,
};

constexpr uint32_t step_desc_header_words = 2;
constexpr uint32_t step_desc_word_count = step_desc_header_words + step_coeff_capacity;
constexpr uint32_t step_k_arg_idx = 1;
constexpr uint32_t step_coeffs_arg_idx = 2;

struct DeviceStepDesc {
    uint32_t type;
    uint32_t k;
    uint32_t coeffs_packed[step_coeff_capacity];
};

}  // namespace ttwv::device_protocol
