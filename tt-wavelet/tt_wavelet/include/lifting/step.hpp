#pragma once

#include <cstdint>

namespace ttwv {

enum class StepType : uint8_t {
    kPredict = 0,
    kUpdate = 1,
    kScaleEven = 2,
    kScaleOdd = 3,
    kSwap = 4,
};

}  // namespace ttwv
