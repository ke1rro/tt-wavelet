#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct rbio4_4_tag {};

inline constexpr auto rbio4_4_scheme = make_lifting_scheme(
    10,
    2,
    3,
    PredictStep<2>{{1.5861343420367127f, 1.5861343420367127f}, {0x3fcb0673u, 0x3fcb0673u}, 0},
    UpdateStep<2>{{0.052980118574844914f, 0.052980118574844914f}, {0x3d5901aeu, 0x3d5901aeu}, -1},
    PredictStep<2>{{-0.88291107550920522f, -0.88291107550920522f}, {0xbf620676u, 0xbf620676u}, 0},
    SwapStep{{}, {}, 0},
    PredictStep<2>{{-0.44350685204873624f, -0.44350685204873624f}, {0xbee31355u, 0xbee31355u}, -1},
    ScaleEvenStep<1>{{0.86986445162473958f}, {0x3f5eaf70u}, 0},
    ScaleOddStep<1>{{-1.1496043988602962f}, {0xbf93263du}, 0});

template <>
struct scheme_traits<rbio4_4_tag> {
    using SchemeType = decltype(rbio4_4_scheme);
    static constexpr const char* name = "rbio4.4";
    static constexpr int id = 84;
    static constexpr int tap_size = 10;
    static constexpr int delay_even = 2;
    static constexpr int delay_odd = 3;
    static constexpr int num_steps = 7;
    static constexpr const auto& scheme = rbio4_4_scheme;
};

}  // namespace ttwv
