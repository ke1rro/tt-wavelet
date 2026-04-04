#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct rbio6_8_tag {};

inline constexpr auto rbio6_8_scheme = make_lifting_scheme(
    18,
    4,
    5,
    PredictStep<2>{{0.99715069105509402f, 0.99715069105509402f}, 0},
    UpdateStep<2>{{-0.27351197468493066f, -0.27351197468493066f}, -1},
    PredictStep<2>{{0.38746044045008532f, 0.38746044045008532f}, 0},
    UpdateStep<2>{{0.28650325796898857f, 0.28650325796898857f}, -1},
    PredictStep<2>{{-0.54859416825623497f, -0.54859416825623497f}, 0},
    SwapStep{{}, 0},
    PredictStep<4>{{0.09982321700980705f, -0.34381326275528318f, -0.34381326275528318f, 0.09982321700980705f}, -2},
    ScaleEvenStep<1>{{0.86857869730814941f}, 0},
    ScaleOddStep<1>{{-1.1513061546399239f}, 0});

template <>
struct scheme_traits<rbio6_8_tag> {
    using SchemeType = decltype(rbio6_8_scheme);
    static constexpr const char* name = "rbio6.8";
    static constexpr int id = 86;
    static constexpr int tap_size = 18;
    static constexpr int delay_even = 4;
    static constexpr int delay_odd = 5;
    static constexpr int num_steps = 9;
    static constexpr const auto& scheme = rbio6_8_scheme;
};

}  // namespace ttwv
