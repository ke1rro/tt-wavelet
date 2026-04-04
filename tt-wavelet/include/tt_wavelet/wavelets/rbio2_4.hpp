#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct rbio2_4_tag {};

inline constexpr auto rbio2_4_scheme = make_lifting_scheme(
    10,
    2,
    3,
    PredictStep<2>{{0.5f, 0.5f}, 0},
    SwapStep{{}, 0},
    PredictStep<4>{{0.046874999999999993f, -0.296875f, -0.296875f, 0.046874999999999993f}, -2},
    ScaleEvenStep<1>{{0.70710678118654746f}, 0},
    ScaleOddStep<1>{{-1.4142135623730951f}, 0});

template <>
struct scheme_traits<rbio2_4_tag> {
    using SchemeType = decltype(rbio2_4_scheme);
    static constexpr const char* name = "rbio2.4";
    static constexpr int id = 76;
    static constexpr int tap_size = 10;
    static constexpr int delay_even = 2;
    static constexpr int delay_odd = 3;
    static constexpr int num_steps = 5;
    static constexpr const auto& scheme = rbio2_4_scheme;
};

}  // namespace ttwv
