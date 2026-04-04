#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct bior2_8_tag {};

inline constexpr auto bior2_8_scheme = make_lifting_scheme(
    18,
    4,
    5,
    UpdateStep<2>{{-0.49999999999999994f, -0.49999999999999994f}, -1},
    PredictStep<8>{
        {-0.00213623046875f,
         0.020446777343750003f,
         -0.095397949218750014f,
         0.32708740234375f,
         0.32708740234375f,
         -0.095397949218750014f,
         0.020446777343750003f,
         -0.00213623046875f},
        -3},
    SwapStep{{}, 0},
    ScaleEvenStep<1>{{1.4142135623730949f}, 0},
    ScaleOddStep<1>{{-0.70710678118654757f}, 0});

template <>
struct scheme_traits<bior2_8_tag> {
    using SchemeType = decltype(bior2_8_scheme);
    static constexpr const char* name = "bior2.8";
    static constexpr int id = 6;
    static constexpr int tap_size = 18;
    static constexpr int delay_even = 4;
    static constexpr int delay_odd = 5;
    static constexpr int num_steps = 5;
    static constexpr const auto& scheme = bior2_8_scheme;
};

}  // namespace ttwv
