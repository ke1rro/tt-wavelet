#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct bior4_4_tag {};

inline constexpr auto bior4_4_scheme = make_lifting_scheme(
    10,
    2,
    3,
    UpdateStep<2>{{-1.5861343419225902f, -1.5861343419225902f}, -1},
    PredictStep<2>{{-0.052980118579436512f, -0.052980118579436512f}, 0},
    UpdateStep<2>{{0.88291107541535185f, 0.88291107541535185f}, -1},
    PredictStep<2>{{0.44350685205067092f, 0.44350685205067092f}, 0},
    SwapStep{{}, 0},
    ScaleEvenStep<1>{{1.1496043988613913f}, 0},
    ScaleOddStep<1>{{-0.86986445162391102f}, 0});

template <>
struct scheme_traits<bior4_4_tag> {
    using SchemeType = decltype(bior4_4_scheme);
    static constexpr const char* name = "bior4.4";
    static constexpr int id = 12;
    static constexpr int tap_size = 10;
    static constexpr int delay_even = 2;
    static constexpr int delay_odd = 3;
    static constexpr int num_steps = 7;
    static constexpr const auto& scheme = bior4_4_scheme;
};

}  // namespace ttwv
