#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct rbio2_2_tag {};

inline constexpr auto rbio2_2_scheme = make_lifting_scheme(
    6,
    1,
    2,
    PredictStep<2>{{0.5f, 0.5f}, {0x3f000000u, 0x3f000000u}, 0},
    SwapStep{{}, {}, 0},
    PredictStep<2>{{-0.25f, -0.25f}, {0xbe800000u, 0xbe800000u}, -1},
    ScaleEvenStep<1>{{0.70710678118654757f}, {0x3f3504f3u}, 0},
    ScaleOddStep<1>{{-1.4142135623730951f}, {0xbfb504f3u}, 0});

template <>
struct scheme_traits<rbio2_2_tag> {
    using SchemeType = decltype(rbio2_2_scheme);
    static constexpr const char* name = "rbio2.2";
    static constexpr int id = 75;
    static constexpr int tap_size = 6;
    static constexpr int delay_even = 1;
    static constexpr int delay_odd = 2;
    static constexpr int num_steps = 5;
    static constexpr const auto& scheme = rbio2_2_scheme;
};

}  // namespace ttwv
