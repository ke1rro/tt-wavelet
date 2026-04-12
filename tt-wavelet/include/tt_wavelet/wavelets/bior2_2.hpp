#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct bior2_2_tag {};

inline constexpr auto bior2_2_scheme = make_lifting_scheme(
    6,
    1,
    2,
    UpdateStep<2>{{-0.5f, -0.5f}, {0xbf000000u, 0xbf000000u}, -1},
    PredictStep<2>{{0.25f, 0.25f}, {0x3e800000u, 0x3e800000u}, 0},
    SwapStep{{}, {}, 0},
    ScaleEvenStep<1>{{1.4142135623730949f}, {0x3fb504f3u}, 0},
    ScaleOddStep<1>{{-0.70710678118654757f}, {0xbf3504f3u}, 0});

template <>
struct scheme_traits<bior2_2_tag> {
    using SchemeType = decltype(bior2_2_scheme);
    static constexpr const char* name = "bior2.2";
    static constexpr int id = 3;
    static constexpr int tap_size = 6;
    static constexpr int delay_even = 1;
    static constexpr int delay_odd = 2;
    static constexpr int num_steps = 5;
    static constexpr const auto& scheme = bior2_2_scheme;
};

}  // namespace ttwv
