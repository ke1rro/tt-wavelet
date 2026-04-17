#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct bior1_3_tag {};

inline constexpr auto bior1_3_scheme = make_lifting_scheme(
    6,
    1,
    2,
    PredictStep<1>{{-1.0f}, {0xbf800000u}, 0},
    UpdateStep<3>{{0.0625f, 0.49999999999999994f, -0.0625f}, {0x3d800000u, 0x3f000000u, 0xbd800000u}, -1},
    ScaleEvenStep<1>{{1.4142135623730951f}, {0x3fb504f3u}, 0},
    ScaleOddStep<1>{{0.70710678118654746f}, {0x3f3504f3u}, 0});

template <>
struct scheme_traits<bior1_3_tag> {
    using SchemeType = decltype(bior1_3_scheme);
    static constexpr const char* name = "bior1.3";
    static constexpr int id = 1;
    static constexpr int tap_size = 6;
    static constexpr int delay_even = 1;
    static constexpr int delay_odd = 2;
    static constexpr int num_steps = 4;
    static constexpr const auto& scheme = bior1_3_scheme;
};

}  // namespace ttwv
