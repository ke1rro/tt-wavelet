#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct bior1_5_tag {};

inline constexpr auto bior1_5_scheme = make_lifting_scheme(
    10,
    2,
    3,
    PredictStep<1>{{-1.0f}, {0xbf800000u}, 0},
    UpdateStep<5>{
        {-0.011718749999999998f,
         0.085937499999999986f,
         0.49999999999999994f,
         -0.085937499999999986f,
         0.011718749999999998f},
        {0xbc400000u, 0x3db00000u, 0x3f000000u, 0xbdb00000u, 0x3c400000u},
        -2},
    ScaleEvenStep<1>{{1.4142135623730951f}, {0x3fb504f3u}, 0},
    ScaleOddStep<1>{{0.70710678118654746f}, {0x3f3504f3u}, 0});

template <>
struct scheme_traits<bior1_5_tag> {
    using SchemeType = decltype(bior1_5_scheme);
    static constexpr const char* name = "bior1.5";
    static constexpr int id = 2;
    static constexpr int tap_size = 10;
    static constexpr int delay_even = 2;
    static constexpr int delay_odd = 3;
    static constexpr int num_steps = 4;
    static constexpr const auto& scheme = bior1_5_scheme;
};

}  // namespace ttwv
