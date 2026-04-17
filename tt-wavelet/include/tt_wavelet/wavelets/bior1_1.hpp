#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct bior1_1_tag {};

inline constexpr auto bior1_1_scheme = make_lifting_scheme(
    2,
    0,
    1,
    PredictStep<1>{{1.0f}, {0x3f800000u}, 0},
    SwapStep{{}, {}, 0},
    PredictStep<1>{{-0.49999999999999994f}, {0xbf000000u}, 0},
    ScaleEvenStep<1>{{0.70710678118654746f}, {0x3f3504f3u}, 0},
    ScaleOddStep<1>{{-1.4142135623730951f}, {0xbfb504f3u}, 0});

template <>
struct scheme_traits<bior1_1_tag> {
    using SchemeType = decltype(bior1_1_scheme);
    static constexpr const char* name = "bior1.1";
    static constexpr int id = 0;
    static constexpr int tap_size = 2;
    static constexpr int delay_even = 0;
    static constexpr int delay_odd = 1;
    static constexpr int num_steps = 5;
    static constexpr const auto& scheme = bior1_1_scheme;
};

}  // namespace ttwv
