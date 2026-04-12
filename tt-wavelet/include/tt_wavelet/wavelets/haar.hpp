#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct haar_tag {};

inline constexpr auto haar_scheme = make_lifting_scheme(
    2,
    0,
    1,
    PredictStep<1>{{1.0f}, {0x3f800000u}, 0},
    SwapStep{{}, {}, 0},
    PredictStep<1>{{-0.49999999999999994f}, {0xbf000000u}, 0},
    ScaleEvenStep<1>{{0.70710678118654746f}, {0x3f3504f3u}, 0},
    ScaleOddStep<1>{{-1.4142135623730951f}, {0xbfb504f3u}, 0});

template <>
struct scheme_traits<haar_tag> {
    using SchemeType = decltype(haar_scheme);
    static constexpr const char* name = "haar";
    static constexpr int id = 71;
    static constexpr int tap_size = 2;
    static constexpr int delay_even = 0;
    static constexpr int delay_odd = 1;
    static constexpr int num_steps = 5;
    static constexpr const auto& scheme = haar_scheme;
};

}  // namespace ttwv
