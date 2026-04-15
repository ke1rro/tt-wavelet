#pragma once

#include "../scheme.hpp"

namespace ttwv {

struct bior2_6_tag {};

inline constexpr auto bior2_6_scheme = make_lifting_scheme(
    14,
    3,
    4,
    UpdateStep<2>{ { -0.49999999999999994f, -0.49999999999999994f }, -1 },
    PredictStep<6>{ { 0.009765625f, -0.076171875f, 0.31640625f, 0.31640625f, -0.076171875f, 0.009765625f }, -2 },
    SwapStep{ {}, 0 },
    ScaleEvenStep<1>{ { 1.4142135623730949f }, 0 },
    ScaleOddStep<1>{ { -0.70710678118654757f }, 0 }
);

template <>
struct scheme_traits<bior2_6_tag> {
    using SchemeType = decltype(bior2_6_scheme);
    static constexpr const char* name = "bior2.6";
    static constexpr int id         = 5;
    static constexpr int tap_size   = 14;
    static constexpr int delay_even = 3;
    static constexpr int delay_odd  = 4;
    static constexpr int num_steps  = 5;
    static constexpr const auto& scheme = bior2_6_scheme;
};

}
