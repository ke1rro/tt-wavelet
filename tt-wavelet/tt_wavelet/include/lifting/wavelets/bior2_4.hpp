#pragma once

#include "../scheme.hpp"

namespace ttwv {

struct bior2_4_tag {};

inline constexpr auto bior2_4_scheme = make_lifting_scheme(
    10,
    2,
    3,
    UpdateStep<2>{ { -0.5f, -0.5f }, -1 },
    PredictStep<4>{ { -0.046875f, 0.296875f, 0.296875f, -0.046875f }, -1 },
    SwapStep{ {}, 0 },
    ScaleEvenStep<1>{ { 1.4142135623730949f }, 0 },
    ScaleOddStep<1>{ { -0.70710678118654757f }, 0 }
);

template <>
struct scheme_traits<bior2_4_tag> {
    using SchemeType = decltype(bior2_4_scheme);
    static constexpr const char* name = "bior2.4";
    static constexpr int id         = 4;
    static constexpr int tap_size   = 10;
    static constexpr int delay_even = 2;
    static constexpr int delay_odd  = 3;
    static constexpr int num_steps  = 5;
    static constexpr const auto& scheme = bior2_4_scheme;
};

}
