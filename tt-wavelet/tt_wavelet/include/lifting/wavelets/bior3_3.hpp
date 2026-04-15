#pragma once

#include "../scheme.hpp"

namespace ttwv {

struct bior3_3_tag {};

inline constexpr auto bior3_3_scheme = make_lifting_scheme(
    8,
    2,
    2,
    PredictStep<1>{ { -0.33333333333333331f }, -1 },
    UpdateStep<2>{ { -1.1250000000000002f, -0.375f }, 0 },
    PredictStep<3>{ { 0.083333333333333329f, 0.44444444444444442f, -0.083333333333333329f }, -1 },
    SwapStep{ {}, 0 },
    ScaleEvenStep<1>{ { 2.1213203435596428f }, 0 },
    ScaleOddStep<1>{ { -0.47140452079103168f }, 0 }
);

template <>
struct scheme_traits<bior3_3_tag> {
    using SchemeType = decltype(bior3_3_scheme);
    static constexpr const char* name = "bior3.3";
    static constexpr int id         = 8;
    static constexpr int tap_size   = 8;
    static constexpr int delay_even = 2;
    static constexpr int delay_odd  = 2;
    static constexpr int num_steps  = 6;
    static constexpr const auto& scheme = bior3_3_scheme;
};

}
