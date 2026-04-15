#pragma once

#include "../scheme.hpp"

namespace ttwv {

struct bior3_1_tag {};

inline constexpr auto bior3_1_scheme = make_lifting_scheme(
    4,
    1,
    1,
    PredictStep<1>{ { -0.33333333333333331f }, -1 },
    UpdateStep<2>{ { 1.1250000000000002f, -0.37500000000000006f }, 0 },
    PredictStep<1>{ { -0.44444444444444442f }, 0 },
    ScaleEvenStep<1>{ { 0.94280904158206336f }, 0 },
    ScaleOddStep<1>{ { 1.0606601717798214f }, 0 }
);

template <>
struct scheme_traits<bior3_1_tag> {
    using SchemeType = decltype(bior3_1_scheme);
    static constexpr const char* name = "bior3.1";
    static constexpr int id         = 7;
    static constexpr int tap_size   = 4;
    static constexpr int delay_even = 1;
    static constexpr int delay_odd  = 1;
    static constexpr int num_steps  = 5;
    static constexpr const auto& scheme = bior3_1_scheme;
};

}
