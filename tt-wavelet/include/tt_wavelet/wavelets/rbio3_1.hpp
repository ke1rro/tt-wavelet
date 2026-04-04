#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct rbio3_1_tag {};

inline constexpr auto rbio3_1_scheme = make_lifting_scheme(
    4,
    1,
    1,
    PredictStep<1>{{0.33333333333333331f}, -1},
    UpdateStep<2>{{1.1250000000000002f, 0.37500000000000006f}, 0},
    PredictStep<1>{{-0.44444444444444442f}, 0},
    ScaleEvenStep<1>{{0.47140452079103168f}, 0},
    ScaleOddStep<1>{{2.1213203435596428f}, 0});

template <>
struct scheme_traits<rbio3_1_tag> {
    using SchemeType = decltype(rbio3_1_scheme);
    static constexpr const char* name = "rbio3.1";
    static constexpr int id = 79;
    static constexpr int tap_size = 4;
    static constexpr int delay_even = 1;
    static constexpr int delay_odd = 1;
    static constexpr int num_steps = 5;
    static constexpr const auto& scheme = rbio3_1_scheme;
};

}  // namespace ttwv
