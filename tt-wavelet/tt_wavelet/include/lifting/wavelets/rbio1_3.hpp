#pragma once

#include "../scheme.hpp"

namespace ttwv {

struct rbio1_3_tag {};

inline constexpr auto rbio1_3_scheme = make_lifting_scheme(
    6,
    1,
    2,
    PredictStep<1>{ { 1.0f }, 0 },
    SwapStep{ {}, 0 },
    PredictStep<3>{ { -0.0625f, -0.49999999999999994f, 0.0625f }, -1 },
    ScaleEvenStep<1>{ { 0.70710678118654746f }, 0 },
    ScaleOddStep<1>{ { -1.4142135623730951f }, 0 }
);

template <>
struct scheme_traits<rbio1_3_tag> {
    using SchemeType = decltype(rbio1_3_scheme);
    static constexpr const char* name = "rbio1.3";
    static constexpr int id         = 73;
    static constexpr int tap_size   = 6;
    static constexpr int delay_even = 1;
    static constexpr int delay_odd  = 2;
    static constexpr int num_steps  = 5;
    static constexpr const auto& scheme = rbio1_3_scheme;
};

}
