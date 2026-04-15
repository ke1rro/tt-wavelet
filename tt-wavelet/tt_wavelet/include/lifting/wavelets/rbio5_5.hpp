#pragma once

#include "../scheme.hpp"

namespace ttwv {

struct rbio5_5_tag {};

inline constexpr auto rbio5_5_scheme = make_lifting_scheme(
    12,
    3,
    3,
    UpdateStep<2>{ { -4.9932742047211152f, -4.9932742047211152f }, 0 },
    PredictStep<2>{ { 0.0043674458537323888f, 0.0043674458537323888f }, -1 },
    UpdateStep<2>{ { 5.5857858612712104f, 5.5857858612712104f }, 0 },
    PredictStep<2>{ { -0.35223144311517335f, -0.35223144311517335f }, -1 },
    UpdateStep<2>{ { 0.2900930732061176f, 0.2900930732061176f }, 0 },
    ScaleEvenStep<1>{ { 1.0811255707903031f }, 0 },
    ScaleOddStep<1>{ { 0.92496193505903268f }, 0 }
);

template <>
struct scheme_traits<rbio5_5_tag> {
    using SchemeType = decltype(rbio5_5_scheme);
    static constexpr const char* name = "rbio5.5";
    static constexpr int id         = 85;
    static constexpr int tap_size   = 12;
    static constexpr int delay_even = 3;
    static constexpr int delay_odd  = 3;
    static constexpr int num_steps  = 7;
    static constexpr const auto& scheme = rbio5_5_scheme;
};

}
