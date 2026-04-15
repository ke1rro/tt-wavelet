#pragma once

#include "../scheme.hpp"

namespace ttwv {

struct rbio3_9_tag {};

inline constexpr auto rbio3_9_scheme = make_lifting_scheme(
    20,
    5,
    5,
    PredictStep<1>{ { 0.33333333333333331f }, -1 },
    UpdateStep<2>{ { 1.1250000000000002f, 0.375f }, 0 },
    PredictStep<9>{ { 0.00085449218749999989f, -0.0089246961805555542f, 0.044514973958333329f, -0.14900716145833331f, -0.44444444444444448f, 0.14900716145833331f, -0.044514973958333329f, 0.0089246961805555559f, -0.00085449218750000011f }, -4 },
    ScaleEvenStep<1>{ { 0.47140452079103168f }, 0 },
    ScaleOddStep<1>{ { 2.1213203435596428f }, 0 }
);

template <>
struct scheme_traits<rbio3_9_tag> {
    using SchemeType = decltype(rbio3_9_scheme);
    static constexpr const char* name = "rbio3.9";
    static constexpr int id         = 83;
    static constexpr int tap_size   = 20;
    static constexpr int delay_even = 5;
    static constexpr int delay_odd  = 5;
    static constexpr int num_steps  = 5;
    static constexpr const auto& scheme = rbio3_9_scheme;
};

}
