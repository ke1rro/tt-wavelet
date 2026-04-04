#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct rbio2_6_tag {};

inline constexpr auto rbio2_6_scheme = make_lifting_scheme(
    14,
    3,
    4,
    PredictStep<2>{{0.50000000000000011f, 0.50000000000000011f}, 0},
    SwapStep{{}, 0},
    PredictStep<6>{
        {-0.0097656249999999983f,
         0.076171874999999972f,
         -0.31640624999999994f,
         -0.31640624999999994f,
         0.076171874999999972f,
         -0.0097656249999999983f},
        -3},
    ScaleEvenStep<1>{{0.70710678118654746f}, 0},
    ScaleOddStep<1>{{-1.4142135623730951f}, 0});

template <>
struct scheme_traits<rbio2_6_tag> {
    using SchemeType = decltype(rbio2_6_scheme);
    static constexpr const char* name = "rbio2.6";
    static constexpr int id = 77;
    static constexpr int tap_size = 14;
    static constexpr int delay_even = 3;
    static constexpr int delay_odd = 4;
    static constexpr int num_steps = 5;
    static constexpr const auto& scheme = rbio2_6_scheme;
};

}  // namespace ttwv
