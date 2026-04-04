#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct rbio2_8_tag {};

inline constexpr auto rbio2_8_scheme = make_lifting_scheme(
    18,
    4,
    5,
    PredictStep<2>{{0.5f, 0.5f}, 0},
    SwapStep{{}, 0},
    PredictStep<8>{
        {0.0021362304687499996f,
         -0.020446777343749997f,
         0.095397949218749986f,
         -0.32708740234374994f,
         -0.32708740234374994f,
         0.095397949218749986f,
         -0.020446777343749997f,
         0.0021362304687499996f},
        -4},
    ScaleEvenStep<1>{{0.70710678118654746f}, 0},
    ScaleOddStep<1>{{-1.4142135623730951f}, 0});

template <>
struct scheme_traits<rbio2_8_tag> {
    using SchemeType = decltype(rbio2_8_scheme);
    static constexpr const char* name = "rbio2.8";
    static constexpr int id = 78;
    static constexpr int tap_size = 18;
    static constexpr int delay_even = 4;
    static constexpr int delay_odd = 5;
    static constexpr int num_steps = 5;
    static constexpr const auto& scheme = rbio2_8_scheme;
};

}  // namespace ttwv
