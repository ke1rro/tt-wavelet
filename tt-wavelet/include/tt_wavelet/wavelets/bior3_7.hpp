#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct bior3_7_tag {};

inline constexpr auto bior3_7_scheme = make_lifting_scheme(
    16,
    4,
    4,
    PredictStep<1>{{-0.33333333333333331f}, {0xbeaaaaabu}, -1},
    UpdateStep<2>{{-1.125f, -0.375f}, {0xbf900000u, 0xbec00000u}, 0},
    PredictStep<7>{
        {0.0037977430555555551f,
         -0.032552083333333329f,
         0.13704427083333331f,
         0.44444444444444448f,
         -0.13704427083333331f,
         0.032552083333333336f,
         -0.0037977430555555555f},
        {0x3b78e38eu, 0xbd055555u, 0x3e0c5555u, 0x3ee38e39u, 0xbe0c5555u, 0x3d055555u, 0xbb78e38eu},
        -3},
    SwapStep{{}, {}, 0},
    ScaleEvenStep<1>{{2.1213203435596428f}, {0x4007c3b6u}, 0},
    ScaleOddStep<1>{{-0.47140452079103168f}, {0xbef15befu}, 0});

template <>
struct scheme_traits<bior3_7_tag> {
    using SchemeType = decltype(bior3_7_scheme);
    static constexpr const char* name = "bior3.7";
    static constexpr int id = 10;
    static constexpr int tap_size = 16;
    static constexpr int delay_even = 4;
    static constexpr int delay_odd = 4;
    static constexpr int num_steps = 6;
    static constexpr const auto& scheme = bior3_7_scheme;
};

}  // namespace ttwv
