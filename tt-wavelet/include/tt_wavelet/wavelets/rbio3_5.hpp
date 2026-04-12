#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct rbio3_5_tag {};

inline constexpr auto rbio3_5_scheme = make_lifting_scheme(
    12,
    3,
    3,
    PredictStep<1>{{0.33333333333333331f}, {0x3eaaaaabu}, -1},
    UpdateStep<2>{{1.125f, 0.37499999999999994f}, {0x3f900000u, 0x3ec00000u}, 0},
    PredictStep<5>{
        {0.017361111111111108f,
         -0.11805555555555554f,
         -0.44444444444444448f,
         0.11805555555555555f,
         -0.017361111111111112f},
        {0x3c8e38e4u, 0xbdf1c71cu, 0xbee38e39u, 0x3df1c71cu, 0xbc8e38e4u},
        -2},
    ScaleEvenStep<1>{{0.47140452079103168f}, {0x3ef15befu}, 0},
    ScaleOddStep<1>{{2.1213203435596428f}, {0x4007c3b6u}, 0});

template <>
struct scheme_traits<rbio3_5_tag> {
    using SchemeType = decltype(rbio3_5_scheme);
    static constexpr const char* name = "rbio3.5";
    static constexpr int id = 81;
    static constexpr int tap_size = 12;
    static constexpr int delay_even = 3;
    static constexpr int delay_odd = 3;
    static constexpr int num_steps = 5;
    static constexpr const auto& scheme = rbio3_5_scheme;
};

}  // namespace ttwv
