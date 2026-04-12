#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct sym9_tag {};

inline constexpr auto sym9_scheme = make_lifting_scheme(
    18,
    4,
    5,
    PredictStep<1>{{-0.44241132145920276f}, {0xbee283bdu}, 0},
    UpdateStep<2>{{0.36999334613244811f, 0.24766291376392885f}, {0x3ebd6fc5u, 0x3e7d9b59u}, 0},
    PredictStep<2>{{-0.32554987599702756f, -0.30523013138741117f}, {0xbea6ae79u, 0xbe9c4720u}, -1},
    UpdateStep<2>{{0.23449108763411461f, -0.53855817168882569f}, {0x3e701e6fu, 0xbf09def3u}, 0},
    PredictStep<2>{{0.53678548594586772f, -1.4370656379766482f}, {0x3f096ac6u, 0xbfb7f1c4u}, -1},
    UpdateStep<2>{{0.51809907926241139f, -0.77332617146123661f}, {0x3f04a224u, 0xbf45f8b4u}, 0},
    PredictStep<2>{{1.1205475512662801f, 0.025922981463428654f}, {0x3f8f6e1au, 0x3cd45c6fu}, -1},
    UpdateStep<2>{{-0.13359434344986246f, 38.083235677324794f}, {0xbe08ccf5u, 0x4218553cu}, 0},
    PredictStep<2>{{-0.026165619950111896f, -0.0011886670398210493f}, {0xbcd65948u, 0xba9bcd0cu}, -1},
    SwapStep{{}, {}, 0},
    PredictStep<1>{{309.77019284436136f}, {0x439ae296u}, 0},
    ScaleEvenStep<1>{{-20.14361519718268f}, {0xc1a12620u}, 0},
    ScaleOddStep<1>{{0.049643521791453885f}, {0x3d4b5701u}, 0});

template <>
struct scheme_traits<sym9_tag> {
    using SchemeType = decltype(sym9_scheme);
    static constexpr const char* name = "sym9";
    static constexpr int id = 105;
    static constexpr int tap_size = 18;
    static constexpr int delay_even = 4;
    static constexpr int delay_odd = 5;
    static constexpr int num_steps = 13;
    static constexpr const auto& scheme = sym9_scheme;
};

}  // namespace ttwv
