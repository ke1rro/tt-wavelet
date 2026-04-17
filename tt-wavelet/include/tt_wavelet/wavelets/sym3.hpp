#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct sym3_tag {};

inline constexpr auto sym3_scheme = make_lifting_scheme(
    6,
    1,
    2,
    PredictStep<1>{{2.4254972439264524f}, {0x401b3b59u}, 0},
    UpdateStep<2>{{-0.35238765767323654f, -0.26604223490661466f}, {0xbeb46c28u, 0xbe8836b0u}, 0},
    PredictStep<2>{{2.8953474541226774f, -14.931981221241225f}, {0x40394d5fu, 0xc16ee965u}, -1},
    SwapStep{{}, {}, 0},
    PredictStep<1>{{0.066227766012329339f}, {0x3d87a26cu}, 0},
    ScaleEvenStep<1>{{0.083742580129641767f}, {0x3dab813bu}, 0},
    ScaleOddStep<1>{{-11.941356457514223f}, {0xc13f0fccu}, 0});

template <>
struct scheme_traits<sym3_tag> {
    using SchemeType = decltype(sym3_scheme);
    static constexpr const char* name = "sym3";
    static constexpr int id = 99;
    static constexpr int tap_size = 6;
    static constexpr int delay_even = 1;
    static constexpr int delay_odd = 2;
    static constexpr int num_steps = 7;
    static constexpr const auto& scheme = sym3_scheme;
};

}  // namespace ttwv
