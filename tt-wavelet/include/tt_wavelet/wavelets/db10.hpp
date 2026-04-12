#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct db10_tag {};

inline constexpr auto db10_scheme = make_lifting_scheme(
    20,
    5,
    5,
    PredictStep<1>{{-0.14172872474469653f}, {0xbe112156u}, -1},
    UpdateStep<2>{{-0.43800064247219772f, 0.13893787527377985f}, {0xbee0419fu, 0x3e0e45bbu}, 0},
    PredictStep<2>{{-0.70489896355877613f, 0.37992877874268477f}, {0xbf347442u, 0x3ec28606u}, -1},
    UpdateStep<2>{{-0.9717825548868525f, 0.57780843639226609f}, {0xbf78c6beu, 0x3f13eb41u}, 0},
    PredictStep<2>{{-1.1711091875922104f, 0.67947192784387223f}, {0xbf95e6e8u, 0x3f2df1dfu}, -1},
    UpdateStep<2>{{-1.4040918716399233f, 0.72784174508199972f}, {0xbfb3b948u, 0x3f3a53d6u}, 0},
    PredictStep<2>{{-1.6126772331704493f, 0.68053486002603614f}, {0xbfce6c35u, 0x3f2e3788u}, -1},
    UpdateStep<2>{{-1.9947170905426341f, 0.61499050567076452f}, {0xbfff52e4u, 0x3f1d7005u}, 0},
    PredictStep<2>{{-2.5729098417245022f, 0.50092963698584669f}, {0xc024aa8eu, 0x3f003cedu}, -1},
    UpdateStep<2>{{1.0562227980310385e-06f, 0.38865339387972236f}, {0x358dc38au, 0x3ec6fd94u}, 0},
    PredictStep<1>{{-0.23418482571008753f}, {0xbe6fce26u}, 0},
    ScaleEvenStep<1>{{21.699552444721558f}, {0x41ad98afu}, 0},
    ScaleOddStep<1>{{0.046083899773852306f}, {0x3d3cc279u}, 0});

template <>
struct scheme_traits<db10_tag> {
    using SchemeType = decltype(db10_scheme);
    static constexpr const char* name = "db10";
    static constexpr int id = 33;
    static constexpr int tap_size = 20;
    static constexpr int delay_even = 5;
    static constexpr int delay_odd = 5;
    static constexpr int num_steps = 13;
    static constexpr const auto& scheme = db10_scheme;
};

}  // namespace ttwv
