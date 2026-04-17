#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct coif4_tag {};

inline constexpr auto coif4_scheme = make_lifting_scheme(
    24,
    6,
    6,
    PredictStep<1>{{0.54760236299522191f}, {0x3f0c2fabu}, -1},
    UpdateStep<2>{{1.1212935937063142f, -0.42127524980163228f}, {0x3f8f868cu, 0xbed7b164u}, 0},
    PredictStep<2>{{0.32121198358760272f, -0.606388118604409f}, {0x3ea475e6u, 0xbf1b3c40u}, -1},
    UpdateStep<2>{{0.19716919124568619f, -1.1626996493797885f}, {0x3e49e6b8u, 0xbf94d358u}, 0},
    PredictStep<2>{{-0.11165973910032083f, -0.085475548774440385f}, {0xbde4addcu, 0xbdaf0dceu}, -1},
    UpdateStep<2>{{-0.44018485730592999f, 0.25453832714798075f}, {0xbee15fe9u, 0x3e8252d9u}, 0},
    PredictStep<2>{{-0.49868137258145051f, 0.18275836101951604f}, {0xbeff532au, 0x3e3b2502u}, -1},
    UpdateStep<2>{{-3.0641413286805763f, 0.7909701912155177f}, {0xc0441ae4u, 0x3f4a7d06u}, 0},
    PredictStep<2>{{-1.0677827335252994f, 0.29606798866338807f}, {0xbf88ad1bu, 0x3e979639u}, -1},
    UpdateStep<2>{{-1.0437580890270115f, 0.92921196523070726f}, {0xbf8599ddu, 0x3f6de0d6u}, 0},
    PredictStep<2>{{-4.0568583086485575f, 0.95214136944124328f}, {0xc081d1c9u, 0x3f73bf89u}, -1},
    UpdateStep<2>{{9.1338114548968003e-06f, 0.24641155429125014f}, {0x37193d6cu, 0x3e7c534fu}, 0},
    PredictStep<1>{{-0.43810975220387677f}, {0xbee04fecu}, 0},
    ScaleEvenStep<1>{{14.798997099569736f}, {0x416cc8b1u}, 0},
    ScaleOddStep<1>{{0.067572146495594218f}, {0x3d8a6344u}, 0});

template <>
struct scheme_traits<coif4_tag> {
    using SchemeType = decltype(coif4_scheme);
    static constexpr const char* name = "coif4";
    static constexpr int id = 26;
    static constexpr int tap_size = 24;
    static constexpr int delay_even = 6;
    static constexpr int delay_odd = 6;
    static constexpr int num_steps = 15;
    static constexpr const auto& scheme = coif4_scheme;
};

}  // namespace ttwv
