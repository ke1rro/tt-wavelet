#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct db26_tag {};

inline constexpr auto db26_scheme = make_lifting_scheme(
    52,
    13,
    13,
    PredictStep<1>{{-0.30791794533139116f}, {0xbe9da76cu}, -1},
    UpdateStep<2>{{-0.44282761343908694f, 0.16943380066921138f}, {0xbee2ba4du, 0x3e2d800eu}, 0},
    PredictStep<2>{{-0.51619296909957235f, 0.39930564644094979f}, {0xbf042539u, 0x3ecc71cau}, -1},
    UpdateStep<2>{{-0.62954516687942719f, 0.54351634877815935f}, {0xbf2129dfu, 0x3f0b23e3u}, 0},
    PredictStep<2>{{-0.70456561490213832f, 0.5953011340337947f}, {0xbf345e6au, 0x3f1865a8u}, -1},
    UpdateStep<2>{{-0.81506951202642275f, 0.67652228516249691f}, {0xbf50a865u, 0x3f2d3091u}, 0},
    PredictStep<2>{{-0.87319390745897774f, 0.70748019848117671f}, {0xbf5f89a3u, 0x3f351d6cu}, -1},
    UpdateStep<2>{{-0.9802273203527021f, 0.68809073123067288f}, {0xbf7af02du, 0x3f3026b7u}, 0},
    PredictStep<2>{{-1.118319741450412f, 0.27701286347794052f}, {0xbf8f251au, 0x3e8dd4a1u}, -1},
    UpdateStep<2>{{-2.4438948299289658f, 0.17284304409453011f}, {0xc01c68c6u, 0x3e30fdc4u}, 0},
    PredictStep<2>{{-3.5770917910816982f, 0.27153193342421095f}, {0xc064ef12u, 0x3e8b063cu}, -1},
    UpdateStep<2>{{0.0051322627971474985f, 0.26570744439661453f}, {0x3ba82c8au, 0x3e880aceu}, 0},
    PredictStep<2>{{1.5713391853181813f, 68.652367936435141f}, {0x3fc921a4u, 0x42894e03u}, -1},
    UpdateStep<2>{{-0.014618666726126035f, 0.0072731029927715742f}, {0xbc6f8322u, 0x3bee5336u}, 0},
    PredictStep<2>{{-144.76608359808017f, 90.89578361418468f}, {0xc310c41eu, 0x42b5caa4u}, -1},
    UpdateStep<2>{{-0.011980156691732724f, 0.0068488414096477736f}, {0xbc44486bu, 0x3be06c3fu}, 0},
    PredictStep<2>{{-163.05593551247927f, 83.199573526481487f}, {0xc3230e52u, 0x42a6662fu}, -1},
    UpdateStep<2>{{-0.013698457315341691f, 0.0061276425876670782f}, {0xbc606f7fu, 0x3bc8ca64u}, 0},
    PredictStep<2>{{-189.45486833629852f, 72.988473845924489f}, {0xc33d7472u, 0x4291fa19u}, -1},
    UpdateStep<2>{{-0.016211620313866776f, 0.0052781683482031171f}, {0xbc84ce3bu, 0x3bacf47cu}, 0},
    PredictStep<2>{{-229.23065135462028f, 61.683983316540932f}, {0xc3653b0cu, 0x4276bc66u}, -1},
    UpdateStep<2>{{-0.020188079859829476f, 0.0043624175031211155f}, {0xbca56179u, 0x3b8ef29cu}, 0},
    PredictStep<2>{{-297.35401092550796f, 49.534180460524645f}, {0xc394ad50u, 0x42462300u}, -1},
    UpdateStep<2>{{-0.027923060905478161f, 0.0033629948243703912f}, {0xbce4bee7u, 0x3b5c65b1u}, 0},
    PredictStep<2>{{-462.41496993430354f, 35.812692719520619f}, {0xc3e7351eu, 0x420f4033u}, -1},
    UpdateStep<2>{{1.7704621762076593e-18f, 0.0021625597461562515f}, {0x2202a316u, 0x3b0db9bbu}, 0},
    PredictStep<1>{{-17.284694825178242f}, {0xc18a470eu}, 0},
    ScaleEvenStep<1>{{55897.677767484471f}, {0x475a59aeu}, 0},
    ScaleOddStep<1>{{1.7889830846992671e-05f}, {0x3796121fu}, 0});

template <>
struct scheme_traits<db26_tag> {
    using SchemeType = decltype(db26_scheme);
    static constexpr const char* name = "db26";
    static constexpr int id = 50;
    static constexpr int tap_size = 52;
    static constexpr int delay_even = 13;
    static constexpr int delay_odd = 13;
    static constexpr int num_steps = 29;
    static constexpr const auto& scheme = db26_scheme;
};

}  // namespace ttwv
