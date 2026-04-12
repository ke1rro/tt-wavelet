#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct sym14_tag {};

inline constexpr auto sym14_scheme = make_lifting_scheme(
    28,
    7,
    7,
    PredictStep<1>{{-2.3083935448305213f}, {0xc013bcb8u}, -1},
    UpdateStep<2>{{0.044256178063431857f, 0.36475114374934603f}, {0x3d3545f7u, 0x3ebac0a9u}, 0},
    PredictStep<2>{{-3.2250626346014504f, -1.6436208172458648f}, {0xc04e676du, 0xbfd2622bu}, -1},
    UpdateStep<2>{{-0.094279121166079935f, 0.071926470115798585f}, {0xbdc11569u, 0x3d934e2fu}, 0},
    PredictStep<2>{{3.1218410000420582f, 3.623683275487187f}, {0x4047cc3eu, 0x4067ea6du}, -1},
    UpdateStep<2>{{-0.053660684705479959f, -0.089066938949500074f}, {0xbd5bcb4eu, 0xbdb668bau}, 0},
    PredictStep<2>{{-1.9886024475679511f, 2.285688870238284f}, {0xbffe8a86u, 0x401248bau}, -1},
    UpdateStep<2>{{0.20686077921705359f, 0.048120597453893459f}, {0x3e53d350u, 0x3d451a1bu}, 0},
    PredictStep<2>{{0.58950814749339453f, -3.1985158585804938f}, {0x3f16ea02u, 0xc04cb47cu}, -1},
    UpdateStep<2>{{-0.30960800310405207f, -0.10566209408734417f}, {0xbe9e84f1u, 0xbdd8655eu}, 0},
    PredictStep<2>{{56.894745162060929f, 1.1730670561985774f}, {0x42639438u, 0x3f962710u}, -1},
    UpdateStep<2>{{6.5348451358654796e-05f, -0.017563224442510694f}, {0x38890bafu, 0xbc8fe0c0u}, 0},
    PredictStep<2>{{-2592.4491603578672f, -279.03519293200111f}, {0xc5220730u, 0xc38b8481u}, -1},
    UpdateStep<2>{{0.00010881914227928249f, 0.00023755842633969938f}, {0x38e435d5u, 0x3979191bu}, 0},
    PredictStep<1>{{-2313.2166362149178f}, {0xc5109377u}, 0},
    ScaleEvenStep<1>{{63.152923520146949f}, {0x427c9c98u}, 0},
    ScaleOddStep<1>{{0.015834579687842663f}, {0x3c81b785u}, 0});

template <>
struct scheme_traits<sym14_tag> {
    using SchemeType = decltype(sym14_scheme);
    static constexpr const char* name = "sym14";
    static constexpr int id = 91;
    static constexpr int tap_size = 28;
    static constexpr int delay_even = 7;
    static constexpr int delay_odd = 7;
    static constexpr int num_steps = 17;
    static constexpr const auto& scheme = sym14_scheme;
};

}  // namespace ttwv
