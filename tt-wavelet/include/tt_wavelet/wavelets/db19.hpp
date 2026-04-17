#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct db19_tag {};

inline constexpr auto db19_scheme = make_lifting_scheme(
    38,
    9,
    10,
    PredictStep<1>{{12.881149553385246f}, {0x414e1930u}, 0},
    UpdateStep<2>{{-0.075159410807509833f, -0.0013657137388346995f}, {0xbd99ed2du, 0xbab301c0u}, 0},
    PredictStep<2>{{94.707485775876592f, -60.624378996231627f}, {0x42bd6a3cu, 0xc2727f5du}, -1},
    UpdateStep<2>{{0.0042639161308487775f, -0.0029590941884320766f}, {0x3b8bb852u, 0xbb41ed5du}, 0},
    PredictStep<2>{{316.06797053295338f, -98.670369256403617f}, {0x439e08b3u, 0xc2c5573bu}, -1},
    UpdateStep<2>{{-3.6033022277885786e-05f, -0.0023750076300292113f}, {0xb817222au, 0xbb1ba604u}, 0},
    PredictStep<2>{{-304.814512904867f, -1305.725175423479f}, {0xc3986842u, 0xc4a33735u}, -1},
    UpdateStep<2>{{0.00054732982026483949f, -0.00066903912853195631f}, {0x3a0f7aafu, 0xba2f6275u}, 0},
    PredictStep<2>{{1207.0924256713108f, -1715.7689006640021f}, {0x4496e2f5u, 0xc4d6789bu}, -1},
    UpdateStep<2>{{0.00052150389814170194f, -0.00079168178805079012f}, {0x3a08b589u, 0xba4f88deu}, 0},
    PredictStep<2>{{1201.8136506378387f, -1997.7357399477216f}, {0x44963a09u, 0xc4f9b78bu}, -1},
    UpdateStep<2>{{0.00049149103180141712f, -0.00092252506604972916f}, {0x3a00d768u, 0xba71d59cu}, 0},
    PredictStep<2>{{1078.329548880344f, -2360.5296373896867f}, {0x4486ca8cu, 0xc5138879u}, -1},
    UpdateStep<2>{{0.00042317089293933126f, -0.0011167006871620081f}, {0x39dddd09u, 0xba925e42u}, 0},
    PredictStep<2>{{895.3544766791257f, -2952.5988200018469f}, {0x445fd6b0u, 0xc5388995u}, -1},
    UpdateStep<2>{{0.00033867979296810485f, -0.0014597879644390121f}, {0x39b190d5u, 0xbabf565bu}, 0},
    PredictStep<2>{{685.03046122625369f, -4123.3982447159897f}, {0x442b41f3u, 0xc580db30u}, -1},
    UpdateStep<2>{{0.00024251840816599267f, -0.0022940328551947419f}, {0x397e4c8au, 0xbb16577cu}, 0},
    PredictStep<2>{{435.91354740004522f, -8635.1767134340316f}, {0x43d9f4efu, 0xc606ecb5u}, -1},
    SwapStep{{}, {}, 0},
    PredictStep<1>{{0.00011580538918719008f}, {0x38f2dc8bu}, 0},
    ScaleEvenStep<1>{{9.5146736041093986e-06f}, {0x371fa136u}, 0},
    ScaleOddStep<1>{{-105100.82022867279f}, {0xc7cd4669u}, 0});

template <>
struct scheme_traits<db19_tag> {
    using SchemeType = decltype(db19_scheme);
    static constexpr const char* name = "db19";
    static constexpr int id = 42;
    static constexpr int tap_size = 38;
    static constexpr int delay_even = 9;
    static constexpr int delay_odd = 10;
    static constexpr int num_steps = 23;
    static constexpr const auto& scheme = db19_scheme;
};

}  // namespace ttwv
