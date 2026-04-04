#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct db19_tag {};

inline constexpr auto db19_scheme = make_lifting_scheme(
    38,
    9,
    10,
    PredictStep<1>{{12.881149553385246f}, 0},
    UpdateStep<2>{{-0.075159410807509833f, -0.0013657137388346995f}, 0},
    PredictStep<2>{{94.707485775876592f, -60.624378996231627f}, -1},
    UpdateStep<2>{{0.0042639161308487775f, -0.0029590941884320766f}, 0},
    PredictStep<2>{{316.06797053295338f, -98.670369256403617f}, -1},
    UpdateStep<2>{{-3.6033022277885786e-05f, -0.0023750076300292113f}, 0},
    PredictStep<2>{{-304.814512904867f, -1305.725175423479f}, -1},
    UpdateStep<2>{{0.00054732982026483949f, -0.00066903912853195631f}, 0},
    PredictStep<2>{{1207.0924256713108f, -1715.7689006640021f}, -1},
    UpdateStep<2>{{0.00052150389814170194f, -0.00079168178805079012f}, 0},
    PredictStep<2>{{1201.8136506378387f, -1997.7357399477216f}, -1},
    UpdateStep<2>{{0.00049149103180141712f, -0.00092252506604972916f}, 0},
    PredictStep<2>{{1078.329548880344f, -2360.5296373896867f}, -1},
    UpdateStep<2>{{0.00042317089293933126f, -0.0011167006871620081f}, 0},
    PredictStep<2>{{895.3544766791257f, -2952.5988200018469f}, -1},
    UpdateStep<2>{{0.00033867979296810485f, -0.0014597879644390121f}, 0},
    PredictStep<2>{{685.03046122625369f, -4123.3982447159897f}, -1},
    UpdateStep<2>{{0.00024251840816599267f, -0.0022940328551947419f}, 0},
    PredictStep<2>{{435.91354740004522f, -8635.1767134340316f}, -1},
    SwapStep{{}, 0},
    PredictStep<1>{{0.00011580538918719008f}, 0},
    ScaleEvenStep<1>{{9.5146736041093986e-06f}, 0},
    ScaleOddStep<1>{{-105100.82022867279f}, 0});

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
