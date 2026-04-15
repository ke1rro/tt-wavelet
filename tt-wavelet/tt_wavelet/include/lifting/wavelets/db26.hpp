#pragma once

#include "../scheme.hpp"

namespace ttwv {

struct db26_tag {};

inline constexpr auto db26_scheme = make_lifting_scheme(
    52,
    13,
    13,
    PredictStep<1>{ { -0.30791794533139116f }, -1 },
    UpdateStep<2>{ { -0.44282761343908694f, 0.16943380066921138f }, 0 },
    PredictStep<2>{ { -0.51619296909957235f, 0.39930564644094979f }, -1 },
    UpdateStep<2>{ { -0.62954516687942719f, 0.54351634877815935f }, 0 },
    PredictStep<2>{ { -0.70456561490213832f, 0.5953011340337947f }, -1 },
    UpdateStep<2>{ { -0.81506951202642275f, 0.67652228516249691f }, 0 },
    PredictStep<2>{ { -0.87319390745897774f, 0.70748019848117671f }, -1 },
    UpdateStep<2>{ { -0.9802273203527021f, 0.68809073123067288f }, 0 },
    PredictStep<2>{ { -1.118319741450412f, 0.27701286347794052f }, -1 },
    UpdateStep<2>{ { -2.4438948299289658f, 0.17284304409453011f }, 0 },
    PredictStep<2>{ { -3.5770917910816982f, 0.27153193342421095f }, -1 },
    UpdateStep<2>{ { 0.0051322627971474985f, 0.26570744439661453f }, 0 },
    PredictStep<2>{ { 1.5713391853181813f, 68.652367936435141f }, -1 },
    UpdateStep<2>{ { -0.014618666726126035f, 0.0072731029927715742f }, 0 },
    PredictStep<2>{ { -144.76608359808017f, 90.89578361418468f }, -1 },
    UpdateStep<2>{ { -0.011980156691732724f, 0.0068488414096477736f }, 0 },
    PredictStep<2>{ { -163.05593551247927f, 83.199573526481487f }, -1 },
    UpdateStep<2>{ { -0.013698457315341691f, 0.0061276425876670782f }, 0 },
    PredictStep<2>{ { -189.45486833629852f, 72.988473845924489f }, -1 },
    UpdateStep<2>{ { -0.016211620313866776f, 0.0052781683482031171f }, 0 },
    PredictStep<2>{ { -229.23065135462028f, 61.683983316540932f }, -1 },
    UpdateStep<2>{ { -0.020188079859829476f, 0.0043624175031211155f }, 0 },
    PredictStep<2>{ { -297.35401092550796f, 49.534180460524645f }, -1 },
    UpdateStep<2>{ { -0.027923060905478161f, 0.0033629948243703912f }, 0 },
    PredictStep<2>{ { -462.41496993430354f, 35.812692719520619f }, -1 },
    UpdateStep<2>{ { 1.7704621762076593e-18f, 0.0021625597461562515f }, 0 },
    PredictStep<1>{ { -17.284694825178242f }, 0 },
    ScaleEvenStep<1>{ { 55897.677767484471f }, 0 },
    ScaleOddStep<1>{ { 1.7889830846992671e-05f }, 0 }
);

template <>
struct scheme_traits<db26_tag> {
    using SchemeType = decltype(db26_scheme);
    static constexpr const char* name = "db26";
    static constexpr int id         = 50;
    static constexpr int tap_size   = 52;
    static constexpr int delay_even = 13;
    static constexpr int delay_odd  = 13;
    static constexpr int num_steps  = 29;
    static constexpr const auto& scheme = db26_scheme;
};

}
