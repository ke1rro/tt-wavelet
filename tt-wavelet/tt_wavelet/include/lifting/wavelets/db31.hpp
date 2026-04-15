#pragma once

#include "../scheme.hpp"

namespace ttwv {

struct db31_tag {};

inline constexpr auto db31_scheme = make_lifting_scheme(
    62,
    15,
    16,
    PredictStep<1>{ { 6.7999480975072508f }, 0 },
    UpdateStep<2>{ { -0.13669849028236414f, -0.0076161462816134173f }, 0 },
    PredictStep<2>{ { 28.117429703516081f, -21.376535696406219f }, -1 },
    UpdateStep<2>{ { 0.014488184589878179f, -0.011437324882576694f }, 0 },
    PredictStep<2>{ { 33.227297647912842f, -27.385748649932623f }, -1 },
    UpdateStep<2>{ { 0.014266397271637261f, -0.014215106573958311f }, 0 },
    PredictStep<2>{ { 15.297641054077108f, -35.789151856737249f }, -1 },
    UpdateStep<2>{ { 0.0057000145319884175f, -0.023953623314032495f }, 0 },
    PredictStep<2>{ { 20.897723898535087f, -41.364473614649334f }, -1 },
    UpdateStep<2>{ { 0.016918708166313464f, -0.022269515965861228f }, 0 },
    PredictStep<2>{ { 35.419878788192527f, -44.624158016075874f }, -1 },
    UpdateStep<2>{ { 0.019137324100687306f, -0.024184349936321913f }, 0 },
    PredictStep<2>{ { 37.388795953492036f, -27.491130237742407f }, -1 },
    UpdateStep<2>{ { 0.032811135342873776f, -0.0018711056173121633f }, 0 },
    PredictStep<2>{ { 231.37129093230732f, -13.679277857301795f }, -1 },
    UpdateStep<2>{ { 0.0098803464902259327f, -0.0041023994204559908f }, 0 },
    PredictStep<2>{ { 161.35450048426688f, -99.699928450526187f }, -1 },
    UpdateStep<2>{ { -0.05373982556748836f, -0.0061609196034318253f }, 0 },
    PredictStep<2>{ { -0.017887826139852143f, 18.615565073729034f }, -1 },
    UpdateStep<2>{ { -0.32555919547579948f, 71.553415107305639f }, 0 },
    PredictStep<2>{ { -0.014003647757496294f, -0.10910320630120229f }, -1 },
    UpdateStep<2>{ { 9.1645168288802523f, -20.694833366850975f }, 0 },
    PredictStep<2>{ { 0.048320148091100426f, -0.13211236484651345f }, -1 },
    UpdateStep<2>{ { 7.5692913518714642f, -24.051718329673232f }, 0 },
    PredictStep<2>{ { 0.041577058539438966f, -0.15595785351895999f }, -1 },
    UpdateStep<2>{ { 6.4119885993802708f, -28.987730146567262f }, 0 },
    PredictStep<2>{ { 0.034497354360067879f, -0.19329826822656018f }, -1 },
    UpdateStep<2>{ { 5.173352090225686f, -37.410567640232017f }, 0 },
    PredictStep<2>{ { 0.026730415042511894f, -0.26595747299445321f }, -1 },
    UpdateStep<2>{ { 3.7599996297936311f, -57.872581570187904f }, 0 },
    PredictStep<2>{ { 0.017279339764499694f, -0.54821647249577776f }, -1 },
    SwapStep{ {}, 0 },
    PredictStep<1>{ { 1.8240969583556281f }, 0 },
    ScaleEvenStep<1>{ { 1.7285256822476851e-05f }, 0 },
    ScaleOddStep<1>{ { -57852.770732318648f }, 0 }
);

template <>
struct scheme_traits<db31_tag> {
    using SchemeType = decltype(db31_scheme);
    static constexpr const char* name = "db31";
    static constexpr int id         = 56;
    static constexpr int tap_size   = 62;
    static constexpr int delay_even = 15;
    static constexpr int delay_odd  = 16;
    static constexpr int num_steps  = 35;
    static constexpr const auto& scheme = db31_scheme;
};

}
