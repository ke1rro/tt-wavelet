#pragma once

#include "../scheme.hpp"

namespace ttwv {

struct db29_tag {};

inline constexpr auto db29_scheme = make_lifting_scheme(
    58,
    14,
    15,
    PredictStep<1>{ { 6.2556077969234165f }, 0 },
    UpdateStep<2>{ { -0.14539087653320201f, -0.0095639085194801875f }, 0 },
    PredictStep<2>{ { 47.509844256179214f, -18.742770930794855f }, -1 },
    UpdateStep<2>{ { -0.001784464467468842f, -0.011022652811415993f }, 0 },
    PredictStep<2>{ { -20.476708484884803f, -84.248598570494877f }, -1 },
    UpdateStep<2>{ { 0.0065536061542196848f, -0.0046440142208326492f }, 0 },
    PredictStep<2>{ { 111.74140412920461f, -127.75684945295538f }, -1 },
    UpdateStep<2>{ { 0.0046711997174206732f, -0.0057010310049971577f }, 0 },
    PredictStep<2>{ { 120.27128244206924f, -149.51674861220812f }, -1 },
    UpdateStep<2>{ { 0.0051407115236466296f, -0.0067767814401327392f }, 0 },
    PredictStep<2>{ { 124.76815639249615f, 214.84087227797497f }, -1 },
    UpdateStep<2>{ { -0.0050764893904451398f, 5.355845741752987e-05f }, 0 },
    PredictStep<2>{ { 3002.0775893583041f, 411.68090436809644f }, -1 },
    UpdateStep<2>{ { 0.0003552542898035845f, -0.00031637553764263612f }, 0 },
    PredictStep<2>{ { 4010.0936665339455f, -2723.3485739396542f }, -1 },
    UpdateStep<2>{ { 0.00020991284621504951f, -0.00024560699693975492f }, 0 },
    PredictStep<2>{ { 67.229411986819386f, -4711.3648224267008f }, -1 },
    UpdateStep<2>{ { 0.00010004805759216591f, -0.010325688021008625f }, 0 },
    PredictStep<2>{ { 96.249180422775339f, -587.71763479362426f }, -1 },
    UpdateStep<2>{ { 0.0017006159279610471f, -0.0038670092514657911f }, 0 },
    PredictStep<2>{ { 258.57112320391542f, -654.83872650063006f }, -1 },
    UpdateStep<2>{ { 0.0015270690025670649f, -0.0044920647246490036f }, 0 },
    PredictStep<2>{ { 222.61436623980643f, -773.99913539252907f }, -1 },
    UpdateStep<2>{ { 0.0012919909087183046f, -0.0054227907705018509f }, 0 },
    PredictStep<2>{ { 184.4068918745792f, -961.07148750244335f }, -1 },
    UpdateStep<2>{ { 0.0010405053241954714f, -0.0070119060252413299f }, 0 },
    PredictStep<2>{ { 142.61457532276452f, -1324.8981084290226f }, -1 },
    UpdateStep<2>{ { 0.00075477502280205718f, -0.010867975715585806f }, 0 },
    PredictStep<2>{ { 92.013455510937135f, -2736.1319260991695f }, -1 },
    SwapStep{ {}, 0 },
    PredictStep<1>{ { 0.00036547945311455558f }, 0 },
    ScaleEvenStep<1>{ { -4.943419141279731e-07f }, 0 },
    ScaleOddStep<1>{ { 2022891.3863474752f }, 0 }
);

template <>
struct scheme_traits<db29_tag> {
    using SchemeType = decltype(db29_scheme);
    static constexpr const char* name = "db29";
    static constexpr int id         = 53;
    static constexpr int tap_size   = 58;
    static constexpr int delay_even = 14;
    static constexpr int delay_odd  = 15;
    static constexpr int num_steps  = 33;
    static constexpr const auto& scheme = db29_scheme;
};

}
