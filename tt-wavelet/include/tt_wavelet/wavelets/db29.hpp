#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct db29_tag {};

inline constexpr auto db29_scheme = make_lifting_scheme(
    58,
    14,
    15,
    PredictStep<1>{{6.2556077969234165f}, {0x40c82df0u}, 0},
    UpdateStep<2>{{-0.14539087653320201f, -0.0095639085194801875f}, {0xbe14e159u, 0xbc1cb1f1u}, 0},
    PredictStep<2>{{47.509844256179214f, -18.742770930794855f}, {0x423e0a15u, 0xc195f132u}, -1},
    UpdateStep<2>{{-0.001784464467468842f, -0.011022652811415993f}, {0xbae9e4b1u, 0xbc34985bu}, 0},
    PredictStep<2>{{-20.476708484884803f, -84.248598570494877f}, {0xc1a3d04du, 0xc2a87f48u}, -1},
    UpdateStep<2>{{0.0065536061542196848f, -0.0046440142208326492f}, {0x3bd6bfa2u, 0xbb982cd1u}, 0},
    PredictStep<2>{{111.74140412920461f, -127.75684945295538f}, {0x42df7b99u, 0xc2ff8382u}, -1},
    UpdateStep<2>{{0.0046711997174206732f, -0.0057010310049971577f}, {0x3b9910ddu, 0xbbbacfb7u}, 0},
    PredictStep<2>{{120.27128244206924f, -149.51674861220812f}, {0x42f08ae6u, 0xc315844au}, -1},
    UpdateStep<2>{{0.0051407115236466296f, -0.0067767814401327392f}, {0x3ba8736au, 0xbbde0fc3u}, 0},
    PredictStep<2>{{124.76815639249615f, 214.84087227797497f}, {0x42f9894cu, 0x4356d743u}, -1},
    UpdateStep<2>{{-0.0050764893904451398f, 5.355845741752987e-05f}, {0xbba658aeu, 0x3860a3f5u}, 0},
    PredictStep<2>{{3002.0775893583041f, 411.68090436809644f}, {0x453ba13eu, 0x43cdd728u}, -1},
    UpdateStep<2>{{0.0003552542898035845f, -0.00031637553764263612f}, {0x39ba416cu, 0xb9a5df35u}, 0},
    PredictStep<2>{{4010.0936665339455f, -2723.3485739396542f}, {0x457aa180u, 0xc52a3594u}, -1},
    UpdateStep<2>{{0.00020991284621504951f, -0.00024560699693975492f}, {0x395c1c0du, 0xb980c4d0u}, 0},
    PredictStep<2>{{67.229411986819386f, -4711.3648224267008f}, {0x42867575u, 0xc5933aebu}, -1},
    UpdateStep<2>{{0.00010004805759216591f, -0.010325688021008625f}, {0x38d1d0e4u, 0xbc292d13u}, 0},
    PredictStep<2>{{96.249180422775339f, -587.71763479362426f}, {0x42c07f95u, 0xc412edeeu}, -1},
    UpdateStep<2>{{0.0017006159279610471f, -0.0038670092514657911f}, {0x3adee734u, 0xbb7d6da6u}, 0},
    PredictStep<2>{{258.57112320391542f, -654.83872650063006f}, {0x4381491bu, 0xc423b5aeu}, -1},
    UpdateStep<2>{{0.0015270690025670649f, -0.0044920647246490036f}, {0x3ac827efu, 0xbb93322cu}, 0},
    PredictStep<2>{{222.61436623980643f, -773.99913539252907f}, {0x435e9d47u, 0xc4417ff2u}, -1},
    UpdateStep<2>{{0.0012919909087183046f, -0.0054227907705018509f}, {0x3aa95805u, 0xbbb1b1abu}, 0},
    PredictStep<2>{{184.4068918745792f, -961.07148750244335f}, {0x4338682au, 0xc4704493u}, -1},
    UpdateStep<2>{{0.0010405053241954714f, -0.0070119060252413299f}, {0x3a886191u, 0xbbe5c422u}, 0},
    PredictStep<2>{{142.61457532276452f, -1324.8981084290226f}, {0x430e9d55u, 0xc4a59cbdu}, -1},
    UpdateStep<2>{{0.00075477502280205718f, -0.010867975715585806f}, {0x3a45dc18u, 0xbc320f98u}, 0},
    PredictStep<2>{{92.013455510937135f, -2736.1319260991695f}, {0x42b806e4u, 0xc52b021cu}, -1},
    SwapStep{{}, {}, 0},
    PredictStep<1>{{0.00036547945311455558f}, {0x39bf9dd2u}, 0},
    ScaleEvenStep<1>{{-4.943419141279731e-07f}, {0xb504b2ebu}, 0},
    ScaleOddStep<1>{{2022891.3863474752f}, {0x49f6ef5bu}, 0});

template <>
struct scheme_traits<db29_tag> {
    using SchemeType = decltype(db29_scheme);
    static constexpr const char* name = "db29";
    static constexpr int id = 53;
    static constexpr int tap_size = 58;
    static constexpr int delay_even = 14;
    static constexpr int delay_odd = 15;
    static constexpr int num_steps = 33;
    static constexpr const auto& scheme = db29_scheme;
};

}  // namespace ttwv
