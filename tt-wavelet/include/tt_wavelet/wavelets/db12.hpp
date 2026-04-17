#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct db12_tag {};

inline constexpr auto db12_scheme = make_lifting_scheme(
    24,
    6,
    6,
    PredictStep<1>{{-0.11967421811249769f}, {0xbdf517c2u}, -1},
    UpdateStep<2>{{-0.36680379770882249f, 0.11798445371722027f}, {0xbebbcdb5u, 0x3df1a1d5u}, 0},
    PredictStep<2>{{-0.59592431487647668f, 0.33149764333372977f}, {0xbf188e7fu, 0x3ea9ba0fu}, -1},
    UpdateStep<2>{{-0.82314354286488767f, 0.51865436620416794f}, {0xbf52b989u, 0x3f04c688u}, 0},
    PredictStep<2>{{-1.0032487583850327f, 0.64363456495666793f}, {0xbf806a75u, 0x3f24c53cu}, -1},
    UpdateStep<2>{{-1.1900515768631985f, 0.73007868519888097f}, {0xbf98539cu, 0x3f3ae670u}, 0},
    PredictStep<2>{{-1.33599513381916f, 0.73873923878783743f}, {0xbfab01e3u, 0x3f3d1e04u}, -1},
    UpdateStep<2>{{-1.5397695309437902f, 0.71831182425503004f}, {0xbfc5172bu, 0x3f37e349u}, 0},
    PredictStep<2>{{-1.7618599488552713f, 0.6435606236085617f}, {0xbfe184a0u, 0x3f24c064u}, -1},
    UpdateStep<2>{{-2.1649521242289773f, 0.56689008390782403f}, {0xc00a8e93u, 0x3f111fb5u}, 0},
    PredictStep<2>{{-2.803251578457794f, 0.4618653266319796f}, {0xc0336879u, 0x3eec799du}, -1},
    UpdateStep<2>{{6.2700299224856329e-08f, 0.35672778566100999f}, {0x3386a5dbu, 0x3eb6a506u}, 0},
    PredictStep<1>{{-0.21688581342080118f}, {0xbe5e1751u}, 0},
    ScaleEvenStep<1>{{43.12611945993504f}, {0x422c8125u}, 0},
    ScaleOddStep<1>{{0.023187803876697471f}, {0x3cbdf459u}, 0});

template <>
struct scheme_traits<db12_tag> {
    using SchemeType = decltype(db12_scheme);
    static constexpr const char* name = "db12";
    static constexpr int id = 35;
    static constexpr int tap_size = 24;
    static constexpr int delay_even = 6;
    static constexpr int delay_odd = 6;
    static constexpr int num_steps = 15;
    static constexpr const auto& scheme = db12_scheme;
};

}  // namespace ttwv
