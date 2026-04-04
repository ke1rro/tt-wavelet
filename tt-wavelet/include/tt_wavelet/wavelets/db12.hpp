#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct db12_tag {};

inline constexpr auto db12_scheme = make_lifting_scheme(
    24,
    6,
    6,
    PredictStep<1>{{-0.11967421811249769f}, -1},
    UpdateStep<2>{{-0.36680379770882249f, 0.11798445371722027f}, 0},
    PredictStep<2>{{-0.59592431487647668f, 0.33149764333372977f}, -1},
    UpdateStep<2>{{-0.82314354286488767f, 0.51865436620416794f}, 0},
    PredictStep<2>{{-1.0032487583850327f, 0.64363456495666793f}, -1},
    UpdateStep<2>{{-1.1900515768631985f, 0.73007868519888097f}, 0},
    PredictStep<2>{{-1.33599513381916f, 0.73873923878783743f}, -1},
    UpdateStep<2>{{-1.5397695309437902f, 0.71831182425503004f}, 0},
    PredictStep<2>{{-1.7618599488552713f, 0.6435606236085617f}, -1},
    UpdateStep<2>{{-2.1649521242289773f, 0.56689008390782403f}, 0},
    PredictStep<2>{{-2.803251578457794f, 0.4618653266319796f}, -1},
    UpdateStep<2>{{6.2700299224856329e-08f, 0.35672778566100999f}, 0},
    PredictStep<1>{{-0.21688581342080118f}, 0},
    ScaleEvenStep<1>{{43.12611945993504f}, 0},
    ScaleOddStep<1>{{0.023187803876697471f}, 0});

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
