#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct db22_tag {};

inline constexpr auto db22_scheme = make_lifting_scheme(
    44,
    11,
    11,
    PredictStep<1>{{-0.36557896835151732f}, -1},
    UpdateStep<2>{{-0.52529428031802827f, 0.3112316219328683f}, 0},
    PredictStep<2>{{-0.56179128424079239f, -4.2066594467040268f}, -1},
    UpdateStep<2>{{0.29448865001113733f, -0.0012542378228807621f}, 0},
    PredictStep<2>{{-10.774740636442822f, -11.893363848762659f}, -1},
    UpdateStep<2>{{-0.080773678978255672f, 0.049840303899345371f}, 0},
    PredictStep<2>{{-17.761953642289662f, 7.3909306836485742f}, -1},
    UpdateStep<2>{{0.006502492039799441f, 0.041230088224165129f}, 0},
    PredictStep<2>{{1.5953971383466559f, 71.351352202275436f}, -1},
    UpdateStep<2>{{-0.012365484754481779f, 0.0057225751652041868f}, 0},
    PredictStep<2>{{-161.48112216129343f, 112.75131761267052f}, -1},
    UpdateStep<2>{{-0.0089011631235751641f, 0.0058255091716807247f}, 0},
    PredictStep<2>{{-182.80112472501656f, 109.23314224649263f}, -1},
    UpdateStep<2>{{-0.010173033631892616f, 0.0054152910820283772f}, 0},
    PredictStep<2>{{-211.89938055279262f, 98.016312497601589f}, -1},
    UpdateStep<2>{{-0.012028525999239321f, 0.0047163096211582482f}, 0},
    PredictStep<2>{{-256.7696256543129f, 83.127816982535805f}, -1},
    UpdateStep<2>{{-0.015030716959180625f, 0.0038945029066282891f}, 0},
    PredictStep<2>{{-334.62950619389466f, 66.530381415216581f}, -1},
    UpdateStep<2>{{-0.020899075749656687f, 0.0029883795513981522f}, 0},
    PredictStep<2>{{-523.23620066131934f, 47.849005929208587f}, -1},
    UpdateStep<2>{{3.7864016416070248e-16f, 0.0019111827483105804f}, 0},
    PredictStep<1>{{-22.967903959970986f}, 0},
    ScaleEvenStep<1>{{-15693.639455406194f}, 0},
    ScaleOddStep<1>{{-6.3720082447511363e-05f}, 0});

template <>
struct scheme_traits<db22_tag> {
    using SchemeType = decltype(db22_scheme);
    static constexpr const char* name = "db22";
    static constexpr int id = 46;
    static constexpr int tap_size = 44;
    static constexpr int delay_even = 11;
    static constexpr int delay_odd = 11;
    static constexpr int num_steps = 25;
    static constexpr const auto& scheme = db22_scheme;
};

}  // namespace ttwv
