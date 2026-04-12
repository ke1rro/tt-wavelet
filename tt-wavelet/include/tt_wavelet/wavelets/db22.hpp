#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct db22_tag {};

inline constexpr auto db22_scheme = make_lifting_scheme(
    44,
    11,
    11,
    PredictStep<1>{{-0.36557896835151732f}, {0xbebb2d2bu}, -1},
    UpdateStep<2>{{-0.52529428031802827f, 0.3112316219328683f}, {0xbf0679b0u, 0x3e9f59c0u}, 0},
    PredictStep<2>{{-0.56179128424079239f, -4.2066594467040268f}, {0xbf0fd18eu, 0xc0869cf4u}, -1},
    UpdateStep<2>{{0.29448865001113733f, -0.0012542378228807621f}, {0x3e96c737u, 0xbaa4653du}, 0},
    PredictStep<2>{{-10.774740636442822f, -11.893363848762659f}, {0xc12c6556u, 0xc13e4b38u}, -1},
    UpdateStep<2>{{-0.080773678978255672f, 0.049840303899345371f}, {0xbda56cacu, 0x3d4c2559u}, 0},
    PredictStep<2>{{-17.761953642289662f, 7.3909306836485742f}, {0xc18e187bu, 0x40ec8281u}, -1},
    UpdateStep<2>{{0.006502492039799441f, 0.041230088224165129f}, {0x3bd512dbu, 0x3d28e0e2u}, 0},
    PredictStep<2>{{1.5953971383466559f, 71.351352202275436f}, {0x3fcc35f9u, 0x428eb3e4u}, -1},
    UpdateStep<2>{{-0.012365484754481779f, 0.0057225751652041868f}, {0xbc4a989au, 0x3bbb8471u}, 0},
    PredictStep<2>{{-161.48112216129343f, 112.75131761267052f}, {0xc3217b2bu, 0x42e180adu}, -1},
    UpdateStep<2>{{-0.0089011631235751641f, 0.0058255091716807247f}, {0xbc11d62fu, 0x3bbee3eau}, 0},
    PredictStep<2>{{-182.80112472501656f, 109.23314224649263f}, {0xc336cd17u, 0x42da775eu}, -1},
    UpdateStep<2>{{-0.010173033631892616f, 0.0054152910820283772f}, {0xbc26acccu, 0x3bb172c1u}, 0},
    PredictStep<2>{{-211.89938055279262f, 98.016312497601589f}, {0xc353e63eu, 0x42c4085au}, -1},
    UpdateStep<2>{{-0.012028525999239321f, 0.0047163096211582482f}, {0xbc45134bu, 0x3b9a8b46u}, 0},
    PredictStep<2>{{-256.7696256543129f, 83.127816982535805f}, {0xc3806283u, 0x42a64171u}, -1},
    UpdateStep<2>{{-0.015030716959180625f, 0.0038945029066282891f}, {0xbc764365u, 0x3b7f3aebu}, 0},
    PredictStep<2>{{-334.62950619389466f, 66.530381415216581f}, {0xc3a75094u, 0x42850f8eu}, -1},
    UpdateStep<2>{{-0.020899075749656687f, 0.0029883795513981522f}, {0xbcab348au, 0x3b43d8b0u}, 0},
    PredictStep<2>{{-523.23620066131934f, 47.849005929208587f}, {0xc402cf1eu, 0x423f6562u}, -1},
    UpdateStep<2>{{3.7864016416070248e-16f, 0.0019111827483105804f}, {0x25da456du, 0x3afa80a7u}, 0},
    PredictStep<1>{{-22.967903959970986f}, {0xc1b7be44u}, 0},
    ScaleEvenStep<1>{{-15693.639455406194f}, {0xc675368fu}, 0},
    ScaleOddStep<1>{{-6.3720082447511363e-05f}, {0xb885a175u}, 0});

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
