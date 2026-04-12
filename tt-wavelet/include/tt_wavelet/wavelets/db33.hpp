#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct db33_tag {};

inline constexpr auto db33_scheme = make_lifting_scheme(
    66,
    16,
    17,
    PredictStep<1>{{4.2357714448187158f}, {0x40878b71u}, 0},
    UpdateStep<2>{{-0.21195549465751357f, -0.029074639996635603f}, {0xbe590adcu, 0xbcee2df0u}, 0},
    PredictStep<2>{{9.8932078251416442f, -10.831527801129132f}, {0x411e4a94u, 0xc12d4df0u}, -1},
    UpdateStep<2>{{0.034594681668146283f, -0.028975927843083973f}, {0x3d0db327u, 0xbced5eedu}, 0},
    PredictStep<2>{{12.907278533617816f, -7.5539051213152186f}, {0x414e8436u, 0xc0f1b997u}, -1},
    UpdateStep<2>{{0.044829431240088091f, -0.025400298468783073f}, {0x3d379f11u, 0xbcd01449u}, 0},
    PredictStep<2>{{14.90011693354964f, -10.428033404814805f}, {0x416e66e1u, 0xc126d93au}, -1},
    UpdateStep<2>{{0.051325783533251963f, -0.039322283394074002f}, {0x3d523afcu, 0xbd211067u}, 0},
    PredictStep<2>{{39.464512977756442f, -13.294663985221256f}, {0x421ddba9u, 0xc154b6f2u}, -1},
    UpdateStep<2>{{-0.00090229836447291935f, -0.022344806536200208f}, {0xba6c8838u, 0xbcb70c75u}, 0},
    PredictStep<2>{{-42.725342534879928f, -489.50812416229343f}, {0xc22ae6c0u, 0xc3f4c10au}, -1},
    UpdateStep<2>{{0.0018006845190256637f, -0.0013056196147095136f}, {0x3aec04f2u, 0xbaab2153u}, 0},
    PredictStep<2>{{666.4061371015016f, -824.83964259727827f}, {0x442699feu, 0xc44e35bdu}, -1},
    UpdateStep<2>{{0.0011071531693948493f, -0.0014630532363956654f}, {0x3a911de5u, 0xbabfc3ecu}, 0},
    PredictStep<2>{{648.03195868105956f, -1347.208891519876f}, {0x4422020cu, 0xc4a866afu}, -1},
    UpdateStep<2>{{0.00072830518217835395f, 3.1413647883518602e-05f}, {0x3a3eebbcu, 0x3803c226u}, 0},
    PredictStep<2>{{-46105.007742771617f, 636.23165485919446f}, {0xc7341902u, 0x441f0ed3u}, -1},
    UpdateStep<2>{{7.6901195362909662e-05f, 2.2332930817844923e-05f}, {0x38a14604u, 0x37bb579bu}, 0},
    PredictStep<2>{{19828.593362425931f, -12933.816075014971f}, {0x469ae930u, 0xc64a1744u}, -1},
    UpdateStep<2>{{-2.6421618116832556e-05f, -5.0326311220888503e-05f}, {0xb7dda3feu, 0xb8531577u}, 0},
    PredictStep<2>{{-152.02298544388702f, 37930.668976140129f}, {0xc31805e2u, 0x47142aabu}, -1},
    UpdateStep<2>{{-3.3977528224327719e-05f, 0.0089972098498205689f}, {0xb80e8318u, 0x3c136909u}, 0},
    PredictStep<2>{{-111.19192214164994f, -702.44027440342381f}, {0xc2de6244u, 0xc42f9c2du}, -1},
    UpdateStep<2>{{0.0014235672307349523f, -0.0035053073566306096f}, {0x3aba96fdu, 0xbb65b94cu}, 0},
    PredictStep<2>{{285.2803796568611f, -839.60501961793989f}, {0x438ea3e3u, 0xc451e6b9u}, -1},
    UpdateStep<2>{{0.001191035602958748f, -0.0040690025409124645f}, {0x3a9c1c86u, 0xbb855544u}, 0},
    PredictStep<2>{{245.76046453729631f, -989.77355142604233f}, {0x4375c2aeu, 0xc4777182u}, -1},
    UpdateStep<2>{{0.0010103321065346421f, -0.0048962204167357264f}, {0x3a846d1fu, 0xbba07079u}, 0},
    PredictStep<2>{{204.23917119067571f, -1224.656242212053f}, {0x434c3d3au, 0xc4991500u}, -1},
    UpdateStep<2>{{0.00081655567132081486f, -0.00630784434781991f}, {0x3a560e20u, 0xbbceb209u}, 0},
    PredictStep<2>{{158.53276410435845f, -1682.051212386408f}, {0x431e8863u, 0xc4d241a4u}, -1},
    UpdateStep<2>{{0.00059451221974463596f, -0.0097412054333906838f}, {0x3a1bd90au, 0xbc1f9994u}, 0},
    PredictStep<2>{{102.65669960847173f, -3461.3990719250919f}, {0x42cd503bu, 0xc5585663u}, -1},
    SwapStep{{}, {}, 0},
    PredictStep<1>{{0.00028890052236705546f}, {0x39977792u}, 0},
    ScaleEvenStep<1>{{-1.0774208983879711e-07f}, {0xb3e75fd7u}, 0},
    ScaleOddStep<1>{{9281423.8288508449f}, {0x4b0d9f90u}, 0});

template <>
struct scheme_traits<db33_tag> {
    using SchemeType = decltype(db33_scheme);
    static constexpr const char* name = "db33";
    static constexpr int id = 58;
    static constexpr int tap_size = 66;
    static constexpr int delay_even = 16;
    static constexpr int delay_odd = 17;
    static constexpr int num_steps = 37;
    static constexpr const auto& scheme = db33_scheme;
};

}  // namespace ttwv
