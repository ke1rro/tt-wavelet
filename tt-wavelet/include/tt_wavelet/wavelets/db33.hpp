#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct db33_tag {};

inline constexpr auto db33_scheme = make_lifting_scheme(
    66,
    16,
    17,
    PredictStep<1>{{4.2357714448187158f}, 0},
    UpdateStep<2>{{-0.21195549465751357f, -0.029074639996635603f}, 0},
    PredictStep<2>{{9.8932078251416442f, -10.831527801129132f}, -1},
    UpdateStep<2>{{0.034594681668146283f, -0.028975927843083973f}, 0},
    PredictStep<2>{{12.907278533617816f, -7.5539051213152186f}, -1},
    UpdateStep<2>{{0.044829431240088091f, -0.025400298468783073f}, 0},
    PredictStep<2>{{14.90011693354964f, -10.428033404814805f}, -1},
    UpdateStep<2>{{0.051325783533251963f, -0.039322283394074002f}, 0},
    PredictStep<2>{{39.464512977756442f, -13.294663985221256f}, -1},
    UpdateStep<2>{{-0.00090229836447291935f, -0.022344806536200208f}, 0},
    PredictStep<2>{{-42.725342534879928f, -489.50812416229343f}, -1},
    UpdateStep<2>{{0.0018006845190256637f, -0.0013056196147095136f}, 0},
    PredictStep<2>{{666.4061371015016f, -824.83964259727827f}, -1},
    UpdateStep<2>{{0.0011071531693948493f, -0.0014630532363956654f}, 0},
    PredictStep<2>{{648.03195868105956f, -1347.208891519876f}, -1},
    UpdateStep<2>{{0.00072830518217835395f, 3.1413647883518602e-05f}, 0},
    PredictStep<2>{{-46105.007742771617f, 636.23165485919446f}, -1},
    UpdateStep<2>{{7.6901195362909662e-05f, 2.2332930817844923e-05f}, 0},
    PredictStep<2>{{19828.593362425931f, -12933.816075014971f}, -1},
    UpdateStep<2>{{-2.6421618116832556e-05f, -5.0326311220888503e-05f}, 0},
    PredictStep<2>{{-152.02298544388702f, 37930.668976140129f}, -1},
    UpdateStep<2>{{-3.3977528224327719e-05f, 0.0089972098498205689f}, 0},
    PredictStep<2>{{-111.19192214164994f, -702.44027440342381f}, -1},
    UpdateStep<2>{{0.0014235672307349523f, -0.0035053073566306096f}, 0},
    PredictStep<2>{{285.2803796568611f, -839.60501961793989f}, -1},
    UpdateStep<2>{{0.001191035602958748f, -0.0040690025409124645f}, 0},
    PredictStep<2>{{245.76046453729631f, -989.77355142604233f}, -1},
    UpdateStep<2>{{0.0010103321065346421f, -0.0048962204167357264f}, 0},
    PredictStep<2>{{204.23917119067571f, -1224.656242212053f}, -1},
    UpdateStep<2>{{0.00081655567132081486f, -0.00630784434781991f}, 0},
    PredictStep<2>{{158.53276410435845f, -1682.051212386408f}, -1},
    UpdateStep<2>{{0.00059451221974463596f, -0.0097412054333906838f}, 0},
    PredictStep<2>{{102.65669960847173f, -3461.3990719250919f}, -1},
    SwapStep{{}, 0},
    PredictStep<1>{{0.00028890052236705546f}, 0},
    ScaleEvenStep<1>{{-1.0774208983879711e-07f}, 0},
    ScaleOddStep<1>{{9281423.8288508449f}, 0});

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
