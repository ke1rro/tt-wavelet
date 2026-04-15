#pragma once

#include "../scheme.hpp"

namespace ttwv {

struct db28_tag {};

inline constexpr auto db28_scheme = make_lifting_scheme(
    56,
    14,
    14,
    PredictStep<1>{ { -0.53530364886318982f }, -1 },
    UpdateStep<2>{ { -0.58711802344597197f, 0.15217894218045955f }, 0 },
    PredictStep<2>{ { -0.38625655277786169f, 0.36267847440778356f }, -1 },
    UpdateStep<2>{ { -0.37084055441495356f, 0.54709361027222325f }, 0 },
    PredictStep<2>{ { -0.51141342887053109f, 0.56169246050930166f }, -1 },
    UpdateStep<2>{ { -0.74528038211926761f, 0.65911287374932814f }, 0 },
    PredictStep<2>{ { -0.80205595737591506f, 0.6713633444217918f }, -1 },
    UpdateStep<2>{ { -0.9386891402817884f, 0.77105365827169248f }, 0 },
    PredictStep<2>{ { -0.942541831818972f, 0.75990329974006832f }, -1 },
    UpdateStep<2>{ { -1.0729514780800964f, 0.83747036675672981f }, 0 },
    PredictStep<2>{ { -1.065618678978131f, 0.68206057426776245f }, -1 },
    UpdateStep<2>{ { -1.3839972956761748f, 0.21373716154232639f }, 0 },
    PredictStep<2>{ { -0.61476062150923094f, 0.16625768622617648f }, -1 },
    UpdateStep<2>{ { -0.013298174482636796f, 0.65532105456810053f }, 0 },
    PredictStep<2>{ { -1.1324406262895566f, 1.4560074049591982f }, -1 },
    UpdateStep<2>{ { -0.71438369809105728f, 0.46032555359591071f }, 0 },
    PredictStep<2>{ { -2.3467521354903487f, 1.380397212457563f }, -1 },
    UpdateStep<2>{ { -0.80083643637847757f, 0.4246597802035651f }, 0 },
    PredictStep<2>{ { -2.6508568486116606f, 1.2474936336966507f }, -1 },
    UpdateStep<2>{ { -0.91688476442869304f, 0.37715612045388092f }, 0 },
    PredictStep<2>{ { -3.081000204598789f, 1.090610262420431f }, -1 },
    UpdateStep<2>{ { -1.0843187388326063f, 0.32456845643488763f }, 0 },
    PredictStep<2>{ { -3.7223454760766739f, 0.92223767722833938f }, -1 },
    UpdateStep<2>{ { -1.3476750923044902f, 0.26864781401169108f }, 0 },
    PredictStep<2>{ { -4.8180865642740605f, 0.74201861021437443f }, -1 },
    UpdateStep<2>{ { -1.8598058796095529f, 0.20755127302849599f }, 0 },
    PredictStep<2>{ { -7.4755156017438527f, 0.5376905251045061f }, -1 },
    UpdateStep<2>{ { 7.0250799452919618e-18f, 0.13377003718201369f }, 0 },
    PredictStep<1>{ { -0.2600956094272952f }, 0 },
    ScaleEvenStep<1>{ { 13871.404541741156f }, 0 },
    ScaleOddStep<1>{ { 7.2090753102243445e-05f }, 0 }
);

template <>
struct scheme_traits<db28_tag> {
    using SchemeType = decltype(db28_scheme);
    static constexpr const char* name = "db28";
    static constexpr int id         = 52;
    static constexpr int tap_size   = 56;
    static constexpr int delay_even = 14;
    static constexpr int delay_odd  = 14;
    static constexpr int num_steps  = 31;
    static constexpr const auto& scheme = db28_scheme;
};

}
