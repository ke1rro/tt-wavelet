#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct db28_tag {};

inline constexpr auto db28_scheme = make_lifting_scheme(
    56,
    14,
    14,
    PredictStep<1>{{-0.53530364886318982f}, {0xbf0909a9u}, -1},
    UpdateStep<2>{{-0.58711802344597197f, 0.15217894218045955f}, {0xbf164d5eu, 0x3e1bd4ccu}, 0},
    PredictStep<2>{{-0.38625655277786169f, 0.36267847440778356f}, {0xbec5c36bu, 0x3eb9b0feu}, -1},
    UpdateStep<2>{{-0.37084055441495356f, 0.54709361027222325f}, {0xbebdded0u, 0x3f0c0e54u}, 0},
    PredictStep<2>{{-0.51141342887053109f, 0.56169246050930166f}, {0xbf02ebfeu, 0x3f0fcb14u}, -1},
    UpdateStep<2>{{-0.74528038211926761f, 0.65911287374932814f}, {0xbf3ecab2u, 0x3f28bb9fu}, 0},
    PredictStep<2>{{-0.80205595737591506f, 0.6713633444217918f}, {0xbf4d538au, 0x3f2bde78u}, -1},
    UpdateStep<2>{{-0.9386891402817884f, 0.77105365827169248f}, {0xbf704deeu, 0x3f4563c6u}, 0},
    PredictStep<2>{{-0.942541831818972f, 0.75990329974006832f}, {0xbf714a6cu, 0x3f428906u}, -1},
    UpdateStep<2>{{-1.0729514780800964f, 0.83747036675672981f}, {0xbf895679u, 0x3f566475u}, 0},
    PredictStep<2>{{-1.065618678978131f, 0.68206057426776245f}, {0xbf886631u, 0x3f2e9b86u}, -1},
    UpdateStep<2>{{-1.3839972956761748f, 0.21373716154232639f}, {0xbfb126d3u, 0x3e5addeau}, 0},
    PredictStep<2>{{-0.61476062150923094f, 0.16625768622617648f}, {0xbf1d60f4u, 0x3e2a3f74u}, -1},
    UpdateStep<2>{{-0.013298174482636796f, 0.65532105456810053f}, {0xbc59e096u, 0x3f27c31fu}, 0},
    PredictStep<2>{{-1.1324406262895566f, 1.4560074049591982f}, {0xbf90f3d0u, 0x3fba5e73u}, -1},
    UpdateStep<2>{{-0.71438369809105728f, 0.46032555359591071f}, {0xbf36e1dau, 0x3eebafcau}, 0},
    PredictStep<2>{{-2.3467521354903487f, 1.380397212457563f}, {0xc0163130u, 0x3fb0b0dbu}, -1},
    UpdateStep<2>{{-0.80083643637847757f, 0.4246597802035651f}, {0xbf4d039eu, 0x3ed96d02u}, 0},
    PredictStep<2>{{-2.6508568486116606f, 1.2474936336966507f}, {0xc029a7a3u, 0x3f9faddfu}, -1},
    UpdateStep<2>{{-0.91688476442869304f, 0.37715612045388092f}, {0xbf6ab8f6u, 0x3ec11a9bu}, 0},
    PredictStep<2>{{-3.081000204598789f, 1.090610262420431f}, {0xc0452f1bu, 0x3f8b991eu}, -1},
    UpdateStep<2>{{-1.0843187388326063f, 0.32456845643488763f}, {0xbf8acaf5u, 0x3ea62dd6u}, 0},
    PredictStep<2>{{-3.7223454760766739f, 0.92223767722833938f}, {0xc06e3ae9u, 0x3f6c17c5u}, -1},
    UpdateStep<2>{{-1.3476750923044902f, 0.26864781401169108f}, {0xbfac809eu, 0x3e898c35u}, 0},
    PredictStep<2>{{-4.8180865642740605f, 0.74201861021437443f}, {0xc09a2dc4u, 0x3f3df4eeu}, -1},
    UpdateStep<2>{{-1.8598058796095529f, 0.20755127302849599f}, {0xbfee0e1eu, 0x3e548852u}, 0},
    PredictStep<2>{{-7.4755156017438527f, 0.5376905251045061f}, {0xc0ef376cu, 0x3f09a616u}, -1},
    UpdateStep<2>{{7.0250799452919618e-18f, 0.13377003718201369f}, {0x23019701u, 0x3e08fb03u}, 0},
    PredictStep<1>{{-0.2600956094272952f}, {0xbe852b40u}, 0},
    ScaleEvenStep<1>{{13871.404541741156f}, {0x4658bd9eu}, 0},
    ScaleOddStep<1>{{7.2090753102243445e-05f}, {0x38972f6eu}, 0});

template <>
struct scheme_traits<db28_tag> {
    using SchemeType = decltype(db28_scheme);
    static constexpr const char* name = "db28";
    static constexpr int id = 52;
    static constexpr int tap_size = 56;
    static constexpr int delay_even = 14;
    static constexpr int delay_odd = 14;
    static constexpr int num_steps = 31;
    static constexpr const auto& scheme = db28_scheme;
};

}  // namespace ttwv
