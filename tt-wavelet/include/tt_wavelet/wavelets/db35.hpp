#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct db35_tag {};

inline constexpr auto db35_scheme = make_lifting_scheme(
    70,
    17,
    18,
    PredictStep<1>{{4.3485545935046064f}, 0},
    UpdateStep<2>{{-0.20824101938231618f, -0.02764661683165625f}, 0},
    PredictStep<2>{{9.8337240895095288f, -12.369295746879304f}, -1},
    UpdateStep<2>{{0.03020985069250921f, -0.037298618243981799f}, 0},
    PredictStep<2>{{11.75082550953868f, -14.11411958562006f}, -1},
    UpdateStep<2>{{0.035642147441871927f, -0.039104248501616479f}, 0},
    PredictStep<2>{{14.152886763518625f, -10.185575662941879f}, -1},
    UpdateStep<2>{{0.050492060697702859f, -0.016791835251891152f}, 0},
    PredictStep<2>{{20.605376406607618f, -7.672732874423267f}, -1},
    UpdateStep<2>{{0.047589056338738583f, -0.031237291469713834f}, 0},
    PredictStep<2>{{20.455933705879207f, -16.037437044877855f}, -1},
    UpdateStep<2>{{0.064948917910796614f, -0.040655507106671658f}, 0},
    PredictStep<2>{{-2.7405795139733069f, -13.954657531617885f}, -1},
    UpdateStep<2>{{-0.012933564057333341f, 0.63904946696337006f}, 0},
    PredictStep<2>{{-1.7147925675389444f, -2.5241293938024807f}, -1},
    UpdateStep<2>{{0.37901631872230113f, -0.46187893667537688f}, 0},
    PredictStep<2>{{2.1075154772618521f, -7.4650415049560301f}, -1},
    UpdateStep<2>{{0.13324070577021138f, 0.0017012731730240221f}, 0},
    PredictStep<2>{{-814.54418530652833f, 10.718649490465449f}, -1},
    UpdateStep<2>{{0.0054606722721337346f, 0.0012456200370110505f}, 0},
    PredictStep<2>{{291.5837210677098f, -182.76133489734804f}, -1},
    UpdateStep<2>{{0.0025801800356416373f, -0.0034271332245373208f}, 0},
    PredictStep<2>{{2.9045604015304325f, -387.38515937329942f}, -1},
    UpdateStep<2>{{0.0015939968190745966f, -0.33452682900195052f}, 0},
    PredictStep<2>{{2.9852390216744888f, -182.67883631469024f}, -1},
    UpdateStep<2>{{0.0054740379681681543f, -0.020718529644743137f}, 0},
    PredictStep<2>{{48.265931230617333f, -153.0547752405146f}, -1},
    UpdateStep<2>{{0.0065336080576529586f, -0.023896788724457042f}, 0},
    PredictStep<2>{{41.846626549269281f, -180.16361867291596f}, -1},
    UpdateStep<2>{{0.0055505101807689951f, -0.028711132051938166f}, 0},
    PredictStep<2>{{34.82969595827366f, -222.56614098461262f}, -1},
    UpdateStep<2>{{0.0044930464066805838f, -0.036929915254358375f}, 0},
    PredictStep<2>{{27.07831829865837f, -305.20983134091932f}, -1},
    UpdateStep<2>{{0.0032764344307212052f, -0.056942751936641618f}, 0},
    PredictStep<2>{{17.561497574136705f, -627.1306023277906f}, -1},
    SwapStep{{}, 0},
    PredictStep<1>{{0.0015945641885249874f}, 0},
    ScaleEvenStep<1>{{1.2544597865856584e-07f}, 0},
    ScaleOddStep<1>{{-7971558.839058226f}, 0});

template <>
struct scheme_traits<db35_tag> {
    using SchemeType = decltype(db35_scheme);
    static constexpr const char* name = "db35";
    static constexpr int id = 60;
    static constexpr int tap_size = 70;
    static constexpr int delay_even = 17;
    static constexpr int delay_odd = 18;
    static constexpr int num_steps = 39;
    static constexpr const auto& scheme = db35_scheme;
};

}  // namespace ttwv
