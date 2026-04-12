#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct db27_tag {};

inline constexpr auto db27_scheme = make_lifting_scheme(
    54,
    13,
    14,
    PredictStep<1>{{5.8152725316691587f}, {0x40ba16b6u}, 0},
    UpdateStep<2>{{-0.16423913237336468f, -0.012547154611349419f}, {0xbe282e4eu, 0xbc4d9295u}, 0},
    PredictStep<2>{{13.481498432675139f, -18.546079772373421f}, {0x4157b438u, 0xc1945e5fu}, -1},
    UpdateStep<2>{{0.014683083636757603f, -0.018628977982641293f}, {0x3c709151u, 0xbc989bccu}, 0},
    PredictStep<2>{{19.19409562150652f, -24.065844528016207f}, {0x41998d82u, 0xc1c086d9u}, -1},
    UpdateStep<2>{{0.018782270096427592f, -0.023506840482236658f}, {0x3c99dd46u, 0xbcc0916bu}, 0},
    PredictStep<2>{{23.508548120065399f, -29.695755017248928f}, {0x41bc1182u, 0xc1ed90e8u}, -1},
    UpdateStep<2>{{0.022024627153054244f, -0.03750578293433475f}, {0x3cb46cfeu, 0xbd199faau}, 0},
    PredictStep<2>{{21.274664057571197f, 2.6178170407640775f}, {0x41aa3283u, 0x40278a50u}, -1},
    UpdateStep<2>{{0.50163187933696407f, 0.0099460832329504763f}, {0x3f006af2u, 0x3c22f4e6u}, 0},
    PredictStep<2>{{0.87047127614218955f, -1.8477239741420792f}, {0x3f5ed735u, 0xbfec8238u}, -1},
    UpdateStep<2>{{1.334858022732476f, -1.0179858324278719f}, {0x3faadca1u, 0xbf824d5cu}, 0},
    PredictStep<2>{{0.9955886385753463f, -0.6983715996363139f}, {0x3f7edee6u, 0xbf32c87bu}, -1},
    UpdateStep<2>{{-10.923342152299432f, -0.96956018358296847f}, {0xc12ec602u, 0xbf783519u}, 0},
    PredictStep<2>{{-8.293887877790156e-05f, 0.091740439205346566f}, {0xb8adef79u, 0x3dbbe269u}, -1},
    UpdateStep<2>{{-90.264491189735196f, -27281.347564438533f}, {0xc2b4876bu, 0xc6d522b2u}, 0},
    PredictStep<2>{{3.6590571774214571e-05f, -1.9972083214461381e-05f}, {0x381978d4u, 0xb7a789b9u}, -1},
    UpdateStep<2>{{49980.984586722487f, -101030.41578250722f}, {0x47433cfcu, 0xc7c55335u}, 0},
    PredictStep<2>{{9.8937436897010589e-06f, -2.307151113925093e-05f}, {0x3725fd4eu, 0xb7c189b1u}, -1},
    UpdateStep<2>{{43340.062402489646f, -117435.35550487753f}, {0x47294c10u, 0xc7e55daeu}, 0},
    PredictStep<2>{{8.5152311980596346e-06f, -2.7296486998966708e-05f}, {0x370edca4u, 0xb7e4fac2u}, -1},
    UpdateStep<2>{{36634.712346456479f, -141989.92258597811f}, {0x470f1ab6u, 0xc80aa97bu}, 0},
    PredictStep<2>{{7.042752836554307e-06f, -3.3958967053227064e-05f}, {0x36ec50c9u, 0xb80e6f2au}, -1},
    UpdateStep<2>{{29447.303135699218f, -183983.32111239916f}, {0x46e60e9bu, 0xc833abd5u}, 0},
    PredictStep<2>{{5.4352752949383505e-06f, -4.6915567097635372e-05f}, {0x36b660a9u, 0xb844c735u}, -1},
    UpdateStep<2>{{21314.886760675377f, -285776.27256794588f}, {0x46a685c6u, 0xc88b8a09u}, 0},
    PredictStep<2>{{3.4992408257485239e-06f, -9.7093256701926263e-05f}, {0x366ad480u, 0xb8cb9e8cu}, -1},
    SwapStep{{}, {}, 0},
    PredictStep<1>{{10299.376434245825f}, {0x4620ed81u}, 0},
    ScaleEvenStep<1>{{-0.005306351399468323f}, {0xbbade0e7u}, 0},
    ScaleOddStep<1>{{188.45340700583765f}, {0x433c7412u}, 0});

template <>
struct scheme_traits<db27_tag> {
    using SchemeType = decltype(db27_scheme);
    static constexpr const char* name = "db27";
    static constexpr int id = 51;
    static constexpr int tap_size = 54;
    static constexpr int delay_even = 13;
    static constexpr int delay_odd = 14;
    static constexpr int num_steps = 31;
    static constexpr const auto& scheme = db27_scheme;
};

}  // namespace ttwv
