#pragma once

#include "../scheme.hpp"

namespace ttwv {

struct db21_tag {};

inline constexpr auto db21_scheme = make_lifting_scheme(
    42,
    10,
    11,
    PredictStep<1>{ { 14.681137736660679f }, 0 },
    UpdateStep<2>{ { -0.06638242141320562f, -0.00095107351215285721f }, 0 },
    PredictStep<2>{ { 111.94902232586738f, -72.653878539800345f }, -1 },
    UpdateStep<2>{ { 0.0029538523778921714f, -0.0021191422141005065f }, 0 },
    PredictStep<2>{ { 161.62217339876875f, -120.77815335403679f }, -1 },
    UpdateStep<2>{ { 0.0039787641677223413f, -0.003015602350222666f }, 0 },
    PredictStep<2>{ { 207.94146885567346f, -155.22153840114217f }, -1 },
    UpdateStep<2>{ { 0.055153475821330518f, -0.0035516357753463054f }, 0 },
    PredictStep<2>{ { -0.021509861504539176f, -17.827137571504483f }, -1 },
    UpdateStep<2>{ { -0.48599414111221467f, -4.2511845724833295f }, 0 },
    PredictStep<2>{ { 0.21581261056413642f, -0.28193246372228098f }, -1 },
    UpdateStep<2>{ { 3.3899296375106815f, -5.4613581845917096f }, 0 },
    PredictStep<2>{ { 0.17990162682949379f, -0.32384038763885425f }, -1 },
    UpdateStep<2>{ { 3.0710676766765839f, -6.3316525958560934f }, 0 },
    PredictStep<2>{ { 0.15773276913165321f, -0.38267118777481107f }, -1 },
    UpdateStep<2>{ { 2.6126357896902217f, -7.6710920615029199f }, 0 },
    PredictStep<2>{ { 0.13035621513040216f, -0.47845190911879365f }, -1 },
    UpdateStep<2>{ { 2.0900702662218325f, -10.008451585314345f }, 0 },
    PredictStep<2>{ { 0.099915547276383354f, -0.66621090543176775f }, -1 },
    UpdateStep<2>{ { 1.5010261615232185f, -15.674261996139641f }, 0 },
    PredictStep<2>{ { 0.063798857019825003f, -1.3901625923210885f }, -1 },
    SwapStep{ {}, 0 },
    PredictStep<1>{ { 0.71934031711377855f }, 0 },
    ScaleEvenStep<1>{ { 0.00036899321310894105f }, 0 },
    ScaleOddStep<1>{ { -2710.0769457913075f }, 0 }
);

template <>
struct scheme_traits<db21_tag> {
    using SchemeType = decltype(db21_scheme);
    static constexpr const char* name = "db21";
    static constexpr int id         = 45;
    static constexpr int tap_size   = 42;
    static constexpr int delay_even = 10;
    static constexpr int delay_odd  = 11;
    static constexpr int num_steps  = 25;
    static constexpr const auto& scheme = db21_scheme;
};

}
