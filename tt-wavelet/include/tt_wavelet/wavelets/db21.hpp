#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct db21_tag {};

inline constexpr auto db21_scheme = make_lifting_scheme(
    42,
    10,
    11,
    PredictStep<1>{{14.681137736660679f}, {0x416ae5f1u}, 0},
    UpdateStep<2>{{-0.06638242141320562f, -0.00095107351215285721f}, {0xbd87f382u, 0xba795177u}, 0},
    PredictStep<2>{{111.94902232586738f, -72.653878539800345f}, {0x42dfe5e6u, 0xc2914ec9u}, -1},
    UpdateStep<2>{{0.0029538523778921714f, -0.0021191422141005065f}, {0x3b41956bu, 0xbb0ae14fu}, 0},
    PredictStep<2>{{161.62217339876875f, -120.77815335403679f}, {0x43219f47u, 0xc2f18e6au}, -1},
    UpdateStep<2>{{0.0039787641677223413f, -0.003015602350222666f}, {0x3b82604bu, 0xbb45a169u}, 0},
    PredictStep<2>{{207.94146885567346f, -155.22153840114217f}, {0x434ff104u, 0xc31b38b7u}, -1},
    UpdateStep<2>{{0.055153475821330518f, -0.0035516357753463054f}, {0x3d61e89cu, 0xbb68c290u}, 0},
    PredictStep<2>{{-0.021509861504539176f, -17.827137571504483f}, {0xbcb03573u, 0xc18e9dfau}, -1},
    UpdateStep<2>{{-0.48599414111221467f, -4.2511845724833295f}, {0xbef8d439u, 0xc08809b4u}, 0},
    PredictStep<2>{{0.21581261056413642f, -0.28193246372228098f}, {0x3e5cfdfbu, 0xbe905974u}, -1},
    UpdateStep<2>{{3.3899296375106815f, -5.4613581845917096f}, {0x4058f49bu, 0xc0aec372u}, 0},
    PredictStep<2>{{0.17990162682949379f, -0.32384038763885425f}, {0x3e383822u, 0xbea5ce68u}, -1},
    UpdateStep<2>{{3.0710676766765839f, -6.3316525958560934f}, {0x40448c5fu, 0xc0ca9ce6u}, 0},
    PredictStep<2>{{0.15773276913165321f, -0.38267118777481107f}, {0x3e2184b3u, 0xbec3ed7au}, -1},
    UpdateStep<2>{{2.6126357896902217f, -7.6710920615029199f}, {0x4027356du, 0xc0f57996u}, 0},
    PredictStep<2>{{0.13035621513040216f, -0.47845190911879365f}, {0x3e057c1au, 0xbef4f7a6u}, -1},
    UpdateStep<2>{{2.0900702662218325f, -10.008451585314345f}, {0x4005c3b6u, 0xc120229eu}, 0},
    PredictStep<2>{{0.099915547276383354f, -0.66621090543176775f}, {0x3dcca086u, 0xbf2a8cccu}, -1},
    UpdateStep<2>{{1.5010261615232185f, -15.674261996139641f}, {0x3fc021a0u, 0xc17ac9c7u}, 0},
    PredictStep<2>{{0.063798857019825003f, -1.3901625923210885f}, {0x3d82a8fau, 0xbfb1f0d9u}, -1},
    SwapStep{{}, {}, 0},
    PredictStep<1>{{0.71934031711377855f}, {0x3f3826b0u}, 0},
    ScaleEvenStep<1>{{0.00036899321310894105f}, {0x39c1756eu}, 0},
    ScaleOddStep<1>{{-2710.0769457913075f}, {0xc529613bu}, 0});

template <>
struct scheme_traits<db21_tag> {
    using SchemeType = decltype(db21_scheme);
    static constexpr const char* name = "db21";
    static constexpr int id = 45;
    static constexpr int tap_size = 42;
    static constexpr int delay_even = 10;
    static constexpr int delay_odd = 11;
    static constexpr int num_steps = 25;
    static constexpr const auto& scheme = db21_scheme;
};

}  // namespace ttwv
