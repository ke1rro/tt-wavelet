#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct coif5_tag {};

inline constexpr auto coif5_scheme = make_lifting_scheme(
    30,
    7,
    8,
    PredictStep<1>{{-1.6907515695408157f}, {0xbfd86a8cu}, 0},
    UpdateStep<2>{{0.43817282471860691f, 0.44521529359227846f}, {0x3ee05830u, 0x3ee3f342u}, 0},
    PredictStep<2>{{-1.6776530195196462f, 1.1670102230648784f}, {0xbfd6bd56u, 0x3f956097u}, -1},
    UpdateStep<2>{{-0.50398248982942906f, 0.17107566571932797f}, {0xbf0104ffu, 0x3e2f2e76u}, 0},
    PredictStep<2>{{-0.82594833384483668f, 0.26806108995090666f}, {0xbf53715au, 0x3e893f4eu}, -1},
    UpdateStep<2>{{-0.063557174506965602f, -0.058190604302669993f}, {0xbd822a44u, 0xbd6e5945u}, 0},
    PredictStep<2>{{0.24610502897906822f, -0.74955834180025238f}, {0x3e7c02f5u, 0xbf3fe30eu}, -1},
    UpdateStep<2>{{0.15845018313391071f, -0.20194405044052352f}, {0x3e2240c4u, 0xbe4eca6cu}, 0},
    PredictStep<2>{{0.88936148018805417f, -2.2163173264587366f}, {0x3f63ad32u, 0xc00dd825u}, -1},
    UpdateStep<2>{{0.259975581459709f, -0.91418284992013787f}, {0x3e851b85u, 0xbf6a07e3u}, 0},
    PredictStep<2>{{1.0324573621917275f, -3.9321722865504167f}, {0x3f842790u, 0xc07ba8b6u}, -1},
    UpdateStep<2>{{0.25327368881527801f, -0.18816039284859565f}, {0x3e81ad17u, 0xbe40ad1eu}, 0},
    PredictStep<2>{{5.2754164764171643f, -17.090988568094051f}, {0x40a8d036u, 0xc188ba58u}, -1},
    UpdateStep<2>{{0.058468974358543702f, -0.30697247128357286f}, {0x3d6f7d2au, 0xbe9d2b7fu}, 0},
    PredictStep<2>{{3.2575374388850817f, -36.455332655288899f}, {0x40507b7eu, 0xc211d243u}, -1},
    SwapStep{{}, {}, 0},
    PredictStep<1>{{0.027430823478902962f}, {0x3ce0b69bu}, 0},
    ScaleEvenStep<1>{{-0.0035244711758772819f}, {0xbb66fad0u}, 0},
    ScaleOddStep<1>{{283.73050880493821f}, {0x438ddd81u}, 0});

template <>
struct scheme_traits<coif5_tag> {
    using SchemeType = decltype(coif5_scheme);
    static constexpr const char* name = "coif5";
    static constexpr int id = 27;
    static constexpr int tap_size = 30;
    static constexpr int delay_even = 7;
    static constexpr int delay_odd = 8;
    static constexpr int num_steps = 19;
    static constexpr const auto& scheme = coif5_scheme;
};

}  // namespace ttwv
