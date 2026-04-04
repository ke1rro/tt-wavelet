#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct coif5_tag {};

inline constexpr auto coif5_scheme = make_lifting_scheme(
    30,
    7,
    8,
    PredictStep<1>{{-1.6907515695408157f}, 0},
    UpdateStep<2>{{0.43817282471860691f, 0.44521529359227846f}, 0},
    PredictStep<2>{{-1.6776530195196462f, 1.1670102230648784f}, -1},
    UpdateStep<2>{{-0.50398248982942906f, 0.17107566571932797f}, 0},
    PredictStep<2>{{-0.82594833384483668f, 0.26806108995090666f}, -1},
    UpdateStep<2>{{-0.063557174506965602f, -0.058190604302669993f}, 0},
    PredictStep<2>{{0.24610502897906822f, -0.74955834180025238f}, -1},
    UpdateStep<2>{{0.15845018313391071f, -0.20194405044052352f}, 0},
    PredictStep<2>{{0.88936148018805417f, -2.2163173264587366f}, -1},
    UpdateStep<2>{{0.259975581459709f, -0.91418284992013787f}, 0},
    PredictStep<2>{{1.0324573621917275f, -3.9321722865504167f}, -1},
    UpdateStep<2>{{0.25327368881527801f, -0.18816039284859565f}, 0},
    PredictStep<2>{{5.2754164764171643f, -17.090988568094051f}, -1},
    UpdateStep<2>{{0.058468974358543702f, -0.30697247128357286f}, 0},
    PredictStep<2>{{3.2575374388850817f, -36.455332655288899f}, -1},
    SwapStep{{}, 0},
    PredictStep<1>{{0.027430823478902962f}, 0},
    ScaleEvenStep<1>{{-0.0035244711758772819f}, 0},
    ScaleOddStep<1>{{283.73050880493821f}, 0});

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
