#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct db5_tag {};

inline constexpr auto db5_scheme = make_lifting_scheme(
    10,
    2,
    3,
    PredictStep<1>{{3.771519211689268f}, {0x40716092u}, 0},
    UpdateStep<2>{{-0.24772929136032976f, -0.061736570470349854f}, {0xbe7dacbfu, 0xbd7cdf7cu}, 0},
    PredictStep<2>{{7.5975797354057519f, -19.027396275110604f}, {0x40f31f60u, 0xc198381cu}, -1},
    UpdateStep<2>{{0.044520705161802412f, -0.14201121405642173f}, {0x3d365b58u, 0xbe116b63u}, 0},
    PredictStep<2>{{6.9189679172596978f, -44.228550858569371f}, {0x40dd682fu, 0xc230ea09u}, -1},
    SwapStep{{}, {}, 0},
    PredictStep<1>{{0.022600019079444381f}, {0x3cb923adu}, 0},
    ScaleEvenStep<1>{{0.02169954477294184f}, {0x3cb1c33eu}, 0},
    ScaleOddStep<1>{{-46.083916066614727f}, {0xc23855eeu}, 0});

template <>
struct scheme_traits<db5_tag> {
    using SchemeType = decltype(db5_scheme);
    static constexpr const char* name = "db5";
    static constexpr int id = 65;
    static constexpr int tap_size = 10;
    static constexpr int delay_even = 2;
    static constexpr int delay_odd = 3;
    static constexpr int num_steps = 9;
    static constexpr const auto& scheme = db5_scheme;
};

}  // namespace ttwv
