#pragma once

#include "../scheme.hpp"

namespace ttwv {

struct sym6_tag {};

inline constexpr auto sym6_scheme = make_lifting_scheme(
    12,
    3,
    3,
    PredictStep<1>{ { 4.4128845219062072f }, -1 },
    UpdateStep<2>{ { 0.036665591641754106f, -0.21554076182323406f }, 0 },
    PredictStep<2>{ { 1.1660745495199778f, -9.8297752472062747f }, -1 },
    UpdateStep<2>{ { -0.035157056361653401f, -0.0067470277078235689f }, 0 },
    PredictStep<2>{ { -13.988673926361949f, 5.0392826204781018f }, -1 },
    UpdateStep<2>{ { 0.068379527635057225f, 0.044603194228431023f }, 0 },
    PredictStep<1>{ { -11.63939178157182f }, 0 },
    ScaleEvenStep<1>{ { 2.4278053663652512f }, 0 },
    ScaleOddStep<1>{ { 0.41189463284576783f }, 0 }
);

template <>
struct scheme_traits<sym6_tag> {
    using SchemeType = decltype(sym6_scheme);
    static constexpr const char* name = "sym6";
    static constexpr int id         = 102;
    static constexpr int tap_size   = 12;
    static constexpr int delay_even = 3;
    static constexpr int delay_odd  = 3;
    static constexpr int num_steps  = 9;
    static constexpr const auto& scheme = sym6_scheme;
};

}
