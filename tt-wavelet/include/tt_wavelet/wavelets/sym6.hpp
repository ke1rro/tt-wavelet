#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct sym6_tag {};

inline constexpr auto sym6_scheme = make_lifting_scheme(
    12,
    3,
    3,
    PredictStep<1>{{4.4128845219062072f}, {0x408d365au}, -1},
    UpdateStep<2>{{0.036665591641754106f, -0.21554076182323406f}, {0x3d162ea9u, 0xbe5cb6b8u}, 0},
    PredictStep<2>{{1.1660745495199778f, -9.8297752472062747f}, {0x3f9541eeu, 0xc11d46c2u}, -1},
    UpdateStep<2>{{-0.035157056361653401f, -0.0067470277078235689f}, {0xbd1000d8u, 0xbbdd162cu}, 0},
    PredictStep<2>{{-13.988673926361949f, 5.0392826204781018f}, {0xc15fd19cu, 0x40a141ceu}, -1},
    UpdateStep<2>{{0.068379527635057225f, 0.044603194228431023f}, {0x3d8c0a91u, 0x3d36b1d7u}, 0},
    PredictStep<1>{{-11.63939178157182f}, {0xc13a3af3u}, 0},
    ScaleEvenStep<1>{{2.4278053663652512f}, {0x401b612au}, 0},
    ScaleOddStep<1>{{0.41189463284576783f}, {0x3ed2e3dau}, 0});

template <>
struct scheme_traits<sym6_tag> {
    using SchemeType = decltype(sym6_scheme);
    static constexpr const char* name = "sym6";
    static constexpr int id = 102;
    static constexpr int tap_size = 12;
    static constexpr int delay_even = 3;
    static constexpr int delay_odd = 3;
    static constexpr int num_steps = 9;
    static constexpr const auto& scheme = sym6_scheme;
};

}  // namespace ttwv
