#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct db25_tag {};

inline constexpr auto db25_scheme = make_lifting_scheme(
    50,
    12,
    13,
    PredictStep<1>{{5.372238388697542f}, {0x40abe960u}, 0},
    UpdateStep<2>{{-0.17562330142215124f, -0.015657411404777881f}, {0xbe33d698u, 0xbc8043f9u}, 0},
    PredictStep<2>{{12.098443850534734f, -17.055096556676062f}, {0x4141933au, 0xc18870d6u}, -1},
    UpdateStep<2>{{0.017553007523733094f, -0.023477476984443194f}, {0x3c8fcb53u, 0xbcc053d6u}, 0},
    PredictStep<2>{{16.996595405797549f, -22.305310921492797f}, {0x4187f907u, 0xc1b27147u}, -1},
    UpdateStep<2>{{0.022814866597785501f, -0.037888367567111302f}, {0x3cbae63eu, 0xbd1b30d5u}, 0},
    PredictStep<2>{{17.804723831399389f, 3.851785888035276f}, {0x418e7013u, 0x407683a9u}, -1},
    UpdateStep<2>{{0.26121583359618955f, 0.007184500196483856f}, {0x3e85be15u, 0x3beb6bf5u}, 0},
    PredictStep<2>{{1.621191359976293f, -3.3111199108775482f}, {0x3fcf8333u, 0xc053e963u}, -1},
    UpdateStep<2>{{0.62703369913858342f, -0.49426354195925282f}, {0x3f208548u, 0xbefd101cu}, 0},
    PredictStep<2>{{1.8972626322373671f, -1.3878612224705855f}, {0x3ff2d980u, 0xbfb1a570u}, -1},
    UpdateStep<2>{{1.2914134360925682f, -0.48800591933244031f}, {0x3fa54d09u, 0xbef9dbe9u}, 0},
    PredictStep<2>{{-0.0082559175718463999f, -0.75838409621127922f}, {0xbc0743d4u, 0xbf422576u}, -1},
    UpdateStep<2>{{-1.0441115542970403f, -441.73524027032187f}, {0xbf85a573u, 0xc3dcde1cu}, 0},
    PredictStep<2>{{0.0022560754793105558f, -0.00079589379498479565f}, {0x3b13daaau, 0xba50a387u}, -1},
    UpdateStep<2>{{1249.2611455499973f, -2323.2745571853729f}, {0x449c285bu, 0xc5113465u}, 0},
    PredictStep<2>{{0.00042972047223729866f, -0.00091993590535617293f}, {0x39e14c1bu, 0xba7127dbu}, -1},
    UpdateStep<2>{{1086.6425126095949f, -2698.650622353452f}, {0x4487d48fu, 0xc528aa69u}, 0},
    PredictStep<2>{{0.00037053401530857532f, -0.0010888279720140568f}, {0x39c2443cu, 0xba8eb701u}, -1},
    UpdateStep<2>{{918.41254197527189f, -3267.2479467778671f}, {0x44659a67u, 0xc54c33f8u}, 0},
    PredictStep<2>{{0.00030606782348983257f, -0.0013571965678171828f}, {0x39a077bau, 0xbab1e3f6u}, -1},
    UpdateStep<2>{{736.81292500314271f, -4243.0515964614397f}, {0x44383407u, 0xc584986au}, 0},
    PredictStep<2>{{0.00023567943411715808f, -0.0018795001024450504f}, {0x397720b7u, 0xbaf6598fu}, -1},
    UpdateStep<2>{{532.05636897080433f, -6606.6011570946603f}, {0x4405039cu, 0xc5ce74cfu}, 0},
    PredictStep<2>{{0.00015136376121722473f, -0.0038990065165801489f}, {0x391eb766u, 0xbb7f8679u}, -1},
    SwapStep{{}, {}, 0},
    PredictStep<1>{{256.47559083258682f}, {0x43803ce0u}, 0},
    ScaleEvenStep<1>{{0.0016947961267543693f}, {0x3ade23ecu}, 0},
    ScaleOddStep<1>{{-590.04147119161564f}, {0xc41382a7u}, 0});

template <>
struct scheme_traits<db25_tag> {
    using SchemeType = decltype(db25_scheme);
    static constexpr const char* name = "db25";
    static constexpr int id = 49;
    static constexpr int tap_size = 50;
    static constexpr int delay_even = 12;
    static constexpr int delay_odd = 13;
    static constexpr int num_steps = 29;
    static constexpr const auto& scheme = db25_scheme;
};

}  // namespace ttwv
