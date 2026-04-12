#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct coif12_tag {};

inline constexpr auto coif12_scheme = make_lifting_scheme(
    72,
    18,
    18,
    PredictStep<1>{{0.7419735588397337f}, {0x3f3df1fbu}, -1},
    UpdateStep<2>{{1.9011732739725553f, -0.47853060912773071f}, {0x3ff359a5u, 0xbef501f7u}, 0},
    PredictStep<2>{{0.4220310440412221f, -0.4717070225995707f}, {0x3ed81474u, 0xbef18395u}, -1},
    UpdateStep<2>{{1.8122252425707353f, -2.071660376788802f}, {0x3fe7f6ffu, 0xc0049615u}, 0},
    PredictStep<2>{{0.30791909592946892f, -0.46453374076353277f}, {0x3e9da792u, 0xbeedd75eu}, -1},
    UpdateStep<2>{{1.3170445349763216f, -2.2748848028048343f}, {0x3fa894eau, 0xc01197b6u}, 0},
    PredictStep<2>{{0.26341381225398608f, -0.33364557666743949f}, {0x3e86de2du, 0xbeaad398u}, -1},
    UpdateStep<2>{{1.2724095658138674f, -1.2460655527734226f}, {0x3fa2de51u, 0xbf9f7f13u}, 0},
    PredictStep<2>{{0.16604087808781964f, -0.26526230815718982f}, {0x3e2a069fu, 0xbe87d076u}, -1},
    UpdateStep<2>{{0.65660771829549713f, -1.0021917558905151f}, {0x3f281772u, 0xbf8047d2u}, 0},
    PredictStep<2>{{0.086177695627798048f, -0.12020134316630957f}, {0x3db07defu, 0xbdf62c1fu}, -1},
    UpdateStep<2>{{0.1259210312172056f, -0.48952870320318048f}, {0x3e00f171u, 0xbefaa382u}, 0},
    PredictStep<2>{{-0.027614091145744431f, -0.023076493728451851f}, {0xbce236f2u, 0xbcbd0aeau}, -1},
    UpdateStep<2>{{-0.35414459025467604f, 0.15049235715253564f}, {0xbeb55271u, 0x3e1a1aabu}, 0},
    PredictStep<2>{{-0.13267645226364189f, 0.063779890803819281f}, {0xbe07dc56u, 0x3d829f08u}, -1},
    UpdateStep<2>{{-0.90866626390212246f, 0.68520247561447656f}, {0xbf689e5au, 0x3f2f696eu}, 0},
    PredictStep<2>{{-0.19145171520519821f, 0.16459475069717577f}, {0xbe440bebu, 0x3e288b87u}, -1},
    UpdateStep<2>{{-1.7000668396066005f, 1.0039349266748734f}, {0xbfd99bcau, 0x3f8080f1u}, 0},
    PredictStep<2>{{-0.26999329573098868f, 0.23856765709029468f}, {0xbe8a3c90u, 0x3e744b14u}, -1},
    UpdateStep<2>{{-1.6168115365425491f, 1.7272907181568449f}, {0xbfcef3aeu, 0x3fdd17ddu}, 0},
    PredictStep<2>{{-0.36383686744455829f, 0.26821344837010486f}, {0xbeba48d3u, 0x3e895346u}, -1},
    UpdateStep<2>{{-2.7329291973874392f, 1.6075561886868779f}, {0xc02ee850u, 0x3fcdc467u}, 0},
    PredictStep<2>{{-0.4354909907977032f, 0.29376832571914585f}, {0xbedef8adu, 0x3e9668cdu}, -1},
    UpdateStep<2>{{-12.426560544515882f, 2.0654587197126681f}, {0xc146d331u, 0x4004307au}, 0},
    PredictStep<2>{{0.14337001587570933f, 0.080225127348936498f}, {0x3e12cf97u, 0x3da44d12u}, -1},
    UpdateStep<2>{{-2.2898975902453333f, -6.9682235145422888f}, {0xc0128dafu, 0xc0defbb0u}, 0},
    PredictStep<2>{{-0.75462190916756f, 0.43282656491497928f}, {0xbf412ee7u, 0x3edd9b72u}, -1},
    UpdateStep<2>{{-2.7757909289798004f, 1.3212762177137551f}, {0xc031a68fu, 0x3fa91f94u}, 0},
    PredictStep<2>{{-0.87723239136882569f, 0.36001747990858857f}, {0xbf60924du, 0x3eb85436u}, -1},
    UpdateStep<2>{{-3.2711234462841801f, 1.1398206998415334f}, {0xc0515a16u, 0x3f91e5a5u}, 0},
    PredictStep<2>{{-1.0583188989874712f, 0.30570119790858147f}, {0xbf8776feu, 0x3e9c84deu}, -1},
    UpdateStep<2>{{-4.0683213837136032f, 0.94489370125065175f}, {0xc0822fb0u, 0x3f71e48eu}, 0},
    PredictStep<2>{{-1.3727148293205405f, 0.24580160815039451f}, {0xbfafb51fu, 0x3e7bb36bu}, -1},
    UpdateStep<2>{{-5.6308550474910444f, 0.72848342326566629f}, {0xc0b42ff7u, 0x3f3a7de4u}, 0},
    PredictStep<2>{{-2.1371281803165378f, 0.17759292177350866f}, {0xc008c6b5u, 0x3e35daebu}, -1},
    UpdateStep<2>{{4.2649242864934375e-15f, 0.46791765192654877f}, {0x2799a8fcu, 0x3eef92e7u}, 0},
    PredictStep<1>{{-0.085593304157528222f}, {0xbdaf4b8bu}, 0},
    ScaleEvenStep<1>{{-2116.5674672724658f}, {0xc5044914u}, 0},
    ScaleOddStep<1>{{-0.00047246308726867997f}, {0xb9f7b4ecu}, 0});

template <>
struct scheme_traits<coif12_tag> {
    using SchemeType = decltype(coif12_scheme);
    static constexpr const char* name = "coif12";
    static constexpr int id = 18;
    static constexpr int tap_size = 72;
    static constexpr int delay_even = 18;
    static constexpr int delay_odd = 18;
    static constexpr int num_steps = 39;
    static constexpr const auto& scheme = coif12_scheme;
};

}  // namespace ttwv
