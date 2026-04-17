#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct coif11_tag {};

inline constexpr auto coif11_scheme = make_lifting_scheme(
    66,
    16,
    17,
    PredictStep<1>{{-1.3727906085756485f}, {0xbfafb79au}, 0},
    UpdateStep<2>{{0.47591086261292054f, 0.9706732968558156f}, {0x3ef3aa97u, 0x3f787e0cu}, 0},
    PredictStep<2>{{-0.91366985715305282f, 0.81764811937492798f}, {0xbf69e645u, 0x3f515163u}, -1},
    UpdateStep<2>{{-1.0549887629369721f, 0.88531662060543992f}, {0xbf8709dfu, 0x3f62a41cu}, 0},
    PredictStep<2>{{-0.92119094399604573f, 0.56940801598843627f}, {0xbf6bd32bu, 0x3f11c4b9u}, -1},
    UpdateStep<2>{{-1.1031742537900564f, 0.63278450128870889f}, {0xbf8d34d0u, 0x3f21fe2au}, 0},
    PredictStep<2>{{-0.56456859973038298f, 0.49083822788075043f}, {0xbf108791u, 0x3efb4f26u}, -1},
    UpdateStep<2>{{-0.60271274569640043f, 0.54807383667272458f}, {0xbf1a4b62u, 0x3f0c4e91u}, 0},
    PredictStep<2>{{-0.47043034448573212f, 0.24315901572159743f}, {0xbef0dc3fu, 0x3e78feadu}, -1},
    UpdateStep<2>{{-0.34928841218501433f, 0.24770420940514767f}, {0xbeb2d5eeu, 0x3e7da62cu}, 0},
    PredictStep<2>{{-0.18004167144861416f, 0.054184070543505047f}, {0xbe385cd8u, 0x3d5df01eu}, -1},
    UpdateStep<2>{{-0.077698598938008984f, -0.074991173643737252f}, {0xbd9f2071u, 0xbd9994f9u}, 0},
    PredictStep<2>{{0.05231109480399937f, -0.14539886702867888f}, {0x3d564429u, 0xbe14e371u}, -1},
    UpdateStep<2>{{0.20308018438226641f, -0.4126684518692848f}, {0x3e4ff440u, 0xbed34948u}, 0},
    PredictStep<2>{{0.27046668081346437f, -0.29464179421574432f}, {0x3e8a7a9cu, 0xbe96db4au}, -1},
    UpdateStep<2>{{0.44037462669145505f, -0.75162517331294199f}, {0x3ee178c8u, 0xbf406a82u}, 0},
    PredictStep<2>{{0.40288210321829748f, -0.54123157882511386f}, {0x3ece4690u, 0xbf0a8e27u}, -1},
    UpdateStep<2>{{0.81188921388166846f, -0.74502510847301573f}, {0x3f4fd7f9u, 0xbf3eb9f7u}, 0},
    PredictStep<2>{{0.53369493323225825f, -0.68077136441161668f}, {0x3f08a03bu, 0xbf2e4708u}, -1},
    UpdateStep<2>{{0.76066876418804652f, -1.3173094303786146f}, {0x3f42bb30u, 0xbfa89d98u}, 0},
    PredictStep<2>{{0.57927895594522805f, -0.90191721444415451f}, {0x3f144ba0u, 0xbf66e40cu}, -1},
    UpdateStep<2>{{0.98287186679872629f, -5.0672648712709103f}, {0x3f7b9d7eu, 0xc0a22709u}, 0},
    PredictStep<2>{{0.19639880533452486f, 0.62199641012616391f}, {0x3e491cc5u, 0x3f1f3b28u}, -1},
    UpdateStep<2>{{-1.6069543466953335f, -0.37052199762326743f}, {0xbfcdb0aeu, 0xbebdb50fu}, 0},
    PredictStep<2>{{2.6747261499460895f, -4.871280250912533f}, {0x402b2eb7u, 0xc09be187u}, -1},
    UpdateStep<2>{{0.20472711006729705f, -0.45373707687081177f}, {0x3e51a3fcu, 0xbee8503au}, 0},
    PredictStep<2>{{2.2026978657347516f, -5.7641843828303063f}, {0x400cf900u, 0xc0b87433u}, -1},
    UpdateStep<2>{{0.17347103266818978f, -0.5474225279098488f}, {0x3e31a264u, 0xbf0c23e2u}, 0},
    PredictStep<2>{{1.8267276488757496f, -7.1742532615841883f}, {0x3fe9d236u, 0xc0e5937cu}, -1},
    UpdateStep<2>{{0.13938725554740647f, -0.71122766909363622f}, {0x3e0ebb88u, 0xbf361304u}, 0},
    PredictStep<2>{{1.4060195113370635f, -9.9520720118223487f}, {0x3fb3f873u, 0xc11f3bb0u}, -1},
    UpdateStep<2>{{0.10048158799573143f, -1.1101151993034835f}, {0x3dcdc94au, 0xbf8e1841u}, 0},
    PredictStep<2>{{0.90080741226146221f, -20.704299517933499f}, {0x3f669b51u, 0xc1a5a268u}, -1},
    SwapStep{{}, {}, 0},
    PredictStep<1>{{0.048299146712682602f}, {0x3d45d553u}, 0},
    ScaleEvenStep<1>{{6.1839822240622013e-05f}, {0x3881b000u}, 0},
    ScaleOddStep<1>{{-16170.809743096401f}, {0xc67cab3du}, 0});

template <>
struct scheme_traits<coif11_tag> {
    using SchemeType = decltype(coif11_scheme);
    static constexpr const char* name = "coif11";
    static constexpr int id = 17;
    static constexpr int tap_size = 66;
    static constexpr int delay_even = 16;
    static constexpr int delay_odd = 17;
    static constexpr int num_steps = 37;
    static constexpr const auto& scheme = coif11_scheme;
};

}  // namespace ttwv
