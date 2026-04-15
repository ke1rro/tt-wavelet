#pragma once

#include "../scheme.hpp"

namespace ttwv {

struct coif11_tag {};

inline constexpr auto coif11_scheme = make_lifting_scheme(
    66,
    16,
    17,
    PredictStep<1>{ { -1.3727906085756485f }, 0 },
    UpdateStep<2>{ { 0.47591086261292054f, 0.9706732968558156f }, 0 },
    PredictStep<2>{ { -0.91366985715305282f, 0.81764811937492798f }, -1 },
    UpdateStep<2>{ { -1.0549887629369721f, 0.88531662060543992f }, 0 },
    PredictStep<2>{ { -0.92119094399604573f, 0.56940801598843627f }, -1 },
    UpdateStep<2>{ { -1.1031742537900564f, 0.63278450128870889f }, 0 },
    PredictStep<2>{ { -0.56456859973038298f, 0.49083822788075043f }, -1 },
    UpdateStep<2>{ { -0.60271274569640043f, 0.54807383667272458f }, 0 },
    PredictStep<2>{ { -0.47043034448573212f, 0.24315901572159743f }, -1 },
    UpdateStep<2>{ { -0.34928841218501433f, 0.24770420940514767f }, 0 },
    PredictStep<2>{ { -0.18004167144861416f, 0.054184070543505047f }, -1 },
    UpdateStep<2>{ { -0.077698598938008984f, -0.074991173643737252f }, 0 },
    PredictStep<2>{ { 0.05231109480399937f, -0.14539886702867888f }, -1 },
    UpdateStep<2>{ { 0.20308018438226641f, -0.4126684518692848f }, 0 },
    PredictStep<2>{ { 0.27046668081346437f, -0.29464179421574432f }, -1 },
    UpdateStep<2>{ { 0.44037462669145505f, -0.75162517331294199f }, 0 },
    PredictStep<2>{ { 0.40288210321829748f, -0.54123157882511386f }, -1 },
    UpdateStep<2>{ { 0.81188921388166846f, -0.74502510847301573f }, 0 },
    PredictStep<2>{ { 0.53369493323225825f, -0.68077136441161668f }, -1 },
    UpdateStep<2>{ { 0.76066876418804652f, -1.3173094303786146f }, 0 },
    PredictStep<2>{ { 0.57927895594522805f, -0.90191721444415451f }, -1 },
    UpdateStep<2>{ { 0.98287186679872629f, -5.0672648712709103f }, 0 },
    PredictStep<2>{ { 0.19639880533452486f, 0.62199641012616391f }, -1 },
    UpdateStep<2>{ { -1.6069543466953335f, -0.37052199762326743f }, 0 },
    PredictStep<2>{ { 2.6747261499460895f, -4.871280250912533f }, -1 },
    UpdateStep<2>{ { 0.20472711006729705f, -0.45373707687081177f }, 0 },
    PredictStep<2>{ { 2.2026978657347516f, -5.7641843828303063f }, -1 },
    UpdateStep<2>{ { 0.17347103266818978f, -0.5474225279098488f }, 0 },
    PredictStep<2>{ { 1.8267276488757496f, -7.1742532615841883f }, -1 },
    UpdateStep<2>{ { 0.13938725554740647f, -0.71122766909363622f }, 0 },
    PredictStep<2>{ { 1.4060195113370635f, -9.9520720118223487f }, -1 },
    UpdateStep<2>{ { 0.10048158799573143f, -1.1101151993034835f }, 0 },
    PredictStep<2>{ { 0.90080741226146221f, -20.704299517933499f }, -1 },
    SwapStep{ {}, 0 },
    PredictStep<1>{ { 0.048299146712682602f }, 0 },
    ScaleEvenStep<1>{ { 6.1839822240622013e-05f }, 0 },
    ScaleOddStep<1>{ { -16170.809743096401f }, 0 }
);

template <>
struct scheme_traits<coif11_tag> {
    using SchemeType = decltype(coif11_scheme);
    static constexpr const char* name = "coif11";
    static constexpr int id         = 17;
    static constexpr int tap_size   = 66;
    static constexpr int delay_even = 16;
    static constexpr int delay_odd  = 17;
    static constexpr int num_steps  = 37;
    static constexpr const auto& scheme = coif11_scheme;
};

}
