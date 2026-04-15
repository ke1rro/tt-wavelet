#pragma once

#include "../scheme.hpp"

namespace ttwv {

struct db24_tag {};

inline constexpr auto db24_scheme = make_lifting_scheme(
    48,
    12,
    12,
    PredictStep<1>{ { -0.33506232489240084f }, -1 },
    UpdateStep<2>{ { -0.48017495688557604f, 0.18239575579147896f }, 0 },
    PredictStep<2>{ { -0.55374431761547427f, 0.42447415773502223f }, -1 },
    UpdateStep<2>{ { -0.67619594989108966f, 0.58161584624153684f }, 0 },
    PredictStep<2>{ { -0.752143272652811f, 0.65051069110748427f }, -1 },
    UpdateStep<2>{ { -0.85659141586005039f, 1.1276905654578357f }, 0 },
    PredictStep<2>{ { -0.67533427417529623f, -0.066165863982362461f }, -1 },
    UpdateStep<2>{ { -20.110766114423821f, -0.4402623937907274f }, 0 },
    PredictStep<2>{ { -0.031963721358628114f, 0.045206535245998922f }, -1 },
    UpdateStep<2>{ { -2.5485741868079481f, 26.465551394330816f }, 0 },
    PredictStep<2>{ { -0.0039467471127879843f, 0.11216441653075979f }, -1 },
    UpdateStep<2>{ { -8.131675516396303f, 8.1688805221175205f }, 0 },
    PredictStep<2>{ { -0.12302498385025439f, 0.08233603108313485f }, -1 },
    UpdateStep<2>{ { -12.854792463745287f, 7.8939898007456373f }, 0 },
    PredictStep<2>{ { -0.13921394488051425f, 0.077011663484636478f }, -1 },
    UpdateStep<2>{ { -14.678947093820494f, 7.161020540907697f }, 0 },
    PredictStep<2>{ { -0.16157090255335824f, 0.06807462585706596f }, -1 },
    UpdateStep<2>{ { -17.371815486458516f, 6.188423914012013f }, 0 },
    PredictStep<2>{ { -0.19570833615930353f, 0.057563551994512005f }, -1 },
    UpdateStep<2>{ { -21.673403086388262f, 5.1096370438202046f }, 0 },
    PredictStep<2>{ { -0.25445544684060084f, 0.046139497162130047f }, -1 },
    UpdateStep<2>{ { -30.052769875652409f, 3.9299610591657101f }, 0 },
    PredictStep<2>{ { -0.39672235258105332f, 0.033274803091121817f }, -1 },
    UpdateStep<2>{ { 3.2126137973467628e-14f, 2.5206545421344186f }, 0 },
    PredictStep<1>{ { -0.01601884943632427f }, 0 },
    ScaleEvenStep<1>{ { 840.31709710933887f }, 0 },
    ScaleOddStep<1>{ { 0.0011900269593942152f }, 0 }
);

template <>
struct scheme_traits<db24_tag> {
    using SchemeType = decltype(db24_scheme);
    static constexpr const char* name = "db24";
    static constexpr int id         = 48;
    static constexpr int tap_size   = 48;
    static constexpr int delay_even = 12;
    static constexpr int delay_odd  = 12;
    static constexpr int num_steps  = 27;
    static constexpr const auto& scheme = db24_scheme;
};

}
