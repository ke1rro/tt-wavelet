#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct db24_tag {};

inline constexpr auto db24_scheme = make_lifting_scheme(
    48,
    12,
    12,
    PredictStep<1>{{-0.33506232489240084f}, {0xbeab8d4au}, -1},
    UpdateStep<2>{{-0.48017495688557604f, 0.18239575579147896f}, {0xbef5d97eu, 0x3e3ac5f4u}, 0},
    PredictStep<2>{{-0.55374431761547427f, 0.42447415773502223f}, {0xbf0dc230u, 0x3ed954adu}, -1},
    UpdateStep<2>{{-0.67619594989108966f, 0.58161584624153684f}, {0xbf2d1b2eu, 0x3f14e4c7u}, 0},
    PredictStep<2>{{-0.752143272652811f, 0.65051069110748427f}, {0xbf408c76u, 0x3f2687deu}, -1},
    UpdateStep<2>{{-0.85659141586005039f, 1.1276905654578357f}, {0xbf5b4993u, 0x3f90582au}, 0},
    PredictStep<2>{{-0.67533427417529623f, -0.066165863982362461f}, {0xbf2ce2b5u, 0xbd8781f8u}, -1},
    UpdateStep<2>{{-20.110766114423821f, -0.4402623937907274f}, {0xc1a0e2d9u, 0xbee16a13u}, 0},
    PredictStep<2>{{-0.031963721358628114f, 0.045206535245998922f}, {0xbd02ec64u, 0x3d392a7du}, -1},
    UpdateStep<2>{{-2.5485741868079481f, 26.465551394330816f}, {0xc0231bd7u, 0x41d3b973u}, 0},
    PredictStep<2>{{-0.0039467471127879843f, 0.11216441653075979f}, {0xbb8153b7u, 0x3de5b675u}, -1},
    UpdateStep<2>{{-8.131675516396303f, 8.1688805221175205f}, {0xc1021b58u, 0x4102b3bcu}, 0},
    PredictStep<2>{{-0.12302498385025439f, 0.08233603108313485f}, {0xbdfbf486u, 0x3da89fcbu}, -1},
    UpdateStep<2>{{-12.854792463745287f, 7.8939898007456373f}, {0xc14dad3bu, 0x40fc9b90u}, 0},
    PredictStep<2>{{-0.13921394488051425f, 0.077011663484636478f}, {0xbe0e8e1au, 0x3d9db84bu}, -1},
    UpdateStep<2>{{-14.678947093820494f, 7.161020540907697f}, {0xc16adcf8u, 0x40e52715u}, 0},
    PredictStep<2>{{-0.16157090255335824f, 0.06807462585706596f}, {0xbe2572d8u, 0x3d8b6ab6u}, -1},
    UpdateStep<2>{{-17.371815486458516f, 6.188423914012013f}, {0xc18af97au, 0x40c60792u}, 0},
    PredictStep<2>{{-0.19570833615930353f, 0.057563551994512005f}, {0xbe4867c4u, 0x3d6bc7c2u}, -1},
    UpdateStep<2>{{-21.673403086388262f, 5.1096370438202046f}, {0xc1ad6321u, 0x40a38226u}, 0},
    PredictStep<2>{{-0.25445544684060084f, 0.046139497162130047f}, {0xbe8247fcu, 0x3d3cfcc5u}, -1},
    UpdateStep<2>{{-30.052769875652409f, 3.9299610591657101f}, {0xc1f06c13u, 0x407b847bu}, 0},
    PredictStep<2>{{-0.39672235258105332f, 0.033274803091121817f}, {0xbecb1f31u, 0x3d084b29u}, -1},
    UpdateStep<2>{{3.2126137973467628e-14f, 2.5206545421344186f}, {0x2910aeeau, 0x40215267u}, 0},
    PredictStep<1>{{-0.01601884943632427f}, {0xbc8339f6u}, 0},
    ScaleEvenStep<1>{{840.31709710933887f}, {0x4452144bu}, 0},
    ScaleOddStep<1>{{0.0011900269593942152f}, {0x3a9bfaaeu}, 0});

template <>
struct scheme_traits<db24_tag> {
    using SchemeType = decltype(db24_scheme);
    static constexpr const char* name = "db24";
    static constexpr int id = 48;
    static constexpr int tap_size = 48;
    static constexpr int delay_even = 12;
    static constexpr int delay_odd = 12;
    static constexpr int num_steps = 27;
    static constexpr const auto& scheme = db24_scheme;
};

}  // namespace ttwv
