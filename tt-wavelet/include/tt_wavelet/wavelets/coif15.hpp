#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct coif15_tag {};

inline constexpr auto coif15_scheme = make_lifting_scheme(
    90,
    22,
    23,
    PredictStep<1>{{-1.2897741618562026f}, {0xbfa51752u}, 0},
    UpdateStep<2>{{0.48423718452217374f, 1.2639499898685347f}, {0x3ef7edf0u, 0x3fa1c91du}, 0},
    PredictStep<2>{{-0.72702336976677173f, 0.64280333066785356f}, {0xbf3a1e34u, 0x3f248ec2u}, -1},
    UpdateStep<2>{{-1.397907228804814f, 1.3194470386377024f}, {0xbfb2eea0u, 0x3fa8e3a4u}, 0},
    PredictStep<2>{{-0.67267232830471391f, 0.52059816485622323f}, {0xbf2c3441u, 0x3f0545ecu}, -1},
    UpdateStep<2>{{-1.5855070452361419f, 1.0082262565477131f}, {0xbfcaf1e5u, 0x3f810d8fu}, 0},
    PredictStep<2>{{-0.65120792802994854f, 0.43388022263273901f}, {0xbf26b590u, 0x3ede258cu}, -1},
    UpdateStep<2>{{-1.0580611007234606f, 0.99248270649715475f}, {0xbf876e8cu, 0x3f7e1359u}, 0},
    PredictStep<2>{{-0.43073043718854231f, 0.43452691424516299f}, {0xbedc88b3u, 0x3ede7a50u}, -1},
    UpdateStep<2>{{-0.99376662372631031f, 0.66704857862270317f}, {0xbf7e677du, 0x3f2ac3b2u}, 0},
    PredictStep<2>{{-0.38234932922976372f, 0.28514295571405879f}, {0xbec3c34bu, 0x3e91fe42u}, -1},
    UpdateStep<2>{{-0.5610215121565959f, 0.52650627690363994f}, {0xbf0f9f1bu, 0x3f06c91eu}, 0},
    PredictStep<2>{{-0.2728352161796091f, 0.15252736975188622f}, {0xbe8bb10fu, 0x3e1c3023u}, -1},
    UpdateStep<2>{{-0.32638239466529378f, 0.22808720846718547f}, {0xbea71b98u, 0x3e698fb1u}, 0},
    PredictStep<2>{{-0.1095001816745404f, 0.032513625761338136f}, {0xbde041a2u, 0x3d052d02u}, -1},
    UpdateStep<2>{{-0.069237205762130424f, -0.071060566260341865f}, {0xbd8dcc3cu, 0xbd918834u}, 0},
    PredictStep<2>{{0.033367283876259812f, -0.090329481602721973f}, {0x3d08ac22u, 0xbdb8feaau}, -1},
    UpdateStep<2>{{0.1897005733384422f, -0.37591428825355988f}, {0x3e4240deu, 0xbec077d6u}, 0},
    PredictStep<2>{{0.17026581712957981f, -0.19418344535919665f}, {0x3e2e5a2au, 0xbe46d806u}, -1},
    UpdateStep<2>{{0.42089738596889648f, -0.69225295288589683f}, {0x3ed77fddu, 0xbf31377du}, 0},
    PredictStep<2>{{0.28027334058871389f, -0.32171545905656795f}, {0x3e8f7ffdu, 0xbea4b7e3u}, -1},
    UpdateStep<2>{{0.74842494563260942f, -0.79416379177431651f}, {0x3f3f98c7u, 0xbf4b4e51u}, 0},
    PredictStep<2>{{0.33136335444982778f, -0.50721482854371225f}, {0x3ea9a875u, 0xbf01d8d5u}, -1},
    UpdateStep<2>{{0.89812937676737958f, -1.105715028883524f}, {0x3f65ebcfu, 0xbf8d8812u}, 0},
    PredictStep<2>{{0.50567318518250948f, -0.46145766351246464f}, {0x3f0173ccu, 0xbeec442eu}, -1},
    UpdateStep<2>{{1.1131016802234768f, -1.2188572285710393f}, {0x3f8e7a1eu, 0xbf9c0383u}, 0},
    PredictStep<2>{{0.45843207556345145f, -0.68121286790140501f}, {0x3eeab79cu, 0xbf2e63f7u}, -1},
    UpdateStep<2>{{1.081277049003081f, -1.714456745036544f}, {0x3f8a6749u, 0xbfdb7352u}, 0},
    PredictStep<2>{{0.51063739680702791f, -0.67375910123183225f}, {0x3f02b922u, 0xbf2c7b7au}, -1},
    UpdateStep<2>{{1.3720982902714223f, -16.929943370953424f}, {0x3fafa0ebu, 0xc1877086u}, 0},
    PredictStep<2>{{0.059035259240292744f, 0.025228791688864768f}, {0x3d71cef5u, 0x3cceac9cu}, -1},
    UpdateStep<2>{{-39.521112347573059f, -23.187661245448869f}, {0xc21e159eu, 0xc1b98055u}, 0},
    PredictStep<2>{{0.042761335929603982f, -0.067338683475913513f}, {0x3d2f2683u, 0xbd89e8ddu}, -1},
    UpdateStep<2>{{14.799358504406761f, -27.62418098094733f}, {0x416cca2cu, 0xc1dcfe53u}, 0},
    PredictStep<2>{{0.036164435923515585f, -0.075610170311507061f}, {0x3d142129u, 0xbd9ad981u}, -1},
    UpdateStep<2>{{13.222745144271988f, -31.130228903031174f}, {0x4153905du, 0xc1f90ab5u}, 0},
    PredictStep<2>{{0.032121807502188938f, -0.086264918124265164f}, {0x3d039228u, 0xbdb0aba9u}, -1},
    UpdateStep<2>{{11.592132320892309f, -36.048470967339185f}, {0x41397960u, 0xc21031a2u}, 0},
    PredictStep<2>{{0.027740411408545204f, -0.10164250328122518f}, {0x3ce33fdcu, 0xbdd029f2u}, -1},
    UpdateStep<2>{{9.8384034647385086f, -43.396412869306701f}, {0x411d6a1au, 0xc22d95edu}, 0},
    PredictStep<2>{{0.023043379202172459f, -0.12589665617398568f}, {0x3cbcc578u, 0xbe00eb0eu}, -1},
    UpdateStep<2>{{7.9430227165289731f, -55.988947628834993f}, {0x40fe2d3eu, 0xc25ff4afu}, 0},
    PredictStep<2>{{0.017860667905882883f, -0.17320877077439453f}, {0x3c925089u, 0xbe315da4u}, -1},
    UpdateStep<2>{{5.773379693932994f, -86.620993617251841f}, {0x40b8bf87u, 0xc2ad3df3u}, 0},
    PredictStep<2>{{0.011544545476107716f, -0.35710897129418684f}, {0x3c3d2555u, 0xbeb6d6fdu}, -1},
    SwapStep{{}, {}, 0},
    PredictStep<1>{{2.8002656902623673f}, {0x4033378eu}, 0},
    ScaleEvenStep<1>{{2.7647346075936041e-05f}, {0x37e7ec39u}, 0},
    ScaleOddStep<1>{{-36169.836962050744f}, {0xc70d49d6u}, 0});

template <>
struct scheme_traits<coif15_tag> {
    using SchemeType = decltype(coif15_scheme);
    static constexpr const char* name = "coif15";
    static constexpr int id = 21;
    static constexpr int tap_size = 90;
    static constexpr int delay_even = 22;
    static constexpr int delay_odd = 23;
    static constexpr int num_steps = 49;
    static constexpr const auto& scheme = coif15_scheme;
};

}  // namespace ttwv
