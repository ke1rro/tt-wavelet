#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct coif13_tag {};

inline constexpr auto coif13_scheme = make_lifting_scheme(
    78,
    19,
    20,
    PredictStep<1>{{-1.3259810519686346f}, {0xbfa9b9bfu}, 0},
    UpdateStep<2>{{0.48073695634536534f, 1.1206680619385954f}, {0x3ef62328u, 0x3f8f720du}, 0},
    PredictStep<2>{{-0.8077813837373089f, 0.72086256510491398f}, {0xbf4ecac3u, 0x3f388a73u}, -1},
    UpdateStep<2>{{-1.2260912274308338f, 1.1072796993843563f}, {0xbf9cf08fu, 0x3f8dbb57u}, 0},
    PredictStep<2>{{-0.77778316839336537f, 0.54743274551398358f}, {0xbf471cccu, 0x3f0c248du}, -1},
    UpdateStep<2>{{-1.3783579635547847f, 0.81819999945628785f}, {0xbfb06e09u, 0x3f51758eu}, 0},
    PredictStep<2>{{-0.63552548589081803f, 0.46332889179861464f}, {0xbf22b1ccu, 0x3eed3972u}, -1},
    UpdateStep<2>{{-0.78862176519404847f, 0.81896031325736562f}, {0xbf49e31eu, 0x3f51a762u}, 0},
    PredictStep<2>{{-0.46775045601703891f, 0.36006576595625917f}, {0xbeef7cfdu, 0x3eb85a8au}, -1},
    UpdateStep<2>{{-0.74701102410866349f, 0.44954696199256478f}, {0xbf3f3c1du, 0x3ee62b05u}, 0},
    PredictStep<2>{{-0.26155933568150291f, 0.23395128957575803f}, {0xbe85eb1bu, 0x3e6f90edu}, -1},
    UpdateStep<2>{{-0.41177889098228021f, 0.20362289621796378f}, {0xbed2d4afu, 0x3e508285u}, 0},
    PredictStep<2>{{-0.12476818523732612f, 0.045677508421504844f}, {0xbdff8676u, 0x3d3b1857u}, -1},
    UpdateStep<2>{{-0.076223241978223583f, -0.065799514456700112f}, {0xbd9c1aeeu, 0xbd86c1e5u}, 0},
    PredictStep<2>{{0.039466002941631474f, -0.1316979474039017f}, {0x3d21a71au, 0xbe06dbd4u}, -1},
    UpdateStep<2>{{0.21394138737425236f, -0.30541505497479887f}, {0x3e5b1373u, 0xbe9c5f5du}, 0},
    PredictStep<2>{{0.18266571636310072f, -0.33384721129541417f}, {0x3e3b0cb9u, 0xbeaaee06u}, -1},
    UpdateStep<2>{{0.49372664802042016f, -0.51569892232452208f}, {0x3efcc9bdu, 0xbf0404d8u}, 0},
    PredictStep<2>{{0.34355215914110193f, -0.45717878955130448f}, {0x3eafe612u, 0xbeea1357u}, -1},
    UpdateStep<2>{{0.60379813615956957f, -0.9410396401402068f}, {0x3f1a9284u, 0xbf70e7f9u}, 0},
    PredictStep<2>{{0.51096565498262181f, -0.51380203087298704f}, {0x3f02cea5u, 0xbf038888u}, -1},
    UpdateStep<2>{{0.94122509656866082f, -0.92258373615789757f}, {0x3f70f421u, 0xbf6c2e73u}, 0},
    PredictStep<2>{{0.51336247506069788f, -0.72762772448303448f}, {0x3f036bb9u, 0xbf3a45d0u}, -1},
    UpdateStep<2>{{0.8848247815392587f, -1.4712779923987136f}, {0x3f6283e0u, 0xbfbc52d6u}, 0},
    PredictStep<2>{{0.56635874285804777f, -0.80396496626669045f}, {0x3f10fce3u, 0xbf4dd0a6u}, -1},
    UpdateStep<2>{{1.1314831294903154f, -8.2053977084440479f}, {0x3f90d470u, 0xc103494fu}, 0},
    PredictStep<2>{{0.12164132329268246f, 0.13397454964790603f}, {0x3df91f16u, 0x3e0930a0u}, -1},
    UpdateStep<2>{{-7.4525044187069875f, -3.1283852846521163f}, {0xc0ee7aebu, 0xc0483777u}, 0},
    PredictStep<2>{{0.31685491862263604f, -0.53180448725847285f}, {0x3ea23acfu, 0xbf082457u}, -1},
    UpdateStep<2>{{1.8745124691407373f, -3.7636140985928082f}, {0x3feff006u, 0xc070df0eu}, 0},
    PredictStep<2>{{0.26549553948220467f, -0.6095981497095263f}, {0x3e87ef08u, 0xbf1c0ea0u}, -1},
    UpdateStep<2>{{1.6401829269264072f, -4.3551051252747932f}, {0x3fd1f184u, 0xc08b5d05u}, 0},
    PredictStep<2>{{0.22961077736063318f, -0.71853053525290156f}, {0x3e6b1f16u, 0xbf37f19eu}, -1},
    UpdateStep<2>{{1.3917263274771501f, -5.2526441551464931f}, {0x3fb22417u, 0xc0a815a9u}, 0},
    PredictStep<2>{{0.19038027714061015f, -0.89258744729254369f}, {0x3e42f30cu, 0xbf64809cu}, -1},
    UpdateStep<2>{{1.1203384006893005f, -6.8007913184985016f}, {0x3f8f6740u, 0xc0d9a015u}, 0},
    PredictStep<2>{{0.14704171220409792f, -1.2327533537053577f}, {0x3e16921au, 0xbf9dcaddu}, -1},
    UpdateStep<2>{{0.81119227702087138f, -10.56341334987477f}, {0x3f4faa4cu, 0xc12903beu}, 0},
    PredictStep<2>{{0.094666370317870743f, -2.5517266227422764f}, {0x3dc1e071u, 0xc0234f7du}, -1},
    SwapStep{{}, {}, 0},
    PredictStep<1>{{0.39189151027680436f}, {0x3ec8a601u}, 0},
    ScaleEvenStep<1>{{4.2572096959339881e-05f}, {0x38328f71u}, 0},
    ScaleOddStep<1>{{-23489.564090655167f}, {0xc6b78321u}, 0});

template <>
struct scheme_traits<coif13_tag> {
    using SchemeType = decltype(coif13_scheme);
    static constexpr const char* name = "coif13";
    static constexpr int id = 19;
    static constexpr int tap_size = 78;
    static constexpr int delay_even = 19;
    static constexpr int delay_odd = 20;
    static constexpr int num_steps = 43;
    static constexpr const auto& scheme = coif13_scheme;
};

}  // namespace ttwv
