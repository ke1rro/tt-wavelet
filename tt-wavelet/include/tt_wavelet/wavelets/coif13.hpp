#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct coif13_tag {};

inline constexpr auto coif13_scheme = make_lifting_scheme(
    78,
    19,
    20,
    PredictStep<1>{{-1.3259810519686346f}, 0},
    UpdateStep<2>{{0.48073695634536534f, 1.1206680619385954f}, 0},
    PredictStep<2>{{-0.8077813837373089f, 0.72086256510491398f}, -1},
    UpdateStep<2>{{-1.2260912274308338f, 1.1072796993843563f}, 0},
    PredictStep<2>{{-0.77778316839336537f, 0.54743274551398358f}, -1},
    UpdateStep<2>{{-1.3783579635547847f, 0.81819999945628785f}, 0},
    PredictStep<2>{{-0.63552548589081803f, 0.46332889179861464f}, -1},
    UpdateStep<2>{{-0.78862176519404847f, 0.81896031325736562f}, 0},
    PredictStep<2>{{-0.46775045601703891f, 0.36006576595625917f}, -1},
    UpdateStep<2>{{-0.74701102410866349f, 0.44954696199256478f}, 0},
    PredictStep<2>{{-0.26155933568150291f, 0.23395128957575803f}, -1},
    UpdateStep<2>{{-0.41177889098228021f, 0.20362289621796378f}, 0},
    PredictStep<2>{{-0.12476818523732612f, 0.045677508421504844f}, -1},
    UpdateStep<2>{{-0.076223241978223583f, -0.065799514456700112f}, 0},
    PredictStep<2>{{0.039466002941631474f, -0.1316979474039017f}, -1},
    UpdateStep<2>{{0.21394138737425236f, -0.30541505497479887f}, 0},
    PredictStep<2>{{0.18266571636310072f, -0.33384721129541417f}, -1},
    UpdateStep<2>{{0.49372664802042016f, -0.51569892232452208f}, 0},
    PredictStep<2>{{0.34355215914110193f, -0.45717878955130448f}, -1},
    UpdateStep<2>{{0.60379813615956957f, -0.9410396401402068f}, 0},
    PredictStep<2>{{0.51096565498262181f, -0.51380203087298704f}, -1},
    UpdateStep<2>{{0.94122509656866082f, -0.92258373615789757f}, 0},
    PredictStep<2>{{0.51336247506069788f, -0.72762772448303448f}, -1},
    UpdateStep<2>{{0.8848247815392587f, -1.4712779923987136f}, 0},
    PredictStep<2>{{0.56635874285804777f, -0.80396496626669045f}, -1},
    UpdateStep<2>{{1.1314831294903154f, -8.2053977084440479f}, 0},
    PredictStep<2>{{0.12164132329268246f, 0.13397454964790603f}, -1},
    UpdateStep<2>{{-7.4525044187069875f, -3.1283852846521163f}, 0},
    PredictStep<2>{{0.31685491862263604f, -0.53180448725847285f}, -1},
    UpdateStep<2>{{1.8745124691407373f, -3.7636140985928082f}, 0},
    PredictStep<2>{{0.26549553948220467f, -0.6095981497095263f}, -1},
    UpdateStep<2>{{1.6401829269264072f, -4.3551051252747932f}, 0},
    PredictStep<2>{{0.22961077736063318f, -0.71853053525290156f}, -1},
    UpdateStep<2>{{1.3917263274771501f, -5.2526441551464931f}, 0},
    PredictStep<2>{{0.19038027714061015f, -0.89258744729254369f}, -1},
    UpdateStep<2>{{1.1203384006893005f, -6.8007913184985016f}, 0},
    PredictStep<2>{{0.14704171220409792f, -1.2327533537053577f}, -1},
    UpdateStep<2>{{0.81119227702087138f, -10.56341334987477f}, 0},
    PredictStep<2>{{0.094666370317870743f, -2.5517266227422764f}, -1},
    SwapStep{{}, 0},
    PredictStep<1>{{0.39189151027680436f}, 0},
    ScaleEvenStep<1>{{4.2572096959339881e-05f}, 0},
    ScaleOddStep<1>{{-23489.564090655167f}, 0});

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
