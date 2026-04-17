#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct coif14_tag {};

inline constexpr auto coif14_scheme = make_lifting_scheme(
    84,
    21,
    21,
    PredictStep<1>{{0.76521979648337257f}, {0x3f43e572u}, -1},
    UpdateStep<2>{{2.0374084626090685f, -0.48261759344277227f}, {0x400264e6u, 0xbef719a7u}, 0},
    PredictStep<2>{{0.39806412700880228f, -0.44789971817584134f}, {0x3ecbcf10u, 0xbee5531du}, -1},
    UpdateStep<2>{{2.073956206853699f, -2.2403573952170572f}, {0x4004bbb3u, 0xc00f6204u}, 0},
    PredictStep<2>{{0.31291007674031601f, -0.42237499006192925f}, {0x3ea035c0u, 0xbed84189u}, -1},
    UpdateStep<2>{{1.5584204819221406f, -2.5404989688164208f}, {0x3fc77a53u, 0xc0229789u}, 0},
    PredictStep<2>{{0.26244987338594916f, -0.38070792300629475f}, {0x3e865fd4u, 0xbec2ec26u}, -1},
    UpdateStep<2>{{1.5582264101513275f, -1.5600422041039557f}, {0x3fc773f7u, 0xbfc7af77u}, 0},
    PredictStep<2>{{0.23836964853394316f, -0.26239541079516321f}, {0x3e74172cu, 0xbe8658b1u}, -1},
    UpdateStep<2>{{0.93132717320274694f, -1.5244507483530496f}, {0x3f6e6b75u, 0xbfc32134u}, 0},
    PredictStep<2>{{0.15988011917763784f, -0.18858176327415055f}, {0x3e23b79du, 0xbe411b94u}, -1},
    UpdateStep<2>{{0.61664827748144935f, -0.83056049438762891f}, {0x3f1ddca9u, 0xbf549f9du}, 0},
    PredictStep<2>{{0.062777799781643717f, -0.12623721452341582f}, {0x3d8091a6u, 0xbe014454u}, -1},
    UpdateStep<2>{{0.1339078781872747f, -0.32570727317182102f}, {0x3e091f26u, 0xbea6c31bu}, 0},
    PredictStep<2>{{-0.019568527394751838f, -0.026258044817136988f}, {0xbca04e2du, 0xbcd71b1cu}, -1},
    UpdateStep<2>{{-0.37522132290520421f, 0.099949181700875536f}, {0xbec01d02u, 0x3dccb228u}, 0},
    PredictStep<2>{{-0.10112877630315804f, 0.07162258134469987f}, {0xbdcf1c9bu, 0x3d92aedcu}, -1},
    UpdateStep<2>{{-0.79588690495200409f, 0.51607685969243822f}, {0xbf4bbf3fu, 0x3f041d9du}, 0},
    PredictStep<2>{{-0.21037162030385459f, 0.14548018265734958f}, {0xbe576ba8u, 0x3e14f8c2u}, -1},
    UpdateStep<2>{{-1.0143045307478267f, 1.0218673254231661f}, {0xbf81d4bbu, 0x3f82cc8cu}, 0},
    PredictStep<2>{{-0.30847375876251176f, 0.20947231117449625f}, {0xbe9df046u, 0x3e567fe9u}, -1},
    UpdateStep<2>{{-1.6596573578304827f, 1.1965972619364931f}, {0xbfd46fa7u, 0x3f992a19u}, 0},
    PredictStep<2>{{-0.29966488052224943f, 0.31905871156057031f}, {0xbe996dadu, 0x3ea35baau}, -1},
    UpdateStep<2>{{-1.7225306981312161f, 1.6625513014926698f}, {0xbfdc7be3u, 0x3fd4ce7bu}, 0},
    PredictStep<2>{{-0.43713845986858796f, 0.29951466743319793f}, {0xbedfd09du, 0x3e9959fdu}, -1},
    UpdateStep<2>{{-2.578930441129839f, 1.5880838475065475f}, {0xc0250d32u, 0x3fcb4655u}, 0},
    PredictStep<2>{{-0.4538071669918945f, 0.33226018015362319f}, {0xbee8596au, 0x3eaa1e02u}, -1},
    UpdateStep<2>{{-18.484791985447387f, 2.0224889400902404f}, {0xc193e0dbu, 0x40017075u}, 0},
    PredictStep<2>{{0.037543215325781094f, 0.054040600202731659f}, {0x3d19c6eau, 0x3d5d59adu}, -1},
    UpdateStep<2>{{-13.433636758966786f, -26.576946357791797f}, {0xc156f02du, 0xc1d49d96u}, 0},
    PredictStep<2>{{-0.11976483005018371f, 0.073798501249187021f}, {0xbdf54743u, 0x3d9723abu}, -1},
    UpdateStep<2>{{-16.070621802317405f, 8.3222285613890765f}, {0xc18090a2u, 0x410527d9u}, 0},
    PredictStep<2>{{-0.1357363099738676f, 0.062170317611729753f}, {0xbe0afe76u, 0x3d7ea64eu}, -1},
    UpdateStep<2>{{-18.327357537528609f, 7.3658577405245316f}, {0xc1929e6eu, 0x40ebb51bu}, 0},
    PredictStep<2>{{-0.15712499086342097f, 0.054561603089160249f}, {0xbe20e560u, 0x3d5f7bfdu}, -1},
    UpdateStep<2>{{-21.604023771834854f, 6.3643369387644713f}, {0xc1acd50au, 0x40cba8a6u}, 0},
    PredictStep<2>{{-0.18935895947583978f, 0.046287659034964165f}, {0xbe41e751u, 0x3d3d9821u}, -1},
    UpdateStep<2>{{-26.799349318386785f, 5.280975265251123f}, {0xc1d66511u, 0x40a8fdc0u}, 0},
    PredictStep<2>{{-0.24472882427912399f, 0.037314338769622854f}, {0xbe7a9a31u, 0x3d18d6ecu}, -1},
    UpdateStep<2>{{-36.93856138458893f, 4.0861553718842929f}, {0xc213c116u, 0x4082c1c9u}, 0},
    PredictStep<2>{{-0.37933515755161873f, 0.027071980134478139f}, {0xbec23838u, 0x3cddc60fu}, -1},
    UpdateStep<2>{{9.6107365212083286e-17f, 2.6361911889591254f}, {0x24dd9bc6u, 0x4028b75bu}, 0},
    PredictStep<1>{{-0.013106145914714576f}, {0xbc56bb29u}, 0},
    ScaleEvenStep<1>{{-3417.272265676947f}, {0xc555945bu}, 0},
    ScaleOddStep<1>{{-0.00029263105841579885f}, {0xb9996c47u}, 0});

template <>
struct scheme_traits<coif14_tag> {
    using SchemeType = decltype(coif14_scheme);
    static constexpr const char* name = "coif14";
    static constexpr int id = 20;
    static constexpr int tap_size = 84;
    static constexpr int delay_even = 21;
    static constexpr int delay_odd = 21;
    static constexpr int num_steps = 45;
    static constexpr const auto& scheme = coif14_scheme;
};

}  // namespace ttwv
