#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct db32_tag {};

inline constexpr auto db32_scheme = make_lifting_scheme(
    64,
    16,
    16,
    PredictStep<1>{{-0.49826628928282496f}, {0xbeff1cc2u}, -1},
    UpdateStep<2>{{-0.64571688916162384f, 0.16319761715556738f}, {0xbf254db4u, 0x3e271d47u}, 0},
    PredictStep<2>{{-0.66324377975928073f, 0.3568621321575825f}, {0xbf29ca58u, 0x3eb6b6a2u}, -1},
    UpdateStep<2>{{-0.76738197172343225f, 0.45690884772776924f}, {0xbf447325u, 0x3ee9eff5u}, 0},
    PredictStep<2>{{-0.83172403425174557f, 0.45571173038356771f}, {0xbf54ebdeu, 0x3ee9530cu}, -1},
    UpdateStep<2>{{-0.94367277783822268f, 0.5121764793863115f}, {0xbf71948au, 0x3f031dffu}, 0},
    PredictStep<2>{{-0.90123779095151524f, 0.55155068260095408f}, {0xbf66b785u, 0x3f0d326du}, -1},
    UpdateStep<2>{{-0.40228379371998413f, 0.65666888367961185f}, {0xbecdf824u, 0x3f281b74u}, 0},
    PredictStep<2>{{-0.19441393277417965f, 1.0745338939066615f}, {0xbe471472u, 0x3f898a54u}, -1},
    UpdateStep<2>{{-0.47586040954581499f, 0.81786961563638838f}, {0xbef3a3fau, 0x3f515fe7u}, 0},
    PredictStep<2>{{-0.93623111339590903f, 0.81367563779715901f}, {0xbf6facd8u, 0x3f504d0cu}, -1},
    UpdateStep<2>{{-1.0449668297638925f, 0.8352803590182033f}, {0xbf85c179u, 0x3f55d4efu}, 0},
    PredictStep<2>{{-1.0943726393355921f, 0.84570635340909572f}, {0xbf8c1467u, 0x3f588036u}, -1},
    UpdateStep<2>{{-1.1436641888944212f, 0.84974187014990665f}, {0xbf926397u, 0x3f5988afu}, 0},
    PredictStep<2>{{-1.1887454317464654f, 0.86296004383992109f}, {0xbf9828cfu, 0x3f5ceaf3u}, -1},
    UpdateStep<2>{{-1.2080604594815054f, 7.9764557038847226f}, {0xbf9aa1bau, 0x40ff3f20u}, 0},
    PredictStep<2>{{-0.1260979588431568f, -0.00031463754953975104f}, {0xbe011fd3u, 0xb9a4f5f0u}, -1},
    UpdateStep<2>{{-81.290298166641449f, -72.299193568768771f}, {0xc2a294a2u, 0xc2909930u}, 0},
    PredictStep<2>{{-6.3417601952339041e-05f, 0.0048539272742149685f}, {0xb884ff11u, 0x3b9f0db1u}, -1},
    UpdateStep<2>{{-249.27278439741747f, 215.11926368472942f}, {0xc37945d5u, 0x43571e88u}, 0},
    PredictStep<2>{{-0.005141547887583647f, 0.0026292389768268216f}, {0xbba87a6eu, 0x3b2c4f4fu}, -1},
    UpdateStep<2>{{-425.41405914258638f, 194.1479117949288f}, {0xc3d4b500u, 0x434225deu}, 0},
    PredictStep<2>{{-0.0058286788340899095f, 0.0023504937338808467f}, {0xbbbefe80u, 0x3b1a0abeu}, -1},
    UpdateStep<2>{{-487.2532963870637f, 171.56370097293035f}, {0xc3f3a06cu, 0x432b904fu}, 0},
    PredictStep<2>{{-0.0067658410780222578f, 0.0020523180304839908f}, {0xbbddb3fdu, 0x3b06802fu}, -1},
    UpdateStep<2>{{-574.79841839461346f, 147.80126686512045f}, {0xc40fb319u, 0x4313cd20u}, 0},
    PredictStep<2>{{-0.0081477735613048486f, 0.0017397403337327307f}, {0xbc057e3du, 0x3ae40800u}, -1},
    UpdateStep<2>{{-711.80031527711697f, 122.73291496411819f}, {0xc431f338u, 0x42f57741u}, 0},
    PredictStep<2>{{-0.010505824226723795f, 0.0014048883914868349f}, {0xbc2c209fu, 0x3ab8243bu}, -1},
    UpdateStep<2>{{-978.48236825560662f, 95.185297071323504f}, {0xc4749edfu, 0x42be5edfu}, 0},
    PredictStep<2>{{-0.016237724557235138f, 0.0010219908221573296f}, {0xbc8504fau, 0x3a85f452u}, -1},
    UpdateStep<2>{{1.3258790745075123e-17f, 61.584983565595962f}, {0x237494deu, 0x42765706u}, 0},
    PredictStep<1>{{-0.00049622754886128934f}, {0xba021544u}, 0},
    ScaleEvenStep<1>{{2473.3993292028217f}, {0x451a9664u}, 0},
    ScaleOddStep<1>{{0.00040430188049024037f}, {0x39d3f87bu}, 0});

template <>
struct scheme_traits<db32_tag> {
    using SchemeType = decltype(db32_scheme);
    static constexpr const char* name = "db32";
    static constexpr int id = 57;
    static constexpr int tap_size = 64;
    static constexpr int delay_even = 16;
    static constexpr int delay_odd = 16;
    static constexpr int num_steps = 35;
    static constexpr const auto& scheme = db32_scheme;
};

}  // namespace ttwv
