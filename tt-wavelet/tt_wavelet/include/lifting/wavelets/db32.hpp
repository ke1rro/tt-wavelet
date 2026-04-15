#pragma once

#include "../scheme.hpp"

namespace ttwv {

struct db32_tag {};

inline constexpr auto db32_scheme = make_lifting_scheme(
    64,
    16,
    16,
    PredictStep<1>{ { -0.49826628928282496f }, -1 },
    UpdateStep<2>{ { -0.64571688916162384f, 0.16319761715556738f }, 0 },
    PredictStep<2>{ { -0.66324377975928073f, 0.3568621321575825f }, -1 },
    UpdateStep<2>{ { -0.76738197172343225f, 0.45690884772776924f }, 0 },
    PredictStep<2>{ { -0.83172403425174557f, 0.45571173038356771f }, -1 },
    UpdateStep<2>{ { -0.94367277783822268f, 0.5121764793863115f }, 0 },
    PredictStep<2>{ { -0.90123779095151524f, 0.55155068260095408f }, -1 },
    UpdateStep<2>{ { -0.40228379371998413f, 0.65666888367961185f }, 0 },
    PredictStep<2>{ { -0.19441393277417965f, 1.0745338939066615f }, -1 },
    UpdateStep<2>{ { -0.47586040954581499f, 0.81786961563638838f }, 0 },
    PredictStep<2>{ { -0.93623111339590903f, 0.81367563779715901f }, -1 },
    UpdateStep<2>{ { -1.0449668297638925f, 0.8352803590182033f }, 0 },
    PredictStep<2>{ { -1.0943726393355921f, 0.84570635340909572f }, -1 },
    UpdateStep<2>{ { -1.1436641888944212f, 0.84974187014990665f }, 0 },
    PredictStep<2>{ { -1.1887454317464654f, 0.86296004383992109f }, -1 },
    UpdateStep<2>{ { -1.2080604594815054f, 7.9764557038847226f }, 0 },
    PredictStep<2>{ { -0.1260979588431568f, -0.00031463754953975104f }, -1 },
    UpdateStep<2>{ { -81.290298166641449f, -72.299193568768771f }, 0 },
    PredictStep<2>{ { -6.3417601952339041e-05f, 0.0048539272742149685f }, -1 },
    UpdateStep<2>{ { -249.27278439741747f, 215.11926368472942f }, 0 },
    PredictStep<2>{ { -0.005141547887583647f, 0.0026292389768268216f }, -1 },
    UpdateStep<2>{ { -425.41405914258638f, 194.1479117949288f }, 0 },
    PredictStep<2>{ { -0.0058286788340899095f, 0.0023504937338808467f }, -1 },
    UpdateStep<2>{ { -487.2532963870637f, 171.56370097293035f }, 0 },
    PredictStep<2>{ { -0.0067658410780222578f, 0.0020523180304839908f }, -1 },
    UpdateStep<2>{ { -574.79841839461346f, 147.80126686512045f }, 0 },
    PredictStep<2>{ { -0.0081477735613048486f, 0.0017397403337327307f }, -1 },
    UpdateStep<2>{ { -711.80031527711697f, 122.73291496411819f }, 0 },
    PredictStep<2>{ { -0.010505824226723795f, 0.0014048883914868349f }, -1 },
    UpdateStep<2>{ { -978.48236825560662f, 95.185297071323504f }, 0 },
    PredictStep<2>{ { -0.016237724557235138f, 0.0010219908221573296f }, -1 },
    UpdateStep<2>{ { 1.3258790745075123e-17f, 61.584983565595962f }, 0 },
    PredictStep<1>{ { -0.00049622754886128934f }, 0 },
    ScaleEvenStep<1>{ { 2473.3993292028217f }, 0 },
    ScaleOddStep<1>{ { 0.00040430188049024037f }, 0 }
);

template <>
struct scheme_traits<db32_tag> {
    using SchemeType = decltype(db32_scheme);
    static constexpr const char* name = "db32";
    static constexpr int id         = 57;
    static constexpr int tap_size   = 64;
    static constexpr int delay_even = 16;
    static constexpr int delay_odd  = 16;
    static constexpr int num_steps  = 35;
    static constexpr const auto& scheme = db32_scheme;
};

}
