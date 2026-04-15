#pragma once

#include "../scheme.hpp"

namespace ttwv {

struct sym17_tag {};

inline constexpr auto sym17_scheme = make_lifting_scheme(
    34,
    8,
    9,
    PredictStep<1>{ { -0.6469407750679157f }, 0 },
    UpdateStep<2>{ { 0.45606346531210418f, -0.15793165712934559f }, 0 },
    PredictStep<2>{ { 0.30260759391315445f, -1.3603548304100614f }, -1 },
    UpdateStep<2>{ { 0.37015847013579722f, 0.081358883639484306f }, 0 },
    PredictStep<2>{ { -0.57413725799660142f, -4.4113458878303762f }, -1 },
    UpdateStep<2>{ { 0.16845436441586983f, 0.056285902523750944f }, 0 },
    PredictStep<2>{ { -4.3370763943137911f, -15.669598793469076f }, -1 },
    UpdateStep<2>{ { 0.051581978168794312f, 0.0041222814263238055f }, 0 },
    PredictStep<2>{ { -6.3601386107272964f, 62.619534355194965f }, -1 },
    UpdateStep<2>{ { -0.0115457011961176f, -0.00238448914246786f }, 0 },
    PredictStep<2>{ { 42.009169906778531f, 190.56120536809206f }, -1 },
    UpdateStep<2>{ { -0.0036529432461494971f, -0.0016349940447680984f }, 0 },
    PredictStep<2>{ { 192.38376054655799f, 270.75871431524058f }, -1 },
    UpdateStep<2>{ { -0.0017585719205422626f, -0.00025741990749812748f }, 0 },
    PredictStep<2>{ { 74.212900761230969f, -672.04955198264952f }, -1 },
    UpdateStep<2>{ { 0.00091505260997855917f, 0.00026931259913100101f }, 0 },
    PredictStep<2>{ { -451.26632944432856f, -1070.3699595149076f }, -1 },
    SwapStep{ {}, 0 },
    PredictStep<1>{ { 0.00040890127199186385f }, 0 },
    ScaleEvenStep<1>{ { -0.02152870278895368f }, 0 },
    ScaleOddStep<1>{ { 46.449617043953864f }, 0 }
);

template <>
struct scheme_traits<sym17_tag> {
    using SchemeType = decltype(sym17_scheme);
    static constexpr const char* name = "sym17";
    static constexpr int id         = 94;
    static constexpr int tap_size   = 34;
    static constexpr int delay_even = 8;
    static constexpr int delay_odd  = 9;
    static constexpr int num_steps  = 21;
    static constexpr const auto& scheme = sym17_scheme;
};

}
