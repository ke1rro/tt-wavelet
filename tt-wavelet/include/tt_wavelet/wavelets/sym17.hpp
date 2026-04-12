#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct sym17_tag {};

inline constexpr auto sym17_scheme = make_lifting_scheme(
    34,
    8,
    9,
    PredictStep<1>{{-0.6469407750679157f}, {0xbf259de9u}, 0},
    UpdateStep<2>{{0.45606346531210418f, -0.15793165712934559f}, {0x3ee98127u, 0xbe21b8d6u}, 0},
    PredictStep<2>{{0.30260759391315445f, -1.3603548304100614f}, {0x3e9aef62u, 0xbfae201bu}, -1},
    UpdateStep<2>{{0.37015847013579722f, 0.081358883639484306f}, {0x3ebd8569u, 0x3da69f7du}, 0},
    PredictStep<2>{{-0.57413725799660142f, -4.4113458878303762f}, {0xbf12faa9u, 0xc08d29bfu}, -1},
    UpdateStep<2>{{0.16845436441586983f, 0.056285902523750944f}, {0x3e2c7f4du, 0x3d668c0cu}, 0},
    PredictStep<2>{{-4.3370763943137911f, -15.669598793469076f}, {0xc08ac954u, 0xc17ab6adu}, -1},
    UpdateStep<2>{{0.051581978168794312f, 0.0041222814263238055f}, {0x3d5347a0u, 0x3b871434u}, 0},
    PredictStep<2>{{-6.3601386107272964f, 62.619534355194965f}, {0xc0cb8641u, 0x427a7a67u}, -1},
    UpdateStep<2>{{-0.0115457011961176f, -0.00238448914246786f}, {0xbc3d2a2eu, 0xbb1c4517u}, 0},
    PredictStep<2>{{42.009169906778531f, 190.56120536809206f}, {0x42280964u, 0x433e8fabu}, -1},
    UpdateStep<2>{{-0.0036529432461494971f, -0.0016349940447680984f}, {0xbb6f6638u, 0xbad64d4cu}, 0},
    PredictStep<2>{{192.38376054655799f, 270.75871431524058f}, {0x4340623eu, 0x4387611eu}, -1},
    UpdateStep<2>{{-0.0017585719205422626f, -0.00025741990749812748f}, {0xbae67fe2u, 0xb986f651u}, 0},
    PredictStep<2>{{74.212900761230969f, -672.04955198264952f}, {0x42946d01u, 0xc428032cu}, -1},
    UpdateStep<2>{{0.00091505260997855917f, 0.00026931259913100101f}, {0x3a6fe024u, 0x398d3286u}, 0},
    PredictStep<2>{{-451.26632944432856f, -1070.3699595149076f}, {0xc3e1a217u, 0xc485cbd7u}, -1},
    SwapStep{{}, {}, 0},
    PredictStep<1>{{0.00040890127199186385f}, {0x39d661cdu}, 0},
    ScaleEvenStep<1>{{-0.02152870278895368f}, {0xbcb05cf6u}, 0},
    ScaleOddStep<1>{{46.449617043953864f}, {0x4239cc68u}, 0});

template <>
struct scheme_traits<sym17_tag> {
    using SchemeType = decltype(sym17_scheme);
    static constexpr const char* name = "sym17";
    static constexpr int id = 94;
    static constexpr int tap_size = 34;
    static constexpr int delay_even = 8;
    static constexpr int delay_odd = 9;
    static constexpr int num_steps = 21;
    static constexpr const auto& scheme = sym17_scheme;
};

}  // namespace ttwv
