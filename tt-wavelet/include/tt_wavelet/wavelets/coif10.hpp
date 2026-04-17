#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct coif10_tag {};

inline constexpr auto coif10_scheme = make_lifting_scheme(
    60,
    15,
    15,
    PredictStep<1>{{0.7132767894922879f}, {0x3f36994fu}, -1},
    UpdateStep<2>{{1.7540798088726735f, -0.47275577508579603f}, {0x3fe085b0u, 0xbef20d0cu}, 0},
    PredictStep<2>{{0.44496621786329815f, -0.49887195595502387f}, {0x3ee3d29du, 0xbeff6c25u}, -1},
    UpdateStep<2>{{1.5134785487872304f, -1.9054055413375817f}, {0x3fc1b9aau, 0xbff3e454u}, 0},
    PredictStep<2>{{0.2934266713265814f, -0.51440678456355393f}, {0x3e963c05u, 0xbf03b02au}, -1},
    UpdateStep<2>{{1.0623615323309017f, -1.8183624845353186f}, {0x3f87fb76u, 0xbfe8c01au}, 0},
    PredictStep<2>{{0.25018780197300433f, -0.26432420115490501f}, {0x3e80189eu, 0xbe875580u}, -1},
    UpdateStep<2>{{0.72837736425403121f, -1.0358250848794248f}, {0x3f3a76f0u, 0xbf8495ebu}, 0},
    PredictStep<2>{{0.093939894807340718f, -0.20245131910996594f}, {0x3dc0638fu, 0xbe4f4f66u}, -1},
    UpdateStep<2>{{0.16804485684217962f, -0.38220110016211339f}, {0x3e2c13f3u, 0xbec3afddu}, 0},
    PredictStep<2>{{-0.028619420738579787f, -0.042535118830928573f}, {0xbcea7347u, 0xbd2e394eu}, -1},
    UpdateStep<2>{{-0.46082689565462032f, 0.11351170630640733f}, {0xbeebf181u, 0x3de878d3u}, 0},
    PredictStep<2>{{-0.14811445887624602f, 0.11062348113415291f}, {0xbe17ab51u, 0x3de28e90u}, -1},
    UpdateStep<2>{{-0.92350019535611316f, 0.5930398581325812f}, {0xbf6c6a82u, 0x3f17d176u}, 0},
    PredictStep<2>{{-0.33956271160327395f, 0.20499058542599516f}, {0xbeaddb2au, 0x3e51e90du}, -1},
    UpdateStep<2>{{-1.0918515695887512f, 1.1500735300282554f}, {0xbf8bc1cbu, 0x3f93359cu}, 0},
    PredictStep<2>{{-0.39482057852409669f, 0.3352944693644917f}, {0xbeca25ecu, 0x3eababb7u}, -1},
    UpdateStep<2>{{-1.9721713063898181f, 1.1262925555643799f}, {0xbffc701cu, 0x3f902a5bu}, 0},
    PredictStep<2>{{-0.59322357113186308f, 0.36030646050964804f}, {0xbf17dd80u, 0x3eb87a17u}, -1},
    UpdateStep<2>{{-6.6445300520970578f, 1.4655138831652108f}, {0xc0d49ffdu, 0x3fbb95f5u}, 0},
    PredictStep<2>{{1.0845418748876099f, 0.14940770366000031f}, {0x3f8ad245u, 0x3e18fe55u}, -1},
    UpdateStep<2>{{-0.11398380696752623f, -0.92192042573002386f}, {0xbde97058u, 0xbf6c02fau}, 0},
    PredictStep<2>{{-16.660557648355947f, 8.6942810977377594f}, {0xc18548d2u, 0x410b1bc6u}, -1},
    UpdateStep<2>{{-0.14152526973669308f, 0.05987404127341639f}, {0xbe10ec00u, 0x3d753e7cu}, 0},
    PredictStep<2>{{-20.175502878523176f, 7.0627521267771716f}, {0xc1a1676eu, 0x40e20211u}, -1},
    UpdateStep<2>{{-0.17622002603371745f, 0.049562373110663256f}, {0xbe347306u, 0x3d4b01eau}, 0},
    PredictStep<2>{{-26.247153128801219f, 5.6746996071417009f}, {0xc1d1fa2bu, 0x40b59724u}, -1},
    UpdateStep<2>{{-0.24500075235991398f, 0.038099362421490646f}, {0xbe7ae17au, 0x3d1c0e14u}, 0},
    PredictStep<2>{{-41.08186734584033f, 4.0816200992088216f}, {0xc22453d5u, 0x40829ca2u}, -1},
    UpdateStep<2>{{5.5445185208960055e-14f, 0.024341639379092464f}, {0x2979b3f2u, 0x3cc7681eu}, 0},
    PredictStep<1>{{-1.9560136108873891f}, {0xbffa5ea7u}, 0},
    ScaleEvenStep<1>{{-2437.1205319567825f}, {0xc51851eeu}, 0},
    ScaleOddStep<1>{{-0.00041032028858953995f}, {0xb9d72042u}, 0});

template <>
struct scheme_traits<coif10_tag> {
    using SchemeType = decltype(coif10_scheme);
    static constexpr const char* name = "coif10";
    static constexpr int id = 16;
    static constexpr int tap_size = 60;
    static constexpr int delay_even = 15;
    static constexpr int delay_odd = 15;
    static constexpr int num_steps = 33;
    static constexpr const auto& scheme = coif10_scheme;
};

}  // namespace ttwv
