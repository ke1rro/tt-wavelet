#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct sym19_tag {};

inline constexpr auto sym19_scheme = make_lifting_scheme(
    38,
    9,
    10,
    PredictStep<1>{{1.1778363807084933f}, {0x3f96c358u}, 0},
    UpdateStep<2>{{-0.49337624153367698f, 0.10718239644640744f}, {0xbefc9bd0u, 0x3ddb8272u}, 0},
    PredictStep<2>{{-0.57331670044723571f, 17.757542226642059f}, {0xbf12c4e2u, 0x418e0f72u}, -1},
    UpdateStep<2>{{-0.055431587950508322f, -0.001248308377124182f}, {0xbd630c3cu, 0xbaa39e47u}, 0},
    PredictStep<2>{{24.730098517753493f, -653.22564945235763f}, {0x41c5d73eu, 0xc4234e71u}, -1},
    UpdateStep<2>{{0.0014649499135974122f, -0.00020422129308200461f}, {0x3ac00390u, 0xb956243cu}, 0},
    PredictStep<2>{{1476.9960705811795f, 1806.1683842259984f}, {0x44b89fe0u, 0x44e1c563u}, -1},
    UpdateStep<2>{{-0.00021726882356400698f, -0.00026517118927846662f}, {0xb963d2a8u, 0xb98b06adu}, 0},
    PredictStep<2>{{1849.1367685612267f, 283.20656873397149f}, {0x44e72460u, 0x438d9a71u}, -1},
    UpdateStep<2>{{-7.7926694336290638e-05f, 0.00052928445612608881f}, {0xb8a36c93u, 0x3a0abfaeu}, 0},
    PredictStep<2>{{-963.68810434168074f, -210.54836117513807f}, {0xc470ec0au, 0xc3528c61u}, -1},
    UpdateStep<2>{{0.00022485511915643793f, 0.0029444257535002722f}, {0x396bc716u, 0x3b40f744u}, 0},
    PredictStep<2>{{-303.95536993538076f, -37.871315970072395f}, {0xc397fa4au, 0xc2177c3au}, -1},
    UpdateStep<2>{{0.003084950424882933f, 0.04511574939929059f}, {0x3b4a2ce1u, 0x3d38cb4bu}, 0},
    PredictStep<2>{{-21.408529081459903f, -2.4242704809051596f}, {0xc1ab44abu, 0xc01b273fu}, -1},
    UpdateStep<2>{{0.1098123428743676f, 0.095290722493559687f}, {0x3de0e54bu, 0x3dc327c8u}, 0},
    PredictStep<2>{{-2.2517435869776903f, 1.3460695949601784f}, {0xc0101c91u, 0x3fac4c02u}, -1},
    UpdateStep<2>{{-0.066075129588057058f, -0.13300237155989392f}, {0xbd875266u, 0xbe0831c6u}, 0},
    PredictStep<2>{{2.1310670509213989f, -10.811191962466632f}, {0x40086367u, 0xc12cfaa4u}, -1},
    SwapStep{{}, {}, 0},
    PredictStep<1>{{0.08422345335496273f}, {0x3dac7d59u}, 0},
    ScaleEvenStep<1>{{-0.16247171486763512f}, {0xbe265efcu}, 0},
    ScaleOddStep<1>{{6.1549174932676429f}, {0x40c4f516u}, 0});

template <>
struct scheme_traits<sym19_tag> {
    using SchemeType = decltype(sym19_scheme);
    static constexpr const char* name = "sym19";
    static constexpr int id = 96;
    static constexpr int tap_size = 38;
    static constexpr int delay_even = 9;
    static constexpr int delay_odd = 10;
    static constexpr int num_steps = 23;
    static constexpr const auto& scheme = sym19_scheme;
};

}  // namespace ttwv
