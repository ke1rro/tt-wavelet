#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct sym15_tag {};

inline constexpr auto sym15_scheme = make_lifting_scheme(
    30,
    7,
    8,
    PredictStep<1>{{0.75775831325592102f}, {0x3f41fc73u}, 0},
    UpdateStep<2>{{-0.48136160529575922f, 0.14578765418332687f}, {0xbef67507u, 0x3e15495cu}, 0},
    PredictStep<2>{{-0.3431999314082787f, -1.3513976360856272f}, {0xbeafb7e7u, 0xbfacfa99u}, -1},
    UpdateStep<2>{{0.33264372517215479f, -0.073511902207933799f}, {0x3eaa5047u, 0xbd968d68u}, 0},
    PredictStep<2>{{0.52173074267610442f, 6.7891429885457102f}, {0x3f059025u, 0x40d940a9u}, -1},
    UpdateStep<2>{{-0.12829677450018243f, -0.037377207400622642f}, {0xbe03603bu, 0xbd1918d8u}, 0},
    PredictStep<2>{{9.7480213403470692f, 4.8982235184629443f}, {0x411bf7e5u, 0x409cbe3fu}, -1},
    UpdateStep<2>{{-0.025811383414566513f, 0.1104213029019846f}, {0xbcd37265u, 0x3de22490u}, 0},
    PredictStep<2>{{-6.5741928773595557f, -1.0798561478694195f}, {0xc0d25fcau, 0xbf8a38bau}, -1},
    UpdateStep<2>{{0.06176452149213537f, 1.4542485905674891f}, {0x3d7cfccbu, 0x3fba24d1u}, 0},
    PredictStep<2>{{-0.67071046971151149f, -0.042380308588806384f}, {0xbf2bb3aeu, 0xbd2d96f9u}, -1},
    UpdateStep<2>{{3.2225592566274113f, 19.047849893099094f}, {0x404e3e69u, 0x419861ffu}, 0},
    PredictStep<2>{{-0.044455009576793061f, 0.0020897802239007242f}, {0xbd361675u, 0x3b08f4b2u}, -1},
    UpdateStep<2>{{-5.7732080938428689f, -68.477197199020921f}, {0xc0b8be1fu, 0xc288f453u}, 0},
    PredictStep<2>{{0.0092307386193651673f, -0.020781289818896681f}, {0x3c173c86u, 0xbcaa3d86u}, -1},
    SwapStep{{}, {}, 0},
    PredictStep<1>{{43.163455870508663f}, {0x422ca761u}, 0},
    ScaleEvenStep<1>{{-3.8245325787523092f}, {0xc074c524u}, 0},
    ScaleOddStep<1>{{0.26146986054077059f}, {0x3e85df61u}, 0});

template <>
struct scheme_traits<sym15_tag> {
    using SchemeType = decltype(sym15_scheme);
    static constexpr const char* name = "sym15";
    static constexpr int id = 92;
    static constexpr int tap_size = 30;
    static constexpr int delay_even = 7;
    static constexpr int delay_odd = 8;
    static constexpr int num_steps = 19;
    static constexpr const auto& scheme = sym15_scheme;
};

}  // namespace ttwv
