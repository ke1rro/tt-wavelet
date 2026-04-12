#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct sym20_tag {};

inline constexpr auto sym20_scheme = make_lifting_scheme(
    40,
    10,
    10,
    PredictStep<1>{{-1.9434163137621281f}, {0xbff8c1deu}, -1},
    UpdateStep<2>{{0.09324917455266675f, 0.4068391116174333f}, {0x3dbef96cu, 0x3ed04d37u}, 0},
    PredictStep<2>{{19.728621774552529f, -1.7755125967510852f}, {0x419dd438u, 0xbfe343ffu}, -1},
    UpdateStep<2>{{-0.00092867047036839607f, -0.048699829260086543f}, {0xba737205u, 0xbd477979u}, 0},
    PredictStep<2>{{627.87957587331437f, 9.5077414868753216f}, {0x441cf84bu, 0x41181fb6u}, -1},
    UpdateStep<2>{{-0.002029574922537772f, -0.0015526954539577006f}, {0xbb05029eu, 0xbacb83d0u}, 0},
    PredictStep<2>{{-57.502915407804537f, 485.40136147627993f}, {0xc26602fcu, 0x43f2b360u}, -1},
    UpdateStep<2>{{-0.0055900197344689439f, 0.0083870499917810277f}, {0xbbb72c7cu, 0x3c0969d6u}, 0},
    PredictStep<2>{{-37.308064026177121f, 52.360590021358824f}, {0xc2153b75u, 0x4251713fu}, -1},
    UpdateStep<2>{{-0.00067508812860207421f, 0.0046535851983408533f}, {0xba30f866u, 0x3b987d1au}, 0},
    PredictStep<2>{{24.542230423391011f, 6.5204530803863676f}, {0x41c4567du, 0x40d0a78du}, -1},
    UpdateStep<2>{{0.0045921166907157253f, -0.0024017508991247064f}, {0x3b967978u, 0xbb1d66b2u}, 0},
    PredictStep<2>{{94.394802294041639f, -40.573182084223902f}, {0x42bcca24u, 0xc2224af0u}, -1},
    UpdateStep<2>{{0.0073004968543325987f, -0.0058631787189556295f}, {0x3bef3902u, 0xbbc01fe8u}, 0},
    PredictStep<2>{{1039.6489124780501f, -90.092049248909831f}, {0x4481f4c4u, 0xc2b42f21u}, -1},
    UpdateStep<2>{{-1.9554620734215174e-05f, -0.00095811891354331759f}, {0xb7a4093au, 0xba7b2a46u}, 0},
    PredictStep<2>{{5977.2450943685235f, 4925.8113611579111f}, {0x45bac9f6u, 0x4599ee7eu}, -1},
    UpdateStep<2>{{2.4893783839747609e-05f, -2.2695759144272928e-05f}, {0x37d0d2feu, 0xb7be62c6u}, 0},
    PredictStep<2>{{-16728.730223207454f, -6380.3553605142451f}, {0xc682b176u, 0xc5c762d8u}, -1},
    UpdateStep<2>{{1.7304421578720832e-05f, 3.3765084935234443e-05f}, {0x379128f6u, 0x380d9efcu}, 0},
    PredictStep<1>{{-14692.766999446822f}, {0xc6659311u}, 0},
    ScaleEvenStep<1>{{-158.629869059329f}, {0xc31ea13fu}, 0},
    ScaleOddStep<1>{{-0.0063039830136025071f}, {0xbbce91a4u}, 0});

template <>
struct scheme_traits<sym20_tag> {
    using SchemeType = decltype(sym20_scheme);
    static constexpr const char* name = "sym20";
    static constexpr int id = 98;
    static constexpr int tap_size = 40;
    static constexpr int delay_even = 10;
    static constexpr int delay_odd = 10;
    static constexpr int num_steps = 23;
    static constexpr const auto& scheme = sym20_scheme;
};

}  // namespace ttwv
