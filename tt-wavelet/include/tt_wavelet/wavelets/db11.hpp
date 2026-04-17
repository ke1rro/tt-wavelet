#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct db11_tag {};

inline constexpr auto db11_scheme = make_lifting_scheme(
    22,
    5,
    6,
    PredictStep<1>{{7.7064687312047839f}, {0x40f69b64u}, 0},
    UpdateStep<2>{{-0.12761238749161716f, -0.0067210578643736793f}, {0xbe02acd2u, 0xbbdc3c52u}, 0},
    PredictStep<2>{{21.044258231146927f, -38.352439955648663f}, {0x41a85aa4u, 0xc21968e6u}, -1},
    UpdateStep<2>{{0.0092212384258733866f, -0.015002563332857616f}, {0x3c1714adu, 0xbc75cd50u}, 0},
    PredictStep<2>{{39.430635799101921f, -64.124637132163912f}, {0x421db8f9u, 0xc2803fd0u}, -1},
    UpdateStep<2>{{0.012366253295858793f, -0.021627937037283919f}, {0x3c4a9bd3u, 0xbcb12d12u}, 0},
    PredictStep<2>{{42.600626470404222f, -86.224389557954041f}, {0x422a670bu, 0xc2ac72e3u}, -1},
    UpdateStep<2>{{0.011360957218137643f, -0.028910095500299011f}, {0x3c3a234fu, 0xbcecd4ddu}, 0},
    PredictStep<2>{{34.479062680016789f, -121.54934317145921f}, {0x4209ea8fu, 0xc2f31944u}, -1},
    UpdateStep<2>{{0.0082249822155652415f, -0.046052354307299356f}, {0x3c06c213u, 0xbd3ca165u}, 0},
    PredictStep<2>{{21.71423713872737f, -259.55384374295431f}, {0x41adb6c2u, 0xc381c6e4u}, -1},
    SwapStep{{}, {}, 0},
    PredictStep<1>{{0.0038527649129863543f}, {0x3b7c7eabu}, 0},
    ScaleEvenStep<1>{{0.00096241296234657282f}, {0x3a7c4a71u}, 0},
    ScaleOddStep<1>{{-1039.0549993858995f}, {0xc481e1c3u}, 0});

template <>
struct scheme_traits<db11_tag> {
    using SchemeType = decltype(db11_scheme);
    static constexpr const char* name = "db11";
    static constexpr int id = 34;
    static constexpr int tap_size = 22;
    static constexpr int delay_even = 5;
    static constexpr int delay_odd = 6;
    static constexpr int num_steps = 15;
    static constexpr const auto& scheme = db11_scheme;
};

}  // namespace ttwv
