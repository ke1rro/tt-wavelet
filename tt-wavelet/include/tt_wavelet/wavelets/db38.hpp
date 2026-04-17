#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct db38_tag {};

inline constexpr auto db38_scheme = make_lifting_scheme(
    76,
    19,
    19,
    PredictStep<1>{{-0.58692022496048613f}, {0xbf164067u}, -1},
    UpdateStep<2>{{-0.66158611960747793f, 0.18719239452388931f}, {0xbf295db5u, 0x3e3faf5du}, 0},
    PredictStep<2>{{-0.51834627340959105f, 0.42450891741754376f}, {0xbf04b257u, 0x3ed9593cu}, -1},
    UpdateStep<2>{{-0.50644837490359829f, 0.6147347720866243f}, {0xbf01a69au, 0x3f1d5f42u}, 0},
    PredictStep<2>{{-0.54316572562149801f, 0.62463252936875102f}, {0xbf0b0ce9u, 0x3f1fe7ebu}, -1},
    UpdateStep<2>{{-0.69248174233791504f, 0.6889066565814298f}, {0xbf31467cu, 0x3f305c30u}, 0},
    PredictStep<2>{{-0.74922615399857317f, 0.68770086410163389f}, {0xbf3fcd49u, 0x3f300d2au}, -1},
    UpdateStep<2>{{-0.85745482537113549f, 0.76179689808864304f}, {0xbf5b8229u, 0x3f43051fu}, 0},
    PredictStep<2>{{-0.86613236149755524f, 0.7563996353369129f}, {0xbf5dbadau, 0x3f41a368u}, -1},
    UpdateStep<2>{{-0.96186588890707225f, 0.82922652993791823f}, {0xbf763cd8u, 0x3f544831u}, 0},
    PredictStep<2>{{-0.95486781919095043f, 0.82635779153175037f}, {0xbf747238u, 0x3f538c2fu}, -1},
    UpdateStep<2>{{-1.0295283329193781f, 1.1171947536639995f}, {0xbf83c796u, 0x3f8f003du}, 0},
    PredictStep<2>{{-0.77421754182182234f, -0.28651548780556724f}, {0xbf46331fu, 0xbe92b228u}, -1},
    UpdateStep<2>{{-0.90528772462350959f, -0.15061375077390909f}, {0xbf67c0f0u, 0xbe1a3a7eu}, 0},
    PredictStep<2>{{-0.035413119635738337f, 0.50303840590061932f}, {0xbd110d59u, 0x3f00c720u}, -1},
    UpdateStep<2>{{-3.4326268649992882f, 3.8251486673057546f}, {0xc05bb029u, 0x4074cf3cu}, 0},
    PredictStep<2>{{-0.26314829547890944f, 0.20281383287740953f}, {0xbe86bb60u, 0x3e4fae6eu}, -1},
    UpdateStep<2>{{-4.9883458544803299f, 3.6323335388013986f}, {0xc09fa087u, 0x40687827u}, 0},
    PredictStep<2>{{-0.2850086775051246f, 0.19675609113986295f}, {0xbe91eca8u, 0x3e497a6eu}, -1},
    UpdateStep<2>{{-5.3638836846974627f, 3.4776544859504681f}, {0xc0aba4efu, 0x405e91e4u}, 0},
    PredictStep<2>{{-0.30804042537801618f, 0.18526919441918968f}, {0xbe9db77au, 0x3e3db735u}, -1},
    UpdateStep<2>{{-5.8514514310128085f, 2.9591744767242534f}, {0xc0bb3f17u, 0x403d631du}, 0},
    PredictStep<2>{{-0.36076403402376112f, 0.045475537967697738f}, {0xbeb8b610u, 0x3d3a448fu}, -1},
    UpdateStep<2>{{-0.31849852103676729f, 0.44148293098427127f}, {0xbea3123du, 0x3ee20a0du}, 0},
    PredictStep<2>{{-0.012833020047569548f, 0.1678634732242148f}, {0xbc524196u, 0x3e2be467u}, -1},
    UpdateStep<2>{{-5.1685161107518098f, 2.5168014232011258f}, {0xc0a5647cu, 0x40211346u}, 0},
    PredictStep<2>{{-0.4380315814074725f, 0.18235687140841536f}, {0xbee045adu, 0x3e3abbc2u}, -1},
    UpdateStep<2>{{-6.1436712885178304f, 2.2827024645852139f}, {0xc0c498f5u, 0x401217ccu}, 0},
    PredictStep<2>{{-0.49533656735820492f, 0.16276900515128054f}, {0xbefd9cc1u, 0x3e26acebu}, -1},
    UpdateStep<2>{{-7.0186908975066711f, 2.0188292203484917f}, {0xc0e0991eu, 0x4001347fu}, 0},
    PredictStep<2>{{-0.57295231009235958f, 0.14247671101787809f}, {0xbf12ad01u, 0x3e11e56au}, -1},
    UpdateStep<2>{{-8.2457819027539543f, 1.7453459599679406f}, {0xc103eeb9u, 0x3fdf677fu}, 0},
    PredictStep<2>{{-0.68693524412872631f, 0.12127412679341069f}, {0xbf2fdafdu, 0x3df85e92u}, -1},
    UpdateStep<2>{{-10.164741040848345f, 1.4557412922784212f}, {0xc122a2c8u, 0x3fba55bbu}, 0},
    PredictStep<2>{{-0.88169781872062747f, 0.09837928934749636f}, {0xbf61b6f3u, 0x3dc97b15u}, -1},
    UpdateStep<2>{{-13.909970837779319f, 1.1341754269632114f}, {0xc15e8f3eu, 0x3f912ca9u}, 0},
    PredictStep<2>{{-1.3567249908280941f, 0.071890876815069352f}, {0xbfada92au, 0x3d933b86u}, -1},
    UpdateStep<2>{{4.1326821323165235e-23f, 0.73706904992561351f}, {0x1a47d820u, 0x3f3cb08fu}, 0},
    PredictStep<1>{{-0.035057160459905271f}, {0xbd0f9819u}, 0},
    ScaleEvenStep<1>{{170661.7456029701f}, {0x4826a970u}, 0},
    ScaleOddStep<1>{{5.8595439561857884e-06f}, {0x36c49d19u}, 0});

template <>
struct scheme_traits<db38_tag> {
    using SchemeType = decltype(db38_scheme);
    static constexpr const char* name = "db38";
    static constexpr int id = 63;
    static constexpr int tap_size = 76;
    static constexpr int delay_even = 19;
    static constexpr int delay_odd = 19;
    static constexpr int num_steps = 41;
    static constexpr const auto& scheme = db38_scheme;
};

}  // namespace ttwv
