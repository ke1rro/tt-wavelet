#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct db34_tag {};

inline constexpr auto db34_scheme = make_lifting_scheme(
    68,
    17,
    17,
    PredictStep<1>{{-0.46720297956375068f}, {0xbeef353bu}, -1},
    UpdateStep<2>{{-0.60413165706843497f, 0.21179549069749168f}, {0xbf1aa85fu, 0x3e58e0ebu}, 0},
    PredictStep<2>{{-0.60629354196068819f, 0.46741761132401549f}, {0xbf1b360eu, 0x3eef515cu}, -1},
    UpdateStep<2>{{-0.69045651424672416f, 0.62893006595637824f}, {0xbf30c1c2u, 0x3f210190u}, 0},
    PredictStep<2>{{-0.73036109176405939f, 0.60339999148019685f}, {0xbf3af8f2u, 0x3f1a786cu}, -1},
    UpdateStep<2>{{-0.8497823095691498f, 0.5278785379615083f}, {0xbf598b55u, 0x3f07230cu}, 0},
    PredictStep<2>{{-0.97059302264567826f, 0.38209866477914584f}, {0xbf7878c9u, 0x3ec3a270u}, -1},
    UpdateStep<2>{{-1.1840358478626407f, 0.40751633589397673f}, {0xbf978e7du, 0x3ed0a5fbu}, 0},
    PredictStep<2>{{-1.462244435345289f, 0.48836203453917115f}, {0xbfbb2ad3u, 0x3efa0a97u}, -1},
    UpdateStep<2>{{0.11167848411017281f, 0.52476607817601484f}, {0x3de4b7b0u, 0x3f065712u}, 0},
    PredictStep<2>{{0.33786249138099611f, 10.233159864565936f}, {0x3eacfc50u, 0x4123bb06u}, -1},
    UpdateStep<2>{{-0.089051683341395585f, 0.028192672574591147f}, {0xbdb660bbu, 0x3ce6f452u}, 0},
    PredictStep<2>{{-30.773187989302567f, 24.001546459316707f}, {0xc1f62f7du, 0x41c0032bu}, -1},
    UpdateStep<2>{{-0.03843255953687074f, 0.029859432879725905f}, {0xbd1d6b76u, 0x3cf49bc5u}, 0},
    PredictStep<2>{{-32.505720665020512f, 24.338366930468247f}, {0xc20205dcu, 0x41c2b4fau}, -1},
    UpdateStep<2>{{-0.041484209073867781f, 0.029725727872616622f}, {0xbd29eb59u, 0x3cf3835fu}, 0},
    PredictStep<2>{{-34.995531282551369f, 27.78189036308466f}, {0xc20bfb6du, 0x41de4150u}, -1},
    UpdateStep<2>{{-0.037928199681602361f, -0.0047371091070260822f}, {0xbd1b5a9au, 0xbb9b39c0u}, 0},
    PredictStep<2>{{155.71045081517133f, -3.4411933733593587f}, {0x431bb5e0u, 0xc05c3c83u}, -1},
    UpdateStep<2>{{0.0011633498152991242f, -0.0085431690379691051f}, {0x3a987b8bu, 0xbc0bf8a6u}, 0},
    PredictStep<2>{{4.2937922958038737f, 398.49916464323462f}, {0x408966bfu, 0x43c73fe5u}, -1},
    UpdateStep<2>{{-0.0026840341967468715f, 0.00099314444841929676f}, {0xbb2fe69fu, 0x3a822c66u}, 0},
    PredictStep<2>{{-1116.5698975308953f, 531.05940264999731f}, {0xc48b923du, 0x4404c3cdu}, -1},
    UpdateStep<2>{{-0.0021096355491019292f, 0.0008961514345138022f}, {0xbb0a41d0u, 0x3a6aebb4u}, 0},
    PredictStep<2>{{-1263.1093792358963f, 474.00943126856731f}, {0xc49de380u, 0x43ed0135u}, -1},
    UpdateStep<2>{{-0.0024147617459825705f, 0.00079169548239804352f}, {0xbb1e40fbu, 0x3a4f89c9u}, 0},
    PredictStep<2>{{-1464.5762902660504f, 414.11944066721185f}, {0xc4b71271u, 0x43cf0f4au}, -1},
    UpdateStep<2>{{-0.0028446356059430278f, 0.00068279132288543852f}, {0xbb3a6d11u, 0x3a32fd5au}, 0},
    PredictStep<2>{{-1760.9589532492828f, 351.53887440413627f}, {0xc4dc1eb0u, 0x43afc4fau}, -1},
    UpdateStep<2>{{-0.0035168578529191836f, 0.00056787240731255844f}, {0xbb667b15u, 0x3a14dd46u}, 0},
    PredictStep<2>{{-2266.8099076815461f, 284.34473095604636f}, {0xc50dacf5u, 0x438e2c20u}, -1},
    UpdateStep<2>{{-0.0048264566369945828f, 0.00044114859239466017f}, {0xbb9e2741u, 0x39e749f6u}, 0},
    PredictStep<2>{{-3497.8590875248779f, 207.19133625588651f}, {0xc55a9dbfu, 0x434f30fbu}, -1},
    UpdateStep<2>{{3.9343734282048946e-24f, 0.00028588916104896914f}, {0x18983426u, 0x3995e365u}, 0},
    PredictStep<1>{{-100.76140716615676f}, {0xc2c985d7u}, 0},
    ScaleEvenStep<1>{{-2249596.6748834746f}, {0xca094df3u}, 0},
    ScaleOddStep<1>{{-4.4452412788696817e-07f}, {0xb4eea6eeu}, 0});

template <>
struct scheme_traits<db34_tag> {
    using SchemeType = decltype(db34_scheme);
    static constexpr const char* name = "db34";
    static constexpr int id = 59;
    static constexpr int tap_size = 68;
    static constexpr int delay_even = 17;
    static constexpr int delay_odd = 17;
    static constexpr int num_steps = 37;
    static constexpr const auto& scheme = db34_scheme;
};

}  // namespace ttwv
