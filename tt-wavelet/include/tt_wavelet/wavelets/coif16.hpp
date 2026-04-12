#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct coif16_tag {};

inline constexpr auto coif16_scheme = make_lifting_scheme(
    96,
    24,
    24,
    PredictStep<1>{{0.78462438483731467f}, {0x3f48dd25u}, -1},
    UpdateStep<2>{{2.1662572116408882f, -0.48564445480673535f}, {0x400aa3f5u, 0xbef8a664u}, 0},
    PredictStep<2>{{0.37513135617811111f, -0.42678385055963142f}, {0x3ec01138u, 0xbeda836au}, -1},
    UpdateStep<2>{{2.3108623189390562f, -2.4109709887953747f}, {0x4013e52bu, 0xc01a4d59u}, 0},
    PredictStep<2>{{0.31178248262765662f, -0.38809926273452133f}, {0x3e9fa1f4u, 0xbec6b4f2u}, -1},
    UpdateStep<2>{{1.7951074078193745f, -2.7222057976263292f}, {0x3fe5c614u, 0xc02e389fu}, 0},
    PredictStep<2>{{0.25891781250578705f, -0.39512960652618934f}, {0x3e8490e0u, 0xbeca4e6eu}, -1},
    UpdateStep<2>{{1.7326020576370749f, -1.9760333764580471f}, {0x3fddc5e7u, 0xbffceea9u}, 0},
    PredictStep<2>{{0.27277111241846508f, -0.2574968891353101f}, {0x3e8ba8a8u, 0xbe83d6a2u}, -1},
    UpdateStep<2>{{1.3240624411674116f, -1.7402233523997699f}, {0x3fa97ae1u, 0xbfdebfa4u}, 0},
    PredictStep<2>{{0.17688581495655603f, -0.25950401725807615f}, {0x3e35218eu, 0xbe84ddb6u}, -1},
    UpdateStep<2>{{1.0828303377523705f, -1.1051467881230623f}, {0x3f8a9a2fu, 0xbf8d7573u}, 0},
    PredictStep<2>{{0.12319255464247061f, -0.17480373429681104f}, {0x3dfc4c61u, 0xbe32ffc0u}, -1},
    UpdateStep<2>{{0.54642753290676749f, -0.84364248358610439f}, {0x3f0be2adu, 0xbf57f8f4u}, 0},
    PredictStep<2>{{0.060345677539742626f, -0.084703252406761992f}, {0x3d772d07u, 0xbdad78e6u}, -1},
    UpdateStep<2>{{0.10478678194580753f, -0.40032726370697658f}, {0x3dd69a74u, 0xbeccf7b2u}, 0},
    PredictStep<2>{{-0.019573608855753523f, -0.016089444389004826f}, {0xbca058d5u, 0xbc83ce03u}, -1},
    UpdateStep<2>{{-0.30160725108658443f, 0.12737059424495242f}, {0xbe9a6c44u, 0x3e026d70u}, 0},
    PredictStep<2>{{-0.09391836387817934f, 0.046005105709544233f}, {0xbdc05845u, 0x3d3c6fdau}, -1},
    UpdateStep<2>{{-0.7589182828821488f, 0.58707591928043978f}, {0xbf424878u, 0x3f164a9cu}, 0},
    PredictStep<2>{{-0.14594008785178847f, 0.11738804451562071f}, {0xbe157152u, 0x3df06925u}, -1},
    UpdateStep<2>{{-1.3755596644063f, 0.89945422928710472f}, {0xbfb01257u, 0x3f6642a2u}, 0},
    PredictStep<2>{{-0.18890637691609727f, 0.18971005095822652f}, {0xbe4170acu, 0x3e42435au}, -1},
    UpdateStep<2>{{-1.6227821353161918f, 1.373750491306218f}, {0xbfcfb753u, 0x3fafd70eu}, 0},
    PredictStep<2>{{-0.30452024672464678f, 0.20229940834152479f}, {0xbe9bea14u, 0x3e4f2793u}, -1},
    UpdateStep<2>{{-1.9149053564143461f, 1.725933550317293f}, {0xbff51b9eu, 0x3fdceb64u}, 0},
    PredictStep<2>{{-0.27469932970212546f, 0.30126438036901543f}, {0xbe8ca564u, 0x3e9a3f53u}, -1},
    UpdateStep<2>{{-2.2274874254825128f, 1.9342123774787243f}, {0xc00e8f27u, 0x3ff79445u}, 0},
    PredictStep<2>{{-0.40447886292521262f, 0.26961338436262627f}, {0xbecf17dbu, 0x3e8a0ac4u}, -1},
    UpdateStep<2>{{-2.9557724846803786f, 1.9083628757157782f}, {0xc03d2b60u, 0x3ff4453cu}, 0},
    PredictStep<2>{{-0.38526167001011663f, 0.30121532562915315f}, {0xbec54105u, 0x3e9a38e5u}, -1},
    UpdateStep<2>{{-45.562104972080334f, 2.4138694686035906f}, {0xc2363f98u, 0x401a7cd6u}, 0},
    PredictStep<2>{{0.0052486807327997457f, 0.021943426093019327f}, {0x3babfd20u, 0x3cb3c2b3u}, -1},
    UpdateStep<2>{{-126.10297351658178f, -189.82280083536205f}, {0xc2fc34b9u, 0xc33dd2a3u}, 0},
    PredictStep<2>{{-0.012052059612729251f, 0.0078641954382848538f}, {0xbc457600u, 0x3c00d8d4u}, -1},
    UpdateStep<2>{{-149.77806614939269f, 82.678683359360818f}, {0xc315c72fu, 0x42a55b7cu}, 0},
    PredictStep<2>{{-0.013427546877936743f, 0.0066693016878112479f}, {0xbc5bff37u, 0x3bda8a28u}, -1},
    UpdateStep<2>{{-167.12071006659275f, 74.453818613005382f}, {0xc3271ee7u, 0x4294e85bu}, 0},
    PredictStep<2>{{-0.015138480336320811f, 0.0059833806574614151f}, {0xbc780763u, 0x3bc4103cu}, -1},
    UpdateStep<2>{{-190.7807881948176f, 66.056279786320246f}, {0xc33ec7e2u, 0x42841cd1u}, 0},
    PredictStep<2>{{-0.017527274719633668f, 0.0052416127092248356f}, {0xbc8f955cu, 0x3babc1d6u}, -1},
    UpdateStep<2>{{-224.60760178572031f, 57.053930341536116f}, {0xc3609b8cu, 0x4264373au}, 0},
    PredictStep<2>{{-0.021073914598892775f, 0.0044522090361256489f}, {0xbcaca334u, 0x3b91e3d6u}, -1},
    UpdateStep<2>{{-277.79034081643812f, 47.452028670365159f}, {0xc38ae52au, 0x423dcee1u}, 0},
    PredictStep<2>{{-0.027144903906415378f, 0.0035998371903553984f}, {0xbcde5efdu, 0x3b6beb3fu}, -1},
    UpdateStep<2>{{-381.54091179200799f, 36.839327317107582f}, {0xc3bec53du, 0x42135b79u}, 0},
    PredictStep<2>{{-0.041924904586175787f, 0.0026209509100956797f}, {0xbd2bb973u, 0x3b2bc442u}, -1},
    UpdateStep<2>{{3.4745776741938201e-18f, 23.852171158661086f}, {0x22803075u, 0x41bed13fu}, 0},
    PredictStep<1>{{-0.0012733777719509563f}, {0xbaa6e778u}, 0},
    ScaleEvenStep<1>{{-4375.3601795890281f}, {0xc588bae2u}, 0},
    ScaleOddStep<1>{{-0.00022855261257461295f}, {0xb96fa7a0u}, 0});

template <>
struct scheme_traits<coif16_tag> {
    using SchemeType = decltype(coif16_scheme);
    static constexpr const char* name = "coif16";
    static constexpr int id = 22;
    static constexpr int tap_size = 96;
    static constexpr int delay_even = 24;
    static constexpr int delay_odd = 24;
    static constexpr int num_steps = 51;
    static constexpr const auto& scheme = coif16_scheme;
};

}  // namespace ttwv
