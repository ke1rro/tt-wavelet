#pragma once

#include "../lifting_scheme.hpp"

namespace ttwv {

struct coif17_tag {};

inline constexpr auto coif17_scheme = make_lifting_scheme(
    102,
    25,
    26,
    PredictStep<1>{{-1.2606934133381571f}, {0xbfa15e67u}, 0},
    UpdateStep<2>{{0.48687679857701316f, 1.4022115186424073f}, {0x3ef947eau, 0x3fb37babu}, 0},
    PredictStep<2>{{-0.66287580090348086f, 0.57894651321246771f}, {0xbf29b23au, 0x3f1435d7u}, -1},
    UpdateStep<2>{{-1.5710441023591091f, 1.5243441215430149f}, {0xbfc917f9u, 0x3fc31db5u}, 0},
    PredictStep<2>{{-0.59334427360627207f, 0.49212980578932608f}, {0xbf17e569u, 0x3efbf870u}, -1},
    UpdateStep<2>{{-1.7615108792441594f, 1.2033282051334329f}, {0xbfe17930u, 0x3f9a06a9u}, 0},
    PredictStep<2>{{-0.62527938632556179f, 0.40804332140548294f}, {0xbf20124fu, 0x3ed0eb0eu}, -1},
    UpdateStep<2>{{-1.3825793998373763f, 1.1405040398131796f}, {0xbfb0f85du, 0x3f91fc09u}, 0},
    PredictStep<2>{{-0.41053597584053897f, 0.4367681673793789f}, {0xbed231c5u, 0x3edfa014u}, -1},
    UpdateStep<2>{{-1.143435530371901f, 0.98671873549986111f}, {0xbf925c18u, 0x3f7c9999u}, 0},
    PredictStep<2>{{-0.43128694209852247f, 0.28546266477483984f}, {0xbedcd1a4u, 0x3e92282au}, -1},
    UpdateStep<2>{{-0.86163571884748302f, 0.79384522073441044f}, {0xbf5c9429u, 0x3f4b3971u}, 0},
    PredictStep<2>{{-0.27161310458083165f, 0.23566730559953233f}, {0xbe8b10dfu, 0x3e7152c5u}, -1},
    UpdateStep<2>{{-0.73288283954188127f, 0.45574727919902458f}, {0xbf3b9e36u, 0x3ee957b5u}, 0},
    PredictStep<2>{{-0.16310939875532324f, 0.14222534910374968f}, {0xbe270626u, 0x3e11a386u}, -1},
    UpdateStep<2>{{-0.40386671819187453f, 0.19985863473926727f}, {0xbecec79eu, 0x3e4ca7beu}, 0},
    PredictStep<2>{{-0.074144619572329803f, 0.028584586471573695f}, {0xbd97d922u, 0x3cea2a39u}, -1},
    UpdateStep<2>{{-0.077449663456854084f, -0.064097154162521375f}, {0xbd9e9deeu, 0xbd83455eu}, 0},
    PredictStep<2>{{0.023671237388042599f, -0.082235256909774471f}, {0x3cc1ea2fu, 0xbda86af5u}, -1},
    UpdateStep<2>{{0.22050450514821937f, -0.31214267676408458f}, {0x3e61cbefu, 0xbe9fd12au}, 0},
    PredictStep<2>{{0.11345028793121005f, -0.19867874579631237f}, {0x3de858a0u, 0xbe4b7271u}, -1},
    UpdateStep<2>{{0.5103930014468735f, -0.54014461335552066f}, {0x3f02a91eu, 0xbf0a46ebu}, 0},
    PredictStep<2>{{0.20854420467643542f, -0.29361886720120212f}, {0x3e558c9du, 0xbe965536u}, -1},
    UpdateStep<2>{{0.6804327775707284f, -0.89387462144612573f}, {0x3f2e30d8u, 0xbf64d4f8u}, 0},
    PredictStep<2>{{0.33880128922194652f, -0.3239042169421894f}, {0x3ead775du, 0xbea5d6c6u}, -1},
    UpdateStep<2>{{0.87413004809576811f, -1.1380062270402458f}, {0x3f5fc6fdu, 0xbf91aa30u}, 0},
    PredictStep<2>{{0.35293443719525508f, -0.50363239280996475f}, {0x3eb4b3d3u, 0xbf00ee0du}, -1},
    UpdateStep<2>{{1.1459721318855942f, -1.1719922072505069f}, {0x3f92af37u, 0xbf9603d7u}, 0},
    PredictStep<2>{{0.50188527120631243f, -0.46567870727743826f}, {0x3f007b8eu, 0xbeee6d71u}, -1},
    UpdateStep<2>{{1.1842407589455666f, -1.424398312207223f}, {0x3f979534u, 0xbfb652afu}, 0},
    PredictStep<2>{{0.44941180921956975f, -0.67652104593078166f}, {0x3ee6194eu, 0xbf2d307cu}, -1},
    UpdateStep<2>{{1.184341080683788f, -1.7933535433685615f}, {0x3f97987du, 0xbfe58c9cu}, 0},
    PredictStep<2>{{0.50317349811261647f, -0.62568559492695941f}, {0x3f00cffau, 0xbf202ceeu}, -1},
    UpdateStep<2>{{1.493725851049891f, -58.938562486180416f}, {0x3fbf3269u, 0xc26bc117u}, 0},
    PredictStep<2>{{0.016966057799402821f, 0.0017000679802771407f}, {0x3c8afc67u, 0x3aded4d1u}, -1},
    UpdateStep<2>{{-585.59058291949862f, -432.67807988265162f}, {0xc41265ccu, 0xc3d856cbu}, 0},
    PredictStep<2>{{0.0022923917080350234f, -0.00342718508829974f}, {0x3b163bf3u, 0xbb609aa0u}, -1},
    UpdateStep<2>{{290.71820691463273f, -512.82823528238339f}, {0x43915beeu, 0xc4003502u}, 0},
    PredictStep<2>{{0.0019476745808307534f, -0.0037933196832589796f}, {0x3aff491du, 0xbb789958u}, -1},
    UpdateStep<2>{{263.53943121600042f, -567.4244764773016f}, {0x4383c50cu, 0xc40ddb2bu}, 0},
    PredictStep<2>{{0.0017622309325985841f, -0.0042342906443759449f}, {0x3ae6faa8u, 0xbb8abfceu}, -1},
    UpdateStep<2>{{236.16432416443661f, -640.20230091241092f}, {0x436c2a11u, 0xc4200cf2u}, 0},
    PredictStep<2>{{0.0015620037891859685f, -0.0048340823273515668f}, {0x3accbc26u, 0xbb9e6739u}, -1},
    UpdateStep<2>{{206.86446098832252f, -740.8042073048681f}, {0x434edd4du, 0xc4393378u}, 0},
    PredictStep<2>{{0.00134988432473079f, -0.0056854915637984307f}, {0x3ab0ee9au, 0xbbba4d5cu}, -1},
    UpdateStep<2>{{175.88628670952048f, -889.57095145990695f}, {0x432fe2e4u, 0xc45e648au}, 0},
    PredictStep<2>{{0.0011241374263969769f, -0.0070216564092020362f}, {0x3a9357cbu, 0xbbe615ecu}, -1},
    UpdateStep<2>{{142.41653845220213f, -1144.1116221447446f}, {0x430e6aa2u, 0xc48f0392u}, 0},
    PredictStep<2>{{0.00087404059240775379f, -0.009629322771473969f}, {0x3a651fdfu, 0xbc1dc44fu}, -1},
    UpdateStep<2>{{103.84946311722075f, -1764.3542024936271f}, {0x42cfb2edu, 0xc4dc8b56u}, 0},
    PredictStep<2>{{0.00056677961748647916f, -0.019789791925917666f}, {0x3a1493f0u, 0xbca21e34u}, -1},
    SwapStep{{}, {}, 0},
    PredictStep<1>{{50.531102284625426f}, {0x424a1fd9u}, 0},
    ScaleEvenStep<1>{{2.8642934616614059e-05f}, {0x37f0463cu}, 0},
    ScaleOddStep<1>{{-34912.623772145176f}, {0xc70860a0u}, 0});

template <>
struct scheme_traits<coif17_tag> {
    using SchemeType = decltype(coif17_scheme);
    static constexpr const char* name = "coif17";
    static constexpr int id = 23;
    static constexpr int tap_size = 102;
    static constexpr int delay_even = 25;
    static constexpr int delay_odd = 26;
    static constexpr int num_steps = 55;
    static constexpr const auto& scheme = coif17_scheme;
};

}  // namespace ttwv
