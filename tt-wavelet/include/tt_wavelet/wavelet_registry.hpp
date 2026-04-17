#pragma once

#include <span>
#include <string_view>

#include "wavelets/bior1_1.hpp"
#include "wavelets/bior1_3.hpp"
#include "wavelets/bior1_5.hpp"
#include "wavelets/bior2_2.hpp"
#include "wavelets/bior2_4.hpp"
#include "wavelets/bior2_6.hpp"
#include "wavelets/bior2_8.hpp"
#include "wavelets/bior3_1.hpp"
#include "wavelets/bior3_3.hpp"
#include "wavelets/bior3_5.hpp"
#include "wavelets/bior3_7.hpp"
#include "wavelets/bior3_9.hpp"
#include "wavelets/bior4_4.hpp"
#include "wavelets/bior5_5.hpp"
#include "wavelets/bior6_8.hpp"
#include "wavelets/coif1.hpp"
#include "wavelets/coif10.hpp"
#include "wavelets/coif11.hpp"
#include "wavelets/coif12.hpp"
#include "wavelets/coif13.hpp"
#include "wavelets/coif14.hpp"
#include "wavelets/coif15.hpp"
#include "wavelets/coif16.hpp"
#include "wavelets/coif17.hpp"
#include "wavelets/coif2.hpp"
#include "wavelets/coif3.hpp"
#include "wavelets/coif4.hpp"
#include "wavelets/coif5.hpp"
#include "wavelets/coif6.hpp"
#include "wavelets/coif7.hpp"
#include "wavelets/coif8.hpp"
#include "wavelets/coif9.hpp"
#include "wavelets/db1.hpp"
#include "wavelets/db10.hpp"
#include "wavelets/db11.hpp"
#include "wavelets/db12.hpp"
#include "wavelets/db13.hpp"
#include "wavelets/db14.hpp"
#include "wavelets/db15.hpp"
#include "wavelets/db16.hpp"
#include "wavelets/db17.hpp"
#include "wavelets/db18.hpp"
#include "wavelets/db19.hpp"
#include "wavelets/db2.hpp"
#include "wavelets/db20.hpp"
#include "wavelets/db21.hpp"
#include "wavelets/db22.hpp"
#include "wavelets/db23.hpp"
#include "wavelets/db24.hpp"
#include "wavelets/db25.hpp"
#include "wavelets/db26.hpp"
#include "wavelets/db27.hpp"
#include "wavelets/db28.hpp"
#include "wavelets/db29.hpp"
#include "wavelets/db3.hpp"
#include "wavelets/db30.hpp"
#include "wavelets/db31.hpp"
#include "wavelets/db32.hpp"
#include "wavelets/db33.hpp"
#include "wavelets/db34.hpp"
#include "wavelets/db35.hpp"
#include "wavelets/db36.hpp"
#include "wavelets/db37.hpp"
#include "wavelets/db38.hpp"
#include "wavelets/db4.hpp"
#include "wavelets/db5.hpp"
#include "wavelets/db6.hpp"
#include "wavelets/db7.hpp"
#include "wavelets/db8.hpp"
#include "wavelets/db9.hpp"
#include "wavelets/dmey.hpp"
#include "wavelets/haar.hpp"
#include "wavelets/rbio1_1.hpp"
#include "wavelets/rbio1_3.hpp"
#include "wavelets/rbio1_5.hpp"
#include "wavelets/rbio2_2.hpp"
#include "wavelets/rbio2_4.hpp"
#include "wavelets/rbio2_6.hpp"
#include "wavelets/rbio2_8.hpp"
#include "wavelets/rbio3_1.hpp"
#include "wavelets/rbio3_3.hpp"
#include "wavelets/rbio3_5.hpp"
#include "wavelets/rbio3_7.hpp"
#include "wavelets/rbio3_9.hpp"
#include "wavelets/rbio4_4.hpp"
#include "wavelets/rbio5_5.hpp"
#include "wavelets/rbio6_8.hpp"
#include "wavelets/sym10.hpp"
#include "wavelets/sym11.hpp"
#include "wavelets/sym12.hpp"
#include "wavelets/sym13.hpp"
#include "wavelets/sym14.hpp"
#include "wavelets/sym15.hpp"
#include "wavelets/sym16.hpp"
#include "wavelets/sym17.hpp"
#include "wavelets/sym18.hpp"
#include "wavelets/sym19.hpp"
#include "wavelets/sym2.hpp"
#include "wavelets/sym20.hpp"
#include "wavelets/sym3.hpp"
#include "wavelets/sym4.hpp"
#include "wavelets/sym5.hpp"
#include "wavelets/sym6.hpp"
#include "wavelets/sym7.hpp"
#include "wavelets/sym8.hpp"
#include "wavelets/sym9.hpp"

namespace ttwv {

template <>
struct WaveletScheme<0> {
    using Tag = bior1_1_tag;
    static constexpr const auto& scheme = bior1_1_scheme;
};

template <>
struct WaveletScheme<1> {
    using Tag = bior1_3_tag;
    static constexpr const auto& scheme = bior1_3_scheme;
};

template <>
struct WaveletScheme<2> {
    using Tag = bior1_5_tag;
    static constexpr const auto& scheme = bior1_5_scheme;
};

template <>
struct WaveletScheme<3> {
    using Tag = bior2_2_tag;
    static constexpr const auto& scheme = bior2_2_scheme;
};

template <>
struct WaveletScheme<4> {
    using Tag = bior2_4_tag;
    static constexpr const auto& scheme = bior2_4_scheme;
};

template <>
struct WaveletScheme<5> {
    using Tag = bior2_6_tag;
    static constexpr const auto& scheme = bior2_6_scheme;
};

template <>
struct WaveletScheme<6> {
    using Tag = bior2_8_tag;
    static constexpr const auto& scheme = bior2_8_scheme;
};

template <>
struct WaveletScheme<7> {
    using Tag = bior3_1_tag;
    static constexpr const auto& scheme = bior3_1_scheme;
};

template <>
struct WaveletScheme<8> {
    using Tag = bior3_3_tag;
    static constexpr const auto& scheme = bior3_3_scheme;
};

template <>
struct WaveletScheme<9> {
    using Tag = bior3_5_tag;
    static constexpr const auto& scheme = bior3_5_scheme;
};

template <>
struct WaveletScheme<10> {
    using Tag = bior3_7_tag;
    static constexpr const auto& scheme = bior3_7_scheme;
};

template <>
struct WaveletScheme<11> {
    using Tag = bior3_9_tag;
    static constexpr const auto& scheme = bior3_9_scheme;
};

template <>
struct WaveletScheme<12> {
    using Tag = bior4_4_tag;
    static constexpr const auto& scheme = bior4_4_scheme;
};

template <>
struct WaveletScheme<13> {
    using Tag = bior5_5_tag;
    static constexpr const auto& scheme = bior5_5_scheme;
};

template <>
struct WaveletScheme<14> {
    using Tag = bior6_8_tag;
    static constexpr const auto& scheme = bior6_8_scheme;
};

template <>
struct WaveletScheme<15> {
    using Tag = coif1_tag;
    static constexpr const auto& scheme = coif1_scheme;
};

template <>
struct WaveletScheme<16> {
    using Tag = coif10_tag;
    static constexpr const auto& scheme = coif10_scheme;
};

template <>
struct WaveletScheme<17> {
    using Tag = coif11_tag;
    static constexpr const auto& scheme = coif11_scheme;
};

template <>
struct WaveletScheme<18> {
    using Tag = coif12_tag;
    static constexpr const auto& scheme = coif12_scheme;
};

template <>
struct WaveletScheme<19> {
    using Tag = coif13_tag;
    static constexpr const auto& scheme = coif13_scheme;
};

template <>
struct WaveletScheme<20> {
    using Tag = coif14_tag;
    static constexpr const auto& scheme = coif14_scheme;
};

template <>
struct WaveletScheme<21> {
    using Tag = coif15_tag;
    static constexpr const auto& scheme = coif15_scheme;
};

template <>
struct WaveletScheme<22> {
    using Tag = coif16_tag;
    static constexpr const auto& scheme = coif16_scheme;
};

template <>
struct WaveletScheme<23> {
    using Tag = coif17_tag;
    static constexpr const auto& scheme = coif17_scheme;
};

template <>
struct WaveletScheme<24> {
    using Tag = coif2_tag;
    static constexpr const auto& scheme = coif2_scheme;
};

template <>
struct WaveletScheme<25> {
    using Tag = coif3_tag;
    static constexpr const auto& scheme = coif3_scheme;
};

template <>
struct WaveletScheme<26> {
    using Tag = coif4_tag;
    static constexpr const auto& scheme = coif4_scheme;
};

template <>
struct WaveletScheme<27> {
    using Tag = coif5_tag;
    static constexpr const auto& scheme = coif5_scheme;
};

template <>
struct WaveletScheme<28> {
    using Tag = coif6_tag;
    static constexpr const auto& scheme = coif6_scheme;
};

template <>
struct WaveletScheme<29> {
    using Tag = coif7_tag;
    static constexpr const auto& scheme = coif7_scheme;
};

template <>
struct WaveletScheme<30> {
    using Tag = coif8_tag;
    static constexpr const auto& scheme = coif8_scheme;
};

template <>
struct WaveletScheme<31> {
    using Tag = coif9_tag;
    static constexpr const auto& scheme = coif9_scheme;
};

template <>
struct WaveletScheme<32> {
    using Tag = db1_tag;
    static constexpr const auto& scheme = db1_scheme;
};

template <>
struct WaveletScheme<33> {
    using Tag = db10_tag;
    static constexpr const auto& scheme = db10_scheme;
};

template <>
struct WaveletScheme<34> {
    using Tag = db11_tag;
    static constexpr const auto& scheme = db11_scheme;
};

template <>
struct WaveletScheme<35> {
    using Tag = db12_tag;
    static constexpr const auto& scheme = db12_scheme;
};

template <>
struct WaveletScheme<36> {
    using Tag = db13_tag;
    static constexpr const auto& scheme = db13_scheme;
};

template <>
struct WaveletScheme<37> {
    using Tag = db14_tag;
    static constexpr const auto& scheme = db14_scheme;
};

template <>
struct WaveletScheme<38> {
    using Tag = db15_tag;
    static constexpr const auto& scheme = db15_scheme;
};

template <>
struct WaveletScheme<39> {
    using Tag = db16_tag;
    static constexpr const auto& scheme = db16_scheme;
};

template <>
struct WaveletScheme<40> {
    using Tag = db17_tag;
    static constexpr const auto& scheme = db17_scheme;
};

template <>
struct WaveletScheme<41> {
    using Tag = db18_tag;
    static constexpr const auto& scheme = db18_scheme;
};

template <>
struct WaveletScheme<42> {
    using Tag = db19_tag;
    static constexpr const auto& scheme = db19_scheme;
};

template <>
struct WaveletScheme<43> {
    using Tag = db2_tag;
    static constexpr const auto& scheme = db2_scheme;
};

template <>
struct WaveletScheme<44> {
    using Tag = db20_tag;
    static constexpr const auto& scheme = db20_scheme;
};

template <>
struct WaveletScheme<45> {
    using Tag = db21_tag;
    static constexpr const auto& scheme = db21_scheme;
};

template <>
struct WaveletScheme<46> {
    using Tag = db22_tag;
    static constexpr const auto& scheme = db22_scheme;
};

template <>
struct WaveletScheme<47> {
    using Tag = db23_tag;
    static constexpr const auto& scheme = db23_scheme;
};

template <>
struct WaveletScheme<48> {
    using Tag = db24_tag;
    static constexpr const auto& scheme = db24_scheme;
};

template <>
struct WaveletScheme<49> {
    using Tag = db25_tag;
    static constexpr const auto& scheme = db25_scheme;
};

template <>
struct WaveletScheme<50> {
    using Tag = db26_tag;
    static constexpr const auto& scheme = db26_scheme;
};

template <>
struct WaveletScheme<51> {
    using Tag = db27_tag;
    static constexpr const auto& scheme = db27_scheme;
};

template <>
struct WaveletScheme<52> {
    using Tag = db28_tag;
    static constexpr const auto& scheme = db28_scheme;
};

template <>
struct WaveletScheme<53> {
    using Tag = db29_tag;
    static constexpr const auto& scheme = db29_scheme;
};

template <>
struct WaveletScheme<54> {
    using Tag = db3_tag;
    static constexpr const auto& scheme = db3_scheme;
};

template <>
struct WaveletScheme<55> {
    using Tag = db30_tag;
    static constexpr const auto& scheme = db30_scheme;
};

template <>
struct WaveletScheme<56> {
    using Tag = db31_tag;
    static constexpr const auto& scheme = db31_scheme;
};

template <>
struct WaveletScheme<57> {
    using Tag = db32_tag;
    static constexpr const auto& scheme = db32_scheme;
};

template <>
struct WaveletScheme<58> {
    using Tag = db33_tag;
    static constexpr const auto& scheme = db33_scheme;
};

template <>
struct WaveletScheme<59> {
    using Tag = db34_tag;
    static constexpr const auto& scheme = db34_scheme;
};

template <>
struct WaveletScheme<60> {
    using Tag = db35_tag;
    static constexpr const auto& scheme = db35_scheme;
};

template <>
struct WaveletScheme<61> {
    using Tag = db36_tag;
    static constexpr const auto& scheme = db36_scheme;
};

template <>
struct WaveletScheme<62> {
    using Tag = db37_tag;
    static constexpr const auto& scheme = db37_scheme;
};

template <>
struct WaveletScheme<63> {
    using Tag = db38_tag;
    static constexpr const auto& scheme = db38_scheme;
};

template <>
struct WaveletScheme<64> {
    using Tag = db4_tag;
    static constexpr const auto& scheme = db4_scheme;
};

template <>
struct WaveletScheme<65> {
    using Tag = db5_tag;
    static constexpr const auto& scheme = db5_scheme;
};

template <>
struct WaveletScheme<66> {
    using Tag = db6_tag;
    static constexpr const auto& scheme = db6_scheme;
};

template <>
struct WaveletScheme<67> {
    using Tag = db7_tag;
    static constexpr const auto& scheme = db7_scheme;
};

template <>
struct WaveletScheme<68> {
    using Tag = db8_tag;
    static constexpr const auto& scheme = db8_scheme;
};

template <>
struct WaveletScheme<69> {
    using Tag = db9_tag;
    static constexpr const auto& scheme = db9_scheme;
};

template <>
struct WaveletScheme<70> {
    using Tag = dmey_tag;
    static constexpr const auto& scheme = dmey_scheme;
};

template <>
struct WaveletScheme<71> {
    using Tag = haar_tag;
    static constexpr const auto& scheme = haar_scheme;
};

template <>
struct WaveletScheme<72> {
    using Tag = rbio1_1_tag;
    static constexpr const auto& scheme = rbio1_1_scheme;
};

template <>
struct WaveletScheme<73> {
    using Tag = rbio1_3_tag;
    static constexpr const auto& scheme = rbio1_3_scheme;
};

template <>
struct WaveletScheme<74> {
    using Tag = rbio1_5_tag;
    static constexpr const auto& scheme = rbio1_5_scheme;
};

template <>
struct WaveletScheme<75> {
    using Tag = rbio2_2_tag;
    static constexpr const auto& scheme = rbio2_2_scheme;
};

template <>
struct WaveletScheme<76> {
    using Tag = rbio2_4_tag;
    static constexpr const auto& scheme = rbio2_4_scheme;
};

template <>
struct WaveletScheme<77> {
    using Tag = rbio2_6_tag;
    static constexpr const auto& scheme = rbio2_6_scheme;
};

template <>
struct WaveletScheme<78> {
    using Tag = rbio2_8_tag;
    static constexpr const auto& scheme = rbio2_8_scheme;
};

template <>
struct WaveletScheme<79> {
    using Tag = rbio3_1_tag;
    static constexpr const auto& scheme = rbio3_1_scheme;
};

template <>
struct WaveletScheme<80> {
    using Tag = rbio3_3_tag;
    static constexpr const auto& scheme = rbio3_3_scheme;
};

template <>
struct WaveletScheme<81> {
    using Tag = rbio3_5_tag;
    static constexpr const auto& scheme = rbio3_5_scheme;
};

template <>
struct WaveletScheme<82> {
    using Tag = rbio3_7_tag;
    static constexpr const auto& scheme = rbio3_7_scheme;
};

template <>
struct WaveletScheme<83> {
    using Tag = rbio3_9_tag;
    static constexpr const auto& scheme = rbio3_9_scheme;
};

template <>
struct WaveletScheme<84> {
    using Tag = rbio4_4_tag;
    static constexpr const auto& scheme = rbio4_4_scheme;
};

template <>
struct WaveletScheme<85> {
    using Tag = rbio5_5_tag;
    static constexpr const auto& scheme = rbio5_5_scheme;
};

template <>
struct WaveletScheme<86> {
    using Tag = rbio6_8_tag;
    static constexpr const auto& scheme = rbio6_8_scheme;
};

template <>
struct WaveletScheme<87> {
    using Tag = sym10_tag;
    static constexpr const auto& scheme = sym10_scheme;
};

template <>
struct WaveletScheme<88> {
    using Tag = sym11_tag;
    static constexpr const auto& scheme = sym11_scheme;
};

template <>
struct WaveletScheme<89> {
    using Tag = sym12_tag;
    static constexpr const auto& scheme = sym12_scheme;
};

template <>
struct WaveletScheme<90> {
    using Tag = sym13_tag;
    static constexpr const auto& scheme = sym13_scheme;
};

template <>
struct WaveletScheme<91> {
    using Tag = sym14_tag;
    static constexpr const auto& scheme = sym14_scheme;
};

template <>
struct WaveletScheme<92> {
    using Tag = sym15_tag;
    static constexpr const auto& scheme = sym15_scheme;
};

template <>
struct WaveletScheme<93> {
    using Tag = sym16_tag;
    static constexpr const auto& scheme = sym16_scheme;
};

template <>
struct WaveletScheme<94> {
    using Tag = sym17_tag;
    static constexpr const auto& scheme = sym17_scheme;
};

template <>
struct WaveletScheme<95> {
    using Tag = sym18_tag;
    static constexpr const auto& scheme = sym18_scheme;
};

template <>
struct WaveletScheme<96> {
    using Tag = sym19_tag;
    static constexpr const auto& scheme = sym19_scheme;
};

template <>
struct WaveletScheme<97> {
    using Tag = sym2_tag;
    static constexpr const auto& scheme = sym2_scheme;
};

template <>
struct WaveletScheme<98> {
    using Tag = sym20_tag;
    static constexpr const auto& scheme = sym20_scheme;
};

template <>
struct WaveletScheme<99> {
    using Tag = sym3_tag;
    static constexpr const auto& scheme = sym3_scheme;
};

template <>
struct WaveletScheme<100> {
    using Tag = sym4_tag;
    static constexpr const auto& scheme = sym4_scheme;
};

template <>
struct WaveletScheme<101> {
    using Tag = sym5_tag;
    static constexpr const auto& scheme = sym5_scheme;
};

template <>
struct WaveletScheme<102> {
    using Tag = sym6_tag;
    static constexpr const auto& scheme = sym6_scheme;
};

template <>
struct WaveletScheme<103> {
    using Tag = sym7_tag;
    static constexpr const auto& scheme = sym7_scheme;
};

template <>
struct WaveletScheme<104> {
    using Tag = sym8_tag;
    static constexpr const auto& scheme = sym8_scheme;
};

template <>
struct WaveletScheme<105> {
    using Tag = sym9_tag;
    static constexpr const auto& scheme = sym9_scheme;
};

[[nodiscard]] inline std::span<const SchemeInfo> all_schemes() noexcept {
    static constexpr SchemeInfo table[106] = {
        {"bior1.1", 0, 2, 0, 1, 5},     {"bior1.3", 1, 6, 1, 2, 4},     {"bior1.5", 2, 10, 2, 3, 4},
        {"bior2.2", 3, 6, 1, 2, 5},     {"bior2.4", 4, 10, 2, 3, 5},    {"bior2.6", 5, 14, 3, 4, 5},
        {"bior2.8", 6, 18, 4, 5, 5},    {"bior3.1", 7, 4, 1, 1, 5},     {"bior3.3", 8, 8, 2, 2, 6},
        {"bior3.5", 9, 12, 3, 3, 6},    {"bior3.7", 10, 16, 4, 4, 6},   {"bior3.9", 11, 20, 5, 5, 6},
        {"bior4.4", 12, 10, 2, 3, 7},   {"bior5.5", 13, 12, 3, 3, 7},   {"bior6.8", 14, 18, 4, 5, 9},
        {"coif1", 15, 6, 1, 2, 7},      {"coif10", 16, 60, 15, 15, 33}, {"coif11", 17, 66, 16, 17, 37},
        {"coif12", 18, 72, 18, 18, 39}, {"coif13", 19, 78, 19, 20, 43}, {"coif14", 20, 84, 21, 21, 45},
        {"coif15", 21, 90, 22, 23, 49}, {"coif16", 22, 96, 24, 24, 51}, {"coif17", 23, 102, 25, 26, 55},
        {"coif2", 24, 12, 3, 3, 9},     {"coif3", 25, 18, 4, 5, 13},    {"coif4", 26, 24, 6, 6, 15},
        {"coif5", 27, 30, 7, 8, 19},    {"coif6", 28, 36, 9, 9, 21},    {"coif7", 29, 42, 10, 11, 25},
        {"coif8", 30, 48, 12, 12, 27},  {"coif9", 31, 54, 13, 14, 31},  {"db1", 32, 2, 0, 1, 5},
        {"db10", 33, 20, 5, 5, 13},     {"db11", 34, 22, 5, 6, 15},     {"db12", 35, 24, 6, 6, 15},
        {"db13", 36, 26, 6, 7, 17},     {"db14", 37, 28, 7, 7, 17},     {"db15", 38, 30, 7, 8, 19},
        {"db16", 39, 32, 8, 8, 19},     {"db17", 40, 34, 8, 9, 21},     {"db18", 41, 36, 9, 9, 21},
        {"db19", 42, 38, 9, 10, 23},    {"db2", 43, 4, 1, 1, 5},        {"db20", 44, 40, 10, 10, 23},
        {"db21", 45, 42, 10, 11, 25},   {"db22", 46, 44, 11, 11, 25},   {"db23", 47, 46, 11, 12, 27},
        {"db24", 48, 48, 12, 12, 27},   {"db25", 49, 50, 12, 13, 29},   {"db26", 50, 52, 13, 13, 29},
        {"db27", 51, 54, 13, 14, 31},   {"db28", 52, 56, 14, 14, 31},   {"db29", 53, 58, 14, 15, 33},
        {"db3", 54, 6, 1, 2, 7},        {"db30", 55, 60, 15, 15, 33},   {"db31", 56, 62, 15, 16, 35},
        {"db32", 57, 64, 16, 16, 35},   {"db33", 58, 66, 16, 17, 37},   {"db34", 59, 68, 17, 17, 37},
        {"db35", 60, 70, 17, 18, 39},   {"db36", 61, 72, 18, 18, 39},   {"db37", 62, 74, 18, 19, 41},
        {"db38", 63, 76, 19, 19, 41},   {"db4", 64, 8, 2, 2, 7},        {"db5", 65, 10, 2, 3, 9},
        {"db6", 66, 12, 3, 3, 9},       {"db7", 67, 14, 3, 4, 11},      {"db8", 68, 16, 4, 4, 11},
        {"db9", 69, 18, 4, 5, 13},      {"dmey", 70, 62, 15, 16, 33},   {"haar", 71, 2, 0, 1, 5},
        {"rbio1.1", 72, 2, 0, 1, 5},    {"rbio1.3", 73, 6, 1, 2, 5},    {"rbio1.5", 74, 10, 2, 3, 5},
        {"rbio2.2", 75, 6, 1, 2, 5},    {"rbio2.4", 76, 10, 2, 3, 5},   {"rbio2.6", 77, 14, 3, 4, 5},
        {"rbio2.8", 78, 18, 4, 5, 5},   {"rbio3.1", 79, 4, 1, 1, 5},    {"rbio3.3", 80, 8, 2, 2, 5},
        {"rbio3.5", 81, 12, 3, 3, 5},   {"rbio3.7", 82, 16, 4, 4, 5},   {"rbio3.9", 83, 20, 5, 5, 5},
        {"rbio4.4", 84, 10, 2, 3, 7},   {"rbio5.5", 85, 12, 3, 3, 7},   {"rbio6.8", 86, 18, 4, 5, 9},
        {"sym10", 87, 20, 5, 5, 13},    {"sym11", 88, 22, 5, 6, 15},    {"sym12", 89, 24, 6, 6, 15},
        {"sym13", 90, 26, 6, 7, 17},    {"sym14", 91, 28, 7, 7, 17},    {"sym15", 92, 30, 7, 8, 19},
        {"sym16", 93, 32, 8, 8, 19},    {"sym17", 94, 34, 8, 9, 21},    {"sym18", 95, 36, 9, 9, 21},
        {"sym19", 96, 38, 9, 10, 23},   {"sym2", 97, 4, 1, 1, 5},       {"sym20", 98, 40, 10, 10, 23},
        {"sym3", 99, 6, 1, 2, 7},       {"sym4", 100, 8, 2, 2, 7},      {"sym5", 101, 10, 2, 3, 9},
        {"sym6", 102, 12, 3, 3, 9},     {"sym7", 103, 14, 3, 4, 11},    {"sym8", 104, 16, 4, 4, 11},
        {"sym9", 105, 18, 4, 5, 13},
    };
    return {table, 106};
}

[[nodiscard]] inline const SchemeInfo* find_scheme(std::string_view name) noexcept {
    for (const auto& info : all_schemes()) {
        if (name == info.name) {
            return &info;
        }
    }
    return nullptr;
}

}  // namespace ttwv
