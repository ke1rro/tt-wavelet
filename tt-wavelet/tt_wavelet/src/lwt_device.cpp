
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <tt_stl/assert.hpp>
#include <vector>

#include "tt-metalium/tensor_accessor_args.hpp"



namespace ttwv::lwt {

    namespace{
constexpr uint32_t kCbHalo = tt::CBIndex::c_0;        // reader -> compute (halo sticks)
constexpr uint32_t kCbCur = tt::CBIndex::c_1;         // reader -> compute (current sticks)
constexpr uint32_t kCbTilizedH = tt::CBIndex::c_2;    // compute internal (tilized halo)
constexpr uint32_t kCbTilizedC = tt::CBIndex::c_3;    // compute internal (tilized current)
constexpr uint32_t kCbTilizedOut = tt::CBIndex::c_4;  // compute internal (tilized output)
constexpr uint32_t kCbOut = tt::CBIndex::c_16;        // compute -> writer (output sticks)
constexpr uint32_t kCbCache = tt::CBIndex::c_5;       // reader DRAM cache

constexpr uint32_t kAlignment = 32;

constexpr const char* kReader = "kernels/dataflow/lwt_reader.cpp";
constexpr const char* kCompute = "kernels/compute/lwt_compute.cpp";
constexpr const char* kWriter = "kernels/dataflow/lwt_writer.cpp";

    }
}// namespace ttwv::lwt