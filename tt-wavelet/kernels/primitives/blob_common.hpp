#pragma once

#include <cstdint>

namespace ttwv::kernels::primitives {

constexpr uint32_t kBlobInvalid = 0xFFFFFFFFU;
constexpr uint32_t kBlobElements = 4096;
constexpr uint32_t kBlobBytes = kBlobElements * sizeof(float);
static_assert(kBlobElements % 16 == 0, "Blob elements must be a multiple of 16 for noc operations.");

}  // namespace ttwv::kernels::primitives
