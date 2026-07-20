#pragma once

#include <cstdint>

#include "../common/constants.hpp"

namespace ttwv::device_protocol {

constexpr uint32_t kStepCoeffCapacity = 17;
constexpr uint32_t kStickBytes = kStickWidth * sizeof(float);
constexpr uint32_t kLwtCacheStickCount = 4;

constexpr uint32_t kLwtRowsPerGroup = 32;
constexpr uint32_t kLwtOutputBlocksPerRow = 3;
constexpr uint32_t kLwtHalfStickElements = 16;
constexpr uint32_t kLwtHalfStickBytes = kLwtHalfStickElements * sizeof(float);
// A native Wormhole 32x16 FP32 tile contains exactly one 16-element block
// for each row in an LWT processing group.  Keeping this relation explicit
// prevents the LWT kernels from silently falling back to padded 32x32 pages.
constexpr uint32_t kLwtNarrowTileElements = kLwtRowsPerGroup * kLwtHalfStickElements;
constexpr uint32_t kLwtNarrowTileBytes = kLwtNarrowTileElements * sizeof(float);
constexpr uint32_t kLwtGroupOutputElements = kLwtRowsPerGroup * kLwtOutputBlocksPerRow * kLwtHalfStickElements;
constexpr uint32_t kIlwtGroupOutputElements = 2 * kLwtGroupOutputElements;

constexpr uint32_t kRouteConfigWordCount = 16;
constexpr uint32_t kRouteConfigPageBytes = kRouteConfigWordCount * sizeof(uint32_t);

enum RouteConfigWord : uint32_t {
    kRouteType = 0,
    kRouteSourceAddr = 1,
    kRouteSourceLength = 2,
    kRouteBaseAddr = 3,
    kRouteBaseLength = 4,
    kRouteOutputAddr = 5,
    kRouteOutputLength = 6,
    kRouteSourceOffset = 7,
    kRouteBaseOffset = 8,
    kRouteSourceLeftPad = 9,
    kRouteOutputOffset = 10,
    kRouteGroupCount = 11,
    kRouteFlags = 12,
};

constexpr uint32_t kRouteFlagFinalDram = 1U << 0;
constexpr uint32_t kRouteFlagIlwtFinalInterleave = 1U << 1;

constexpr uint32_t kLwtChunkConfigWordCount = 16;
constexpr uint32_t kLwtChunkConfigPageBytes = kLwtChunkConfigWordCount * sizeof(uint32_t);

enum LwtChunkConfigWord : uint32_t {
    kLwtInitialEvenBegin = 0,
    kLwtInitialEvenLength = 1,
    kLwtInitialOddBegin = 2,
    kLwtInitialOddLength = 3,

    // ILWT reuses the same 64-byte chunk page with a direction-specific
    // contract.  The first four words describe canonical coefficient inputs.
    kIlwtApproximationBegin = 0,
    kIlwtApproximationLength = 1,
    kIlwtDetailBegin = 2,
    kIlwtDetailLength = 3,
    kIlwtFinalEvenAddr = 4,
    kIlwtFinalEvenStorageLength = 5,
    kIlwtFinalEvenOffset = 6,
    kIlwtFinalEvenBegin = 7,
    kIlwtFinalOddAddr = 8,
    kIlwtFinalOddStorageLength = 9,
    kIlwtFinalOddOffset = 10,
    kIlwtFinalOddBegin = 11,
    kIlwtOutputBegin = 12,
    kIlwtOutputLength = 13,
};

}  // namespace ttwv::device_protocol
