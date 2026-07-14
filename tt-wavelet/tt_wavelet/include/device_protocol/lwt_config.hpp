#pragma once

#include <cstdint>

#include "../common/constants.hpp"

namespace ttwv::device_protocol {

constexpr uint32_t kStepCoeffCapacity = 17;
constexpr uint32_t kStickBytes = kStickWidth * sizeof(float);
constexpr uint32_t kPadSplitCacheStickCount = 8;
constexpr uint32_t kLwtCacheStickCount = 4;

constexpr uint32_t kLwtRowsPerGroup = 32;
constexpr uint32_t kLwtOutputBlocksPerRow = 3;
constexpr uint32_t kLwtHalfStickElements = 16;
constexpr uint32_t kLwtHalfStickBytes = kLwtHalfStickElements * sizeof(float);
constexpr uint32_t kLwtGroupOutputElements = kLwtRowsPerGroup * kLwtOutputBlocksPerRow * kLwtHalfStickElements;

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
};

constexpr uint32_t kConeChunkConfigWordCount = 16;
constexpr uint32_t kConeChunkConfigPageBytes = kConeChunkConfigWordCount * sizeof(uint32_t);

enum ConeChunkConfigWord : uint32_t {
    kConeInitialEvenBegin = 0,
    kConeInitialEvenLength = 1,
    kConeInitialOddBegin = 2,
    kConeInitialOddLength = 3,
};

}  // namespace ttwv::device_protocol
