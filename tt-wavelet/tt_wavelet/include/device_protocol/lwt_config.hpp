#pragma once

#include <cstdint>

namespace ttwv::device_protocol {

constexpr uint32_t step_coeff_capacity = 17;

constexpr uint32_t fused_route_config_word_count = 16;
constexpr uint32_t fused_route_config_page_bytes = fused_route_config_word_count * sizeof(uint32_t);

enum FusedRouteConfigWord : uint32_t {
    kRouteType = 0,
    kRouteSourceAddr = 1,
    kRouteSourceLength = 2,
    kRouteBaseAddr = 3,
    kRouteBaseLength = 4,
    kRouteOutputAddr = 5,
    kRouteOutputLength = 6,
    kRouteOutputGroupCount = 7,
    kRouteSourceOffset = 8,
    kRouteBaseOffset = 9,
    kRouteSourceLeftPad = 10,
};

}  // namespace ttwv::device_protocol
