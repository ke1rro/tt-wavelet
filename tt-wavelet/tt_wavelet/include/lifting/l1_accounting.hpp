#pragma once

#include <cstdint>
#include <tt_stl/assert.hpp>

#include "tt_wavelet/include/device_protocol/lwt_config.hpp"

namespace ttwv {

struct L1Accounting {
    uint64_t slots_bytes{0};
    uint64_t circular_buffers_bytes{0};
    uint64_t cache_bytes{0};
    uint64_t output_bytes{0};
    uint64_t synchronization_bytes{0};
    uint64_t metadata_bytes{0};
    uint64_t alignment_bytes{0};
    uint64_t padding_bytes{0};
    uint64_t architecture_scratch_bytes{0};
    uint64_t total_bytes{0};
    uint64_t capacity_bytes{0};
    uint64_t headroom_bytes{0};
};

namespace l1_detail {

constexpr uint64_t kSourceTileCircularBuffersBytes =
    uint64_t{8} * device_protocol::kLwtNarrowTileBytes;
constexpr uint64_t kBaseTileCircularBufferBytes =
    uint64_t{6} * device_protocol::kLwtNarrowTileBytes;
constexpr uint64_t kOutputTileCircularBufferBytes =
    uint64_t{6} * device_protocol::kLwtNarrowTileBytes;
constexpr uint64_t kInterleaveOutputBytes = device_protocol::kStickBytes;
constexpr uint64_t kCacheBytes =
    uint64_t{device_protocol::kLwtCacheStickCount} * device_protocol::kStickBytes;
constexpr uint64_t kSynchronizationBytes = 32;
constexpr uint64_t kMetadataBytes = uint64_t{2} * device_protocol::kRouteConfigPageBytes;

static_assert(kSourceTileCircularBuffersBytes == 16384);
static_assert(kBaseTileCircularBufferBytes == 12288);
static_assert(kOutputTileCircularBufferBytes == 12288);

}  // namespace l1_detail

[[nodiscard]] inline L1Accounting make_l1_accounting(
    const uint32_t workspace_elements,
    const uint32_t max_workspace_elements,
    const uint32_t architecture_scratch_bytes,
    const uint32_t capacity_bytes) {
    TT_FATAL(
        max_workspace_elements <= workspace_elements,
        "Logical workspace length {} exceeds allocated workspace length {}",
        max_workspace_elements,
        workspace_elements);

    const uint64_t slots_bytes = uint64_t{3} * max_workspace_elements * sizeof(float);
    const uint64_t padding_bytes =
        uint64_t{3} * (workspace_elements - max_workspace_elements) * sizeof(float);
    constexpr uint64_t circular_buffers_bytes =
        l1_detail::kSourceTileCircularBuffersBytes + l1_detail::kBaseTileCircularBufferBytes;
    constexpr uint64_t output_bytes =
        l1_detail::kOutputTileCircularBufferBytes + l1_detail::kInterleaveOutputBytes;
    constexpr uint64_t alignment_bytes = 0;
    const uint64_t total_bytes = slots_bytes + circular_buffers_bytes + l1_detail::kCacheBytes + output_bytes +
                                 l1_detail::kSynchronizationBytes + l1_detail::kMetadataBytes + alignment_bytes +
                                 padding_bytes + architecture_scratch_bytes;
    TT_FATAL(
        total_bytes <= capacity_bytes,
        "tt-wavelet L1 allocation requires {} bytes/core, exceeding capacity {} by {} bytes",
        total_bytes,
        capacity_bytes,
        total_bytes - capacity_bytes);

    return L1Accounting{
        .slots_bytes = slots_bytes,
        .circular_buffers_bytes = circular_buffers_bytes,
        .cache_bytes = l1_detail::kCacheBytes,
        .output_bytes = output_bytes,
        .synchronization_bytes = l1_detail::kSynchronizationBytes,
        .metadata_bytes = l1_detail::kMetadataBytes,
        .alignment_bytes = alignment_bytes,
        .padding_bytes = padding_bytes,
        .architecture_scratch_bytes = architecture_scratch_bytes,
        .total_bytes = total_bytes,
        .capacity_bytes = capacity_bytes,
        .headroom_bytes = capacity_bytes - total_bytes,
    };
}

}  // namespace ttwv
