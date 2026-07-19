#pragma once

#include <optional>
#include <tt_stl/assert.hpp>

#include <umd/device/types/arch.hpp>

#include "tt_wavelet/include/lifting/execution_plan.hpp"

namespace ttwv {

struct ArchitecturePolicy {
    tt::ARCH architecture{tt::ARCH::Invalid};
    WorkspaceLayout ilwt_layout{WorkspaceLayout::kRowMajor};
    bool inverse_scale_inline{true};
    bool final_interleave_direct{false};
    uint32_t l1_scratch_bytes{0};
};

[[nodiscard]] inline ArchitecturePolicy make_architecture_policy(
    const tt::ARCH architecture, const std::optional<WorkspaceLayout> ilwt_layout_override = std::nullopt) {
    switch (architecture) {
        case tt::ARCH::WORMHOLE_B0:
            return ArchitecturePolicy{
                .architecture = architecture,
                .ilwt_layout = ilwt_layout_override.value_or(WorkspaceLayout::kRowMajor),
                .inverse_scale_inline = true,
                .final_interleave_direct = false,
                .l1_scratch_bytes = 0,
            };
        case tt::ARCH::BLACKHOLE: {
            const WorkspaceLayout layout = ilwt_layout_override.value_or(WorkspaceLayout::kTileNative);
            return ArchitecturePolicy{
                .architecture = architecture,
                .ilwt_layout = layout,
                .inverse_scale_inline = true,
                .final_interleave_direct = layout == WorkspaceLayout::kTileNative,
                .l1_scratch_bytes = 0,
            };
        }
        default: TT_THROW("tt-wavelet supports only Wormhole B0 and Blackhole, got {}", architecture);
    }
}

}  // namespace ttwv
