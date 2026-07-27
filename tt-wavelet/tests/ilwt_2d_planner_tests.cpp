// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tt_wavelet/include/lifting/inverse_plan_2d.hpp"
#include "tt_wavelet/include/schemes/generated/bior3_9.hpp"
#include "tt_wavelet/include/schemes/generated/db1.hpp"
#include "tt_wavelet/include/schemes/generated/db7.hpp"

namespace {

constexpr uint64_t kL1Budget = 768 * 1024;
constexpr uint32_t kCoreLimit = 64;
uint64_t g_max_l1_bytes = 0;

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void validate_chunk(const ttwv::Lwt2DChunkPlan& chunk) {
    const size_t x_routes = chunk.x_cone.routes.size();
    const size_t y_routes = chunk.y_cone.routes.size();
    require(chunk.routes.size() == 2 * x_routes + 2 * y_routes, "inverse route count is inconsistent");
    for (size_t route = 0; route < chunk.routes.size(); ++route) {
        const bool horizontal = route < 2 * x_routes;
        require(
            chunk.routes[route].axis ==
                (horizontal ? ttwv::Lwt2DAxis::kHorizontal : ttwv::Lwt2DAxis::kVertical),
            "inverse routes are not ordered horizontal-before-vertical");
        if (!chunk.routes[route].output.empty()) {
            require(
                chunk.routes[route].output_slot != chunk.routes[route].base_slot,
                "inverse predict/update route aliases its base plane");
        }
    }
    const std::array<ttwv::Lwt2DPlaneSlot, 4> parity = {
        chunk.final_bands.ll,
        chunk.final_bands.lh,
        chunk.final_bands.hl,
        chunk.final_bands.hh,
    };
    auto sorted = parity;
    std::sort(sorted.begin(), sorted.end());
    require(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(), "final parity planes alias");
    require(chunk.resources.plane_count == 5, "inverse planner did not retain the five-plane workspace");
    require(chunk.resources.total_l1_bytes <= kL1Budget, "inverse chunk exceeds the L1 budget");
}

template <typename Scheme>
void validate_scheme() {
    constexpr std::array<std::array<size_t, 2>, 4> shapes = {
        std::array<size_t, 2>{32, 32},
        std::array<size_t, 2>{33, 35},
        std::array<size_t, 2>{100, 70},
        std::array<size_t, 2>{1000, 100},
    };
    for (const auto& [height, width] : shapes) {
        const ttwv::Ilwt2DExecutionPlan plan =
            ttwv::make_ilwt_2d_execution_plan<Scheme>(height, width, kCoreLimit, kL1Budget);
        require(plan.output_height == height && plan.output_width == width, "inverse output shape was lost");
        require(!plan.chunks.empty(), "inverse plan has no chunks");
        std::vector<uint8_t> coverage(height * width, 0);
        for (const auto& chunk : plan.chunks) {
            validate_chunk(chunk);
            for (size_t y = chunk.final_band_rect.y.begin; y < chunk.final_band_rect.y.end; ++y) {
                for (size_t x = chunk.final_band_rect.x.begin; x < chunk.final_band_rect.x.end; ++x) {
                    ++coverage[y * width + x];
                }
            }
        }
        require(
            std::all_of(coverage.begin(), coverage.end(), [](const uint8_t count) { return count == 1; }),
            "inverse chunks do not cover the output exactly once");
        require(
            plan.active_core_count == std::min(plan.chunks.size(), static_cast<size_t>(kCoreLimit)),
            "inverse active-core count does not match its chunk grid");
        require(plan.allocated_l1_bytes <= kL1Budget, "inverse uniform L1 allocation exceeds its budget");
        g_max_l1_bytes = std::max(g_max_l1_bytes, plan.allocated_l1_bytes);
        require(
            ttwv::build_ilwt_2d_chunk_config_words(plan).size() ==
                plan.chunks.size() * ttwv::device_protocol::kLwt2DChunkConfigWordCount,
            "inverse chunk config size is wrong");
        require(
            ttwv::build_ilwt_2d_route_config_words(plan).size() ==
                plan.chunks.size() * plan.chunks.front().routes.size() *
                    ttwv::device_protocol::kLwt2DRouteConfigWordCount,
            "inverse route config size is wrong");
        require(
            ttwv::build_ilwt_2d_band_config_words(plan).size() ==
                plan.chunks.size() * ttwv::device_protocol::kLwt2DBandConfigWordCount,
            "inverse terminal config size is wrong");
    }
}

}  // namespace

int main() {
    try {
        validate_scheme<ttwv::schemes::db1>();
        validate_scheme<ttwv::schemes::db7>();
        validate_scheme<ttwv::schemes::bior3_9>();
        std::cout << "2D ILWT dependency-cone planner validation passed; max L1 bytes=" << g_max_l1_bytes << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
