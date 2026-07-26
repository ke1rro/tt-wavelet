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

#include "tt_wavelet/include/lifting/plan_2d.hpp"
#include "tt_wavelet/include/schemes/generated/registry.hpp"

namespace {

constexpr uint64_t kTestL1BudgetBytes = 1024 * 1024;
constexpr uint32_t kTestCoreLimit = 64;
size_t g_five_plane_chunk_count = 0;

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

[[nodiscard]] bool equal(const ttwv::IndexInterval lhs, const ttwv::IndexInterval rhs) {
    return lhs.begin == rhs.begin && lhs.end == rhs.end;
}

[[nodiscard]] bool equal(const ttwv::IndexRectangle lhs, const ttwv::IndexRectangle rhs) {
    return equal(lhs.y, rhs.y) && equal(lhs.x, rhs.x);
}

void validate_axis_cone(const ttwv::AxisConePlan& cone, const ttwv::LiftingForwardPlan& plan) {
    require(cone.routes.size() == plan.routes.size(), "axis cone route count differs from the lifting plan");
    require(equal(cone.initial_even, cone.routes.front().before.even), "axis initial-even interval mismatch");
    require(equal(cone.initial_odd, cone.routes.front().before.odd), "axis initial-odd interval mismatch");
    require(equal(cone.final_even, cone.routes.back().after.even), "axis final-even interval mismatch");
    require(equal(cone.final_odd, cone.routes.back().after.odd), "axis final-odd interval mismatch");

    for (size_t index = 0; index < cone.routes.size(); ++index) {
        const auto& requirement = cone.routes[index];
        if (index > 0) {
            require(
                equal(requirement.before.even, cone.routes[index - 1].after.even) &&
                    equal(requirement.before.odd, cone.routes[index - 1].after.odd),
                "axis cone route states are not contiguous");
        }
        if (ttwv::is_predict_update_step(requirement.type)) {
            require(!requirement.output.empty(), "predict/update route has an empty output requirement");
            require(
                requirement.base.length() == requirement.output.length(),
                "predict/update base and output lengths differ");
            require(
                requirement.source.length() >= requirement.output.length(),
                "predict/update source is shorter than its output");
        }
    }
}

[[nodiscard]] ttwv::IndexInterval active_interval(const ttwv::IndexRectangle rectangle, const ttwv::Lwt2DAxis axis) {
    return axis == ttwv::Lwt2DAxis::kVertical ? rectangle.y : rectangle.x;
}

[[nodiscard]] ttwv::IndexInterval transverse_interval(
    const ttwv::IndexRectangle rectangle, const ttwv::Lwt2DAxis axis) {
    return axis == ttwv::Lwt2DAxis::kVertical ? rectangle.x : rectangle.y;
}

void validate_route_segment(
    const ttwv::Lwt2DChunkPlan& chunk,
    const size_t route_begin,
    const ttwv::AxisConePlan& cone,
    const ttwv::Lwt2DAxis axis,
    const ttwv::IndexInterval transverse) {
    for (size_t index = 0; index < cone.routes.size(); ++index) {
        const auto& expected = cone.routes[index];
        const auto& route = chunk.routes[route_begin + index];
        require(route.axis == axis, "2D route axis differs from its segment");
        require(route.axis_route_index == index, "2D route lost its axis-plan index");
        require(route.type == expected.type, "2D route type differs from its axis cone");
        require(
            equal(transverse_interval(route.source, axis), transverse) &&
                equal(transverse_interval(route.base, axis), transverse),
            "2D route changed the transverse dependency interval");

        if (expected.type == ttwv::StepType::kSwap) {
            require(route.output.empty(), "metadata swap unexpectedly owns an output rectangle");
            require(
                equal(active_interval(route.source, axis), expected.before.even) &&
                    equal(active_interval(route.base, axis), expected.before.odd),
                "metadata swap descriptors differ from the axis state");
            continue;
        }

        require(
            equal(active_interval(route.source, axis), expected.source) &&
                equal(active_interval(route.base, axis), expected.base) &&
                equal(active_interval(route.output, axis), expected.output),
            "2D route rectangles differ from exact axis requirements");
        if (ttwv::is_predict_update_step(expected.type)) {
            require(!route.in_place, "generic five-plane predict/update unexpectedly aliases its base");
            require(route.output_slot != route.base_slot, "generic five-plane route did not use scratch storage");
        } else {
            require(route.in_place, "2D scale route should update its plane in place");
        }
    }
}

void validate_route_schedule(const ttwv::Lwt2DChunkPlan& chunk) {
    const size_t y_route_count = chunk.y_cone.routes.size();
    const size_t x_route_count = chunk.x_cone.routes.size();
    require(
        chunk.routes.size() == 2 * y_route_count + 2 * x_route_count,
        "2D route schedule does not contain two transforms per axis");
    validate_route_segment(chunk, 0, chunk.y_cone, ttwv::Lwt2DAxis::kVertical, chunk.x_cone.initial_even);
    validate_route_segment(chunk, y_route_count, chunk.y_cone, ttwv::Lwt2DAxis::kVertical, chunk.x_cone.initial_odd);
    validate_route_segment(
        chunk, 2 * y_route_count, chunk.x_cone, ttwv::Lwt2DAxis::kHorizontal, chunk.y_cone.final_even);
    validate_route_segment(
        chunk, 2 * y_route_count + x_route_count, chunk.x_cone, ttwv::Lwt2DAxis::kHorizontal, chunk.y_cone.final_odd);

    std::array<ttwv::Lwt2DPlaneSlot, 4> bands = {
        chunk.final_bands.ll,
        chunk.final_bands.lh,
        chunk.final_bands.hl,
        chunk.final_bands.hh,
    };
    std::sort(bands.begin(), bands.end());
    require(
        std::adjacent_find(bands.begin(), bands.end()) == bands.end(),
        "two final bands alias the same workspace plane");
}

void validate_chunk(const ttwv::Lwt2DChunkPlan& chunk, const ttwv::Lwt2DExecutionPlan& plan) {
    validate_axis_cone(chunk.y_cone, plan.y_plan);
    validate_axis_cone(chunk.x_cone, plan.x_plan);

    require(
        equal(chunk.initial.ee, ttwv::interval_product(chunk.y_cone.initial_even, chunk.x_cone.initial_even)),
        "EE dependency is not the Cartesian product of the even axis cones");
    require(
        equal(chunk.initial.eo, ttwv::interval_product(chunk.y_cone.initial_even, chunk.x_cone.initial_odd)),
        "EO dependency is not the Cartesian product of the y-even/x-odd axis cones");
    require(
        equal(chunk.initial.oe, ttwv::interval_product(chunk.y_cone.initial_odd, chunk.x_cone.initial_even)),
        "OE dependency is not the Cartesian product of the y-odd/x-even axis cones");
    require(
        equal(chunk.initial.oo, ttwv::interval_product(chunk.y_cone.initial_odd, chunk.x_cone.initial_odd)),
        "OO dependency is not the Cartesian product of the odd axis cones");

    require(
        chunk.workspace_policy == ttwv::Lwt2DWorkspacePolicy::kFivePlaneGeneric,
        "initial 2D production planner selected the out-of-scope four-plane policy");
    ++g_five_plane_chunk_count;
    require(chunk.resources.plane_count == 5, "workspace plane-count policy mismatch");
    require(
        chunk.resources.plane_height_elements % 32 == 0 && chunk.resources.plane_width_elements % 32 == 0,
        "workspace plane is not full-tile aligned");
    uint64_t expected_workspace_bytes = 0;
    for (size_t slot = 0; slot < chunk.resources.plane_count; ++slot) {
        require(
            chunk.resources.plane_heights_elements[slot] % 32 == 0 &&
                chunk.resources.plane_widths_elements[slot] % 32 == 0,
            "workspace slot is not full-tile aligned");
        require(
            chunk.resources.plane_slot_bytes[slot] ==
                static_cast<uint64_t>(chunk.resources.plane_heights_elements[slot]) *
                    chunk.resources.plane_widths_elements[slot] * sizeof(float),
            "workspace slot byte accounting is inconsistent");
        expected_workspace_bytes += chunk.resources.plane_slot_bytes[slot];
    }
    require(chunk.resources.workspace_bytes == expected_workspace_bytes, "workspace byte accounting is inconsistent");
    require(chunk.resources.total_l1_bytes <= kTestL1BudgetBytes, "planned chunk exceeds the L1 test budget");
    validate_route_schedule(chunk);
}

void validate_coverage(const ttwv::Lwt2DExecutionPlan& plan) {
    ttwv::validate_lwt_2d_tiling_contract(plan.tiling);
    require(
        plan.tiling.input.logical == ttwv::Shape2D{.height = plan.input_height, .width = plan.input_width},
        "input tiling contract lost the logical shape");
    require(
        plan.tiling.band.logical == ttwv::Shape2D{.height = plan.band_height, .width = plan.band_width},
        "band tiling contract lost the logical shape");

    std::vector<uint8_t> coverage(plan.band_height * plan.band_width, 0);
    const size_t band_tile_rows = plan.tiling.band.storage.height / ttwv::kTileHeight2D;
    const size_t band_tile_columns = plan.tiling.band.storage.width / ttwv::kTileWidth2D;
    std::vector<uint8_t> tile_coverage(band_tile_rows * band_tile_columns, 0);
    for (const auto& chunk : plan.chunks) {
        require(
            chunk.final_band_rect.y.end <= plan.band_height && chunk.final_band_rect.x.end <= plan.band_width,
            "chunk final rectangle exceeds the band");
        require(
            chunk.execution_band_rect.y.begin % ttwv::kTileHeight2D == 0 &&
                chunk.execution_band_rect.y.end % ttwv::kTileHeight2D == 0 &&
                chunk.execution_band_rect.x.begin % ttwv::kTileWidth2D == 0 &&
                chunk.execution_band_rect.x.end % ttwv::kTileWidth2D == 0,
            "chunk execution rectangle violates the 32x32 tiling contract");
        require(
            chunk.execution_band_rect.y.begin <= chunk.final_band_rect.y.begin &&
                chunk.execution_band_rect.y.end >= chunk.final_band_rect.y.end &&
                chunk.execution_band_rect.x.begin <= chunk.final_band_rect.x.begin &&
                chunk.execution_band_rect.x.end >= chunk.final_band_rect.x.end,
            "chunk execution rectangle does not contain its logical output");
        require(
            chunk.execution_band_rect.y.end <= plan.tiling.band.storage.height &&
                chunk.execution_band_rect.x.end <= plan.tiling.band.storage.width,
            "chunk execution rectangle exceeds padded band storage");
        for (size_t y = chunk.final_band_rect.y.begin; y < chunk.final_band_rect.y.end; ++y) {
            for (size_t x = chunk.final_band_rect.x.begin; x < chunk.final_band_rect.x.end; ++x) {
                ++coverage[y * plan.band_width + x];
            }
        }
        for (size_t tile_y = chunk.execution_band_rect.y.begin / ttwv::kTileHeight2D;
             tile_y < chunk.execution_band_rect.y.end / ttwv::kTileHeight2D;
             ++tile_y) {
            for (size_t tile_x = chunk.execution_band_rect.x.begin / ttwv::kTileWidth2D;
                 tile_x < chunk.execution_band_rect.x.end / ttwv::kTileWidth2D;
                 ++tile_x) {
                ++tile_coverage[tile_y * band_tile_columns + tile_x];
            }
        }
        validate_chunk(chunk, plan);
    }
    require(
        std::all_of(coverage.begin(), coverage.end(), [](const uint8_t count) { return count == 1; }),
        "2D chunks do not cover every band element exactly once");
    require(
        std::all_of(tile_coverage.begin(), tile_coverage.end(), [](const uint8_t count) { return count == 1; }),
        "2D chunks do not cover every padded band tile exactly once");
    require(
        plan.active_core_count == std::min(plan.chunks.size(), static_cast<size_t>(kTestCoreLimit)),
        "active core count does not match the selected chunk grid");
    require(plan.max_l1_bytes <= kTestL1BudgetBytes, "execution-plan maximum L1 footprint exceeds budget");

    const std::vector<uint32_t> chunk_words = ttwv::build_lwt_2d_chunk_config_words(plan);
    const std::vector<uint32_t> route_words = ttwv::build_lwt_2d_route_config_words(plan);
    const std::vector<uint32_t> band_words = ttwv::build_lwt_2d_band_config_words(plan);
    require(
        chunk_words.size() == plan.chunks.size() * ttwv::device_protocol::kLwt2DChunkConfigWordCount,
        "serialized 2D chunk protocol has the wrong size");
    require(
        route_words.size() ==
            plan.chunks.size() * plan.chunks.front().routes.size() * ttwv::device_protocol::kLwt2DRouteConfigWordCount,
        "serialized 2D route protocol has the wrong size");
    require(
        band_words.size() == plan.chunks.size() * ttwv::device_protocol::kLwt2DBandConfigWordCount,
        "serialized 2D band protocol has the wrong size");
    for (size_t chunk_index = 0; chunk_index < plan.chunks.size(); ++chunk_index) {
        const auto& chunk = plan.chunks[chunk_index];
        const size_t offset = chunk_index * ttwv::device_protocol::kLwt2DChunkConfigWordCount;
        require(
            chunk_words[offset + ttwv::device_protocol::kLwt2DFinalYBegin] == chunk.final_band_rect.y.begin &&
                chunk_words[offset + ttwv::device_protocol::kLwt2DFinalYLength] == chunk.final_band_rect.height() &&
                chunk_words[offset + ttwv::device_protocol::kLwt2DFinalXBegin] == chunk.final_band_rect.x.begin &&
                chunk_words[offset + ttwv::device_protocol::kLwt2DFinalXLength] == chunk.final_band_rect.width(),
            "serialized 2D chunk lost its logical final rectangle");
    }
}

template <typename Scheme>
void validate_scheme() {
    constexpr std::array<std::array<size_t, 2>, 23> shapes = {
        std::array<size_t, 2>{1, 1},     std::array<size_t, 2>{1, 7},     std::array<size_t, 2>{7, 1},
        std::array<size_t, 2>{2, 2},     std::array<size_t, 2>{2, 3},     std::array<size_t, 2>{3, 2},
        std::array<size_t, 2>{5, 7},     std::array<size_t, 2>{15, 17},   std::array<size_t, 2>{17, 15},
        std::array<size_t, 2>{31, 31},   std::array<size_t, 2>{31, 32},   std::array<size_t, 2>{32, 31},
        std::array<size_t, 2>{32, 32},   std::array<size_t, 2>{32, 33},   std::array<size_t, 2>{33, 32},
        std::array<size_t, 2>{33, 33},   std::array<size_t, 2>{63, 65},   std::array<size_t, 2>{64, 64},
        std::array<size_t, 2>{65, 63},   std::array<size_t, 2>{65, 97},   std::array<size_t, 2>{127, 129},
        std::array<size_t, 2>{256, 256}, std::array<size_t, 2>{513, 769},
    };
    for (const auto& [height, width] : shapes) {
        const ttwv::Lwt2DExecutionPlan plan = ttwv::make_lwt_2d_execution_plan<Scheme>(
            height, width, kTestCoreLimit, kTestL1BudgetBytes, ttwv::BoundaryMode::kSymmetric);
        require(plan.input_height == height && plan.input_width == width, "input shape was not retained");
        validate_coverage(plan);
    }
}

void validate_zero_padding() {
    constexpr ttwv::Shape2D logical{.height = 3, .width = 5};
    std::vector<float> input(logical.height * logical.width);
    for (size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<float>(index + 1);
    }
    const ttwv::TiledShape2D shape = ttwv::make_tiled_shape_2d(logical);
    const std::vector<float> padded = ttwv::zero_pad_row_major_to_tiles_2d(input, logical);
    require(shape.storage == ttwv::Shape2D{.height = 32, .width = 32}, "small input was not padded to one tile");
    require(ttwv::has_zero_tile_padding_2d(padded, shape), "2D input padding contains non-zero values");
    for (size_t row = 0; row < logical.height; ++row) {
        for (size_t column = 0; column < logical.width; ++column) {
            require(
                padded[row * shape.storage.width + column] == input[row * logical.width + column],
                "2D zero padding changed a logical input value");
        }
    }
}

}  // namespace

int main() {
    try {
        validate_zero_padding();
        for (const ttwv::SchemeInfo& info : ttwv::available_wavelets()) {
            const auto validate = [&]<typename Scheme>() {
                validate_scheme<Scheme>();
                return 0;
            };
            static_cast<void>(ttwv::dispatch_scheme(info.name, validate));
        }
        require(g_five_plane_chunk_count > 0, "planner suite did not exercise the five-plane policy");
        std::cout << "2D dependency-cone planner validation passed for " << ttwv::available_wavelets().size()
                  << " schemes (" << g_five_plane_chunk_count << " five-plane chunks)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
