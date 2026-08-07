// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <tt_stl/assert.hpp>
#include <utility>
#include <vector>

#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/distributed.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt_wavelet/include/lifting/inverse_plan_2d.hpp"
#include "tt_wavelet/include/lifting/plan_2d.hpp"
#include "tt_wavelet/include/lifting/policy.hpp"
#include "tt_wavelet/include/lifting/static_scheme.hpp"

namespace ttwv {

struct Lwt2DSchedulerTelemetry {
    tt::ARCH architecture{tt::ARCH::Invalid};
    BoundaryMode boundary_mode{BoundaryMode::kSymmetric};
    Shape2D logical_input{};
    Shape2D padded_input{};
    Shape2D logical_band{};
    Shape2D padded_band{};
    uint32_t available_worker_core_count{0};
    uint32_t active_core_count{0};
    uint32_t batch_count{1};
    uint32_t chunks_per_sample{0};
    uint32_t total_work_items{0};
    uint32_t min_work_items_per_core{0};
    uint32_t max_work_items_per_core{0};
    uint32_t chunk_count{0};
    uint32_t chunk_tiles_y{0};
    uint32_t chunk_tiles_x{0};
    uint32_t route_count{0};
    uint32_t executable_route_count{0};
    uint32_t scale_routes_removed{0};
    uint64_t estimated_latency_cycles{0};
    double max_dependency_overhead{0.0};
    uint64_t l1_workspace_bytes{0};
    uint64_t l1_circular_buffer_bytes{0};
    uint64_t l1_metadata_bytes{0};
    uint64_t l1_synchronization_bytes{0};
    uint64_t l1_total_bytes{0};
    uint64_t l1_capacity_bytes{0};
    uint64_t l1_headroom_bytes{0};
    uint64_t exact_initial_elements{0};
    uint64_t internal_initial_elements{0};
    uint64_t exact_route_elements{0};
    uint64_t internal_route_elements{0};
    uint64_t exact_final_elements{0};
    uint64_t internal_final_elements{0};
};

struct Lwt2DWorkingBuffers {
    std::array<std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>, device_protocol::kLwt2DPlaneCount> planes{};
    std::array<std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>, device_protocol::kLwt2DBandCount> outputs{};
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> chunk_config{};
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> route_config{};
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> band_config{};
    std::vector<tt::tt_metal::CoreCoord> cores;
    Lwt2DSchedulerTelemetry scheduler{};
};

struct Lwt2DExecutable {
    Lwt2DExecutionPlan plan{};
    Lwt2DWorkingBuffers buffers{};
    tt::tt_metal::distributed::MeshWorkload workload{};
};

struct Ilwt2DExecutable {
    Ilwt2DExecutionPlan plan{};
    Lwt2DWorkingBuffers buffers{};
    tt::tt_metal::distributed::MeshWorkload workload{};
};

[[nodiscard]] Lwt2DExecutable create_lwt_2d_executable_impl(
    const std::filesystem::path& kernel_root,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const tt::tt_metal::Buffer& input_buffer,
    uint32_t batch_count,
    uint32_t core_limit,
    Lwt2DExecutionPlan plan,
    const char* compute_scheme_header,
    const char* compute_scheme_type);

template <typename Scheme>
[[nodiscard]] Lwt2DExecutable create_lwt_2d_executable(
    const std::filesystem::path& kernel_root,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const tt::tt_metal::Buffer& input_buffer,
    const size_t logical_height,
    const size_t logical_width,
    const uint32_t core_limit = 1,
    const BoundaryMode boundary_mode = BoundaryMode::kSymmetric,
    const uint32_t batch_count = 1) {
    TT_FATAL(logical_height > 0 && logical_width > 0, "2D LWT input shape must be positive");
    TT_FATAL(batch_count > 0, "2D LWT batch count must be positive");
    Lwt2DExecutionPlan plan = make_lwt_2d_execution_plan<Scheme>(
        logical_height,
        logical_width,
        core_limit,
        768 * 1024,
        boundary_mode,
        true,
        true,
        Lwt2DRouteDomainPolicy::kExact);
    TT_FATAL(
        input_buffer.size() >= static_cast<uint64_t>(batch_count) *
                                   checked_shape_area_2d(plan.tiling.input.storage, "2D input storage") * sizeof(float),
        "2D input buffer is smaller than its padded tile shape");
    return create_lwt_2d_executable_impl(
        kernel_root,
        mesh_device,
        input_buffer,
        batch_count,
        core_limit,
        std::move(plan),
        Scheme::compute_scheme_header,
        Scheme::compute_scheme_type);
}

void prepare_lwt_2d(tt::tt_metal::distributed::MeshCommandQueue& command_queue, Lwt2DExecutable& executable);

void enqueue_lwt_2d(tt::tt_metal::distributed::MeshCommandQueue& command_queue, Lwt2DExecutable& executable);

void execute_lwt_2d(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    Lwt2DExecutable& executable);

[[nodiscard]] Ilwt2DExecutable create_ilwt_2d_executable_impl(
    const std::filesystem::path& kernel_root,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const std::array<const tt::tt_metal::Buffer*, device_protocol::kLwt2DBandCount>& band_buffers,
    uint32_t batch_count,
    uint32_t core_limit,
    Ilwt2DExecutionPlan plan,
    const char* inverse_compute_scheme_header,
    const char* inverse_compute_scheme_type);

template <typename Scheme>
[[nodiscard]] Ilwt2DExecutable create_ilwt_2d_executable(
    const std::filesystem::path& kernel_root,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const tt::tt_metal::Buffer& ll,
    const tt::tt_metal::Buffer& lh,
    const tt::tt_metal::Buffer& hl,
    const tt::tt_metal::Buffer& hh,
    const size_t output_height,
    const size_t output_width,
    const uint32_t core_limit = 1,
    const BoundaryMode boundary_mode = BoundaryMode::kSymmetric,
    const uint32_t batch_count = 1) {
    TT_FATAL(batch_count > 0, "2D ILWT batch count must be positive");
    using InverseScheme = typename Scheme::inverse;
    const ArchitecturePolicy architecture_policy = make_architecture_policy(mesh_device.arch());
    Ilwt2DExecutionPlan plan = make_ilwt_2d_execution_plan<Scheme>(
        output_height,
        output_width,
        core_limit,
        768 * 1024,
        boundary_mode,
        architecture_policy.inverse_2d_coordination_penalty_cycles_per_core);
    const size_t required_band_bytes =
        checked_shape_area_2d(plan.tiling.band.storage, "2D ILWT band storage") * sizeof(float);
    const std::array<const tt::tt_metal::Buffer*, device_protocol::kLwt2DBandCount> bands = {&ll, &lh, &hl, &hh};
    for (const auto* band : bands) {
        TT_FATAL(
            band->size() >= static_cast<uint64_t>(batch_count) * required_band_bytes,
            "2D ILWT input band is smaller than its batched tiled storage shape");
    }
    return create_ilwt_2d_executable_impl(
        kernel_root,
        mesh_device,
        bands,
        batch_count,
        core_limit,
        std::move(plan),
        InverseScheme::compute_scheme_header,
        InverseScheme::compute_scheme_type);
}

void prepare_ilwt_2d(tt::tt_metal::distributed::MeshCommandQueue& command_queue, Ilwt2DExecutable& executable);

void enqueue_ilwt_2d(tt::tt_metal::distributed::MeshCommandQueue& command_queue, Ilwt2DExecutable& executable);

void execute_ilwt_2d(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    Ilwt2DExecutable& executable);

}  // namespace ttwv
