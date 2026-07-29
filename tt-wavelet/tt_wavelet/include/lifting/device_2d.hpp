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
#include "tt_wavelet/include/lifting/static_scheme.hpp"

namespace ttwv {

enum class Lwt2DSplitImplementation : uint8_t {
    kScalar,
    kTiled,
};

enum class Lwt2DRouteStagingImplementation : uint8_t {
    kScalar,
    kOptimized,
};

enum class Lwt2DRoutePersistenceImplementation : uint8_t {
    kScalar,
    kFullTile,
};

enum class Lwt2DTerminalWriteImplementation : uint8_t {
    kFragmented,
    kTiled,
};

enum class Lwt2DScalePolicy : uint8_t {
    kExplicit,
    kFused,
};

enum class Lwt2DPlannerPolicy : uint8_t {
    kMaxCores,
    kLatency,
};

enum class Lwt2DRouteConfigImplementation : uint8_t {
    kPerRoute,
    kPreloaded,
};

enum class Lwt2DExactTransferImplementation : uint8_t {
    kLocalNoc,
    kL1Copy,
};

struct Lwt2DTransportPolicy {
    Lwt2DRouteStagingImplementation route_staging{Lwt2DRouteStagingImplementation::kOptimized};
    Lwt2DRoutePersistenceImplementation route_persistence{Lwt2DRoutePersistenceImplementation::kFullTile};
    Lwt2DTerminalWriteImplementation terminal_writes{Lwt2DTerminalWriteImplementation::kTiled};
    Lwt2DScalePolicy scale{Lwt2DScalePolicy::kFused};
    Lwt2DPlannerPolicy planner{Lwt2DPlannerPolicy::kLatency};
    Lwt2DRouteConfigImplementation route_config{Lwt2DRouteConfigImplementation::kPreloaded};
    Lwt2DExactTransferImplementation exact_transfer{Lwt2DExactTransferImplementation::kLocalNoc};
    Lwt2DRouteDomainPolicy route_domain{Lwt2DRouteDomainPolicy::kExact};
    // Benchmark-only: feed persistent-zero tiles directly to the real
    // unpack/SFPU/pack route kernel, bypassing workspace plane assembly.
    bool compute_only_benchmark{false};
    // Diagnostic-only: run initial EE/EO/OE/OO construction for every chunk
    // without entering the downstream lifting route pipeline.
    bool split_only_benchmark{false};
    // Validation-only: compare every optimized fast-path CB page against the
    // scalar gather before publishing it to the compute kernel.
    bool validate_route_staging{false};
};

struct Lwt2DSchedulerTelemetry {
    tt::ARCH architecture{tt::ARCH::Invalid};
    BoundaryMode boundary_mode{BoundaryMode::kSymmetric};
    Shape2D logical_input{};
    Shape2D padded_input{};
    Shape2D logical_band{};
    Shape2D padded_band{};
    uint32_t active_core_count{0};
    uint32_t chunk_count{0};
    uint32_t chunk_tiles_y{0};
    uint32_t chunk_tiles_x{0};
    uint32_t route_count{0};
    uint32_t executable_route_count{0};
    uint32_t scale_routes_removed{0};
    bool latency_oriented_planner{false};
    Lwt2DRouteDomainPolicy route_domain{Lwt2DRouteDomainPolicy::kExact};
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

struct Lwt2DSplitMetricsSummary {
    uint64_t max_core_cycles{0};
    uint64_t raw_input_bytes{0};
    uint64_t local_output_bytes{0};
    uint64_t noc_read_calls{0};
    uint64_t noc_read_barriers{0};
    uint64_t interior_macro_tiles{0};
    uint64_t boundary_macro_tiles{0};
    uint64_t max_core_macro_tiles{0};
};

struct Lwt2DRouteTransportMetric {
    uint32_t chunk_index{0};
    uint32_t route_index{0};
    uint32_t axis{0};
    uint32_t step_type{0};
    uint32_t coefficient_count{0};
    uint32_t output_tiles{0};
    uint64_t reader_config_cycles{0};
    uint64_t staging_cycles{0};
    uint64_t compute_cycles{0};
    uint64_t persistence_cycles{0};
    uint64_t synchronization_wait_cycles{0};
    uint64_t writer_config_cycles{0};
    uint64_t exact_source_tiles{0};
    uint64_t shifted_source_tiles{0};
    uint64_t generic_source_tiles{0};
    uint64_t exact_base_tiles{0};
    uint64_t shifted_base_tiles{0};
    uint64_t generic_base_tiles{0};
    uint64_t persistence_tiles{0};
};

struct Lwt2DTransportMetricsSummary {
    std::vector<Lwt2DRouteTransportMetric> routes;
    uint64_t max_reader_kernel_cycles{0};
    uint64_t max_writer_kernel_cycles{0};
    uint64_t max_route_staging_cycles{0};
    uint64_t max_route_compute_cycles{0};
    uint64_t max_route_persistence_cycles{0};
    uint64_t max_route_synchronization_wait_cycles{0};
    uint64_t total_route_tiles{0};
    uint64_t exact_source_tiles{0};
    uint64_t shifted_source_tiles{0};
    uint64_t generic_source_tiles{0};
    uint64_t exact_base_tiles{0};
    uint64_t shifted_base_tiles{0};
    uint64_t generic_base_tiles{0};
    uint64_t exact_terminal_tiles{0};
    uint64_t fragmented_terminal_tiles{0};
    uint64_t validated_staging_tiles{0};
    uint64_t staging_validation_mismatches{0};
    uint64_t validation_exact_mismatches{0};
    uint64_t validation_shifted_mismatches{0};
    uint64_t validation_two_axis_mismatches{0};
    uint64_t validation_partial_mismatches{0};
    uint64_t validation_empty_mismatches{0};
    uint64_t validated_persistence_tiles{0};
    uint64_t persistence_validation_mismatches{0};
    uint64_t max_terminal_write_cycles{0};
    uint64_t max_core_cycles{0};
    double mean_active_core_cycles{0.0};
    uint64_t max_core_route_tiles{0};
    uint64_t max_core_config_cycles{0};
    uint64_t max_core_staging_cycles{0};
    uint64_t max_core_compute_pipeline_cycles{0};
    uint64_t max_core_persistence_cycles{0};
    uint64_t max_core_synchronization_wait_cycles{0};
    uint64_t max_core_terminal_write_cycles{0};
    double mean_active_core_config_cycles{0.0};
    double mean_active_core_staging_cycles{0.0};
    double mean_active_core_compute_pipeline_cycles{0.0};
    double mean_active_core_persistence_cycles{0.0};
    double mean_active_core_synchronization_wait_cycles{0.0};
    double mean_active_core_terminal_write_cycles{0.0};
};

struct Lwt2DWorkingBuffers {
    std::array<std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>, device_protocol::kLwt2DPlaneCount> planes{};
    std::array<std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>, device_protocol::kLwt2DBandCount> outputs{};
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> chunk_config{};
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> route_config{};
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> band_config{};
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> split_metrics{};
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> transport_metrics{};
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> split_snapshots{};
    uint32_t split_snapshot_tiles_per_plane{0};
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> route_snapshots{};
    uint32_t snapshot_tiles_per_route{0};
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
    Lwt2DExecutionPlan plan,
    const char* compute_scheme_header,
    const char* compute_scheme_type,
    bool capture_route_snapshots,
    Lwt2DSplitImplementation split_implementation,
    bool capture_split_metrics,
    bool capture_split_snapshots,
    Lwt2DTransportPolicy transport_policy,
    bool capture_transport_metrics);

template <typename Scheme>
[[nodiscard]] Lwt2DExecutable create_lwt_2d_executable(
    const std::filesystem::path& kernel_root,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const tt::tt_metal::Buffer& input_buffer,
    const size_t logical_height,
    const size_t logical_width,
    const uint32_t core_limit = 1,
    const bool capture_route_snapshots = false,
    const Lwt2DSplitImplementation split_implementation = Lwt2DSplitImplementation::kTiled,
    const bool capture_split_metrics = false,
    const bool capture_split_snapshots = false,
    const Lwt2DTransportPolicy transport_policy = {},
    const bool capture_transport_metrics = false,
    const BoundaryMode boundary_mode = BoundaryMode::kSymmetric) {
    TT_FATAL(logical_height > 0 && logical_width > 0, "2D LWT input shape must be positive");
    // Preserve the benchmarked production search space for the exact domain.
    // The deliberately larger tile-closed A/B domain must instead be judged
    // against the architecture's actual per-core L1 capacity.
    const uint64_t initial_l1_budget_bytes =
        transport_policy.route_domain == Lwt2DRouteDomainPolicy::kTileClosed
            ? mesh_device.l1_size_per_core()
            : 768 * 1024;
    Lwt2DExecutionPlan plan = make_lwt_2d_execution_plan<Scheme>(
        logical_height,
        logical_width,
        core_limit,
        initial_l1_budget_bytes,
        boundary_mode,
        transport_policy.scale == Lwt2DScalePolicy::kFused,
        transport_policy.planner == Lwt2DPlannerPolicy::kLatency,
        transport_policy.route_domain);
    TT_FATAL(
        input_buffer.size() >= checked_shape_area_2d(plan.tiling.input.storage, "2D input storage") * sizeof(float),
        "2D input buffer is smaller than its padded tile shape");
    return create_lwt_2d_executable_impl(
        kernel_root,
        mesh_device,
        input_buffer,
        std::move(plan),
        Scheme::compute_scheme_header,
        Scheme::compute_scheme_type,
        capture_route_snapshots,
        split_implementation,
        capture_split_metrics,
        capture_split_snapshots,
        transport_policy,
        capture_transport_metrics);
}

void prepare_lwt_2d(tt::tt_metal::distributed::MeshCommandQueue& command_queue, Lwt2DExecutable& executable);

void execute_lwt_2d(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    Lwt2DExecutable& executable);

[[nodiscard]] Ilwt2DExecutable create_ilwt_2d_executable_impl(
    const std::filesystem::path& kernel_root,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const std::array<const tt::tt_metal::Buffer*, device_protocol::kLwt2DBandCount>& band_buffers,
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
    const BoundaryMode boundary_mode = BoundaryMode::kSymmetric) {
    using InverseScheme = typename Scheme::inverse;
    Ilwt2DExecutionPlan plan =
        make_ilwt_2d_execution_plan<Scheme>(
            output_height, output_width, core_limit, 768 * 1024, boundary_mode);
    const size_t required_band_bytes =
        checked_shape_area_2d(plan.tiling.band.storage, "2D ILWT band storage") * sizeof(float);
    const std::array<const tt::tt_metal::Buffer*, device_protocol::kLwt2DBandCount> bands = {&ll, &lh, &hl, &hh};
    for (const auto* band : bands) {
        TT_FATAL(band->size() >= required_band_bytes, "2D ILWT input band is smaller than its tiled storage shape");
    }
    return create_ilwt_2d_executable_impl(
        kernel_root,
        mesh_device,
        bands,
        std::move(plan),
        InverseScheme::compute_scheme_header,
        InverseScheme::compute_scheme_type);
}

void prepare_ilwt_2d(
    tt::tt_metal::distributed::MeshCommandQueue& command_queue, Ilwt2DExecutable& executable);

void execute_ilwt_2d(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    Ilwt2DExecutable& executable);

[[nodiscard]] Lwt2DSplitMetricsSummary read_lwt_2d_split_metrics(
    tt::tt_metal::distributed::MeshCommandQueue& command_queue, Lwt2DExecutable& executable);

[[nodiscard]] Lwt2DTransportMetricsSummary read_lwt_2d_transport_metrics(
    tt::tt_metal::distributed::MeshCommandQueue& command_queue, Lwt2DExecutable& executable);

}  // namespace ttwv
