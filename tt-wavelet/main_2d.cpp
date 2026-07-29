// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt-metalium/tilize_utils.hpp"
#include "tt_wavelet/include/common/boundary_parse.hpp"
#include "tt_wavelet/include/common/tiling_2d.hpp"
#include "tt_wavelet/include/lifting/device_2d.hpp"
#include "tt_wavelet/include/lifting/reference_2d.hpp"
#include "tt_wavelet/include/schemes/generated/registry.hpp"
#include "tt_wavelet/include/schemes/testing/synthetic_k17.hpp"

namespace {

struct Options {
    bool benchmark{false};
    bool include_transfers{false};
    bool binary_input{false};
    bool quiet{false};
    bool split_metrics{false};
    bool transport_metrics{false};
    std::string microbenchmark_mode{"full"};
    size_t repeats{1};
    size_t warmup_runs{1};
    uint32_t core_limit{1};
    ttwv::BoundaryMode boundary_mode{ttwv::BoundaryMode::kSymmetric};
    ttwv::Lwt2DSplitImplementation split_implementation{ttwv::Lwt2DSplitImplementation::kTiled};
    ttwv::Lwt2DTransportPolicy transport_policy{};
    std::string wavelet;
    size_t height{0};
    size_t width{0};
    std::filesystem::path input_path;
    std::optional<std::filesystem::path> output_prefix;
    std::optional<std::filesystem::path> split_snapshot_prefix;
    std::optional<std::filesystem::path> route_snapshot_prefix;
    std::optional<std::filesystem::path> alignment_csv;
    std::optional<std::filesystem::path> transport_metrics_csv;
};

struct DeviceInput {
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> buffer;
};

struct DeviceOutput {
    double execution_time_ms{0.0};
    ttwv::Lwt2DSchedulerTelemetry scheduler{};
    ttwv::Lwt2DReferenceOutput values{};
};

[[nodiscard]] std::string usage() {
    return "Usage: lwt_2d "
           "[--boundary-mode zero|constant|symmetric|reflect|periodic|smooth|antisymmetric|antireflect] "
           "[--binary-input] "
           "[--cores N] "
           "[--output-prefix PATH] [--quiet] [--split-implementation scalar|tiled] [--split-metrics] "
           "[--route-staging scalar|optimized] [--route-persistence scalar|full-tile] "
           "[--terminal-writes fragmented|tiled] [--scale-policy explicit|fused] "
           "[--planner max-cores|latency] [--route-config per-route|preloaded] "
           "[--exact-transfer local-noc|l1-copy] "
           "[--route-domain exact|tile-closed] "
           "[--transport-metrics [--transport-metrics-csv PATH]] [--validate-route-staging] "
           "[--alignment-csv PATH] "
           "[--microbenchmark empty|split|staging|compute|persistence|terminal|full] "
           "[--split-snapshot-prefix PATH] [--route-snapshot-prefix PATH] "
           "[--benchmark [--include-transfers] [--repeats N] "
           "[--warmup-runs N]] WAVELET HEIGHT WIDTH INPUT_FILE";
}

[[nodiscard]] size_t parse_positive(const std::string& text, const char* label) {
    if (text.empty() || text.front() == '-') {
        throw std::runtime_error(std::string{label} + " must be positive");
    }
    size_t consumed = 0;
    const unsigned long long value = std::stoull(text, &consumed);
    if (consumed != text.size() || value == 0 || value > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error(std::string{label} + " must be positive");
    }
    return static_cast<size_t>(value);
}

[[nodiscard]] size_t parse_nonnegative(const std::string& text, const char* label) {
    if (text.empty() || text.front() == '-') {
        throw std::runtime_error(std::string{label} + " must be non-negative");
    }
    size_t consumed = 0;
    const unsigned long long value = std::stoull(text, &consumed);
    if (consumed != text.size() || value > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error(std::string{label} + " must be non-negative");
    }
    return static_cast<size_t>(value);
}

[[nodiscard]] Options parse_options(const int argc, char** argv) {
    Options options;
    std::vector<std::string> positional;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--boundary-mode") {
            if (++index >= argc || !ttwv::parse_boundary_mode(argv[index], options.boundary_mode)) {
                throw std::runtime_error(
                    "--boundary-mode requires zero, constant, symmetric, reflect, periodic, smooth, "
                    "antisymmetric, or antireflect");
            }
        } else if (argument == "--benchmark") {
            options.benchmark = true;
        } else if (argument == "--include-transfers") {
            options.include_transfers = true;
        } else if (argument == "--binary-input") {
            options.binary_input = true;
        } else if (argument == "--quiet") {
            options.quiet = true;
        } else if (argument == "--split-metrics") {
            options.split_metrics = true;
        } else if (argument == "--transport-metrics") {
            options.transport_metrics = true;
        } else if (argument == "--transport-metrics-csv") {
            if (++index >= argc || std::string_view{argv[index]}.empty()) {
                throw std::runtime_error("--transport-metrics-csv requires a path");
            }
            options.transport_metrics = true;
            options.transport_metrics_csv = std::filesystem::path{argv[index]};
        } else if (argument == "--validate-route-staging") {
            options.transport_metrics = true;
            options.transport_policy.validate_route_staging = true;
        } else if (argument == "--microbenchmark") {
            if (++index >= argc) {
                throw std::runtime_error("--microbenchmark requires a phase");
            }
            options.microbenchmark_mode = argv[index];
            const std::array<std::string_view, 7> modes = {
                "empty", "split", "staging", "compute", "persistence", "terminal", "full"};
            if (std::find(modes.begin(), modes.end(), options.microbenchmark_mode) == modes.end()) {
                throw std::runtime_error(
                    "--microbenchmark requires empty, split, staging, compute, persistence, terminal, or full");
            }
            options.benchmark = true;
            if (options.microbenchmark_mode == "split") {
                options.split_metrics = true;
                options.transport_policy.split_only_benchmark = true;
            } else if (options.microbenchmark_mode == "compute") {
                options.transport_metrics = true;
                options.transport_policy.compute_only_benchmark = true;
            } else if (options.microbenchmark_mode != "empty" && options.microbenchmark_mode != "full") {
                options.transport_metrics = true;
            }
        } else if (argument == "--alignment-csv") {
            if (++index >= argc || std::string_view{argv[index]}.empty()) {
                throw std::runtime_error("--alignment-csv requires a path");
            }
            options.alignment_csv = std::filesystem::path{argv[index]};
        } else if (argument == "--cores") {
            if (++index >= argc) {
                throw std::runtime_error("--cores requires a value");
            }
            const size_t cores = parse_positive(argv[index], "--cores");
            if (cores > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("--cores exceeds uint32_t");
            }
            options.core_limit = static_cast<uint32_t>(cores);
        } else if (argument == "--split-implementation") {
            if (++index >= argc) {
                throw std::runtime_error("--split-implementation requires scalar or tiled");
            }
            const std::string_view implementation = argv[index];
            if (implementation == "scalar") {
                options.split_implementation = ttwv::Lwt2DSplitImplementation::kScalar;
            } else if (implementation == "tiled") {
                options.split_implementation = ttwv::Lwt2DSplitImplementation::kTiled;
            } else {
                throw std::runtime_error("--split-implementation requires scalar or tiled");
            }
        } else if (argument == "--route-staging") {
            if (++index >= argc) {
                throw std::runtime_error("--route-staging requires scalar or optimized");
            }
            const std::string_view implementation = argv[index];
            if (implementation == "scalar") {
                options.transport_policy.route_staging = ttwv::Lwt2DRouteStagingImplementation::kScalar;
            } else if (implementation == "optimized") {
                options.transport_policy.route_staging = ttwv::Lwt2DRouteStagingImplementation::kOptimized;
            } else {
                throw std::runtime_error("--route-staging requires scalar or optimized");
            }
        } else if (argument == "--route-persistence") {
            if (++index >= argc) {
                throw std::runtime_error("--route-persistence requires scalar or full-tile");
            }
            const std::string_view implementation = argv[index];
            if (implementation == "scalar") {
                options.transport_policy.route_persistence = ttwv::Lwt2DRoutePersistenceImplementation::kScalar;
            } else if (implementation == "full-tile") {
                options.transport_policy.route_persistence = ttwv::Lwt2DRoutePersistenceImplementation::kFullTile;
            } else {
                throw std::runtime_error("--route-persistence requires scalar or full-tile");
            }
        } else if (argument == "--terminal-writes") {
            if (++index >= argc) {
                throw std::runtime_error("--terminal-writes requires fragmented or tiled");
            }
            const std::string_view implementation = argv[index];
            if (implementation == "fragmented") {
                options.transport_policy.terminal_writes = ttwv::Lwt2DTerminalWriteImplementation::kFragmented;
            } else if (implementation == "tiled") {
                options.transport_policy.terminal_writes = ttwv::Lwt2DTerminalWriteImplementation::kTiled;
            } else {
                throw std::runtime_error("--terminal-writes requires fragmented or tiled");
            }
        } else if (argument == "--scale-policy") {
            if (++index >= argc) {
                throw std::runtime_error("--scale-policy requires explicit or fused");
            }
            const std::string_view policy = argv[index];
            if (policy == "explicit") {
                options.transport_policy.scale = ttwv::Lwt2DScalePolicy::kExplicit;
            } else if (policy == "fused") {
                options.transport_policy.scale = ttwv::Lwt2DScalePolicy::kFused;
            } else {
                throw std::runtime_error("--scale-policy requires explicit or fused");
            }
        } else if (argument == "--planner") {
            if (++index >= argc) {
                throw std::runtime_error("--planner requires max-cores or latency");
            }
            const std::string_view policy = argv[index];
            if (policy == "max-cores") {
                options.transport_policy.planner = ttwv::Lwt2DPlannerPolicy::kMaxCores;
            } else if (policy == "latency") {
                options.transport_policy.planner = ttwv::Lwt2DPlannerPolicy::kLatency;
            } else {
                throw std::runtime_error("--planner requires max-cores or latency");
            }
        } else if (argument == "--route-config") {
            if (++index >= argc) {
                throw std::runtime_error("--route-config requires per-route or preloaded");
            }
            const std::string_view policy = argv[index];
            if (policy == "per-route") {
                options.transport_policy.route_config = ttwv::Lwt2DRouteConfigImplementation::kPerRoute;
            } else if (policy == "preloaded") {
                options.transport_policy.route_config = ttwv::Lwt2DRouteConfigImplementation::kPreloaded;
            } else {
                throw std::runtime_error("--route-config requires per-route or preloaded");
            }
        } else if (argument == "--exact-transfer") {
            if (++index >= argc) {
                throw std::runtime_error("--exact-transfer requires local-noc or l1-copy");
            }
            const std::string_view policy = argv[index];
            if (policy == "local-noc") {
                options.transport_policy.exact_transfer = ttwv::Lwt2DExactTransferImplementation::kLocalNoc;
            } else if (policy == "l1-copy") {
                options.transport_policy.exact_transfer = ttwv::Lwt2DExactTransferImplementation::kL1Copy;
            } else {
                throw std::runtime_error("--exact-transfer requires local-noc or l1-copy");
            }
        } else if (argument == "--route-domain") {
            if (++index >= argc) {
                throw std::runtime_error("--route-domain requires exact or tile-closed");
            }
            const std::string_view policy = argv[index];
            if (policy == "exact") {
                options.transport_policy.route_domain = ttwv::Lwt2DRouteDomainPolicy::kExact;
            } else if (policy == "tile-closed") {
                options.transport_policy.route_domain = ttwv::Lwt2DRouteDomainPolicy::kTileClosed;
            } else {
                throw std::runtime_error("--route-domain requires exact or tile-closed");
            }
        } else if (argument == "--route-snapshot-prefix") {
            if (++index >= argc || std::string_view{argv[index]}.empty()) {
                throw std::runtime_error("--route-snapshot-prefix requires a path");
            }
            options.route_snapshot_prefix = std::filesystem::path{argv[index]};
        } else if (argument == "--split-snapshot-prefix") {
            if (++index >= argc || std::string_view{argv[index]}.empty()) {
                throw std::runtime_error("--split-snapshot-prefix requires a path");
            }
            options.split_snapshot_prefix = std::filesystem::path{argv[index]};
        } else if (argument == "--output-prefix") {
            if (++index >= argc || std::string_view{argv[index]}.empty()) {
                throw std::runtime_error("--output-prefix requires a path");
            }
            options.output_prefix = std::filesystem::path{argv[index]};
        } else if (argument == "--repeats") {
            if (++index >= argc) {
                throw std::runtime_error("--repeats requires a value");
            }
            options.repeats = parse_positive(argv[index], "--repeats");
        } else if (argument == "--warmup-runs") {
            if (++index >= argc) {
                throw std::runtime_error("--warmup-runs requires a value");
            }
            options.warmup_runs = parse_nonnegative(argv[index], "--warmup-runs");
        } else if (argument == "--help" || argument == "-h") {
            std::cout << usage() << '\n';
            std::exit(EXIT_SUCCESS);
        } else if (argument.starts_with("--")) {
            throw std::runtime_error("Unknown option: " + argument);
        } else {
            positional.push_back(argument);
        }
    }
    if (positional.size() != 4) {
        throw std::runtime_error(usage());
    }
    options.wavelet = positional[0];
    options.height = parse_positive(positional[1], "HEIGHT");
    options.width = parse_positive(positional[2], "WIDTH");
    if (ttwv::boundary_mode_requires_multiple_samples(options.boundary_mode) &&
        (options.height <= 1 || options.width <= 1)) {
        throw std::runtime_error("2D reflect and antireflect modes require HEIGHT and WIDTH greater than one");
    }
    options.input_path = positional[3];
    if (!options.benchmark && (options.repeats != 1 || options.warmup_runs != 1)) {
        throw std::runtime_error("--repeats and --warmup-runs require --benchmark");
    }
    if (options.include_transfers && !options.benchmark) {
        throw std::runtime_error("--include-transfers requires --benchmark");
    }
    if (options.benchmark && options.route_snapshot_prefix) {
        throw std::runtime_error("--route-snapshot-prefix cannot be used with --benchmark");
    }
    if (options.benchmark && options.split_snapshot_prefix) {
        throw std::runtime_error("--split-snapshot-prefix cannot be used with --benchmark");
    }
    if (options.benchmark && options.output_prefix) {
        throw std::runtime_error("--output-prefix cannot be used with --benchmark");
    }
    if (options.split_metrics && !options.benchmark) {
        throw std::runtime_error("--split-metrics requires --benchmark");
    }
    if (options.transport_metrics && !options.benchmark) {
        throw std::runtime_error("--transport-metrics requires --benchmark");
    }
    if (options.transport_policy.validate_route_staging &&
        options.transport_policy.route_staging != ttwv::Lwt2DRouteStagingImplementation::kOptimized) {
        throw std::runtime_error("--validate-route-staging requires --route-staging optimized");
    }
    if (options.transport_policy.compute_only_benchmark &&
        options.transport_policy.route_staging != ttwv::Lwt2DRouteStagingImplementation::kOptimized) {
        throw std::runtime_error("--microbenchmark compute requires --route-staging optimized");
    }
    if (options.transport_policy.compute_only_benchmark && options.transport_policy.validate_route_staging) {
        throw std::runtime_error("--microbenchmark compute cannot be combined with --validate-route-staging");
    }
    if (options.quiet && !options.output_prefix && !options.split_snapshot_prefix && !options.route_snapshot_prefix &&
        !options.benchmark) {
        throw std::runtime_error("--quiet requires an output or snapshot prefix");
    }
    return options;
}

[[nodiscard]] std::vector<float> read_input(
    const std::filesystem::path& path, const size_t height, const size_t width, const bool binary) {
    if (height > std::numeric_limits<size_t>::max() / width) {
        throw std::runtime_error("2D input shape overflows size_t");
    }
    std::ifstream input(path, binary ? std::ios::binary : std::ios::in);
    if (!input.good()) {
        throw std::runtime_error("Failed to open input file: " + path.string());
    }
    std::vector<float> values(height * width);
    if (binary) {
        input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
        if (input.gcount() != static_cast<std::streamsize>(values.size() * sizeof(float))) {
            throw std::runtime_error("Binary input contains fewer than HEIGHT x WIDTH FP32 values");
        }
    } else {
        values.clear();
        values.reserve(height * width);
        for (float value = 0.0F; input >> value;) {
            values.push_back(value);
        }
        if (!input.eof()) {
            throw std::runtime_error("Input file contains a non-numeric token");
        }
    }
    if (values.size() != height * width) {
        throw std::runtime_error("Input element count does not match HEIGHT x WIDTH");
    }
    return values;
}

[[nodiscard]] std::vector<float> tilize_padded(const std::vector<float>& row_major, const ttwv::Shape2D shape) {
    std::vector<float> tiled;
    tiled.reserve(row_major.size());
    for (size_t tile_y = 0; tile_y < shape.height / ttwv::kTileHeight2D; ++tile_y) {
        for (size_t tile_x = 0; tile_x < shape.width / ttwv::kTileWidth2D; ++tile_x) {
            std::vector<float> tile(ttwv::kTileHeight2D * ttwv::kTileWidth2D);
            for (size_t row = 0; row < ttwv::kTileHeight2D; ++row) {
                const size_t source = (tile_y * ttwv::kTileHeight2D + row) * shape.width + tile_x * ttwv::kTileWidth2D;
                std::copy_n(
                    row_major.begin() + static_cast<std::ptrdiff_t>(source),
                    ttwv::kTileWidth2D,
                    tile.begin() + static_cast<std::ptrdiff_t>(row * ttwv::kTileWidth2D));
            }
            std::vector<float> tile_nfaces = tilize_nfaces(tile, 32, 32);
            tiled.insert(tiled.end(), tile_nfaces.begin(), tile_nfaces.end());
        }
    }
    return tiled;
}

[[nodiscard]] std::vector<float> untilize_padded(const std::vector<float>& tiled, const ttwv::Shape2D shape) {
    std::vector<float> row_major(shape.height * shape.width);
    size_t tile_index = 0;
    for (size_t tile_y = 0; tile_y < shape.height / ttwv::kTileHeight2D; ++tile_y) {
        for (size_t tile_x = 0; tile_x < shape.width / ttwv::kTileWidth2D; ++tile_x, ++tile_index) {
            const auto begin =
                tiled.begin() + static_cast<std::ptrdiff_t>(tile_index * ttwv::kTileHeight2D * ttwv::kTileWidth2D);
            std::vector<float> tile_nfaces(
                begin, begin + static_cast<std::ptrdiff_t>(ttwv::kTileHeight2D * ttwv::kTileWidth2D));
            const std::vector<float> tile = untilize_nfaces(tile_nfaces, 32, 32);
            for (size_t row = 0; row < ttwv::kTileHeight2D; ++row) {
                const size_t destination =
                    (tile_y * ttwv::kTileHeight2D + row) * shape.width + tile_x * ttwv::kTileWidth2D;
                std::copy_n(
                    tile.begin() + static_cast<std::ptrdiff_t>(row * ttwv::kTileWidth2D),
                    ttwv::kTileWidth2D,
                    row_major.begin() + static_cast<std::ptrdiff_t>(destination));
            }
        }
    }
    return row_major;
}

[[nodiscard]] std::vector<float> crop(
    const std::vector<float>& padded, const ttwv::Shape2D padded_shape, const ttwv::Shape2D logical_shape) {
    std::vector<float> output(logical_shape.height * logical_shape.width);
    for (size_t row = 0; row < logical_shape.height; ++row) {
        std::copy_n(
            padded.begin() + static_cast<std::ptrdiff_t>(row * padded_shape.width),
            logical_shape.width,
            output.begin() + static_cast<std::ptrdiff_t>(row * logical_shape.width));
    }
    return output;
}

[[nodiscard]] DeviceInput create_input(
    tt::tt_metal::distributed::MeshDevice& mesh_device, const ttwv::Shape2D padded_shape) {
    const size_t tiles = padded_shape.height / ttwv::kTileHeight2D * (padded_shape.width / ttwv::kTileWidth2D);
    auto buffer = tt::tt_metal::distributed::MeshBuffer::create(
        tt::tt_metal::distributed::ReplicatedBufferConfig{
            .size = static_cast<uint64_t>(tiles * ttwv::device_protocol::kLwt2DFullTileBytes),
        },
        tt::tt_metal::distributed::DeviceLocalBufferConfig{
            .page_size = ttwv::device_protocol::kLwt2DFullTileBytes,
            .buffer_type = tt::tt_metal::BufferType::DRAM,
        },
        &mesh_device);
    return DeviceInput{.buffer = std::move(buffer)};
}

[[nodiscard]] const char* step_name(const ttwv::StepType type) {
    switch (type) {
        case ttwv::StepType::kPredict: return "predict";
        case ttwv::StepType::kUpdate: return "update";
        case ttwv::StepType::kSwap: return "swap";
        case ttwv::StepType::kScaleEven: return "scale_even";
        case ttwv::StepType::kScaleOdd: return "scale_odd";
        default: return "unknown";
    }
}

void write_split_snapshots(
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    ttwv::Lwt2DExecutable& executable,
    const std::filesystem::path& prefix) {
    if (!executable.buffers.split_snapshots) {
        throw std::runtime_error("Split snapshots were requested but not captured");
    }
    std::vector<float> all_tiles;
    tt::tt_metal::distributed::EnqueueReadMeshBuffer(
        command_queue, all_tiles, executable.buffers.split_snapshots, true);
    if (!prefix.parent_path().empty()) {
        std::filesystem::create_directories(prefix.parent_path());
    }
    const std::filesystem::path manifest_path = std::filesystem::path{prefix.string() + "_manifest.json"};
    std::ofstream manifest(manifest_path);
    if (!manifest.good()) {
        throw std::runtime_error("Failed to create split snapshot manifest");
    }
    constexpr std::array<const char*, 4> plane_names = {"EE", "EO", "OE", "OO"};
    manifest << "{\n  \"format\": \"row-major-fp32-little-endian\",\n"
             << "  \"snapshot_tiles_per_plane\": " << executable.buffers.split_snapshot_tiles_per_plane << ",\n"
             << "  \"snapshots\": [\n";

    bool first = true;
    for (size_t chunk_index = 0; chunk_index < executable.plan.chunks.size(); ++chunk_index) {
        const auto& chunk = executable.plan.chunks[chunk_index];
        const std::array<ttwv::IndexRectangle, 4> rectangles = {
            chunk.initial.ee,
            chunk.initial.eo,
            chunk.initial.oe,
            chunk.initial.oo,
        };
        for (size_t plane = 0; plane < rectangles.size(); ++plane) {
            const auto& rectangle = rectangles[plane];
            const size_t y_origin = (rectangle.y.begin / ttwv::kTileHeight2D) * ttwv::kTileHeight2D;
            const size_t x_origin = (rectangle.x.begin / ttwv::kTileWidth2D) * ttwv::kTileWidth2D;
            const size_t padded_height = ttwv::round_up(rectangle.y.end, ttwv::kTileHeight2D) - y_origin;
            const size_t padded_width = ttwv::round_up(rectangle.x.end, ttwv::kTileWidth2D) - x_origin;
            const size_t tile_count = padded_height / ttwv::kTileHeight2D * (padded_width / ttwv::kTileWidth2D);
            const size_t snapshot_tile =
                (chunk_index * rectangles.size() + plane) * executable.buffers.split_snapshot_tiles_per_plane;
            const auto begin = all_tiles.begin() + static_cast<std::ptrdiff_t>(
                                                       snapshot_tile * ttwv::device_protocol::kLwt2DFullTileElements);
            std::vector<float> plane_tiles(
                begin, begin + static_cast<std::ptrdiff_t>(tile_count * ttwv::device_protocol::kLwt2DFullTileElements));
            const ttwv::Shape2D padded_shape{
                .height = padded_height,
                .width = padded_width,
            };
            const std::vector<float> padded = untilize_padded(plane_tiles, padded_shape);
            std::vector<float> logical(rectangle.height() * rectangle.width());
            const size_t y_offset = rectangle.y.begin - y_origin;
            const size_t x_offset = rectangle.x.begin - x_origin;
            for (size_t row = 0; row < rectangle.height(); ++row) {
                std::copy_n(
                    padded.begin() + static_cast<std::ptrdiff_t>((y_offset + row) * padded_width + x_offset),
                    rectangle.width(),
                    logical.begin() + static_cast<std::ptrdiff_t>(row * rectangle.width()));
            }

            std::ostringstream suffix;
            suffix << "_chunk" << std::setfill('0') << std::setw(4) << chunk_index << '_' << plane_names[plane]
                   << ".f32";
            const std::filesystem::path data_path = std::filesystem::path{prefix.string() + suffix.str()};
            std::ofstream data(data_path, std::ios::binary);
            data.write(
                reinterpret_cast<const char*>(logical.data()),
                static_cast<std::streamsize>(logical.size() * sizeof(float)));
            if (!data.good()) {
                throw std::runtime_error("Failed to write split snapshot: " + data_path.string());
            }

            if (!first) {
                manifest << ",\n";
            }
            first = false;
            manifest << "    {\"chunk\": " << chunk_index << ", \"plane\": \"" << plane_names[plane]
                     << "\", \"parity_y\": " << (plane / 2) << ", \"parity_x\": " << (plane % 2)
                     << ", \"y_begin\": " << rectangle.y.begin << ", \"height\": " << rectangle.height()
                     << ", \"x_begin\": " << rectangle.x.begin << ", \"width\": " << rectangle.width()
                     << ", \"file\": \"" << data_path.filename().string() << "\"}";
        }
    }
    manifest << "\n  ]\n}\n";
}

void write_route_snapshots(
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    ttwv::Lwt2DExecutable& executable,
    const std::filesystem::path& prefix) {
    if (!executable.buffers.route_snapshots) {
        throw std::runtime_error("Route snapshots were requested but not captured");
    }
    std::vector<float> all_tiles;
    tt::tt_metal::distributed::EnqueueReadMeshBuffer(
        command_queue, all_tiles, executable.buffers.route_snapshots, true);
    if (!prefix.parent_path().empty()) {
        std::filesystem::create_directories(prefix.parent_path());
    }
    const std::filesystem::path manifest_path = std::filesystem::path{prefix.string() + "_manifest.json"};
    std::ofstream manifest(manifest_path);
    if (!manifest.good()) {
        throw std::runtime_error("Failed to create route snapshot manifest");
    }
    manifest << "{\n  \"format\": \"row-major-fp32-little-endian\",\n"
             << "  \"snapshot_tiles_per_route\": " << executable.buffers.snapshot_tiles_per_route << ",\n"
             << "  \"snapshots\": [\n";

    const size_t route_count = executable.plan.chunks.front().routes.size();
    bool first = true;
    for (size_t chunk_index = 0; chunk_index < executable.plan.chunks.size(); ++chunk_index) {
        const auto& chunk = executable.plan.chunks[chunk_index];
        for (size_t route_index = 0; route_index < route_count; ++route_index) {
            const auto& route = chunk.routes[route_index];
            if (route.output.empty()) {
                continue;
            }
            const size_t y_origin = (route.output.y.begin / ttwv::kTileHeight2D) * ttwv::kTileHeight2D;
            const size_t x_origin = (route.output.x.begin / ttwv::kTileWidth2D) * ttwv::kTileWidth2D;
            const size_t padded_height = ttwv::round_up(route.output.y.end, ttwv::kTileHeight2D) - y_origin;
            const size_t padded_width = ttwv::round_up(route.output.x.end, ttwv::kTileWidth2D) - x_origin;
            const size_t tile_count = padded_height / ttwv::kTileHeight2D * (padded_width / ttwv::kTileWidth2D);
            const size_t snapshot_tile =
                (chunk_index * route_count + route_index) * executable.buffers.snapshot_tiles_per_route;
            const auto begin = all_tiles.begin() + static_cast<std::ptrdiff_t>(
                                                       snapshot_tile * ttwv::device_protocol::kLwt2DFullTileElements);
            std::vector<float> route_tiles(
                begin, begin + static_cast<std::ptrdiff_t>(tile_count * ttwv::device_protocol::kLwt2DFullTileElements));
            const ttwv::Shape2D padded_shape{
                .height = padded_height,
                .width = padded_width,
            };
            const std::vector<float> padded = untilize_padded(route_tiles, padded_shape);
            std::vector<float> logical(route.output.height() * route.output.width());
            const size_t y_offset = route.output.y.begin - y_origin;
            const size_t x_offset = route.output.x.begin - x_origin;
            for (size_t row = 0; row < route.output.height(); ++row) {
                std::copy_n(
                    padded.begin() + static_cast<std::ptrdiff_t>((y_offset + row) * padded_width + x_offset),
                    route.output.width(),
                    logical.begin() + static_cast<std::ptrdiff_t>(row * route.output.width()));
            }

            std::ostringstream suffix;
            suffix << "_chunk" << std::setfill('0') << std::setw(4) << chunk_index << "_route" << std::setw(3)
                   << route_index << '_' << (route.axis == ttwv::Lwt2DAxis::kVertical ? 'y' : 'x') << '_'
                   << step_name(route.type) << ".f32";
            const std::filesystem::path data_path = std::filesystem::path{prefix.string() + suffix.str()};
            std::ofstream data(data_path, std::ios::binary);
            data.write(
                reinterpret_cast<const char*>(logical.data()),
                static_cast<std::streamsize>(logical.size() * sizeof(float)));
            if (!data.good()) {
                throw std::runtime_error("Failed to write route snapshot: " + data_path.string());
            }

            if (!first) {
                manifest << ",\n";
            }
            first = false;
            manifest << "    {\"chunk\": " << chunk_index << ", \"route\": " << route_index << ", \"axis\": \""
                     << (route.axis == ttwv::Lwt2DAxis::kVertical ? "y" : "x") << "\", \"step\": \""
                     << step_name(route.type) << "\", \"output_slot\": " << static_cast<uint32_t>(route.output_slot)
                     << ", \"source_slot\": " << static_cast<uint32_t>(route.source_slot)
                     << ", \"source_y_begin\": " << route.source.y.begin
                     << ", \"source_height\": " << route.source.height()
                     << ", \"source_x_begin\": " << route.source.x.begin
                     << ", \"source_width\": " << route.source.width()
                     << ", \"base_slot\": " << static_cast<uint32_t>(route.base_slot)
                     << ", \"base_y_begin\": " << route.base.y.begin << ", \"base_height\": " << route.base.height()
                     << ", \"base_x_begin\": " << route.base.x.begin << ", \"base_width\": " << route.base.width()
                     << ", \"y_begin\": " << route.output.y.begin << ", \"height\": " << route.output.height()
                     << ", \"x_begin\": " << route.output.x.begin << ", \"width\": " << route.output.width()
                     << ", \"file\": \"" << data_path.filename().string() << "\"}";
        }
    }
    manifest << "\n  ]\n}\n";
}

template <typename Scheme>
[[nodiscard]] DeviceOutput run_once(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    const tt::tt_metal::Buffer& input_buffer,
    const size_t height,
    const size_t width,
    const uint32_t core_limit,
    const bool read_outputs,
    const std::optional<std::filesystem::path>& split_snapshot_prefix,
    const std::optional<std::filesystem::path>& snapshot_prefix,
    const ttwv::Lwt2DSplitImplementation split_implementation,
    const ttwv::Lwt2DTransportPolicy transport_policy,
    const ttwv::BoundaryMode boundary_mode) {
    ttwv::Lwt2DTransportPolicy effective_transport_policy = transport_policy;
    effective_transport_policy.split_only_benchmark =
        split_snapshot_prefix.has_value() && !snapshot_prefix.has_value() && !read_outputs;
    ttwv::Lwt2DExecutable executable = ttwv::create_lwt_2d_executable<Scheme>(
        TT_WAVELET_SOURCE_DIR,
        mesh_device,
        input_buffer,
        height,
        width,
        core_limit,
        snapshot_prefix.has_value(),
        split_implementation,
        false,
        split_snapshot_prefix.has_value(),
        effective_transport_policy,
        false,
        boundary_mode);
    ttwv::prepare_lwt_2d(command_queue, executable);
    const auto start = std::chrono::steady_clock::now();
    ttwv::execute_lwt_2d(mesh_device, command_queue, executable);
    const auto stop = std::chrono::steady_clock::now();
    DeviceOutput output{
        .execution_time_ms = std::chrono::duration<double, std::milli>(stop - start).count(),
        .scheduler = executable.buffers.scheduler,
    };
    if (snapshot_prefix) {
        write_route_snapshots(command_queue, executable, *snapshot_prefix);
    }
    if (split_snapshot_prefix) {
        write_split_snapshots(command_queue, executable, *split_snapshot_prefix);
    }
    if (!read_outputs) {
        return output;
    }

    std::array<std::vector<float>, 4> tiled_bands;
    for (size_t band = 0; band < tiled_bands.size(); ++band) {
        tt::tt_metal::distributed::EnqueueReadMeshBuffer(
            command_queue, tiled_bands[band], executable.buffers.outputs[band], true);
    }
    const ttwv::Shape2D padded_band = executable.plan.tiling.band.storage;
    const ttwv::Shape2D logical_band = executable.plan.tiling.band.logical;
    std::array<std::vector<float>, 4> bands;
    for (size_t band = 0; band < bands.size(); ++band) {
        bands[band] = crop(untilize_padded(tiled_bands[band], padded_band), padded_band, logical_band);
    }
    output.values = ttwv::Lwt2DReferenceOutput{
        .band_height = logical_band.height,
        .band_width = logical_band.width,
        .ll = std::move(bands[0]),
        .lh = std::move(bands[1]),
        .hl = std::move(bands[2]),
        .hh = std::move(bands[3]),
    };
    return output;
}

void print_band(const char* name, const std::vector<float>& values, const size_t height, const size_t width) {
    std::cout << "tt-wavelet device " << name << " (" << height << "x" << width << "): [";
    for (size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            std::cout << ", ";
        }
        std::cout << std::scientific << std::setprecision(8) << values[index];
    }
    std::cout << std::defaultfloat << "]\n";
}

void write_output_bands(const std::filesystem::path& prefix, const ttwv::Lwt2DReferenceOutput& output) {
    if (!prefix.parent_path().empty()) {
        std::filesystem::create_directories(prefix.parent_path());
    }
    const std::array<std::pair<const char*, const std::vector<float>*>, 4> bands = {{
        {"LL", &output.ll},
        {"LH", &output.lh},
        {"HL", &output.hl},
        {"HH", &output.hh},
    }};
    for (const auto& [name, values] : bands) {
        const std::filesystem::path path = std::filesystem::path{prefix.string() + "_" + name + ".f32"};
        std::ofstream stream(path, std::ios::binary);
        stream.write(
            reinterpret_cast<const char*>(values->data()),
            static_cast<std::streamsize>(values->size() * sizeof(float)));
        if (!stream.good()) {
            throw std::runtime_error("Failed to write device band: " + path.string());
        }
    }
    const std::filesystem::path shape_path = std::filesystem::path{prefix.string() + "_shape.txt"};
    std::ofstream shape(shape_path);
    shape << output.band_height << ' ' << output.band_width << '\n';
    if (!shape.good()) {
        throw std::runtime_error("Failed to write device output shape");
    }
}

void print_telemetry(const ttwv::Lwt2DSchedulerTelemetry& telemetry) {
    const auto ratio = [](const uint64_t internal, const uint64_t exact) {
        return exact == 0 ? 1.0 : static_cast<double>(internal) / static_cast<double>(exact);
    };
    std::cerr << "lwt_2d_boundary_mode: " << ttwv::boundary_mode_name(telemetry.boundary_mode) << '\n'
              << "lwt_2d_active_core_count: " << telemetry.active_core_count << '\n'
              << "lwt_2d_chunk_count: " << telemetry.chunk_count << '\n'
              << "lwt_2d_chunk_tiles: " << telemetry.chunk_tiles_y << 'x' << telemetry.chunk_tiles_x << '\n'
              << "lwt_2d_route_count: " << telemetry.route_count << '\n'
              << "lwt_2d_executable_route_count: " << telemetry.executable_route_count << '\n'
              << "lwt_2d_scale_routes_removed: " << telemetry.scale_routes_removed << '\n'
              << "lwt_2d_planner: " << (telemetry.latency_oriented_planner ? "latency" : "max-cores") << '\n'
              << "lwt_2d_route_domain: "
              << (telemetry.route_domain == ttwv::Lwt2DRouteDomainPolicy::kTileClosed ? "tile-closed" : "exact")
              << '\n'
              << "lwt_2d_estimated_latency_cycles: " << telemetry.estimated_latency_cycles << '\n'
              << "lwt_2d_l1_total_bytes: " << telemetry.l1_total_bytes << '\n'
              << "lwt_2d_l1_headroom_bytes: " << telemetry.l1_headroom_bytes << '\n'
              << "lwt_2d_exact_initial_elements: " << telemetry.exact_initial_elements << '\n'
              << "lwt_2d_internal_initial_elements: " << telemetry.internal_initial_elements << '\n'
              << "lwt_2d_initial_overcompute_ratio: "
              << ratio(telemetry.internal_initial_elements, telemetry.exact_initial_elements) << '\n'
              << "lwt_2d_exact_route_elements: " << telemetry.exact_route_elements << '\n'
              << "lwt_2d_internal_route_elements: " << telemetry.internal_route_elements << '\n'
              << "lwt_2d_route_overcompute_ratio: "
              << ratio(telemetry.internal_route_elements, telemetry.exact_route_elements) << '\n'
              << "lwt_2d_exact_final_elements: " << telemetry.exact_final_elements << '\n'
              << "lwt_2d_internal_final_elements: " << telemetry.internal_final_elements << '\n'
              << "lwt_2d_final_overcompute_ratio: "
              << ratio(telemetry.internal_final_elements, telemetry.exact_final_elements) << '\n';
}

void print_split_metrics(
    const ttwv::Lwt2DSplitMetricsSummary& metrics, const size_t input_elements, const uint32_t active_core_count) {
    // The attached N150 runs Tensix at 1000 MHz.  Keep the cycle count as the
    // architecture-independent primary measurement and report the direct
    // 1 GHz conversion used by this benchmark environment.
    const double milliseconds_at_1ghz = static_cast<double>(metrics.max_core_cycles) / 1.0e6;
    const double elements_per_second =
        milliseconds_at_1ghz > 0.0 ? static_cast<double>(input_elements) * 1000.0 / milliseconds_at_1ghz : 0.0;
    static_cast<void>(active_core_count);
    std::cerr << std::fixed << std::setprecision(6) << "lwt_2d_split_max_core_cycles: " << metrics.max_core_cycles
              << '\n'
              << "lwt_2d_split_time_ms_at_1ghz: " << milliseconds_at_1ghz << '\n'
              << "lwt_2d_split_input_elements_per_second_at_1ghz: " << elements_per_second << '\n'
              << "lwt_2d_split_raw_input_bytes: " << metrics.raw_input_bytes << '\n'
              << "lwt_2d_split_local_output_bytes: " << metrics.local_output_bytes << '\n'
              << "lwt_2d_split_noc_read_calls: " << metrics.noc_read_calls << '\n'
              << "lwt_2d_split_noc_read_barriers: " << metrics.noc_read_barriers << '\n'
              << "lwt_2d_split_interior_macro_tiles: " << metrics.interior_macro_tiles << '\n'
              << "lwt_2d_split_boundary_macro_tiles: " << metrics.boundary_macro_tiles << '\n'
              << "lwt_2d_split_max_macro_tiles_per_core: " << metrics.max_core_macro_tiles << '\n';
}

void print_transport_metrics(const ttwv::Lwt2DTransportMetricsSummary& metrics) {
    std::cerr << "lwt_2d_transport_route_records: " << metrics.routes.size() << '\n'
              << "lwt_2d_transport_total_route_tiles: " << metrics.total_route_tiles << '\n'
              << "lwt_2d_transport_exact_source_tiles: " << metrics.exact_source_tiles << '\n'
              << "lwt_2d_transport_shifted_source_tiles: " << metrics.shifted_source_tiles << '\n'
              << "lwt_2d_transport_generic_source_tiles: " << metrics.generic_source_tiles << '\n'
              << "lwt_2d_transport_exact_base_tiles: " << metrics.exact_base_tiles << '\n'
              << "lwt_2d_transport_shifted_base_tiles: " << metrics.shifted_base_tiles << '\n'
              << "lwt_2d_transport_generic_base_tiles: " << metrics.generic_base_tiles << '\n'
              << "lwt_2d_transport_exact_terminal_tiles: " << metrics.exact_terminal_tiles << '\n'
              << "lwt_2d_transport_fragmented_terminal_tiles: " << metrics.fragmented_terminal_tiles << '\n'
              << "lwt_2d_transport_validated_staging_tiles: " << metrics.validated_staging_tiles << '\n'
              << "lwt_2d_transport_staging_validation_mismatches: " << metrics.staging_validation_mismatches << '\n'
              << "lwt_2d_transport_validation_exact_mismatches: " << metrics.validation_exact_mismatches << '\n'
              << "lwt_2d_transport_validation_shifted_mismatches: " << metrics.validation_shifted_mismatches << '\n'
              << "lwt_2d_transport_validation_two_axis_mismatches: " << metrics.validation_two_axis_mismatches << '\n'
              << "lwt_2d_transport_validation_partial_mismatches: " << metrics.validation_partial_mismatches << '\n'
              << "lwt_2d_transport_validation_empty_mismatches: " << metrics.validation_empty_mismatches << '\n'
              << "lwt_2d_transport_validated_persistence_tiles: " << metrics.validated_persistence_tiles << '\n'
              << "lwt_2d_transport_persistence_validation_mismatches: " << metrics.persistence_validation_mismatches
              << '\n'
              << "lwt_2d_transport_max_route_staging_cycles: " << metrics.max_route_staging_cycles << '\n'
              << "lwt_2d_transport_max_route_compute_cycles: " << metrics.max_route_compute_cycles << '\n'
              << "lwt_2d_transport_max_route_persistence_cycles: " << metrics.max_route_persistence_cycles << '\n'
              << "lwt_2d_transport_max_route_sync_wait_cycles: " << metrics.max_route_synchronization_wait_cycles
              << '\n'
              << "lwt_2d_transport_max_terminal_write_cycles: " << metrics.max_terminal_write_cycles << '\n'
              << "lwt_2d_transport_max_reader_kernel_cycles: " << metrics.max_reader_kernel_cycles << '\n'
              << "lwt_2d_transport_max_writer_kernel_cycles: " << metrics.max_writer_kernel_cycles << '\n'
              << "lwt_2d_transport_max_core_cycles: " << metrics.max_core_cycles << '\n'
              << "lwt_2d_transport_mean_active_core_cycles: " << metrics.mean_active_core_cycles << '\n'
              << "lwt_2d_transport_max_route_tiles_per_core: " << metrics.max_core_route_tiles << '\n'
              << "lwt_2d_transport_max_core_config_cycles: " << metrics.max_core_config_cycles << '\n'
              << "lwt_2d_transport_max_core_staging_cycles: " << metrics.max_core_staging_cycles << '\n'
              << "lwt_2d_transport_max_core_compute_pipeline_cycles: " << metrics.max_core_compute_pipeline_cycles
              << '\n'
              << "lwt_2d_transport_max_core_persistence_cycles: " << metrics.max_core_persistence_cycles << '\n'
              << "lwt_2d_transport_max_core_sync_wait_cycles: " << metrics.max_core_synchronization_wait_cycles << '\n'
              << "lwt_2d_transport_max_core_terminal_write_cycles: " << metrics.max_core_terminal_write_cycles << '\n'
              << "lwt_2d_transport_mean_active_core_config_cycles: " << metrics.mean_active_core_config_cycles << '\n'
              << "lwt_2d_transport_mean_active_core_staging_cycles: " << metrics.mean_active_core_staging_cycles << '\n'
              << "lwt_2d_transport_mean_active_core_compute_pipeline_cycles: "
              << metrics.mean_active_core_compute_pipeline_cycles << '\n'
              << "lwt_2d_transport_mean_active_core_persistence_cycles: " << metrics.mean_active_core_persistence_cycles
              << '\n'
              << "lwt_2d_transport_mean_active_core_sync_wait_cycles: "
              << metrics.mean_active_core_synchronization_wait_cycles << '\n'
              << "lwt_2d_transport_mean_active_core_terminal_write_cycles: "
              << metrics.mean_active_core_terminal_write_cycles << '\n';
}

void write_transport_metrics_csv(const std::filesystem::path& path, const ttwv::Lwt2DTransportMetricsSummary& metrics) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path);
    if (!output.good()) {
        throw std::runtime_error("Failed to create transport-metrics CSV: " + path.string());
    }
    output << "chunk,route,axis,step,coefficient_count,output_tiles,reader_config_cycles,staging_cycles,"
              "compute_pipeline_cycles,persistence_cycles,synchronization_wait_cycles,writer_config_cycles,"
              "exact_source_tiles,shifted_source_tiles,generic_source_tiles,exact_base_tiles,"
              "shifted_base_tiles,generic_base_tiles,persistence_tiles\n";
    for (const ttwv::Lwt2DRouteTransportMetric& metric : metrics.routes) {
        output << metric.chunk_index << ',' << metric.route_index << ',' << metric.axis << ',' << metric.step_type
               << ',' << metric.coefficient_count << ',' << metric.output_tiles << ',' << metric.reader_config_cycles
               << ',' << metric.staging_cycles << ',' << metric.compute_cycles << ',' << metric.persistence_cycles
               << ',' << metric.synchronization_wait_cycles << ',' << metric.writer_config_cycles << ','
               << metric.exact_source_tiles << ',' << metric.shifted_source_tiles << ',' << metric.generic_source_tiles
               << ',' << metric.exact_base_tiles << ',' << metric.shifted_base_tiles << ',' << metric.generic_base_tiles
               << ',' << metric.persistence_tiles << '\n';
    }
}

int run_empty_prepared_benchmark(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    const size_t repeats,
    const size_t warmup_runs) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    const tt::tt_metal::CoreCoord core{0, 0};
    static_cast<void>(tt::tt_metal::CreateKernel(
        program,
        std::filesystem::path{TT_WAVELET_SOURCE_DIR} / "kernels/dataflow/lwt_2d_empty.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig{}));
    tt::tt_metal::distributed::MeshWorkload workload;
    workload.add_program(tt::tt_metal::distributed::MeshCoordinateRange(mesh_device.shape()), std::move(program));
    const auto execute = [&]() {
        const auto start = std::chrono::steady_clock::now();
        tt::tt_metal::distributed::EnqueueMeshWorkload(command_queue, workload, false);
        tt::tt_metal::distributed::Finish(command_queue);
        const auto stop = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(stop - start).count();
    };
    for (size_t warmup = 0; warmup < warmup_runs; ++warmup) {
        static_cast<void>(execute());
    }
    std::vector<double> times;
    times.reserve(repeats);
    for (size_t repeat = 0; repeat < repeats; ++repeat) {
        times.push_back(execute());
    }
    const double mean = std::accumulate(times.begin(), times.end(), 0.0) / static_cast<double>(times.size());
    std::cerr << std::fixed << std::setprecision(6) << "lwt_2d_microbenchmark_mode: empty\n"
              << "lwt_2d_execution_time_ms: " << mean << '\n'
              << "lwt_2d_min_execution_time_ms: " << *std::min_element(times.begin(), times.end()) << '\n';
    return EXIT_SUCCESS;
}

[[nodiscard]] int64_t modulo_tile(const int64_t value) {
    const int64_t remainder = value % static_cast<int64_t>(ttwv::kTileHeight2D);
    return remainder < 0 ? remainder + static_cast<int64_t>(ttwv::kTileHeight2D) : remainder;
}

void write_alignment_csv(
    const std::filesystem::path& path, const std::string& wavelet, const ttwv::Lwt2DExecutionPlan& plan) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path);
    if (!output.good()) {
        throw std::runtime_error("Failed to create route-alignment CSV: " + path.string());
    }
    output << "wavelet,height,width,chunk,route,axis,step,coefficient_count,role,"
              "requested_y,requested_x,stored_y_begin,stored_y_end,stored_x_begin,stored_x_end,"
              "route_output_y_begin,route_output_y_end,route_output_x_begin,route_output_x_end,"
              "requested_y_mod32,requested_x_mod32,stored_y_begin_mod32,stored_x_begin_mod32,"
              "translation_y_mod32,translation_x_mod32,full_logical_coverage,zero_fill_required,"
              "valid_height,valid_width,valid_area,valid_fraction,physical_plane_tiles_intersected,classification\n";

    constexpr int64_t tile_side = static_cast<int64_t>(ttwv::kTileHeight2D);
    const auto rectangle_contains = [](const ttwv::IndexRectangle& rectangle, const int64_t y, const int64_t x) {
        return y >= static_cast<int64_t>(rectangle.y.begin) && x >= static_cast<int64_t>(rectangle.x.begin) &&
               y + tile_side <= static_cast<int64_t>(rectangle.y.end) &&
               x + tile_side <= static_cast<int64_t>(rectangle.x.end);
    };
    const auto rectangle_intersects = [](const ttwv::IndexRectangle& rectangle, const int64_t y, const int64_t x) {
        return y < static_cast<int64_t>(rectangle.y.end) && y + tile_side > static_cast<int64_t>(rectangle.y.begin) &&
               x < static_cast<int64_t>(rectangle.x.end) && x + tile_side > static_cast<int64_t>(rectangle.x.begin);
    };

    for (size_t chunk_index = 0; chunk_index < plan.chunks.size(); ++chunk_index) {
        const ttwv::Lwt2DChunkPlan& chunk = plan.chunks[chunk_index];
        std::array<ttwv::IndexRectangle, ttwv::device_protocol::kLwt2DPlaneCount> stored = {
            chunk.initial.ee,
            chunk.initial.eo,
            chunk.initial.oe,
            chunk.initial.oo,
            ttwv::IndexRectangle{},
        };
        for (size_t route_index = 0; route_index < chunk.routes.size(); ++route_index) {
            const ttwv::Lwt2DRoutePlan& route = chunk.routes[route_index];
            const ttwv::LiftingStepRoute& axis_route = route.axis == ttwv::Lwt2DAxis::kVertical
                                                           ? plan.y_plan.routes[route.axis_route_index]
                                                           : plan.x_plan.routes[route.axis_route_index];
            const uint32_t coefficient_count = ttwv::is_predict_update_step(route.type)
                                                   ? ttwv::execution_detail::coefficient_count(axis_route)
                                                   : (ttwv::is_scale_step(route.type) ? 1U : 0U);
            if (route.output.empty()) {
                continue;
            }
            const int64_t output_y_origin =
                static_cast<int64_t>((route.output.y.begin / ttwv::kTileHeight2D) * ttwv::kTileHeight2D);
            const int64_t output_x_origin =
                static_cast<int64_t>((route.output.x.begin / ttwv::kTileWidth2D) * ttwv::kTileWidth2D);
            const size_t tile_rows = ttwv::round_up(route.output.y.end, ttwv::kTileHeight2D) / ttwv::kTileHeight2D -
                                     route.output.y.begin / ttwv::kTileHeight2D;
            const size_t tile_columns = ttwv::round_up(route.output.x.end, ttwv::kTileWidth2D) / ttwv::kTileWidth2D -
                                        route.output.x.begin / ttwv::kTileWidth2D;

            const auto write_record = [&](const char* role,
                                          const ttwv::IndexRectangle& stored_rectangle,
                                          const int64_t requested_y,
                                          const int64_t requested_x,
                                          const int64_t output_tile_y,
                                          const int64_t output_tile_x) {
                const bool full = rectangle_contains(stored_rectangle, requested_y, requested_x);
                const bool intersects = rectangle_intersects(stored_rectangle, requested_y, requested_x);
                const int64_t valid_y_begin = std::max(requested_y, static_cast<int64_t>(stored_rectangle.y.begin));
                const int64_t valid_y_end =
                    std::min(requested_y + tile_side, static_cast<int64_t>(stored_rectangle.y.end));
                const int64_t valid_x_begin = std::max(requested_x, static_cast<int64_t>(stored_rectangle.x.begin));
                const int64_t valid_x_end =
                    std::min(requested_x + tile_side, static_cast<int64_t>(stored_rectangle.x.end));
                const uint32_t valid_height =
                    intersects ? static_cast<uint32_t>(std::max<int64_t>(0, valid_y_end - valid_y_begin)) : 0U;
                const uint32_t valid_width =
                    intersects ? static_cast<uint32_t>(std::max<int64_t>(0, valid_x_end - valid_x_begin)) : 0U;
                const uint32_t valid_area = valid_height * valid_width;
                const bool y_aligned = modulo_tile(requested_y) == 0;
                const bool x_aligned = modulo_tile(requested_x) == 0;
                const char* classification = nullptr;
                if (!full) {
                    classification = intersects ? "partial_logical_edge" : "out_of_stored_range_zero_fill";
                } else if (y_aligned && x_aligned) {
                    classification = "exact_full_tile";
                } else if (y_aligned != x_aligned) {
                    classification = "one_axis_shifted";
                } else {
                    classification = "two_axis_shifted";
                }
                const uint32_t physical_tiles = static_cast<uint32_t>((y_aligned ? 1 : 2) * (x_aligned ? 1 : 2));
                output << wavelet << ',' << plan.input_height << ',' << plan.input_width << ',' << chunk_index << ','
                       << route_index << ',' << (route.axis == ttwv::Lwt2DAxis::kVertical ? "vertical" : "horizontal")
                       << ',' << step_name(route.type) << ',' << coefficient_count << ',' << role << ','
                       << requested_y << ',' << requested_x << ',' << stored_rectangle.y.begin << ','
                       << stored_rectangle.y.end << ',' << stored_rectangle.x.begin << ','
                       << stored_rectangle.x.end << ',' << route.output.y.begin << ',' << route.output.y.end << ','
                       << route.output.x.begin << ',' << route.output.x.end << ','
                       << modulo_tile(requested_y) << ',' << modulo_tile(requested_x) << ','
                       << stored_rectangle.y.begin % ttwv::kTileHeight2D << ','
                       << stored_rectangle.x.begin % ttwv::kTileWidth2D << ','
                       << modulo_tile(requested_y - output_tile_y) << ',' << modulo_tile(requested_x - output_tile_x)
                       << ',' << (full ? 1 : 0) << ',' << (full ? 0 : 1) << ',' << valid_height << ','
                       << valid_width << ',' << valid_area << ','
                       << static_cast<double>(valid_area) / static_cast<double>(ttwv::kTileHeight2D * ttwv::kTileWidth2D)
                       << ',' << physical_tiles << ',' << classification << '\n';
            };

            for (size_t tile_y = 0; tile_y < tile_rows; ++tile_y) {
                for (size_t tile_x = 0; tile_x < tile_columns; ++tile_x) {
                    const int64_t output_tile_y = output_y_origin + static_cast<int64_t>(tile_y) * tile_side;
                    const int64_t output_tile_x = output_x_origin + static_cast<int64_t>(tile_x) * tile_side;
                    const auto requested_origin = [&](const ttwv::IndexRectangle& source) {
                        return std::pair{
                            static_cast<int64_t>(source.y.begin) + output_tile_y -
                                static_cast<int64_t>(route.output.y.begin),
                            static_cast<int64_t>(source.x.begin) + output_tile_x -
                                static_cast<int64_t>(route.output.x.begin),
                        };
                    };
                    if (ttwv::is_predict_update_step(route.type)) {
                        const auto [base_y, base_x] = requested_origin(route.base);
                        write_record(
                            "base",
                            stored[static_cast<size_t>(route.base_slot)],
                            base_y,
                            base_x,
                            output_tile_y,
                            output_tile_x);
                        const auto [source_y, source_x] = requested_origin(route.source);
                        for (uint32_t source_tile = 0; source_tile < 2; ++source_tile) {
                            const int64_t requested_y = source_y + (route.axis == ttwv::Lwt2DAxis::kVertical
                                                                        ? static_cast<int64_t>(source_tile) * tile_side
                                                                        : 0);
                            const int64_t requested_x =
                                source_x + (route.axis == ttwv::Lwt2DAxis::kHorizontal
                                                ? static_cast<int64_t>(source_tile) * tile_side -
                                                      static_cast<int64_t>(17 - coefficient_count)
                                                : 0);
                            write_record(
                                source_tile == 0 ? "source0" : "source1",
                                stored[static_cast<size_t>(route.source_slot)],
                                requested_y,
                                requested_x,
                                output_tile_y,
                                output_tile_x);
                        }
                    } else {
                        const auto [source_y, source_x] = requested_origin(route.source);
                        write_record(
                            "scale_source",
                            stored[static_cast<size_t>(route.source_slot)],
                            source_y,
                            source_x,
                            output_tile_y,
                            output_tile_x);
                    }
                    write_record("output", route.output, output_tile_y, output_tile_x, output_tile_y, output_tile_x);
                }
            }
            stored[static_cast<size_t>(route.output_slot)] = route.output;
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const std::vector<float> logical_input =
            read_input(options.input_path, options.height, options.width, options.binary_input);
        const ttwv::TiledShape2D input_shape = ttwv::make_tiled_shape_2d(ttwv::Shape2D{
            .height = options.height,
            .width = options.width,
        });
        const std::vector<float> padded = ttwv::zero_pad_row_major_to_tiles_2d(logical_input, input_shape.logical);
        if (!ttwv::has_zero_tile_padding_2d(padded, input_shape)) {
            throw std::runtime_error("2D input preprocessing violated zero-padding contract");
        }
        const std::vector<float> tiled = tilize_padded(padded, input_shape.storage);

        auto mesh_device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);
        if (options.benchmark) {
            mesh_device->enable_program_cache();
        }
        auto& command_queue = mesh_device->mesh_command_queue();
        DeviceInput input = create_input(*mesh_device, input_shape.storage);
        tt::tt_metal::distributed::EnqueueWriteMeshBuffer(command_queue, input.buffer, tiled, false);
        tt::tt_metal::distributed::Finish(command_queue);
        if (options.microbenchmark_mode == "empty") {
            return run_empty_prepared_benchmark(*mesh_device, command_queue, options.repeats, options.warmup_runs);
        }

        const auto run = [&]<typename Scheme>() {
            if (options.benchmark) {
                ttwv::Lwt2DExecutable executable = ttwv::create_lwt_2d_executable<Scheme>(
                    TT_WAVELET_SOURCE_DIR,
                    *mesh_device,
                    *input.buffer->get_backing_buffer(),
                    options.height,
                    options.width,
                    options.core_limit,
                    false,
                    options.split_implementation,
                    options.split_metrics,
                    false,
                    options.transport_policy,
                    options.transport_metrics,
                    options.boundary_mode);
                if (options.alignment_csv) {
                    write_alignment_csv(*options.alignment_csv, options.wavelet, executable.plan);
                }
                ttwv::prepare_lwt_2d(command_queue, executable);
                const auto execute_prepared = [&]() {
                    const auto start = std::chrono::steady_clock::now();
                    ttwv::execute_lwt_2d(*mesh_device, command_queue, executable);
                    const auto stop = std::chrono::steady_clock::now();
                    return std::chrono::duration<double, std::milli>(stop - start).count();
                };
                for (size_t warmup = 0; warmup < options.warmup_runs; ++warmup) {
                    static_cast<void>(execute_prepared());
                }
                std::vector<double> times;
                times.reserve(options.repeats);
                for (size_t repeat = 0; repeat < options.repeats; ++repeat) {
                    times.push_back(execute_prepared());
                }
                const double total = std::accumulate(times.begin(), times.end(), 0.0);
                std::vector<double> sorted_times = times;
                std::sort(sorted_times.begin(), sorted_times.end());
                const auto percentile = [&sorted_times](const double probability) {
                    const double position = probability * static_cast<double>(sorted_times.size() - 1);
                    const size_t lower = static_cast<size_t>(position);
                    const size_t upper = std::min(lower + 1, sorted_times.size() - 1);
                    const double fraction = position - static_cast<double>(lower);
                    return sorted_times[lower] + fraction * (sorted_times[upper] - sorted_times[lower]);
                };
                const double mean = total / static_cast<double>(times.size());
                const double squared_error =
                    std::accumulate(times.begin(), times.end(), 0.0, [mean](const double sum, const double value) {
                        const double difference = value - mean;
                        return sum + difference * difference;
                    });
                std::cerr << std::fixed << std::setprecision(6)
                          << "lwt_2d_microbenchmark_mode: " << options.microbenchmark_mode << '\n'
                          << "lwt_2d_execution_time_ms: " << mean << '\n'
                          << "lwt_2d_min_execution_time_ms: " << sorted_times.front() << '\n'
                          << "lwt_2d_median_time_ms: " << percentile(0.5) << '\n'
                          << "lwt_2d_p10_time_ms: " << percentile(0.1) << '\n'
                          << "lwt_2d_p90_time_ms: " << percentile(0.9) << '\n'
                          << "lwt_2d_stddev_time_ms: " << std::sqrt(squared_error / static_cast<double>(times.size()))
                          << '\n';
                if (options.include_transfers) {
                    std::array<std::vector<float>, ttwv::device_protocol::kLwt2DBandCount> transfer_outputs;
                    const auto execute_end_to_end = [&]() {
                        const auto start = std::chrono::steady_clock::now();
                        tt::tt_metal::distributed::EnqueueWriteMeshBuffer(
                            command_queue, input.buffer, tiled, false);
                        ttwv::execute_lwt_2d(*mesh_device, command_queue, executable);
                        for (size_t band = 0; band < transfer_outputs.size(); ++band) {
                            tt::tt_metal::distributed::EnqueueReadMeshBuffer(
                                command_queue, transfer_outputs[band], executable.buffers.outputs[band], true);
                        }
                        const auto stop = std::chrono::steady_clock::now();
                        return std::chrono::duration<double, std::milli>(stop - start).count();
                    };
                    for (size_t warmup = 0; warmup < options.warmup_runs; ++warmup) {
                        static_cast<void>(execute_end_to_end());
                    }
                    std::vector<double> end_to_end_times;
                    end_to_end_times.reserve(options.repeats);
                    for (size_t repeat = 0; repeat < options.repeats; ++repeat) {
                        end_to_end_times.push_back(execute_end_to_end());
                    }
                    const double end_to_end_total =
                        std::accumulate(end_to_end_times.begin(), end_to_end_times.end(), 0.0);
                    std::sort(end_to_end_times.begin(), end_to_end_times.end());
                    std::cerr << "lwt_2d_end_to_end_time_ms: "
                              << end_to_end_total / static_cast<double>(end_to_end_times.size()) << '\n'
                              << "lwt_2d_min_end_to_end_time_ms: " << end_to_end_times.front() << '\n';
                }
                print_telemetry(executable.buffers.scheduler);
                if (options.split_metrics) {
                    print_split_metrics(
                        ttwv::read_lwt_2d_split_metrics(command_queue, executable),
                        options.height * options.width,
                        executable.buffers.scheduler.active_core_count);
                }
                if (options.transport_metrics) {
                    const ttwv::Lwt2DTransportMetricsSummary metrics =
                        ttwv::read_lwt_2d_transport_metrics(command_queue, executable);
                    print_transport_metrics(metrics);
                    if (options.transport_policy.validate_route_staging) {
                        if (metrics.staging_validation_mismatches != 0) {
                            throw std::runtime_error(
                                "Direct route-staging validation found " +
                                std::to_string(metrics.staging_validation_mismatches) + " mismatched FP32 words");
                        }
                        if (metrics.persistence_validation_mismatches != 0) {
                            throw std::runtime_error(
                                "Direct route-persistence validation found " +
                                std::to_string(metrics.persistence_validation_mismatches) + " mismatched FP32 words");
                        }
                    }
                    if (options.transport_metrics_csv) {
                        write_transport_metrics_csv(*options.transport_metrics_csv, metrics);
                    }
                }
                return EXIT_SUCCESS;
            }

            const bool read_outputs =
                !options.split_snapshot_prefix.has_value() || options.route_snapshot_prefix.has_value() ||
                options.output_prefix.has_value() || !options.quiet;
            DeviceOutput output = run_once<Scheme>(
                *mesh_device,
                command_queue,
                *input.buffer->get_backing_buffer(),
                options.height,
                options.width,
                options.core_limit,
                read_outputs,
                options.split_snapshot_prefix,
                options.route_snapshot_prefix,
                options.split_implementation,
                options.transport_policy,
                options.boundary_mode);
            if (options.output_prefix) {
                write_output_bands(*options.output_prefix, output.values);
            }
            if (!options.quiet) {
                print_band("LL", output.values.ll, output.values.band_height, output.values.band_width);
                print_band("LH", output.values.lh, output.values.band_height, output.values.band_width);
                print_band("HL", output.values.hl, output.values.band_height, output.values.band_width);
                print_band("HH", output.values.hh, output.values.band_height, output.values.band_width);
            }
            std::cerr << std::fixed << std::setprecision(6) << "lwt_2d_execution_time_ms: " << output.execution_time_ms
                      << '\n';
            print_telemetry(output.scheduler);
            return EXIT_SUCCESS;
        };
        if (options.wavelet == ttwv::schemes::testing::synthetic_k17::name) {
            return run.template operator()<ttwv::schemes::testing::synthetic_k17>();
        }
        return ttwv::dispatch_scheme(options.wavelet, run);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
