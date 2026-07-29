// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt-metalium/tilize_utils.hpp"
#include "tt_wavelet/include/common/boundary_parse.hpp"
#include "tt_wavelet/include/common/tiling_2d.hpp"
#include "tt_wavelet/include/lifting/device_2d.hpp"
#include "tt_wavelet/include/schemes/generated/registry.hpp"
#include "tt_wavelet/include/schemes/testing/synthetic_k17.hpp"

namespace {

struct Options {
    std::string wavelet;
    size_t height{0};
    size_t width{0};
    std::array<std::filesystem::path, 4> bands;
    std::filesystem::path output{"ilwt_2d_output.f32"};
    uint32_t core_limit{1};
    size_t repeats{1};
    size_t warmup_runs{0};
    ttwv::BoundaryMode boundary_mode{ttwv::BoundaryMode::kSymmetric};
};

[[nodiscard]] size_t parse_positive(const std::string& text, const char* label) {
    size_t consumed = 0;
    const unsigned long long value = std::stoull(text, &consumed);
    if (consumed != text.size() || value == 0 || value > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error(std::string{label} + " must be positive");
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
        } else if (argument == "--cores") {
            if (++index >= argc) {
                throw std::runtime_error("--cores requires a value");
            }
            const size_t value = parse_positive(argv[index], "--cores");
            if (value > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("--cores exceeds uint32_t");
            }
            options.core_limit = static_cast<uint32_t>(value);
        } else if (argument == "--output") {
            if (++index >= argc) {
                throw std::runtime_error("--output requires a path");
            }
            options.output = argv[index];
        } else if (argument == "--repeats") {
            if (++index >= argc) {
                throw std::runtime_error("--repeats requires a value");
            }
            options.repeats = parse_positive(argv[index], "--repeats");
        } else if (argument == "--warmup-runs") {
            if (++index >= argc) {
                throw std::runtime_error("--warmup-runs requires a value");
            }
            options.warmup_runs = static_cast<size_t>(std::stoull(argv[index]));
        } else if (argument == "--help" || argument == "-h") {
            std::cout << "Usage: ilwt_2d [--boundary-mode MODE] [--cores N] [--output PATH] "
                         "[--repeats N] [--warmup-runs N] "
                         "WAVELET HEIGHT WIDTH LL.f32 LH.f32 HL.f32 HH.f32\n";
            std::exit(EXIT_SUCCESS);
        } else if (argument.starts_with("--")) {
            throw std::runtime_error("Unknown option: " + argument);
        } else {
            positional.push_back(argument);
        }
    }
    if (positional.size() != 7) {
        throw std::runtime_error(
            "Usage: ilwt_2d [--cores N] [--output PATH] WAVELET HEIGHT WIDTH LL.f32 LH.f32 HL.f32 HH.f32");
    }
    options.wavelet = positional[0];
    options.height = parse_positive(positional[1], "HEIGHT");
    options.width = parse_positive(positional[2], "WIDTH");
    if (ttwv::boundary_mode_requires_multiple_samples(options.boundary_mode) &&
        (options.height <= 1 || options.width <= 1)) {
        throw std::runtime_error("2D reflect and antireflect modes require HEIGHT and WIDTH greater than one");
    }
    for (size_t band = 0; band < options.bands.size(); ++band) {
        options.bands[band] = positional[3 + band];
    }
    return options;
}

[[nodiscard]] std::vector<float> read_binary(const std::filesystem::path& path, const size_t elements) {
    std::ifstream input(path, std::ios::binary);
    if (!input.good()) {
        throw std::runtime_error("Failed to open input band: " + path.string());
    }
    std::vector<float> values(elements);
    input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (input.gcount() != static_cast<std::streamsize>(values.size() * sizeof(float))) {
        throw std::runtime_error("Input band size does not match the planned coefficient shape: " + path.string());
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
                const size_t source =
                    (tile_y * ttwv::kTileHeight2D + row) * shape.width + tile_x * ttwv::kTileWidth2D;
                std::copy_n(
                    row_major.begin() + static_cast<std::ptrdiff_t>(source),
                    ttwv::kTileWidth2D,
                    tile.begin() + static_cast<std::ptrdiff_t>(row * ttwv::kTileWidth2D));
            }
            const std::vector<float> tile_nfaces = tilize_nfaces(tile, 32, 32);
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
            const std::vector<float> tile =
                untilize_nfaces(std::vector<float>(begin, begin + ttwv::kTileHeight2D * ttwv::kTileWidth2D), 32, 32);
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

[[nodiscard]] std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> create_tiled_dram(
    tt::tt_metal::distributed::MeshDevice& mesh_device, const ttwv::Shape2D shape) {
    const size_t tiles = shape.height / ttwv::kTileHeight2D * (shape.width / ttwv::kTileWidth2D);
    return tt::tt_metal::distributed::MeshBuffer::create(
        tt::tt_metal::distributed::ReplicatedBufferConfig{
            .size = static_cast<uint64_t>(tiles * ttwv::device_protocol::kLwt2DFullTileBytes),
        },
        tt::tt_metal::distributed::DeviceLocalBufferConfig{
            .page_size = ttwv::device_protocol::kLwt2DFullTileBytes,
            .buffer_type = tt::tt_metal::BufferType::DRAM,
        },
        &mesh_device);
}

template <typename Scheme>
int run(const Options& options) {
    const ttwv::Ilwt2DExecutionPlan host_plan =
        ttwv::make_ilwt_2d_execution_plan<Scheme>(
            options.height, options.width, options.core_limit, 768 * 1024, options.boundary_mode);
    std::array<std::vector<float>, 4> tiled_bands;
    for (size_t band = 0; band < tiled_bands.size(); ++band) {
        const std::vector<float> logical =
            read_binary(options.bands[band], host_plan.band_height * host_plan.band_width);
        tiled_bands[band] = tilize_padded(
            ttwv::zero_pad_row_major_to_tiles_2d(logical, host_plan.tiling.band.logical),
            host_plan.tiling.band.storage);
    }

    auto mesh_device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);
    mesh_device->enable_program_cache();
    auto& queue = mesh_device->mesh_command_queue();
    std::array<std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>, 4> band_buffers;
    for (size_t band = 0; band < band_buffers.size(); ++band) {
        band_buffers[band] = create_tiled_dram(*mesh_device, host_plan.tiling.band.storage);
        tt::tt_metal::distributed::EnqueueWriteMeshBuffer(queue, band_buffers[band], tiled_bands[band], false);
    }
    tt::tt_metal::distributed::Finish(queue);

    ttwv::Ilwt2DExecutable executable = ttwv::create_ilwt_2d_executable<Scheme>(
        TT_WAVELET_SOURCE_DIR,
        *mesh_device,
        *band_buffers[0]->get_backing_buffer(),
        *band_buffers[1]->get_backing_buffer(),
        *band_buffers[2]->get_backing_buffer(),
        *band_buffers[3]->get_backing_buffer(),
        options.height,
        options.width,
        options.core_limit,
        options.boundary_mode);
    ttwv::prepare_ilwt_2d(queue, executable);
    for (size_t warmup = 0; warmup < options.warmup_runs; ++warmup) {
        ttwv::execute_ilwt_2d(*mesh_device, queue, executable);
    }
    std::vector<double> times;
    for (size_t repeat = 0; repeat < options.repeats; ++repeat) {
        const auto start = std::chrono::steady_clock::now();
        ttwv::execute_ilwt_2d(*mesh_device, queue, executable);
        const auto stop = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    }

    std::vector<float> tiled_output;
    tt::tt_metal::distributed::EnqueueReadMeshBuffer(queue, tiled_output, executable.buffers.outputs[0], true);
    const std::vector<float> padded = untilize_padded(tiled_output, executable.plan.tiling.input.storage);
    std::vector<float> logical(options.height * options.width);
    for (size_t row = 0; row < options.height; ++row) {
        std::copy_n(
            padded.begin() + static_cast<std::ptrdiff_t>(row * executable.plan.tiling.input.storage.width),
            options.width,
            logical.begin() + static_cast<std::ptrdiff_t>(row * options.width));
    }
    if (!options.output.parent_path().empty()) {
        std::filesystem::create_directories(options.output.parent_path());
    }
    std::ofstream output(options.output, std::ios::binary);
    output.write(
        reinterpret_cast<const char*>(logical.data()), static_cast<std::streamsize>(logical.size() * sizeof(float)));
    if (!output.good()) {
        throw std::runtime_error("Failed to write reconstructed output: " + options.output.string());
    }
    const double mean = std::accumulate(times.begin(), times.end(), 0.0) / static_cast<double>(times.size());
    std::vector<double> sorted_times = times;
    std::sort(sorted_times.begin(), sorted_times.end());
    const auto percentile = [&sorted_times](const double probability) {
        const double position = probability * static_cast<double>(sorted_times.size() - 1);
        const size_t lower = static_cast<size_t>(position);
        const size_t upper = std::min(lower + 1, sorted_times.size() - 1);
        const double fraction = position - static_cast<double>(lower);
        return sorted_times[lower] + fraction * (sorted_times[upper] - sorted_times[lower]);
    };
    const double squared_error =
        std::accumulate(times.begin(), times.end(), 0.0, [mean](const double sum, const double value) {
            const double difference = value - mean;
            return sum + difference * difference;
        });
    for (size_t repeat = 0; repeat < times.size(); ++repeat) {
        std::cerr << std::fixed << std::setprecision(6) << "ilwt_2d_repeat_time_ms[" << repeat
                  << "]: " << times[repeat] << '\n';
    }
    const size_t route_count = executable.plan.chunks.empty() ? 0 : executable.plan.chunks.front().routes.size();
    const size_t scale_routes_removed =
        executable.plan.chunks.empty()
            ? 0
            : static_cast<size_t>(std::count_if(
                  executable.plan.chunks.front().routes.begin(),
                  executable.plan.chunks.front().routes.end(),
                  [](const ttwv::Lwt2DRoutePlan& route) { return ttwv::is_scale_step(route.type); }));
    std::cerr << "ilwt_2d_architecture: "
              << tt::arch_to_str(executable.buffers.scheduler.architecture) << '\n'
              << "ilwt_2d_boundary_mode: " << ttwv::boundary_mode_name(options.boundary_mode) << '\n'
              << "ilwt_2d_available_worker_core_count: "
              << executable.buffers.scheduler.available_worker_core_count << '\n'
              << std::fixed << std::setprecision(6) << "ilwt_2d_execution_time_ms: " << mean << '\n'
              << "ilwt_2d_min_execution_time_ms: " << sorted_times.front() << '\n'
              << "ilwt_2d_median_time_ms: " << percentile(0.5) << '\n'
              << "ilwt_2d_p10_time_ms: " << percentile(0.1) << '\n'
              << "ilwt_2d_p90_time_ms: " << percentile(0.9) << '\n'
              << "ilwt_2d_stddev_time_ms: "
              << std::sqrt(squared_error / static_cast<double>(times.size())) << '\n'
              << "ilwt_2d_active_core_count: " << executable.plan.active_core_count << '\n'
              << "ilwt_2d_chunk_count: " << executable.plan.chunks.size() << '\n'
              << "ilwt_2d_chunk_tiles: " << executable.plan.chunk_tiles_y << 'x'
              << executable.plan.chunk_tiles_x << '\n'
              << "ilwt_2d_route_count: " << route_count << '\n'
              << "ilwt_2d_executable_route_count: " << executable.plan.executable_route_count << '\n'
              << "ilwt_2d_scale_routes_removed: " << scale_routes_removed << '\n'
              << "ilwt_2d_l1_total_bytes: " << executable.plan.allocated_l1_bytes << '\n';
    return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const auto dispatch = [&]<typename Scheme>() { return run<Scheme>(options); };
        if (options.wavelet == ttwv::schemes::testing::synthetic_k17::name) {
            return dispatch.template operator()<ttwv::schemes::testing::synthetic_k17>();
        }
        return ttwv::dispatch_scheme(options.wavelet, dispatch);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
