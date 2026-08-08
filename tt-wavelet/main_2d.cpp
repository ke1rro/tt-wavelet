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
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt_wavelet/include/benchmark/timing.hpp"
#include "tt-metalium/tilize_utils.hpp"
#include "tt_wavelet/include/common/boundary_parse.hpp"
#include "tt_wavelet/include/common/tiling_2d.hpp"
#include "tt_wavelet/include/lifting/device_2d.hpp"
#include "tt_wavelet/include/schemes/generated/registry.hpp"
#include "tt_wavelet/include/schemes/testing/synthetic_k17.hpp"

namespace {

struct Options {
    bool benchmark{false};
    bool include_transfers{false};
    bool binary_input{false};
    bool quiet{false};
    size_t repeats{1};
    size_t warmup_runs{1};
    uint32_t core_limit{64};
    uint32_t batch_count{1};
    ttwv::BoundaryMode boundary_mode{ttwv::BoundaryMode::kSymmetric};
    std::string wavelet;
    size_t height{0};
    size_t width{0};
    std::filesystem::path input_path;
    std::optional<std::filesystem::path> output_prefix;
};

struct DeviceBands {
    uint32_t batch_count{1};
    size_t height{0};
    size_t width{0};
    std::array<std::vector<float>, ttwv::device_protocol::kLwt2DBandCount> values;
};

[[nodiscard]] std::string usage() {
    return "Usage: lwt_2d "
           "[--boundary-mode zero|constant|symmetric|reflect|periodic|smooth|antisymmetric|antireflect] "
           "[--binary-input] [--cores N] [--batch-count B] [--output-prefix PATH] [--quiet] "
           "[--benchmark [--include-transfers] [--repeats N] [--warmup-runs N]] "
           "WAVELET HEIGHT WIDTH INPUT_FILE";
}

[[nodiscard]] size_t parse_unsigned(const std::string& text, const char* label, const bool allow_zero) {
    if (text.empty() || text.front() == '-') {
        throw std::runtime_error(std::string{label} + (allow_zero ? " must be non-negative" : " must be positive"));
    }
    size_t consumed = 0;
    const unsigned long long value = std::stoull(text, &consumed);
    if (consumed != text.size() || (!allow_zero && value == 0) || value > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error(std::string{label} + (allow_zero ? " must be non-negative" : " must be positive"));
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
        } else if (argument == "--binary-input") {
            options.binary_input = true;
        } else if (argument == "--cores") {
            if (++index >= argc) {
                throw std::runtime_error("--cores requires a value");
            }
            const size_t cores = parse_unsigned(argv[index], "--cores", false);
            if (cores > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("--cores exceeds uint32_t");
            }
            options.core_limit = static_cast<uint32_t>(cores);
        } else if (argument == "--batch-count") {
            if (++index >= argc) {
                throw std::runtime_error("--batch-count requires a value");
            }
            const size_t batch_count = parse_unsigned(argv[index], "--batch-count", false);
            if (batch_count > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("--batch-count exceeds uint32_t");
            }
            options.batch_count = static_cast<uint32_t>(batch_count);
        } else if (argument == "--output-prefix") {
            if (++index >= argc || std::string_view{argv[index]}.empty()) {
                throw std::runtime_error("--output-prefix requires a path");
            }
            options.output_prefix = std::filesystem::path{argv[index]};
        } else if (argument == "--quiet") {
            options.quiet = true;
        } else if (argument == "--benchmark") {
            options.benchmark = true;
        } else if (argument == "--include-transfers") {
            options.include_transfers = true;
        } else if (argument == "--repeats") {
            if (++index >= argc) {
                throw std::runtime_error("--repeats requires a value");
            }
            options.repeats = parse_unsigned(argv[index], "--repeats", false);
        } else if (argument == "--warmup-runs") {
            if (++index >= argc) {
                throw std::runtime_error("--warmup-runs requires a value");
            }
            options.warmup_runs = parse_unsigned(argv[index], "--warmup-runs", true);
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
    options.height = parse_unsigned(positional[1], "HEIGHT", false);
    options.width = parse_unsigned(positional[2], "WIDTH", false);
    options.input_path = positional[3];
    if (ttwv::boundary_mode_requires_multiple_samples(options.boundary_mode) &&
        (options.height <= 1 || options.width <= 1)) {
        throw std::runtime_error("2D reflect and antireflect modes require HEIGHT and WIDTH greater than one");
    }
    if (!options.benchmark && (options.repeats != 1 || options.warmup_runs != 1)) {
        throw std::runtime_error("--repeats and --warmup-runs require --benchmark");
    }
    if (options.include_transfers && !options.benchmark) {
        throw std::runtime_error("--include-transfers requires --benchmark");
    }
    if (options.quiet && !options.output_prefix && !options.benchmark) {
        throw std::runtime_error("--quiet requires --output-prefix or --benchmark");
    }
    return options;
}

[[nodiscard]] std::vector<float> read_input(
    const std::filesystem::path& path,
    const size_t height,
    const size_t width,
    const uint32_t batch_count,
    const bool binary) {
    if (height > std::numeric_limits<size_t>::max() / width) {
        throw std::runtime_error("2D input shape overflows size_t");
    }
    std::ifstream input(path, binary ? std::ios::binary : std::ios::in);
    if (!input.good()) {
        throw std::runtime_error("Failed to open input file: " + path.string());
    }
    const size_t sample_elements = height * width;
    if (sample_elements > std::numeric_limits<size_t>::max() / batch_count) {
        throw std::runtime_error("2D batched input shape overflows size_t");
    }
    const size_t total_elements = static_cast<size_t>(batch_count) * sample_elements;
    std::vector<float> values(total_elements);
    if (binary) {
        input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
        if (input.gcount() != static_cast<std::streamsize>(values.size() * sizeof(float))) {
            throw std::runtime_error("Binary input contains fewer than B x HEIGHT x WIDTH FP32 values");
        }
    } else {
        values.clear();
        values.reserve(total_elements);
        for (float value = 0.0F; input >> value;) {
            values.push_back(value);
        }
        if (!input.eof()) {
            throw std::runtime_error("Input file contains a non-numeric token");
        }
        if (values.size() != total_elements) {
            throw std::runtime_error("Input element count does not match B x HEIGHT x WIDTH");
        }
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
            const std::vector<float> tile = untilize_nfaces(
                std::vector<float>(
                    begin, begin + static_cast<std::ptrdiff_t>(ttwv::kTileHeight2D * ttwv::kTileWidth2D)),
                32,
                32);
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

[[nodiscard]] std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> create_input(
    tt::tt_metal::distributed::MeshDevice& mesh_device, const ttwv::Shape2D padded_shape, const uint32_t batch_count) {
    const size_t tiles = padded_shape.height / ttwv::kTileHeight2D * (padded_shape.width / ttwv::kTileWidth2D);
    return tt::tt_metal::distributed::MeshBuffer::create(
        tt::tt_metal::distributed::ReplicatedBufferConfig{
            .size = static_cast<uint64_t>(batch_count) * tiles * ttwv::device_protocol::kLwt2DFullTileBytes,
        },
        tt::tt_metal::distributed::DeviceLocalBufferConfig{
            .page_size = ttwv::device_protocol::kLwt2DFullTileBytes,
            .buffer_type = tt::tt_metal::BufferType::DRAM,
        },
        &mesh_device);
}

[[nodiscard]] double percentile(const std::vector<double>& sorted, const double probability) {
    const double position = probability * static_cast<double>(sorted.size() - 1);
    const size_t lower = static_cast<size_t>(position);
    const size_t upper = std::min(lower + 1, sorted.size() - 1);
    const double fraction = position - static_cast<double>(lower);
    return sorted[lower] + fraction * (sorted[upper] - sorted[lower]);
}

void print_timings(const std::string_view prefix, const std::string_view metric, std::vector<double> times) {
    const double mean = std::accumulate(times.begin(), times.end(), 0.0) / static_cast<double>(times.size());
    const double squared_error =
        std::accumulate(times.begin(), times.end(), 0.0, [mean](const double sum, const double value) {
            const double difference = value - mean;
            return sum + difference * difference;
        });
    for (size_t repeat = 0; repeat < times.size(); ++repeat) {
        std::cerr << std::fixed << std::setprecision(6) << prefix << '_' << metric << "_repeat_ms[" << repeat
                  << "]: " << times[repeat] << '\n';
    }
    std::sort(times.begin(), times.end());
    std::cerr << std::fixed << std::setprecision(6) << prefix << '_' << metric << "_mean_ms: " << mean << '\n'
              << prefix << '_' << metric << "_min_ms: " << times.front() << '\n'
              << prefix << '_' << metric << "_median_ms: " << percentile(times, 0.5) << '\n'
              << prefix << '_' << metric << "_p10_ms: " << percentile(times, 0.1) << '\n'
              << prefix << '_' << metric << "_p90_ms: " << percentile(times, 0.9) << '\n'
              << prefix << '_' << metric
              << "_stddev_ms: " << std::sqrt(squared_error / static_cast<double>(times.size())) << '\n';
}

void print_telemetry(const ttwv::Lwt2DSchedulerTelemetry& telemetry) {
    const auto ratio = [](const uint64_t internal, const uint64_t exact) {
        return exact == 0 ? 1.0 : static_cast<double>(internal) / static_cast<double>(exact);
    };
    std::cerr << "lwt_2d_architecture: " << tt::arch_to_str(telemetry.architecture) << '\n'
              << "lwt_2d_boundary_mode: " << ttwv::boundary_mode_name(telemetry.boundary_mode) << '\n'
              << "lwt_2d_available_worker_core_count: " << telemetry.available_worker_core_count << '\n'
              << "lwt_2d_active_core_count: " << telemetry.active_core_count << '\n'
              << "lwt_2d_batch_count: " << telemetry.batch_count << '\n'
              << "lwt_2d_chunks_per_sample: " << telemetry.chunks_per_sample << '\n'
              << "lwt_2d_total_work_items: " << telemetry.total_work_items << '\n'
              << "lwt_2d_min_work_items_per_core: " << telemetry.min_work_items_per_core << '\n'
              << "lwt_2d_max_work_items_per_core: " << telemetry.max_work_items_per_core << '\n'
              << "lwt_2d_chunk_count: " << telemetry.chunk_count << '\n'
              << "lwt_2d_chunk_tiles: " << telemetry.chunk_tiles_y << 'x' << telemetry.chunk_tiles_x << '\n'
              << "lwt_2d_route_count: " << telemetry.route_count << '\n'
              << "lwt_2d_executable_route_count: " << telemetry.executable_route_count << '\n'
              << "lwt_2d_scale_routes_removed: " << telemetry.scale_routes_removed << '\n'
              << "lwt_2d_estimated_latency_cycles: " << telemetry.estimated_latency_cycles << '\n'
              << "lwt_2d_l1_workspace_bytes: " << telemetry.l1_workspace_bytes << '\n'
              << "lwt_2d_l1_circular_buffer_bytes: " << telemetry.l1_circular_buffer_bytes << '\n'
              << "lwt_2d_l1_metadata_bytes: " << telemetry.l1_metadata_bytes << '\n'
              << "lwt_2d_l1_synchronization_bytes: " << telemetry.l1_synchronization_bytes << '\n'
              << "lwt_2d_l1_total_bytes: " << telemetry.l1_total_bytes << '\n'
              << "lwt_2d_l1_capacity_bytes: " << telemetry.l1_capacity_bytes << '\n'
              << "lwt_2d_l1_headroom_bytes: " << telemetry.l1_headroom_bytes << '\n'
              << "lwt_2d_initial_overcompute_ratio: "
              << ratio(telemetry.internal_initial_elements, telemetry.exact_initial_elements) << '\n'
              << "lwt_2d_route_overcompute_ratio: "
              << ratio(telemetry.internal_route_elements, telemetry.exact_route_elements) << '\n'
              << "lwt_2d_final_overcompute_ratio: "
              << ratio(telemetry.internal_final_elements, telemetry.exact_final_elements) << '\n';
}

[[nodiscard]] DeviceBands read_bands(
    tt::tt_metal::distributed::MeshCommandQueue& queue, ttwv::Lwt2DExecutable& executable) {
    DeviceBands result{
        .batch_count = executable.buffers.scheduler.batch_count,
        .height = executable.plan.tiling.band.logical.height,
        .width = executable.plan.tiling.band.logical.width,
    };
    for (size_t band = 0; band < result.values.size(); ++band) {
        std::vector<float> tiled;
        tt::tt_metal::distributed::EnqueueReadMeshBuffer(queue, tiled, executable.buffers.outputs[band], true);
        const size_t tiled_stride = tiled.size() / result.batch_count;
        for (uint32_t batch = 0; batch < result.batch_count; ++batch) {
            const std::vector<float> sample(
                tiled.begin() + static_cast<std::ptrdiff_t>(batch * tiled_stride),
                tiled.begin() + static_cast<std::ptrdiff_t>((batch + 1) * tiled_stride));
            auto logical = crop(
                untilize_padded(sample, executable.plan.tiling.band.storage),
                executable.plan.tiling.band.storage,
                executable.plan.tiling.band.logical);
            result.values[band].insert(result.values[band].end(), logical.begin(), logical.end());
        }
    }
    return result;
}

void write_output_bands(const std::filesystem::path& prefix, const DeviceBands& output) {
    if (!prefix.parent_path().empty()) {
        std::filesystem::create_directories(prefix.parent_path());
    }
    constexpr std::array<std::string_view, 4> names = {"LL", "LH", "HL", "HH"};
    for (size_t band = 0; band < output.values.size(); ++band) {
        const std::filesystem::path path = prefix.string() + "_" + std::string{names[band]} + ".f32";
        std::ofstream stream(path, std::ios::binary);
        stream.write(
            reinterpret_cast<const char*>(output.values[band].data()),
            static_cast<std::streamsize>(output.values[band].size() * sizeof(float)));
        if (!stream.good()) {
            throw std::runtime_error("Failed to write device band: " + path.string());
        }
    }
    std::ofstream shape(prefix.string() + "_shape.txt");
    if (output.batch_count == 1) {
        shape << output.height << ' ' << output.width << '\n';
    } else {
        shape << output.batch_count << ' ' << output.height << ' ' << output.width << '\n';
    }
    if (!shape.good()) {
        throw std::runtime_error("Failed to write device output shape");
    }
}

void print_bands(const DeviceBands& output) {
    constexpr std::array<std::string_view, 4> names = {"LL", "LH", "HL", "HH"};
    for (size_t band = 0; band < output.values.size(); ++band) {
        std::cout << "tt-wavelet device " << names[band] << " (" << output.height << 'x' << output.width << "): [";
        for (size_t index = 0; index < output.values[band].size(); ++index) {
            if (index != 0) {
                std::cout << ", ";
            }
            std::cout << std::scientific << std::setprecision(8) << output.values[band][index];
        }
        std::cout << std::defaultfloat << "]\n";
    }
}

template <typename Scheme>
int run(
    const Options& options,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& queue,
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& input,
    const std::vector<float>& tiled_input) {
    const uint32_t effective_cores =
        (options.core_limit > 0)
            ? options.core_limit
            : static_cast<uint32_t>(
                  mesh_device.compute_with_storage_grid_size().x * mesh_device.compute_with_storage_grid_size().y);
    ttwv::Lwt2DExecutable executable = ttwv::create_lwt_2d_executable<Scheme>(
        TT_WAVELET_SOURCE_DIR,
        mesh_device,
        *input->get_backing_buffer(),
        options.height,
        options.width,
        effective_cores,
        options.boundary_mode,
        options.batch_count);
    ttwv::prepare_lwt_2d(queue, executable);

    const auto execute = [&]() {
        return ttwv::benchmark::measure_host_timing(queue, [&]() { ttwv::enqueue_lwt_2d(queue, executable); });
    };
    if (options.benchmark) {
        for (size_t warmup = 0; warmup < options.warmup_runs; ++warmup) {
            static_cast<void>(execute());
        }
        ttwv::benchmark::discard_device_profiler_samples(mesh_device);
        std::vector<double> enqueue_times;
        std::vector<double> sync_times;
        std::vector<double> host_total_times;
        std::vector<double> device_times;
        enqueue_times.reserve(options.repeats);
        sync_times.reserve(options.repeats);
        host_total_times.reserve(options.repeats);
        device_times.reserve(options.repeats);
        for (size_t repeat = 0; repeat < options.repeats; ++repeat) {
            const ttwv::benchmark::HostTiming sample = execute();
            enqueue_times.push_back(sample.enqueue_or_dispatch_ms);
            sync_times.push_back(sample.sync_wait_ms);
            host_total_times.push_back(sample.host_api_total_ms);
            const std::vector<double> device_sample = ttwv::benchmark::read_device_kernel_times_ms(mesh_device, 1);
            device_times.insert(device_times.end(), device_sample.begin(), device_sample.end());
        }
        if (device_times.empty()) {
            std::cerr << "lwt_2d_device_time_status: unavailable\n"
                      << "lwt_2d_timing_mechanism: host_steady_clock_enqueue_plus_finish\n";
        } else {
            std::cerr << "lwt_2d_device_time_status: available\n"
                      << "lwt_2d_timing_mechanism: tt_metal_device_profiler_kernel_span\n";
            print_timings("lwt_2d", "device_time", device_times);
        }
        print_timings("lwt_2d", "enqueue_or_dispatch", std::move(enqueue_times));
        print_timings("lwt_2d", "sync_wait", std::move(sync_times));
        print_timings("lwt_2d", "host_api_total", std::move(host_total_times));

        if (options.include_transfers) {
            std::array<std::vector<float>, ttwv::device_protocol::kLwt2DBandCount> outputs;
            const auto execute_end_to_end = [&]() {
                const auto start = std::chrono::steady_clock::now();
                tt::tt_metal::distributed::EnqueueWriteMeshBuffer(queue, input, tiled_input, false);
                ttwv::execute_lwt_2d(mesh_device, queue, executable);
                for (size_t band = 0; band < outputs.size(); ++band) {
                    tt::tt_metal::distributed::EnqueueReadMeshBuffer(
                        queue, outputs[band], executable.buffers.outputs[band], true);
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
            print_timings("lwt_2d", "host_end_to_end", std::move(end_to_end_times));
        }
        if (options.output_prefix) {
            write_output_bands(*options.output_prefix, read_bands(queue, executable));
        }
        print_telemetry(executable.buffers.scheduler);
        return EXIT_SUCCESS;
    }

    const ttwv::benchmark::HostTiming execution_timing = execute();
    const DeviceBands output = read_bands(queue, executable);
    if (options.output_prefix) {
        write_output_bands(*options.output_prefix, output);
    }
    if (!options.quiet) {
        print_bands(output);
    }
    std::cerr << std::fixed << std::setprecision(6)
              << "lwt_2d_host_api_total_ms: " << execution_timing.host_api_total_ms << '\n';
    print_telemetry(executable.buffers.scheduler);
    return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.benchmark) {
            ttwv::benchmark::configure_device_profiler_environment();
        }
        const std::vector<float> logical_input =
            read_input(options.input_path, options.height, options.width, options.batch_count, options.binary_input);
        const ttwv::TiledShape2D input_shape = ttwv::make_tiled_shape_2d({
            .height = options.height,
            .width = options.width,
        });
        std::vector<float> tiled;
        const size_t logical_stride = options.height * options.width;
        for (uint32_t batch = 0; batch < options.batch_count; ++batch) {
            const std::vector<float> sample(
                logical_input.begin() + static_cast<std::ptrdiff_t>(batch * logical_stride),
                logical_input.begin() + static_cast<std::ptrdiff_t>((batch + 1) * logical_stride));
            const std::vector<float> padded = ttwv::zero_pad_row_major_to_tiles_2d(sample, input_shape.logical);
            if (!ttwv::has_zero_tile_padding_2d(padded, input_shape)) {
                throw std::runtime_error("2D input preprocessing violated zero-padding contract");
            }
            const std::vector<float> tiled_sample = tilize_padded(padded, input_shape.storage);
            tiled.insert(tiled.end(), tiled_sample.begin(), tiled_sample.end());
        }

        auto mesh_device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);
        mesh_device->enable_program_cache();
        auto& queue = mesh_device->mesh_command_queue();
        auto input = create_input(*mesh_device, input_shape.storage, options.batch_count);
        tt::tt_metal::distributed::EnqueueWriteMeshBuffer(queue, input, tiled, true);

        const auto dispatch = [&]<typename Scheme>() {
            return run<Scheme>(options, *mesh_device, queue, input, tiled);
        };
        if (options.wavelet == ttwv::schemes::testing::synthetic_k17::name) {
            return dispatch.template operator()<ttwv::schemes::testing::synthetic_k17>();
        }
        return ttwv::dispatch_scheme(options.wavelet, dispatch);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
