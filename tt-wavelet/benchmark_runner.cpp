// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt-metalium/tilize_utils.hpp"
#include "tt_metal/impl/program/program_impl.hpp"
#include "tt_wavelet/include/benchmark/timing.hpp"
#include "tt_wavelet/include/common/boundary_parse.hpp"
#include "tt_wavelet/include/common/tiling_2d.hpp"
#include "tt_wavelet/include/lifting/device.hpp"
#include "tt_wavelet/include/lifting/device_2d.hpp"
#include "tt_wavelet/include/schemes/generated/registry.hpp"

namespace {

using Json = nlohmann::json;

struct TimingResult {
    std::vector<double> device_time_ms;
    std::vector<double> enqueue_or_dispatch_ms;
    std::vector<double> sync_wait_ms;
    std::vector<double> host_api_total_ms;
};

[[nodiscard]] std::vector<float> read_binary(const std::filesystem::path& path, const size_t elements) {
    std::ifstream input(path, std::ios::binary);
    if (!input.good()) {
        throw std::runtime_error("Failed to open input file: " + path.string());
    }
    std::vector<float> values(elements);
    input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (input.gcount() != static_cast<std::streamsize>(values.size() * sizeof(float))) {
        throw std::runtime_error("Input file size does not match the request: " + path.string());
    }
    char trailing = 0;
    if (input.read(&trailing, 1)) {
        throw std::runtime_error("Input file contains trailing data: " + path.string());
    }
    return values;
}

void write_binary(const std::filesystem::path& path, const std::vector<float>& values) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path, std::ios::binary);
    output.write(
        reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!output.good()) {
        throw std::runtime_error("Failed to write output file: " + path.string());
    }
}

[[nodiscard]] TimingResult measure(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    const uint32_t warmup_runs,
    const uint32_t repeats,
    const auto& enqueue) {
    for (uint32_t warmup = 0; warmup < warmup_runs; ++warmup) {
        static_cast<void>(ttwv::benchmark::measure_host_timing(command_queue, enqueue));
    }
    ttwv::benchmark::discard_device_profiler_samples(mesh_device);

    TimingResult result;
    result.device_time_ms.reserve(repeats);
    result.enqueue_or_dispatch_ms.reserve(repeats);
    result.sync_wait_ms.reserve(repeats);
    result.host_api_total_ms.reserve(repeats);
    for (uint32_t repeat = 0; repeat < repeats; ++repeat) {
        const ttwv::benchmark::HostTiming host = ttwv::benchmark::measure_host_timing(command_queue, enqueue);
        result.enqueue_or_dispatch_ms.push_back(host.enqueue_or_dispatch_ms);
        result.sync_wait_ms.push_back(host.sync_wait_ms);
        result.host_api_total_ms.push_back(host.host_api_total_ms);
        const std::vector<double> device_sample = ttwv::benchmark::read_device_kernel_times_ms(mesh_device, 1);
        result.device_time_ms.insert(result.device_time_ms.end(), device_sample.begin(), device_sample.end());
    }
    return result;
}

void add_timing(Json& result, const TimingResult& timing, const uint32_t warmup_runs, const uint32_t repeats) {
    result["device_time_ms"] = timing.device_time_ms;
    result["enqueue_or_dispatch_ms"] = timing.enqueue_or_dispatch_ms;
    result["sync_wait_ms"] = timing.sync_wait_ms;
    result["host_api_total_ms"] = timing.host_api_total_ms;
    result["warmup_count"] = warmup_runs;
    result["repeat_count"] = repeats;
    result["timing_mechanism"] = timing.device_time_ms.empty() ? "host_steady_clock_enqueue_plus_finish"
                                                               : "tt_metal_device_profiler_kernel_span";
}

void add_program_size(Json& result, tt::tt_metal::distributed::MeshWorkload& workload) {
    std::vector<uint32_t> maximum_sizes;
    for (auto& [_, program] : workload.get_programs()) {
        const std::vector<uint32_t>& sizes = program.impl().get_program_config_sizes();
        if (maximum_sizes.size() < sizes.size()) {
            maximum_sizes.resize(sizes.size(), 0U);
        }
        for (size_t index = 0; index < sizes.size(); ++index) {
            maximum_sizes[index] = std::max(maximum_sizes[index], sizes[index]);
        }
    }
    result["program_config_sizes_bytes"] = maximum_sizes;
    result["program_config_max_bytes"] =
        maximum_sizes.empty() ? 0U : *std::max_element(maximum_sizes.begin(), maximum_sizes.end());
}

struct DeviceInput1D {
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> buffer;
    ttwv::SignalBuffer descriptor{};
};

[[nodiscard]] DeviceInput1D upload_1d(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    const std::vector<float>& values,
    const size_t logical_length,
    const uint32_t batch_count) {
    if (values.size() != logical_length * batch_count) {
        throw std::runtime_error("1D input element count does not match length and batch_count");
    }
    ttwv::SignalBuffer descriptor{.length = logical_length};
    const size_t stride = descriptor.physical_nbytes() / sizeof(float);
    std::vector<float> payload(static_cast<size_t>(batch_count) * stride, 0.0F);
    for (uint32_t batch = 0; batch < batch_count; ++batch) {
        std::copy_n(
            values.begin() + static_cast<std::ptrdiff_t>(batch * logical_length),
            logical_length,
            payload.begin() + static_cast<std::ptrdiff_t>(batch * stride));
    }
    auto buffer = tt::tt_metal::distributed::MeshBuffer::create(
        tt::tt_metal::distributed::ReplicatedBufferConfig{.size = descriptor.physical_nbytes() * batch_count},
        tt::tt_metal::distributed::DeviceLocalBufferConfig{
            .page_size = descriptor.aligned_stick_bytes(),
            .buffer_type = tt::tt_metal::BufferType::DRAM,
        },
        &mesh_device);
    descriptor.dram_address = buffer->get_backing_buffer()->address();
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(command_queue, buffer, payload, true);
    return DeviceInput1D{.buffer = std::move(buffer), .descriptor = descriptor};
}

[[nodiscard]] std::vector<float> crop_stick_output(
    const std::vector<float>& physical, const size_t logical_length, const uint32_t batch_count) {
    const size_t stride = physical.size() / batch_count;
    std::vector<float> logical;
    logical.reserve(static_cast<size_t>(batch_count) * logical_length);
    for (uint32_t batch = 0; batch < batch_count; ++batch) {
        logical.insert(
            logical.end(),
            physical.begin() + static_cast<std::ptrdiff_t>(batch * stride),
            physical.begin() + static_cast<std::ptrdiff_t>(batch * stride + logical_length));
    }
    return logical;
}

void add_scheduler(Json& result, const ttwv::LiftingSchedulerTelemetry& scheduler) {
    result["architecture"] = tt::arch_to_str(scheduler.architecture);
    result["logical_input_size"] = scheduler.signal_length;
    result["layout"] = scheduler.workspace_layout == ttwv::WorkspaceLayout::kTileNative ? "tile-native" : "row-major";
    result["core_count"] = scheduler.active_core_count;
    result["chunk_count"] = scheduler.chunk_count;
    result["route_count"] = scheduler.route_count;
    result["planner_groups_per_chunk"] = scheduler.groups_per_chunk;
    result["memory_config"] = "dram-interleaved-input-output/l1-sharded-workspace";
}

template <typename Scheme>
[[nodiscard]] Json run_1d(
    const Json& request,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    const ttwv::BoundaryMode boundary_mode) {
    const std::string transform = request.at("transform").get<std::string>();
    const size_t length = request.at("length").get<size_t>();
    const uint32_t batch_count = request.value("batch_count", 1U);
    const uint32_t repeats = request.value("repeats", 1U);
    const uint32_t warmup_runs = request.value("warmup_runs", 0U);
    const std::vector<std::string> input_paths = request.at("input_paths").get<std::vector<std::string>>();
    const std::filesystem::path output_prefix = request.value("output_prefix", "");
    const bool capture = request.value("capture_outputs", false);

    Json result;
    if (transform == "lwt") {
        if (input_paths.size() != 1) {
            throw std::runtime_error("1D LWT requires one input path");
        }
        DeviceInput1D input = upload_1d(
            mesh_device, command_queue, read_binary(input_paths[0], length * batch_count), length, batch_count);
        auto executable = ttwv::create_lwt_executable<Scheme>(
            TT_WAVELET_SOURCE_DIR,
            mesh_device,
            *input.buffer->get_backing_buffer(),
            input.descriptor,
            boundary_mode,
            batch_count);
        ttwv::prepare_lwt(command_queue, executable);
        add_timing(
            result,
            measure(
                mesh_device,
                command_queue,
                warmup_runs,
                repeats,
                [&]() { ttwv::enqueue_lwt(command_queue, executable); }),
            warmup_runs,
            repeats);
        add_program_size(result, executable.workload);
        add_scheduler(result, executable.buffers.scheduler);
        result["logical_output_size"] = executable.plan.full_plan.output_length;
        result["physical_input_size"] = input.descriptor.physical_nbytes() / sizeof(float);
        result["physical_output_size"] = {
            executable.buffers.final_even->get_backing_buffer()->size() / sizeof(float),
            executable.buffers.final_odd->get_backing_buffer()->size() / sizeof(float)};
        if (capture) {
            std::vector<float> even;
            std::vector<float> odd;
            tt::tt_metal::distributed::EnqueueReadMeshBuffer(command_queue, even, executable.buffers.final_even, true);
            tt::tt_metal::distributed::EnqueueReadMeshBuffer(command_queue, odd, executable.buffers.final_odd, true);
            const auto& plan = executable.plan.full_plan;
            write_binary(
                output_prefix.string() + ".approximation.f32",
                crop_stick_output(even, plan.output_length, batch_count));
            write_binary(
                output_prefix.string() + ".detail.f32", crop_stick_output(odd, plan.output_length, batch_count));
        }
        return result;
    }

    if (transform != "ilwt" || input_paths.size() != 2) {
        throw std::runtime_error("1D transform must be lwt or ilwt; ILWT requires two coefficient paths");
    }
    const size_t coefficient_length = request.at("coefficient_length").get<size_t>();
    DeviceInput1D approximation = upload_1d(
        mesh_device,
        command_queue,
        read_binary(input_paths[0], coefficient_length * batch_count),
        coefficient_length,
        batch_count);
    DeviceInput1D detail = upload_1d(
        mesh_device,
        command_queue,
        read_binary(input_paths[1], coefficient_length * batch_count),
        coefficient_length,
        batch_count);
    auto executable = ttwv::create_ilwt_executable<Scheme>(
        TT_WAVELET_SOURCE_DIR,
        mesh_device,
        *approximation.buffer->get_backing_buffer(),
        *detail.buffer->get_backing_buffer(),
        coefficient_length,
        length,
        boundary_mode,
        batch_count);
    ttwv::prepare_ilwt(command_queue, executable);
    add_timing(
        result,
        measure(
            mesh_device, command_queue, warmup_runs, repeats, [&]() { ttwv::enqueue_ilwt(command_queue, executable); }),
        warmup_runs,
        repeats);
    add_program_size(result, executable.workload);
    add_scheduler(result, executable.buffers.scheduler);
    result["logical_output_size"] = length;
    result["physical_input_size"] = {
        approximation.descriptor.physical_nbytes() / sizeof(float),
        detail.descriptor.physical_nbytes() / sizeof(float)};
    result["physical_output_size"] = executable.buffers.output->get_backing_buffer()->size() / sizeof(float);
    if (capture) {
        std::vector<float> physical;
        tt::tt_metal::distributed::EnqueueReadMeshBuffer(command_queue, physical, executable.buffers.output, true);
        write_binary(output_prefix.string() + ".reconstructed.f32", crop_stick_output(physical, length, batch_count));
    }
    return result;
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
            const std::vector<float> faces = tilize_nfaces(tile, 32, 32);
            tiled.insert(tiled.end(), faces.begin(), faces.end());
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

[[nodiscard]] std::vector<float> crop_2d(
    const std::vector<float>& padded, const ttwv::Shape2D storage, const ttwv::Shape2D logical) {
    std::vector<float> result;
    result.reserve(logical.height * logical.width);
    for (size_t row = 0; row < logical.height; ++row) {
        result.insert(
            result.end(),
            padded.begin() + static_cast<std::ptrdiff_t>(row * storage.width),
            padded.begin() + static_cast<std::ptrdiff_t>(row * storage.width + logical.width));
    }
    return result;
}

[[nodiscard]] std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> upload_2d(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    const std::vector<float>& logical_values,
    const ttwv::TiledShape2D shape,
    const uint32_t batch_count) {
    const size_t logical_stride = shape.logical.height * shape.logical.width;
    if (logical_values.size() != logical_stride * batch_count) {
        throw std::runtime_error("2D input element count does not match shape and batch_count");
    }
    std::vector<float> tiled;
    for (uint32_t batch = 0; batch < batch_count; ++batch) {
        const std::vector<float> sample(
            logical_values.begin() + static_cast<std::ptrdiff_t>(batch * logical_stride),
            logical_values.begin() + static_cast<std::ptrdiff_t>((batch + 1) * logical_stride));
        const std::vector<float> padded = ttwv::zero_pad_row_major_to_tiles_2d(sample, shape.logical);
        const std::vector<float> tiled_sample = tilize_padded(padded, shape.storage);
        tiled.insert(tiled.end(), tiled_sample.begin(), tiled_sample.end());
    }
    const size_t tiles = shape.storage.height / ttwv::kTileHeight2D * shape.storage.width / ttwv::kTileWidth2D;
    auto buffer = tt::tt_metal::distributed::MeshBuffer::create(
        tt::tt_metal::distributed::ReplicatedBufferConfig{
            .size = static_cast<uint64_t>(batch_count) * tiles * ttwv::device_protocol::kLwt2DFullTileBytes,
        },
        tt::tt_metal::distributed::DeviceLocalBufferConfig{
            .page_size = ttwv::device_protocol::kLwt2DFullTileBytes,
            .buffer_type = tt::tt_metal::BufferType::DRAM,
        },
        &mesh_device);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(command_queue, buffer, tiled, true);
    return buffer;
}

[[nodiscard]] std::vector<float> read_2d(
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> buffer,
    const ttwv::TiledShape2D shape,
    const uint32_t batch_count) {
    std::vector<float> tiled;
    tt::tt_metal::distributed::EnqueueReadMeshBuffer(command_queue, tiled, buffer, true);
    const size_t stride = tiled.size() / batch_count;
    std::vector<float> logical;
    logical.reserve(static_cast<size_t>(batch_count) * shape.logical.height * shape.logical.width);
    for (uint32_t batch = 0; batch < batch_count; ++batch) {
        const std::vector<float> sample(
            tiled.begin() + static_cast<std::ptrdiff_t>(batch * stride),
            tiled.begin() + static_cast<std::ptrdiff_t>((batch + 1) * stride));
        const std::vector<float> cropped =
            crop_2d(untilize_padded(sample, shape.storage), shape.storage, shape.logical);
        logical.insert(logical.end(), cropped.begin(), cropped.end());
    }
    return logical;
}

void add_scheduler(Json& result, const ttwv::Lwt2DSchedulerTelemetry& scheduler) {
    result["architecture"] = tt::arch_to_str(scheduler.architecture);
    result["logical_input_shape"] = {scheduler.logical_input.height, scheduler.logical_input.width};
    result["physical_input_shape"] = {scheduler.padded_input.height, scheduler.padded_input.width};
    result["logical_output_shape"] = {scheduler.logical_band.height, scheduler.logical_band.width};
    result["physical_output_shape"] = {scheduler.padded_band.height, scheduler.padded_band.width};
    result["layout"] = "tile";
    result["core_count"] = scheduler.active_core_count;
    result["chunk_count"] = scheduler.chunk_count;
    result["route_count"] = scheduler.route_count;
    result["memory_config"] = "dram-interleaved-input-output/l1-sharded-workspace";
}

template <typename Scheme>
[[nodiscard]] Json run_2d(
    const Json& request,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    const ttwv::BoundaryMode boundary_mode) {
    const std::string transform = request.at("transform").get<std::string>();
    const size_t height = request.at("height").get<size_t>();
    const size_t width = request.at("width").get<size_t>();
    const uint32_t batch_count = request.value("batch_count", 1U);
    const uint32_t repeats = request.value("repeats", 1U);
    const uint32_t warmup_runs = request.value("warmup_runs", 0U);
    const uint32_t requested_core_limit = request.value("cores", 0U);
    const auto grid = mesh_device.compute_with_storage_grid_size();
    const uint32_t core_limit =
        requested_core_limit == 0U ? static_cast<uint32_t>(grid.x * grid.y) : requested_core_limit;
    const std::vector<std::string> input_paths = request.at("input_paths").get<std::vector<std::string>>();
    const std::filesystem::path output_prefix = request.value("output_prefix", "");
    const bool capture = request.value("capture_outputs", false);

    Json result;
    result["requested_core_limit"] = requested_core_limit;
    if (transform == "lwt_2d") {
        if (input_paths.size() != 1) {
            throw std::runtime_error("2D LWT requires one input path");
        }
        const ttwv::TiledShape2D input_shape = ttwv::make_tiled_shape_2d({.height = height, .width = width});
        auto input = upload_2d(
            mesh_device,
            command_queue,
            read_binary(input_paths[0], height * width * batch_count),
            input_shape,
            batch_count);
        auto executable = ttwv::create_lwt_2d_executable<Scheme>(
            TT_WAVELET_SOURCE_DIR,
            mesh_device,
            *input->get_backing_buffer(),
            height,
            width,
            core_limit,
            boundary_mode,
            batch_count);
        ttwv::prepare_lwt_2d(command_queue, executable);
        add_timing(
            result,
            measure(
                mesh_device,
                command_queue,
                warmup_runs,
                repeats,
                [&]() { ttwv::enqueue_lwt_2d(command_queue, executable); }),
            warmup_runs,
            repeats);
        add_program_size(result, executable.workload);
        add_scheduler(result, executable.buffers.scheduler);
        if (capture) {
            constexpr std::array<std::string_view, 4> names = {"LL", "LH", "HL", "HH"};
            for (size_t band = 0; band < names.size(); ++band) {
                write_binary(
                    output_prefix.string() + "." + std::string{names[band]} + ".f32",
                    read_2d(command_queue, executable.buffers.outputs[band], executable.plan.tiling.band, batch_count));
            }
        }
        return result;
    }

    if (transform != "ilwt_2d" || input_paths.size() != 4) {
        throw std::runtime_error("2D transform must be lwt_2d or ilwt_2d; ILWT requires four band paths");
    }
    const ttwv::Ilwt2DExecutionPlan host_plan =
        ttwv::make_ilwt_2d_execution_plan<Scheme>(height, width, core_limit, 768 * 1024, boundary_mode);
    std::array<std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>, 4> bands;
    for (size_t band = 0; band < bands.size(); ++band) {
        bands[band] = upload_2d(
            mesh_device,
            command_queue,
            read_binary(
                input_paths[band], host_plan.band_height * host_plan.band_width * static_cast<size_t>(batch_count)),
            host_plan.tiling.band,
            batch_count);
    }
    auto executable = ttwv::create_ilwt_2d_executable<Scheme>(
        TT_WAVELET_SOURCE_DIR,
        mesh_device,
        *bands[0]->get_backing_buffer(),
        *bands[1]->get_backing_buffer(),
        *bands[2]->get_backing_buffer(),
        *bands[3]->get_backing_buffer(),
        height,
        width,
        core_limit,
        boundary_mode,
        batch_count);
    ttwv::prepare_ilwt_2d(command_queue, executable);
    add_timing(
        result,
        measure(
            mesh_device,
            command_queue,
            warmup_runs,
            repeats,
            [&]() { ttwv::enqueue_ilwt_2d(command_queue, executable); }),
        warmup_runs,
        repeats);
    add_program_size(result, executable.workload);
    add_scheduler(result, executable.buffers.scheduler);
    result["physical_input_shape"] = {
        {host_plan.tiling.band.storage.height, host_plan.tiling.band.storage.width},
        {host_plan.tiling.band.storage.height, host_plan.tiling.band.storage.width},
        {host_plan.tiling.band.storage.height, host_plan.tiling.band.storage.width},
        {host_plan.tiling.band.storage.height, host_plan.tiling.band.storage.width}};
    result["logical_output_shape"] = {height, width};
    result["physical_output_shape"] = {
        executable.plan.tiling.input.storage.height, executable.plan.tiling.input.storage.width};
    if (capture) {
        write_binary(
            output_prefix.string() + ".reconstructed.f32",
            read_2d(command_queue, executable.buffers.outputs[0], executable.plan.tiling.input, batch_count));
    }
    return result;
}

[[nodiscard]] std::string classify_error(const std::string& message) {
    if (message.find("Program size") != std::string::npos || message.find("compile") != std::string::npos ||
        message.find("kernel config") != std::string::npos) {
        return "compile_failure";
    }
    if (message.find("profiler") != std::string::npos || message.find("timing") != std::string::npos) {
        return "timing_failure";
    }
    return "runtime_failure";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: tt_wavelet_benchmark_runner REQUESTS.jsonl|-\n";
        return EXIT_FAILURE;
    }
    try {
        setenv("TT_LOGGER_LEVEL", "FATAL", 0);
        unsetenv("TT_METAL_SLOW_DISPATCH_MODE");
        ttwv::benchmark::configure_device_profiler_environment();
        std::ifstream request_file;
        std::istream* requests = &std::cin;
        if (std::string_view{argv[1]} != "-") {
            request_file.open(argv[1]);
            if (!request_file.good()) {
                throw std::runtime_error("Failed to open request file");
            }
            requests = &request_file;
        }
        auto mesh_device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);
        mesh_device->enable_program_cache();
        auto& command_queue = mesh_device->mesh_command_queue();

        std::string line;
        size_t line_number = 0;
        while (std::getline(*requests, line)) {
            ++line_number;
            if (line.empty()) {
                continue;
            }
            Json request;
            Json response;
            try {
                request = Json::parse(line);
                response["case_id"] = request.at("case_id");
                const std::string wavelet = request.at("wavelet").get<std::string>();
                const std::string mode_name = request.at("boundary_mode").get<std::string>();
                ttwv::BoundaryMode boundary_mode{};
                if (!ttwv::parse_boundary_mode(mode_name, boundary_mode)) {
                    throw std::runtime_error("Unsupported boundary mode: " + mode_name);
                }
                const uint32_t dimension = request.at("dimension").get<uint32_t>();
                const auto dispatch = [&]<typename Scheme>() {
                    return dimension == 1 ? run_1d<Scheme>(request, *mesh_device, command_queue, boundary_mode)
                                          : run_2d<Scheme>(request, *mesh_device, command_queue, boundary_mode);
                };
                Json payload = ttwv::dispatch_scheme(wavelet, dispatch);
                response.update(payload);
                response["status"] = "ok";
                response["backend"] = "tt-wavelet";
                response["device_lifecycle"] = "persistent-process-device-queue";
            } catch (const std::exception& error) {
                response["case_id"] = request.value("case_id", "line-" + std::to_string(line_number));
                response["status"] = "error";
                response["backend"] = "tt-wavelet";
                response["error_type"] = classify_error(error.what());
                response["error_message"] = error.what();
            }
            std::cout << response.dump() << '\n' << std::flush;
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        Json response{
            {"status", "fatal"},
            {"backend", "tt-wavelet"},
            {"error_type", classify_error(error.what())},
            {"error_message", error.what()},
        };
        std::cout << response.dump() << '\n';
        return EXIT_FAILURE;
    }
}
