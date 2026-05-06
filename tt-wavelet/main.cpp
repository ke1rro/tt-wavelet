#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt_wavelet/include/lifting/device.hpp"

namespace {

[[nodiscard]] std::filesystem::path resolve_scheme_path(const std::string_view scheme) {
    const std::string scheme_name{scheme};
    std::filesystem::path raw_path{scheme_name};
    if (std::filesystem::exists(raw_path)) {
        return raw_path;
    }

    const std::filesystem::path schemes_dir =
        std::filesystem::path(TT_WAVELET_SOURCE_DIR) / "../ttnn-wavelet/lifting_schemes";
    return schemes_dir / (scheme_name.ends_with(".json") ? scheme_name : scheme_name + ".json");
}

[[nodiscard]] std::vector<float> read_signal_file(const std::filesystem::path& path) {
    std::ifstream handle(path);
    if (!handle.good()) {
        throw std::runtime_error("Failed to open signal file: " + path.string());
    }

    std::vector<float> signal;
    for (float value = 0.0F; handle >> value;) {
        signal.push_back(value);
    }

    if (signal.empty()) {
        throw std::runtime_error("Signal file is empty: " + path.string());
    }
    if (!handle.eof()) {
        throw std::runtime_error("Signal file contains a non-numeric token: " + path.string());
    }

    return signal;
}

void print_coeffs(const char* label, const std::vector<float>& values, const size_t logical_length) {
    std::cout << label << " (" << logical_length << "): [";
    for (size_t i = 0; i < logical_length; ++i) {
        if (i > 0) {
            std::cout << ", ";
        }

        std::cout << std::scientific << std::setprecision(8) << static_cast<double>(values[i]);
    }
    std::cout << std::defaultfloat << "]\n";
}

[[nodiscard]] std::vector<float> canonicalize_forward_output(
    const std::vector<float>& values, const size_t logical_length, const size_t output_length) {
    const size_t available_logical = std::min(values.size(), logical_length);
    std::vector<float> logical_values(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(available_logical));

    if (logical_values.size() <= output_length) {
        return logical_values;
    }

    const size_t extra = logical_values.size() - output_length;
    const size_t crop_offset = (extra + 1) / 2;
    return std::vector<float>(
        logical_values.begin() + static_cast<std::ptrdiff_t>(crop_offset),
        logical_values.begin() + static_cast<std::ptrdiff_t>(crop_offset + output_length));
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            throw std::runtime_error("Usage: lwt <scheme|scheme_path> <signal_file>");
        }
        setenv("TT_LOGGER_LEVEL", "error", 0);

        const std::filesystem::path scheme_path = resolve_scheme_path(argv[1]);
        const std::vector<float> original_signal = read_signal_file(argv[2]);
        const auto scheme = ttwv::load_runtime_lifting_scheme(scheme_path);
        const auto start = std::chrono::steady_clock::now();

        constexpr int device_id = 0;
        constexpr tt::tt_metal::CoreCoord core{0, 0};
        const size_t signal_length = original_signal.size();

        auto mesh_device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
        tt::tt_metal::distributed::MeshCommandQueue& command_queue = mesh_device->mesh_command_queue();

        ttwv::SignalBuffer input_desc{
            .dram_address = 0,
            .length = signal_length,
            .stick_width = 32,
            .element_size_bytes = sizeof(float),
        };

        const tt::tt_metal::distributed::DeviceLocalBufferConfig input_local_config{
            .page_size = input_desc.aligned_stick_bytes(32),
            .buffer_type = tt::tt_metal::BufferType::DRAM,
        };
        const tt::tt_metal::distributed::ReplicatedBufferConfig input_replicated_config{
            .size = static_cast<uint64_t>(input_desc.physical_nbytes()),
        };
        auto input_buffer = tt::tt_metal::distributed::MeshBuffer::create(
            input_replicated_config, input_local_config, mesh_device.get());
        input_desc.dram_address = input_buffer->get_backing_buffer()->address();

        auto bundle = ttwv::create_lifting_preprocess_program(
            TT_WAVELET_SOURCE_DIR, *mesh_device, core, *(input_buffer->get_backing_buffer()), input_desc, scheme);

        tt::tt_metal::distributed::EnqueueWriteMeshBuffer(command_queue, input_buffer, original_signal, false);
        ttwv::run_preprocess(command_queue, *mesh_device, bundle);

        const auto active_streams = ttwv::lwt(TT_WAVELET_SOURCE_DIR, *mesh_device, command_queue, core, bundle);

        std::vector<float> device_even_result;
        std::vector<float> device_odd_result;

        auto even_buffer = active_streams.even.family == ttwv::LogicalStream::kEven
                               ? bundle.buffers.even.at(active_streams.even.slot)
                               : bundle.buffers.odd.at(active_streams.even.slot);
        auto odd_buffer = active_streams.odd.family == ttwv::LogicalStream::kEven
                              ? bundle.buffers.even.at(active_streams.odd.slot)
                              : bundle.buffers.odd.at(active_streams.odd.slot);
        tt::tt_metal::distributed::EnqueueReadMeshBuffer(command_queue, device_even_result, even_buffer, true);
        tt::tt_metal::distributed::EnqueueReadMeshBuffer(command_queue, device_odd_result, odd_buffer, true);
        const auto stop = std::chrono::steady_clock::now();

        const size_t canonical_length = bundle.plan.output_length;

        const auto even_canonical =
            canonicalize_forward_output(device_even_result, bundle.plan.final_even_length, canonical_length);
        const auto odd_canonical =
            canonicalize_forward_output(device_odd_result, bundle.plan.final_odd_length, canonical_length);

        print_coeffs("tt-wavelet approximation coefficients", even_canonical, canonical_length);
        print_coeffs("tt-wavelet detail coefficients", odd_canonical, canonical_length);
        std::cout << std::flush;

        const std::chrono::duration<double, std::milli> elapsed_ms = stop - start;
        std::cerr << "lwt_execution_time_ms: " << std::fixed << std::setprecision(6) << elapsed_ms.count() << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << '\n';
        return EXIT_FAILURE;
    }
}
