#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt_wavelet/include/lifting/device.hpp"

namespace {

[[nodiscard]] std::filesystem::path default_scheme_path() {
    return std::filesystem::path(TT_WAVELET_SOURCE_DIR) / "../ttnn-wavelet/lifting_schemes/coif2.json";
}

[[nodiscard]] std::vector<float> default_signal() {
    constexpr size_t default_signal_length = 19;
    std::vector<float> signal(default_signal_length);
    for (size_t i = 0; i < default_signal_length; ++i) {
        signal[i] = static_cast<float>(i + 1);
    }
    return signal;
}

[[nodiscard]] std::vector<float> parse_signal_arg(const char* raw_signal) {
    std::vector<float> signal;
    std::string token;
    std::stringstream stream(raw_signal);
    while (std::getline(stream, token, ',')) {
        if (token.empty()) {
            continue;
        }

        size_t parsed_chars = 0;
        const float value = std::stof(token, &parsed_chars);
        if (parsed_chars != token.size()) {
            throw std::runtime_error("Failed to parse signal token: " + token);
        }
        signal.push_back(value);
    }

    if (signal.empty()) {
        throw std::runtime_error("Signal argument is empty.");
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
    const std::filesystem::path scheme_path = argc > 1 ? std::filesystem::path(argv[1]) : default_scheme_path();
    const std::vector<float> original_signal = argc > 2 ? parse_signal_arg(argv[2]) : default_signal();

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
    auto input_buffer =
        tt::tt_metal::distributed::MeshBuffer::create(input_replicated_config, input_local_config, mesh_device.get());
    input_desc.dram_address = input_buffer->get_backing_buffer()->address();

    const auto scheme = ttwv::load_runtime_lifting_scheme(scheme_path);
    auto bundle = ttwv::create_lifting_preprocess_program(
        TT_WAVELET_SOURCE_DIR, *mesh_device, core, *(input_buffer->get_backing_buffer()), input_desc, scheme);

    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(command_queue, input_buffer, original_signal, false);
    ttwv::run_preprocess(command_queue, *mesh_device, bundle);

    const auto active_streams =
        ttwv::execute_forward_lifting(TT_WAVELET_SOURCE_DIR, *mesh_device, command_queue, core, bundle);

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

    const size_t canonical_length = bundle.plan.output_length;

    const auto even_canonical =
        canonicalize_forward_output(device_even_result, bundle.plan.final_even_length, canonical_length);
    const auto odd_canonical =
        canonicalize_forward_output(device_odd_result, bundle.plan.final_odd_length, canonical_length);

    print_coeffs("input signal", original_signal, original_signal.size());
    print_coeffs("tt-wavelet approximation coefficients", even_canonical, canonical_length);
    print_coeffs("tt-wavelet detail coefficients", odd_canonical, canonical_length);

    std::cout << "Scheme: " << scheme_path << '\n';
    std::cout << "Steps executed: " << bundle.scheme.steps.size() << '\n';
    std::cout << "Even logical length: " << canonical_length << '\n';
    std::cout << "Odd logical length: " << canonical_length << '\n';
    return EXIT_SUCCESS;
}
