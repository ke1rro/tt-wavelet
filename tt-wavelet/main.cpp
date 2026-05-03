#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
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

[[nodiscard]] std::vector<float> generate_signal(const size_t length, const float start, const float step) {
    if (length == 0) {
        throw std::runtime_error("Signal length must be positive.");
    }

    std::vector<float> signal(length);
    for (size_t i = 0; i < length; ++i) {
        signal[i] = start + static_cast<float>(i) * step;
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

struct CliOptions {
    bool quiet = false;
    std::optional<std::filesystem::path> scheme_path;
    std::optional<std::string> raw_signal;
    std::optional<size_t> signal_length;
    float signal_start = 1.0f;
    float signal_step = 1.0f;
    bool signal_start_set = false;
    bool signal_step_set = false;
};

[[noreturn]] void print_usage(const char* program, const int exit_code) {
    std::cout << "Usage: " << program << " [options] [scheme_path] [signal_csv]\n"
              << "\nOptions:\n"
              << "  --quiet            Skip printing coefficient lists and summary.\n"
              << "  --signal-length N  Generate N samples instead of parsing signal_csv.\n"
              << "  --signal-start V   Start value for generated signal (default: 1).\n"
              << "  --signal-step V    Step value for generated signal (default: 1).\n"
              << "  -h, --help         Show this help message.\n";
    std::exit(exit_code);
}

[[nodiscard]] CliOptions parse_cli(const int argc, char** argv) {
    CliOptions options;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--quiet") {
            options.quiet = true;
            continue;
        }

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0], EXIT_SUCCESS);
        }

        if (arg == "--signal-length") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--signal-length expects a value.");
            }
            const std::string value = argv[++i];
            size_t parsed_chars = 0;
            const size_t length = std::stoull(value, &parsed_chars);
            if (parsed_chars != value.size()) {
                throw std::runtime_error("Invalid --signal-length value: " + value);
            }
            options.signal_length = length;
            continue;
        }

        if (arg == "--signal-start") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--signal-start expects a value.");
            }
            const std::string value = argv[++i];
            size_t parsed_chars = 0;
            const float parsed = std::stof(value, &parsed_chars);
            if (parsed_chars != value.size()) {
                throw std::runtime_error("Invalid --signal-start value: " + value);
            }
            options.signal_start = parsed;
            options.signal_start_set = true;
            continue;
        }

        if (arg == "--signal-step") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--signal-step expects a value.");
            }
            const std::string value = argv[++i];
            size_t parsed_chars = 0;
            const float parsed = std::stof(value, &parsed_chars);
            if (parsed_chars != value.size()) {
                throw std::runtime_error("Invalid --signal-step value: " + value);
            }
            options.signal_step = parsed;
            options.signal_step_set = true;
            continue;
        }

        if (!arg.empty() && arg[0] == '-') {
            throw std::runtime_error("Unknown option: " + arg);
        }

        if (!options.scheme_path) {
            options.scheme_path = std::filesystem::path(arg);
            continue;
        }

        if (!options.raw_signal) {
            options.raw_signal = arg;
            continue;
        }

        throw std::runtime_error("Unexpected argument: " + arg);
    }

    if (!options.signal_length && (options.signal_start_set || options.signal_step_set)) {
        throw std::runtime_error("--signal-start/--signal-step require --signal-length.");
    }

    return options;
}

}  // namespace

int main(int argc, char** argv) {
    const CliOptions options = parse_cli(argc, argv);
    const std::filesystem::path scheme_path = options.scheme_path.value_or(default_scheme_path());
    const std::vector<float> original_signal = [&options]() {
        if (options.signal_length) {
            return generate_signal(*options.signal_length, options.signal_start, options.signal_step);
        }
        if (options.raw_signal) {
            return parse_signal_arg(options.raw_signal->c_str());
        }
        return default_signal();
    }();

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

    if (!options.quiet) {
        print_coeffs("input signal", original_signal, original_signal.size());
        print_coeffs("tt-wavelet approximation coefficients", even_canonical, canonical_length);
        print_coeffs("tt-wavelet detail coefficients", odd_canonical, canonical_length);

        std::cout << "Scheme: " << scheme_path << '\n';
        std::cout << "Steps executed: " << bundle.scheme.steps.size() << '\n';
        std::cout << "Even logical length: " << canonical_length << '\n';
        std::cout << "Odd logical length: " << canonical_length << '\n';
    }
    return EXIT_SUCCESS;
}
