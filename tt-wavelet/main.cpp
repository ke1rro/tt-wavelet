#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt_wavelet/include/lifting/device.hpp"
#include "tt_wavelet/include/schemes/generated/registry.hpp"

namespace {

struct CliOptions {
    bool benchmark{false};
    uint32_t repeats{1};
    uint32_t warmup_runs{1};
    std::optional<size_t> generated_length;
    double signal_start{1.0};
    double signal_step{1.0};
    std::string wavelet_name;
    std::optional<std::filesystem::path> signal_file;
};

struct ForwardOutput {
    std::vector<float> approximation;
    std::vector<float> detail;
    size_t logical_length{0};
};

[[nodiscard]] std::string usage() {
    return "Usage:\n"
           "  lwt <scheme|scheme_path> <signal_file>\n"
           "  lwt --benchmark [--repeats N] [--warmup-runs N] "
           "[--length N --signal-start X --signal-step X | <signal_file>] <scheme|scheme_path>";
}

[[nodiscard]] std::string require_option_value(const int argc, char** argv, int& index, const std::string& option) {
    if (index + 1 >= argc) {
        throw std::runtime_error(option + " requires a value.\n" + usage());
    }
    ++index;
    return argv[index];
}

[[nodiscard]] uint32_t parse_u32(const std::string& raw, const char* label, const bool allow_zero) {
    size_t parsed_chars = 0;
    const unsigned long long value = std::stoull(raw, &parsed_chars, 10);
    if (parsed_chars != raw.size() || value > std::numeric_limits<uint32_t>::max() || (!allow_zero && value == 0)) {
        throw std::runtime_error(std::string(label) + " must be a valid uint32 value.");
    }
    return static_cast<uint32_t>(value);
}

[[nodiscard]] size_t parse_size(const std::string& raw, const char* label) {
    size_t parsed_chars = 0;
    const unsigned long long value = std::stoull(raw, &parsed_chars, 10);
    if (parsed_chars != raw.size() || value == 0 ||
        value > static_cast<unsigned long long>(std::numeric_limits<size_t>::max())) {
        throw std::runtime_error(std::string(label) + " must be a positive size value.");
    }
    return static_cast<size_t>(value);
}

[[nodiscard]] double parse_double(const std::string& raw, const char* label) {
    size_t parsed_chars = 0;
    const double value = std::stod(raw, &parsed_chars);
    if (parsed_chars != raw.size() || !std::isfinite(value)) {
        throw std::runtime_error(std::string(label) + " must be a finite floating-point value.");
    }
    return value;
}

[[nodiscard]] CliOptions parse_cli(const int argc, char** argv) {
    CliOptions options;
    std::vector<std::string> positionals;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            std::cout << usage() << '\n';
            std::exit(EXIT_SUCCESS);
        }
        if (arg == "--benchmark") {
            options.benchmark = true;
        } else if (arg == "--repeats") {
            options.repeats = parse_u32(require_option_value(argc, argv, i, arg), "--repeats", false);
        } else if (arg == "--warmup-runs") {
            options.warmup_runs = parse_u32(require_option_value(argc, argv, i, arg), "--warmup-runs", true);
        } else if (arg == "--length") {
            options.generated_length = parse_size(require_option_value(argc, argv, i, arg), "--length");
        } else if (arg == "--signal-start") {
            options.signal_start = parse_double(require_option_value(argc, argv, i, arg), "--signal-start");
        } else if (arg == "--signal-step") {
            options.signal_step = parse_double(require_option_value(argc, argv, i, arg), "--signal-step");
        } else if (arg.rfind("--", 0) == 0) {  // NOLINT
            throw std::runtime_error("Unknown option: " + arg + "\n" + usage());
        } else {
            positionals.push_back(arg);
        }
    }

    if (!options.benchmark && options.generated_length.has_value()) {
        throw std::runtime_error("--length requires --benchmark.\n" + usage());
    }

    if (options.benchmark && options.generated_length.has_value()) {
        if (positionals.size() != 1) {
            throw std::runtime_error(usage());
        }
        options.wavelet_name = positionals.at(0);
    } else {
        if (positionals.size() != 2) {
            throw std::runtime_error(usage());
        }
        options.wavelet_name = positionals.at(0);
        options.signal_file = std::filesystem::path{positionals.at(1)};
    }

    return options;
}

void configure_tt_runtime(const bool benchmark) {
    if (!benchmark) {
        setenv("TT_LOGGER_LEVEL", "error", 0);
        return;
    }

    setenv("TT_LOGGER_LEVEL", "FATAL", 1);
    setenv("TT_METAL_INSPECTOR_RPC", "0", 1);
    unsetenv("TT_METAL_DPRINT_CORES");
    unsetenv("TT_METAL_WATCHER");
    unsetenv("TT_METAL_SLOW_DISPATCH_MODE");
    unsetenv("TT_METAL_DEVICE_PROFILER");
    unsetenv("TT_METAL_DEVICE_PROFILER_DISPATCH");
    unsetenv("TT_METAL_DISPATCH_DATA_COLLECTION");
    unsetenv("TTNN_CONFIG_OVERRIDES");
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

[[nodiscard]] std::vector<float> generate_signal(const size_t length, const double start, const double step) {
    std::vector<float> signal;
    signal.reserve(length);
    for (size_t i = 0; i < length; ++i) {
        signal.push_back(static_cast<float>(start + step * static_cast<double>(i)));
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
    const std::vector<float>& values,
    const size_t logical_length,
    const size_t output_length,
    const int stream_shift,
    const int canonical_start) {
    const size_t available_logical = std::min(values.size(), logical_length);
    std::vector<float> logical_values(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(available_logical));

    std::vector<float> canonical(output_length, 0.0F);
    const int64_t src_offset = static_cast<int64_t>(canonical_start) - static_cast<int64_t>(stream_shift);
    const size_t src_begin = src_offset > 0 ? static_cast<size_t>(src_offset) : size_t{0};
    const size_t dst_begin = src_offset < 0 ? static_cast<size_t>(-src_offset) : size_t{0};

    if (src_begin >= logical_values.size() || dst_begin >= canonical.size()) {
        return canonical;
    }

    const size_t copy_count = std::min(logical_values.size() - src_begin, canonical.size() - dst_begin);
    std::copy_n(
        logical_values.begin() + static_cast<std::ptrdiff_t>(src_begin), copy_count, canonical.begin() + dst_begin);
    return canonical;
}

[[nodiscard]] std::string canonical_wavelet_name(const std::string_view raw_name) {
    const std::filesystem::path raw_path{std::string{raw_name}};
    if (raw_path.extension() == ".json") {
        return raw_path.stem().string();
    }
    return std::string{raw_name};
}

template <typename Scheme>
[[nodiscard]] ForwardOutput run_forward_once(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    const tt::tt_metal::CoreCoord& core,
    const tt::tt_metal::Buffer& input_buffer,
    const ttwv::SignalBuffer& input_desc,
    const bool read_outputs) {
    auto bundle = ttwv::create_lifting_preprocess_program<Scheme>(
        TT_WAVELET_SOURCE_DIR, mesh_device, core, input_buffer, input_desc);

    ttwv::run_preprocess(command_queue, mesh_device, bundle);
    const auto active_streams = ttwv::lwt<Scheme>(TT_WAVELET_SOURCE_DIR, mesh_device, command_queue, bundle);

    if (!read_outputs) {
        return {};
    }

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
    constexpr int canonical_start = static_cast<int>(Scheme::tap_size / 2);

    return ForwardOutput{
        .approximation = canonicalize_forward_output(
            device_even_result,
            bundle.plan.final_even_length,
            canonical_length,
            bundle.plan.final_even_shift,
            canonical_start),
        .detail = canonicalize_forward_output(
            device_odd_result,
            bundle.plan.final_odd_length,
            canonical_length,
            bundle.plan.final_odd_shift,
            canonical_start),
        .logical_length = canonical_length,
    };
}

template <typename Scheme>
[[nodiscard]] double time_lwt_once_ms(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    const tt::tt_metal::CoreCoord& core,
    const tt::tt_metal::Buffer& input_buffer,
    const ttwv::SignalBuffer& input_desc) {
    auto bundle = ttwv::create_lifting_preprocess_program<Scheme>(
        TT_WAVELET_SOURCE_DIR, mesh_device, core, input_buffer, input_desc);
    ttwv::run_preprocess(command_queue, mesh_device, bundle);

    const auto start = std::chrono::steady_clock::now();
    static_cast<void>(ttwv::lwt<Scheme>(TT_WAVELET_SOURCE_DIR, mesh_device, command_queue, bundle));
    const auto stop = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> elapsed_ms = stop - start;
    return elapsed_ms.count();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const CliOptions options = parse_cli(argc, argv);
        configure_tt_runtime(options.benchmark);

        const std::string wavelet_name = canonical_wavelet_name(options.wavelet_name);
        const std::vector<float> original_signal =
            options.generated_length.has_value()
                ? generate_signal(*options.generated_length, options.signal_start, options.signal_step)
                : read_signal_file(*options.signal_file);
        const auto start = std::chrono::steady_clock::now();

        constexpr int device_id = 0;
        constexpr tt::tt_metal::CoreCoord core{0, 0};
        const size_t signal_length = original_signal.size();

        auto mesh_device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
        if (options.benchmark) {
            mesh_device->enable_program_cache();
        }
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

        tt::tt_metal::distributed::EnqueueWriteMeshBuffer(command_queue, input_buffer, original_signal, false);
        if (options.benchmark) {
            tt::tt_metal::distributed::Finish(command_queue);
        }

        return ttwv::dispatch_scheme(wavelet_name, [&]<typename Scheme>() -> int {
            if (options.benchmark) {
                for (uint32_t i = 0; i < options.warmup_runs; ++i) {
                    static_cast<void>(time_lwt_once_ms<Scheme>(
                        *mesh_device, command_queue, core, *(input_buffer->get_backing_buffer()), input_desc));
                }

                std::vector<double> times_ms;
                times_ms.reserve(options.repeats);
                for (uint32_t i = 0; i < options.repeats; ++i) {
                    times_ms.push_back(
                        time_lwt_once_ms<Scheme>(
                            *mesh_device, command_queue, core, *(input_buffer->get_backing_buffer()), input_desc));
                }

                const double mean_ms = std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / times_ms.size();
                const double min_ms = *std::min_element(times_ms.begin(), times_ms.end());
                std::cerr << std::fixed << std::setprecision(6) << "lwt_execution_time_ms: " << mean_ms << '\n'
                          << "lwt_min_time_ms: " << min_ms << '\n'
                          << "lwt_repeats: " << options.repeats << '\n'
                          << "lwt_warmup_runs: " << options.warmup_runs << '\n';
                return EXIT_SUCCESS;
            }

            const ForwardOutput output = run_forward_once<Scheme>(
                *mesh_device, command_queue, core, *(input_buffer->get_backing_buffer()), input_desc, true);
            const auto stop = std::chrono::steady_clock::now();

            print_coeffs("tt-wavelet approximation coefficients", output.approximation, output.logical_length);
            print_coeffs("tt-wavelet detail coefficients", output.detail, output.logical_length);
            std::cout << std::flush;

            const std::chrono::duration<double, std::milli> elapsed_ms = stop - start;
            std::cerr << "lwt_execution_time_ms: " << std::fixed << std::setprecision(6) << elapsed_ms.count() << '\n';
            return EXIT_SUCCESS;
        });
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << '\n';
        return EXIT_FAILURE;
    }
}
