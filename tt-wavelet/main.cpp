#include <algorithm>
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
    ttwv::LwtMemoryMode memory_mode{ttwv::LwtMemoryMode::kConeStreamed};
    std::string wavelet_name;
    std::optional<std::filesystem::path> signal_file;
};

struct ForwardOutput {
    std::vector<float> approximation;
    std::vector<float> detail;
    size_t logical_length{0};
    double execution_time_ms{0.0};
};

struct TimedTransform {
    double execution_time_ms{0.0};
    ttwv::LiftingSchedulerTelemetry scheduler{};
};

[[nodiscard]] double percentile(const std::vector<double>& sorted_values, const double fraction) {
    const double position = fraction * static_cast<double>(sorted_values.size() - 1);
    const size_t lower = static_cast<size_t>(position);
    const size_t upper = std::min(lower + 1, sorted_values.size() - 1);
    const double weight = position - static_cast<double>(lower);
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight;
}

[[nodiscard]] std::string usage() {
    return "Usage:\n"
           "  lwt <scheme|scheme_path> <signal_file>\n"
           "  lwt --benchmark [--repeats N] [--warmup-runs N] "
           "[--memory-mode cone|resident] "
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
        } else if (arg == "--memory-mode") {
            const std::string mode = require_option_value(argc, argv, i, arg);
            if (mode == "cone") {
                options.memory_mode = ttwv::LwtMemoryMode::kConeStreamed;
            } else if (mode == "resident") {
                options.memory_mode = ttwv::LwtMemoryMode::kResidentSharded;
            } else {
                throw std::runtime_error("--memory-mode must be 'cone' or 'resident'.");
            }
        } else if (arg.rfind("--", 0) == 0) {
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

void configure_tt_runtime(const bool benchmark) {
    if (!benchmark) {
        setenv("TT_LOGGER_LEVEL", "error", 0);
        unsetenv("TT_METAL_SLOW_DISPATCH_MODE");
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

struct DeviceInput {
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> buffer;
    ttwv::SignalBuffer desc{};
};

[[nodiscard]] DeviceInput create_device_input(
    tt::tt_metal::distributed::MeshDevice& mesh_device, const std::vector<float>& signal) {
    ttwv::SignalBuffer input_desc{
        .dram_address = 0,
        .length = signal.size(),
        .stick_width = ttwv::kStickWidth,
        .element_size_bytes = sizeof(float),
    };
    const tt::tt_metal::distributed::DeviceLocalBufferConfig input_local_config{
        .page_size = input_desc.aligned_stick_bytes(),
        .buffer_type = tt::tt_metal::BufferType::DRAM,
    };
    const tt::tt_metal::distributed::ReplicatedBufferConfig input_replicated_config{
        .size = static_cast<uint64_t>(input_desc.physical_nbytes()),
    };
    auto input_buffer =
        tt::tt_metal::distributed::MeshBuffer::create(input_replicated_config, input_local_config, &mesh_device);
    input_desc.dram_address = input_buffer->get_backing_buffer()->address();
    return DeviceInput{.buffer = std::move(input_buffer), .desc = input_desc};
}

template <typename Scheme>
[[nodiscard]] ForwardOutput run_forward_once(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    const tt::tt_metal::Buffer& input_buffer,
    const ttwv::SignalBuffer& input_desc,
    const ttwv::LwtMemoryMode memory_mode,
    const bool read_outputs) {
    if (memory_mode == ttwv::LwtMemoryMode::kConeStreamed) {
        auto executable = ttwv::create_cone_streamed_lwt_executable<Scheme>(
            TT_WAVELET_SOURCE_DIR, mesh_device, input_buffer, input_desc);
        ttwv::prepare_cone_streamed_lwt(command_queue, executable);

        const auto execution_start = std::chrono::steady_clock::now();
        ttwv::execute_cone_streamed_lwt(mesh_device, command_queue, executable);
        const auto execution_stop = std::chrono::steady_clock::now();
        const std::chrono::duration<double, std::milli> execution_time_ms = execution_stop - execution_start;
        if (!read_outputs) {
            return ForwardOutput{.execution_time_ms = execution_time_ms.count()};
        }

        std::vector<float> device_even_result;
        std::vector<float> device_odd_result;
        tt::tt_metal::distributed::EnqueueReadMeshBuffer(
            command_queue, device_even_result, executable.buffers.final_even, true);
        tt::tt_metal::distributed::EnqueueReadMeshBuffer(
            command_queue, device_odd_result, executable.buffers.final_odd, true);
        const auto& plan = executable.plan.full_plan;
        const size_t canonical_length = plan.output_length;
        constexpr int canonical_start = static_cast<int>(Scheme::tap_size / 2);
        return ForwardOutput{
            .approximation = canonicalize_forward_output(
                device_even_result, plan.final_even_length, canonical_length, plan.final_even_shift, canonical_start),
            .detail = canonicalize_forward_output(
                device_odd_result, plan.final_odd_length, canonical_length, plan.final_odd_shift, canonical_start),
            .logical_length = canonical_length,
            .execution_time_ms = execution_time_ms.count(),
        };
    }

    auto executable =
        ttwv::create_resident_lwt_executable<Scheme>(TT_WAVELET_SOURCE_DIR, mesh_device, input_buffer, input_desc);
    ttwv::prepare_resident_lwt(command_queue, executable);

    const auto execution_start = std::chrono::steady_clock::now();
    ttwv::execute_resident_lwt(mesh_device, command_queue, executable);
    const auto execution_stop = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> execution_time_ms = execution_stop - execution_start;

    if (!read_outputs) {
        return ForwardOutput{.execution_time_ms = execution_time_ms.count()};
    }

    std::vector<float> device_even_result;
    std::vector<float> device_odd_result;
    tt::tt_metal::distributed::EnqueueReadMeshBuffer(
        command_queue, device_even_result, executable.buffers.final_even, true);
    tt::tt_metal::distributed::EnqueueReadMeshBuffer(
        command_queue, device_odd_result, executable.buffers.final_odd, true);

    const size_t canonical_length = executable.plan.output_length;
    constexpr int canonical_start = static_cast<int>(Scheme::tap_size / 2);

    return ForwardOutput{
        .approximation = canonicalize_forward_output(
            device_even_result,
            executable.plan.final_even_length,
            canonical_length,
            executable.plan.final_even_shift,
            canonical_start),
        .detail = canonicalize_forward_output(
            device_odd_result,
            executable.plan.final_odd_length,
            canonical_length,
            executable.plan.final_odd_shift,
            canonical_start),
        .logical_length = canonical_length,
        .execution_time_ms = execution_time_ms.count(),
    };
}

template <typename Scheme>
[[nodiscard]] TimedTransform time_transform_once(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    const tt::tt_metal::Buffer& input_buffer,
    const ttwv::SignalBuffer& input_desc,
    const ttwv::LwtMemoryMode memory_mode) {
    if (memory_mode == ttwv::LwtMemoryMode::kConeStreamed) {
        auto executable = ttwv::create_cone_streamed_lwt_executable<Scheme>(
            TT_WAVELET_SOURCE_DIR, mesh_device, input_buffer, input_desc);
        ttwv::prepare_cone_streamed_lwt(command_queue, executable);

        const auto start = std::chrono::steady_clock::now();
        ttwv::execute_cone_streamed_lwt(mesh_device, command_queue, executable);
        const auto stop = std::chrono::steady_clock::now();
        const std::chrono::duration<double, std::milli> elapsed_ms = stop - start;
        return TimedTransform{
            .execution_time_ms = elapsed_ms.count(),
            .scheduler = executable.buffers.scheduler,
        };
    }

    auto executable =
        ttwv::create_resident_lwt_executable<Scheme>(TT_WAVELET_SOURCE_DIR, mesh_device, input_buffer, input_desc);
    ttwv::prepare_resident_lwt(command_queue, executable);

    const auto start = std::chrono::steady_clock::now();
    ttwv::execute_resident_lwt(mesh_device, command_queue, executable);
    const auto stop = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> elapsed_ms = stop - start;
    return TimedTransform{
        .execution_time_ms = elapsed_ms.count(),
        .scheduler = executable.buffers.scheduler,
    };
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const CliOptions options = parse_cli(argc, argv);
        configure_tt_runtime(options.benchmark);

        const std::string wavelet_name = canonical_wavelet_name(options.wavelet_name);
        const std::vector<float> signal =
            options.generated_length.has_value()
                ? generate_signal(*options.generated_length, options.signal_start, options.signal_step)
                : read_signal_file(*options.signal_file);

        constexpr int device_id{0};
        auto mesh_device{tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id)};
        if (options.benchmark) {
            mesh_device->enable_program_cache();
        }
        tt::tt_metal::distributed::MeshCommandQueue& command_queue{mesh_device->mesh_command_queue()};

        DeviceInput input = create_device_input(*mesh_device, signal);
        tt::tt_metal::distributed::EnqueueWriteMeshBuffer(command_queue, input.buffer, signal, false);
        if (options.benchmark) {
            tt::tt_metal::distributed::Finish(command_queue);
        }

        return ttwv::dispatch_scheme(wavelet_name, [&]<typename Scheme>() -> int {
            if (options.benchmark) {
                for (uint32_t i = 0; i < options.warmup_runs; ++i) {
                    static_cast<void>(time_transform_once<Scheme>(
                        *mesh_device,
                        command_queue,
                        *(input.buffer->get_backing_buffer()),
                        input.desc,
                        options.memory_mode));
                }

                std::vector<double> times_ms;
                times_ms.reserve(options.repeats);
                ttwv::LiftingSchedulerTelemetry scheduler;
                for (uint32_t i = 0; i < options.repeats; ++i) {
                    TimedTransform sample = time_transform_once<Scheme>(
                        *mesh_device,
                        command_queue,
                        *(input.buffer->get_backing_buffer()),
                        input.desc,
                        options.memory_mode);
                    times_ms.push_back(sample.execution_time_ms);
                    scheduler = std::move(sample.scheduler);
                }

                const double mean_ms = std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / times_ms.size();
                const double min_ms = *std::min_element(times_ms.begin(), times_ms.end());
                std::vector<double> sorted_times = times_ms;
                std::sort(sorted_times.begin(), sorted_times.end());
                const double median_ms = percentile(sorted_times, 0.5);
                const double p10_ms = percentile(sorted_times, 0.1);
                const double p90_ms = percentile(sorted_times, 0.9);
                const double squared_deviation_sum = std::accumulate(
                    times_ms.begin(), times_ms.end(), 0.0, [mean_ms](const double sum, const double value) {
                        const double deviation = value - mean_ms;
                        return sum + deviation * deviation;
                    });
                const double stddev_ms = std::sqrt(squared_deviation_sum / times_ms.size());
                std::cerr << std::fixed << std::setprecision(6) << "lwt_execution_time_ms: " << mean_ms << '\n'
                          << "lwt_min_time_ms: " << min_ms << '\n'
                          << "lwt_median_time_ms: " << median_ms << '\n'
                          << "lwt_p10_time_ms: " << p10_ms << '\n'
                          << "lwt_p90_time_ms: " << p90_ms << '\n'
                          << "lwt_stddev_time_ms: " << stddev_ms << '\n'
                          << "lwt_repeats: " << options.repeats << '\n'
                          << "lwt_warmup_runs: " << options.warmup_runs << '\n'
                          << "lwt_memory_mode: "
                          << (scheduler.memory_mode == ttwv::LwtMemoryMode::kConeStreamed ? "cone" : "resident") << '\n'
                          << "lwt_max_group_count: " << scheduler.max_group_count << '\n'
                          << "lwt_groups_per_shard: " << scheduler.groups_per_shard << '\n'
                          << "lwt_active_core_count: " << scheduler.active_core_count << '\n';
                std::cerr << "lwt_shard_elements: " << scheduler.shard_elements << '\n'
                          << "lwt_chunk_count: " << scheduler.chunk_count << '\n'
                          << "lwt_groups_per_chunk: " << scheduler.groups_per_chunk << '\n'
                          << "lwt_workspace_elements: " << scheduler.workspace_elements << '\n'
                          << "lwt_max_dependency_overhead: " << scheduler.max_dependency_overhead << '\n'
                          << "lwt_zero_work_cores_per_route:";
                for (const uint32_t count : scheduler.zero_work_cores_per_route) {
                    std::cerr << ' ' << count;
                }
                std::cerr << '\n';
                return EXIT_SUCCESS;
            }

            const ForwardOutput output = run_forward_once<Scheme>(
                *mesh_device,
                command_queue,
                *(input.buffer->get_backing_buffer()),
                input.desc,
                options.memory_mode,
                true);

            print_coeffs("tt-wavelet approximation coefficients", output.approximation, output.logical_length);
            print_coeffs("tt-wavelet detail coefficients", output.detail, output.logical_length);

            std::cerr << "lwt_execution_time_ms: " << std::fixed << std::setprecision(6) << output.execution_time_ms
                      << '\n';
            return EXIT_SUCCESS;
        });
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << '\n';
        return EXIT_FAILURE;
    }
}
