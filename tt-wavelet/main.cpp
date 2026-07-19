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
#include "tt_wavelet/include/schemes/testing/synthetic_k17.hpp"

namespace {

struct CliOptions {
    bool benchmark{false};
    bool inverse{false};
    uint32_t repeats{1};
    uint32_t warmup_runs{1};
    std::optional<size_t> generated_length;
    std::optional<size_t> original_length;
    double signal_start{1.0};
    double signal_step{1.0};
    ttwv::BoundaryMode boundary_mode{ttwv::BoundaryMode::kSymmetric};
    std::string wavelet_name;
    std::optional<std::filesystem::path> signal_file;
    std::optional<std::filesystem::path> approximation_file;
    std::optional<std::filesystem::path> detail_file;
    std::optional<std::filesystem::path> output_prefix;
};

struct ForwardOutput {
    std::vector<float> approximation;
    std::vector<float> detail;
    size_t logical_length{0};
    double execution_time_ms{0.0};
    ttwv::LiftingSchedulerTelemetry scheduler{};
};

struct TimedTransform {
    double execution_time_ms{0.0};
    ttwv::LiftingSchedulerTelemetry scheduler{};
};

struct InverseOutput {
    std::vector<float> signal;
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
           "  lwt [--inverse] <scheme|scheme_path> <signal_file>\n"
           "  lwt [--inverse] --length N [--signal-start X --signal-step X] <scheme|scheme_path>\n"
           "  lwt --inverse --original-length N --approximation-file PATH --detail-file PATH "
           "<scheme|scheme_path>\n"
           "  lwt [--inverse] --benchmark [--repeats N] [--warmup-runs N] "
           "[--output-prefix PATH] "
           "[--boundary-mode symmetric|zero|constant|periodic|antisymmetric|smooth|reflect|antireflect] "
           "[--length N --signal-start X --signal-step X | <signal_file>] <scheme|scheme_path>\n"
           "\n"
           "  --inverse  Run 1D ILWT from coefficients produced by an untimed forward transform;\n"
           "             non-benchmark mode prints the reconstructed signal.";
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
        } else if (arg == "--inverse") {
            options.inverse = true;
        } else if (arg == "--repeats") {
            options.repeats = parse_u32(require_option_value(argc, argv, i, arg), "--repeats", false);
        } else if (arg == "--warmup-runs") {
            options.warmup_runs = parse_u32(require_option_value(argc, argv, i, arg), "--warmup-runs", true);
        } else if (arg == "--length") {
            options.generated_length = parse_size(require_option_value(argc, argv, i, arg), "--length");
        } else if (arg == "--original-length") {
            options.original_length = parse_size(require_option_value(argc, argv, i, arg), "--original-length");
        } else if (arg == "--approximation-file") {
            options.approximation_file = require_option_value(argc, argv, i, arg);
        } else if (arg == "--detail-file") {
            options.detail_file = require_option_value(argc, argv, i, arg);
        } else if (arg == "--output-prefix") {
            options.output_prefix = require_option_value(argc, argv, i, arg);
        } else if (arg == "--signal-start") {
            options.signal_start = parse_double(require_option_value(argc, argv, i, arg), "--signal-start");
        } else if (arg == "--signal-step") {
            options.signal_step = parse_double(require_option_value(argc, argv, i, arg), "--signal-step");
        } else if (arg == "--boundary-mode") {
            const std::string mode = require_option_value(argc, argv, i, arg);
            if (mode == "symmetric") {
                options.boundary_mode = ttwv::BoundaryMode::kSymmetric;
            } else if (mode == "zero") {
                options.boundary_mode = ttwv::BoundaryMode::kZero;
            } else if (mode == "constant") {
                options.boundary_mode = ttwv::BoundaryMode::kConstant;
            } else if (mode == "periodic") {
                options.boundary_mode = ttwv::BoundaryMode::kPeriodic;
            } else if (mode == "antisymmetric") {
                options.boundary_mode = ttwv::BoundaryMode::kAntisymmetric;
            } else if (mode == "smooth") {
                options.boundary_mode = ttwv::BoundaryMode::kSmooth;
            } else if (mode == "reflect") {
                options.boundary_mode = ttwv::BoundaryMode::kReflect;
            } else if (mode == "antireflect") {
                options.boundary_mode = ttwv::BoundaryMode::kAntireflect;
            } else {
                throw std::runtime_error(
                    "--boundary-mode must be 'symmetric', 'zero', 'constant', 'periodic', "
                    "'antisymmetric', 'smooth', 'reflect', or 'antireflect'.");
            }
        } else if (arg.starts_with("--")) {
            throw std::runtime_error("Unknown option: " + arg + "\n" + usage());
        } else {
            positionals.push_back(arg);
        }
    }

    const bool any_coefficient_input = options.original_length.has_value() || options.approximation_file.has_value() ||
                                       options.detail_file.has_value();
    if (any_coefficient_input) {
        if (!options.inverse || !options.original_length.has_value() || !options.approximation_file.has_value() ||
            !options.detail_file.has_value() || options.generated_length.has_value() || positionals.size() != 1) {
            throw std::runtime_error(
                "External ILWT input requires --inverse, --original-length, --approximation-file, "
                "--detail-file, and one scheme name.\n" +
                usage());
        }
        options.wavelet_name = positionals.at(0);
    } else if (options.generated_length.has_value()) {
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

[[nodiscard]] std::filesystem::path output_path(const std::filesystem::path& prefix, const std::string_view suffix) {
    return std::filesystem::path{prefix.string() + std::string{suffix}};
}

void write_fp32_file(const std::filesystem::path& path, const std::vector<float>& values) {
    std::ofstream handle(path, std::ios::binary | std::ios::trunc);
    if (!handle.good()) {
        throw std::runtime_error("Failed to open FP32 output file: " + path.string());
    }
    handle.write(
        reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!handle.good()) {
        throw std::runtime_error("Failed to write FP32 output file: " + path.string());
    }
}

void write_forward_outputs(const std::filesystem::path& prefix, const ForwardOutput& output) {
    write_fp32_file(output_path(prefix, ".approximation.f32"), output.approximation);
    write_fp32_file(output_path(prefix, ".detail.f32"), output.detail);
}

void write_inverse_output(const std::filesystem::path& prefix, const InverseOutput& output) {
    write_fp32_file(output_path(prefix, ".reconstructed.f32"), output.signal);
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
    const ttwv::BoundaryMode boundary_mode,
    const bool read_outputs) {
    auto executable = ttwv::create_lwt_executable<Scheme>(
        TT_WAVELET_SOURCE_DIR, mesh_device, input_buffer, input_desc, boundary_mode);
    ttwv::prepare_lwt(command_queue, executable);

    const auto execution_start = std::chrono::steady_clock::now();
    ttwv::execute_lwt(mesh_device, command_queue, executable);
    const auto execution_stop = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> execution_time_ms = execution_stop - execution_start;

    if (!read_outputs) {
        return ForwardOutput{
            .execution_time_ms = execution_time_ms.count(),
            .scheduler = executable.buffers.scheduler,
        };
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
            device_even_result,
            plan.final_even_length,
            canonical_length,
            plan.final_even_shift,
            canonical_start),
        .detail = canonicalize_forward_output(
            device_odd_result,
            plan.final_odd_length,
            canonical_length,
            plan.final_odd_shift,
            canonical_start),
        .logical_length = canonical_length,
        .execution_time_ms = execution_time_ms.count(),
        .scheduler = executable.buffers.scheduler,
    };
}

template <typename Scheme>
[[nodiscard]] TimedTransform time_transform_once(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    const tt::tt_metal::Buffer& input_buffer,
    const ttwv::SignalBuffer& input_desc,
    const ttwv::BoundaryMode boundary_mode) {
    auto executable = ttwv::create_lwt_executable<Scheme>(
        TT_WAVELET_SOURCE_DIR, mesh_device, input_buffer, input_desc, boundary_mode);
    ttwv::prepare_lwt(command_queue, executable);

    const auto start = std::chrono::steady_clock::now();
    ttwv::execute_lwt(mesh_device, command_queue, executable);
    const auto stop = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> elapsed_ms = stop - start;
    return TimedTransform{
        .execution_time_ms = elapsed_ms.count(),
        .scheduler = executable.buffers.scheduler,
    };
}

template <typename Scheme>
[[nodiscard]] InverseOutput run_inverse_once(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshCommandQueue& command_queue,
    const tt::tt_metal::Buffer& approximation_buffer,
    const tt::tt_metal::Buffer& detail_buffer,
    const size_t coefficient_length,
    const size_t original_length,
    const ttwv::BoundaryMode boundary_mode,
    const bool read_output) {
    auto executable = ttwv::create_ilwt_executable<Scheme>(
        TT_WAVELET_SOURCE_DIR,
        mesh_device,
        approximation_buffer,
        detail_buffer,
        coefficient_length,
        original_length,
        boundary_mode);
    ttwv::prepare_ilwt(command_queue, executable);

    const auto execution_start = std::chrono::steady_clock::now();
    ttwv::execute_ilwt(mesh_device, command_queue, executable);
    const auto execution_stop = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> execution_time_ms = execution_stop - execution_start;

    InverseOutput output{
        .execution_time_ms = execution_time_ms.count(),
        .scheduler = executable.buffers.scheduler,
    };
    if (read_output) {
        tt::tt_metal::distributed::EnqueueReadMeshBuffer(command_queue, output.signal, executable.buffers.output, true);
        output.signal.resize(original_length);
    }
    return output;
}

void print_scheduler_telemetry(const char* prefix, const ttwv::LiftingSchedulerTelemetry& scheduler) {
    std::cerr << prefix << "_architecture: " << tt::arch_to_str(scheduler.architecture) << '\n'
              << prefix << "_layout: "
              << (scheduler.workspace_layout == ttwv::WorkspaceLayout::kTileNative ? "tile-native" : "row-major")
              << '\n'
              << prefix << "_signal_length: " << scheduler.signal_length << '\n'
              << prefix << "_max_group_count: " << scheduler.max_group_count << '\n'
              << prefix << "_active_core_count: " << scheduler.active_core_count << '\n'
              << prefix << "_chunk_count: " << scheduler.chunk_count << '\n'
              << prefix << "_groups_per_chunk: " << scheduler.groups_per_chunk << '\n'
              << prefix << "_workspace_elements: " << scheduler.workspace_elements << '\n'
              << prefix << "_max_workspace_elements: " << scheduler.max_workspace_elements << '\n'
              << prefix << "_max_dependency_overhead: " << scheduler.max_dependency_overhead << '\n'
              << prefix << "_terminal_scale_inline: " << (scheduler.terminal_scale_inline ? 1 : 0) << '\n'
              << prefix << "_inverse_scale_inline: " << (scheduler.inverse_scale_inline ? 1 : 0) << '\n'
              << prefix << "_inverse_final_interleave_direct: " << (scheduler.inverse_final_interleave_direct ? 1 : 0)
              << '\n'
              << prefix << "_l1_slots_bytes: " << scheduler.l1_slots_bytes << '\n'
              << prefix << "_l1_circular_buffers_bytes: " << scheduler.l1_circular_buffers_bytes << '\n'
              << prefix << "_l1_cache_bytes: " << scheduler.l1_cache_bytes << '\n'
              << prefix << "_l1_output_bytes: " << scheduler.l1_output_bytes << '\n'
              << prefix << "_l1_synchronization_bytes: " << scheduler.l1_synchronization_bytes << '\n'
              << prefix << "_l1_metadata_bytes: " << scheduler.l1_metadata_bytes << '\n'
              << prefix << "_l1_alignment_bytes: " << scheduler.l1_alignment_bytes << '\n'
              << prefix << "_l1_padding_bytes: " << scheduler.l1_padding_bytes << '\n'
              << prefix << "_l1_architecture_scratch_bytes: " << scheduler.l1_architecture_scratch_bytes << '\n'
              << prefix << "_l1_total_bytes: " << scheduler.l1_total_bytes << '\n'
              << prefix << "_l1_capacity_bytes: " << scheduler.l1_capacity_bytes << '\n'
              << prefix << "_l1_headroom_bytes: " << scheduler.l1_headroom_bytes << '\n';
}

void print_timing_statistics(
    const char* prefix,
    const std::vector<double>& times_ms,
    const uint32_t warmup_runs,
    const ttwv::LiftingSchedulerTelemetry& scheduler) {
    const double mean_ms = std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / times_ms.size();
    const double min_ms = *std::min_element(times_ms.begin(), times_ms.end());
    std::vector<double> sorted_times = times_ms;
    std::sort(sorted_times.begin(), sorted_times.end());
    const double median_ms = percentile(sorted_times, 0.5);
    const double p10_ms = percentile(sorted_times, 0.1);
    const double p90_ms = percentile(sorted_times, 0.9);
    const double squared_deviation_sum =
        std::accumulate(times_ms.begin(), times_ms.end(), 0.0, [mean_ms](const double sum, const double value) {
            const double deviation = value - mean_ms;
            return sum + deviation * deviation;
        });
    const double stddev_ms = std::sqrt(squared_deviation_sum / times_ms.size());

    std::cerr << std::fixed << std::setprecision(6) << prefix << "_execution_time_ms: " << mean_ms << '\n'
              << prefix << "_min_time_ms: " << min_ms << '\n'
              << prefix << "_median_time_ms: " << median_ms << '\n'
              << prefix << "_p10_time_ms: " << p10_ms << '\n'
              << prefix << "_p90_time_ms: " << p90_ms << '\n'
              << prefix << "_stddev_time_ms: " << stddev_ms << '\n'
              << prefix << "_repeats: " << times_ms.size() << '\n'
              << prefix << "_warmup_runs: " << warmup_runs << '\n';
    print_scheduler_telemetry(prefix, scheduler);
}

void print_reconstruction_error(const std::vector<float>& reference, const std::vector<float>& reconstructed) {
    double max_abs_error = 0.0;
    double squared_error_sum = 0.0;
    for (size_t index = 0; index < reference.size(); ++index) {
        const double error = static_cast<double>(reconstructed[index]) - static_cast<double>(reference[index]);
        max_abs_error = std::max(max_abs_error, std::abs(error));
        squared_error_sum += error * error;
    }
    const double rms_error = std::sqrt(squared_error_sum / static_cast<double>(reference.size()));
    std::cerr << std::scientific << std::setprecision(8) << "ilwt_roundtrip_max_abs_error: " << max_abs_error << '\n'
              << "ilwt_roundtrip_rms_error: " << rms_error << '\n'
              << std::defaultfloat;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const CliOptions options = parse_cli(argc, argv);
        configure_tt_runtime(options.benchmark);

        const std::string wavelet_name = canonical_wavelet_name(options.wavelet_name);
        const bool external_coefficients = options.approximation_file.has_value();
        std::vector<float> signal;
        std::vector<float> supplied_approximation;
        std::vector<float> supplied_detail;
        if (external_coefficients) {
            supplied_approximation = read_signal_file(*options.approximation_file);
            supplied_detail = read_signal_file(*options.detail_file);
            if (supplied_approximation.size() != supplied_detail.size()) {
                throw std::runtime_error("Approximation and detail coefficient files must have equal lengths.");
            }
        } else {
            signal = options.generated_length.has_value()
                         ? generate_signal(*options.generated_length, options.signal_start, options.signal_step)
                         : read_signal_file(*options.signal_file);
        }

        const size_t boundary_signal_length = external_coefficients ? *options.original_length : signal.size();
        if (ttwv::boundary_mode_requires_multiple_samples(options.boundary_mode) && boundary_signal_length <= 1) {
            throw std::runtime_error(
                "reflect and antireflect boundary modes require a signal length greater than one.");
        }

        constexpr int device_id{0};
        auto mesh_device{tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id)};
        if (options.benchmark) {
            mesh_device->enable_program_cache();
        }
        tt::tt_metal::distributed::MeshCommandQueue& command_queue{mesh_device->mesh_command_queue()};

        std::optional<DeviceInput> input;
        if (!external_coefficients) {
            input = create_device_input(*mesh_device, signal);
            tt::tt_metal::distributed::EnqueueWriteMeshBuffer(command_queue, input->buffer, signal, false);
            if (options.benchmark) {
                tt::tt_metal::distributed::Finish(command_queue);
            }
        }

        const auto run_scheme = [&]<typename Scheme>() -> int {
            if (options.inverse) {
                std::vector<float> approximation_values;
                std::vector<float> detail_values;
                size_t original_length = 0;
                if (external_coefficients) {
                    approximation_values = supplied_approximation;
                    detail_values = supplied_detail;
                    original_length = *options.original_length;
                } else {
                    // Forward execution and coefficient upload are deliberately
                    // outside the ILWT timing boundary.
                    const ForwardOutput coefficients = run_forward_once<Scheme>(
                        *mesh_device,
                        command_queue,
                        *(input->buffer->get_backing_buffer()),
                        input->desc,
                        options.boundary_mode,
                        true);
                    approximation_values = coefficients.approximation;
                    detail_values = coefficients.detail;
                    original_length = signal.size();
                }

                const size_t coefficient_length = approximation_values.size();
                DeviceInput approximation = create_device_input(*mesh_device, approximation_values);
                DeviceInput detail = create_device_input(*mesh_device, detail_values);
                tt::tt_metal::distributed::EnqueueWriteMeshBuffer(
                    command_queue, approximation.buffer, approximation_values, false);
                tt::tt_metal::distributed::EnqueueWriteMeshBuffer(command_queue, detail.buffer, detail_values, false);
                tt::tt_metal::distributed::Finish(command_queue);

                if (options.benchmark) {
                    for (uint32_t i = 0; i < options.warmup_runs; ++i) {
                        static_cast<void>(run_inverse_once<Scheme>(
                            *mesh_device,
                            command_queue,
                            *(approximation.buffer->get_backing_buffer()),
                            *(detail.buffer->get_backing_buffer()),
                            coefficient_length,
                            original_length,
                            options.boundary_mode,
                            false));
                    }

                    std::vector<double> times_ms;
                    times_ms.reserve(options.repeats);
                    ttwv::LiftingSchedulerTelemetry scheduler;
                    for (uint32_t i = 0; i < options.repeats; ++i) {
                        InverseOutput sample = run_inverse_once<Scheme>(
                            *mesh_device,
                            command_queue,
                            *(approximation.buffer->get_backing_buffer()),
                            *(detail.buffer->get_backing_buffer()),
                            coefficient_length,
                            original_length,
                            options.boundary_mode,
                            false);
                        times_ms.push_back(sample.execution_time_ms);
                        scheduler = std::move(sample.scheduler);
                    }
                    print_timing_statistics("ilwt", times_ms, options.warmup_runs, scheduler);
                    if (options.output_prefix.has_value()) {
                        const InverseOutput output = run_inverse_once<Scheme>(
                            *mesh_device,
                            command_queue,
                            *(approximation.buffer->get_backing_buffer()),
                            *(detail.buffer->get_backing_buffer()),
                            coefficient_length,
                            original_length,
                            options.boundary_mode,
                            true);
                        write_inverse_output(*options.output_prefix, output);
                    }
                    return EXIT_SUCCESS;
                }

                const InverseOutput output = run_inverse_once<Scheme>(
                    *mesh_device,
                    command_queue,
                    *(approximation.buffer->get_backing_buffer()),
                    *(detail.buffer->get_backing_buffer()),
                    coefficient_length,
                    original_length,
                    options.boundary_mode,
                    true);
                print_coeffs("tt-wavelet reconstructed signal", output.signal, original_length);
                if (!external_coefficients) {
                    print_reconstruction_error(signal, output.signal);
                }
                if (options.output_prefix.has_value()) {
                    write_inverse_output(*options.output_prefix, output);
                }
                std::cerr << "ilwt_execution_time_ms: " << std::fixed << std::setprecision(6)
                          << output.execution_time_ms << '\n';
                print_scheduler_telemetry("ilwt", output.scheduler);
                return EXIT_SUCCESS;
            }

            if (options.benchmark) {
                for (uint32_t i = 0; i < options.warmup_runs; ++i) {
                    static_cast<void>(time_transform_once<Scheme>(
                        *mesh_device,
                        command_queue,
                        *(input->buffer->get_backing_buffer()),
                        input->desc,
                        options.boundary_mode));
                }

                std::vector<double> times_ms;
                times_ms.reserve(options.repeats);
                ttwv::LiftingSchedulerTelemetry scheduler;
                for (uint32_t i = 0; i < options.repeats; ++i) {
                    TimedTransform sample = time_transform_once<Scheme>(
                        *mesh_device,
                        command_queue,
                        *(input->buffer->get_backing_buffer()),
                        input->desc,
                        options.boundary_mode);
                    times_ms.push_back(sample.execution_time_ms);
                    scheduler = std::move(sample.scheduler);
                }
                print_timing_statistics("lwt", times_ms, options.warmup_runs, scheduler);
                if (options.output_prefix.has_value()) {
                    const ForwardOutput output = run_forward_once<Scheme>(
                        *mesh_device,
                        command_queue,
                        *(input->buffer->get_backing_buffer()),
                        input->desc,
                        options.boundary_mode,
                        true);
                    write_forward_outputs(*options.output_prefix, output);
                }
                return EXIT_SUCCESS;
            }

            const ForwardOutput output = run_forward_once<Scheme>(
                *mesh_device,
                command_queue,
                *(input->buffer->get_backing_buffer()),
                input->desc,
                options.boundary_mode,
                true);

            print_coeffs("tt-wavelet approximation coefficients", output.approximation, output.logical_length);
            print_coeffs("tt-wavelet detail coefficients", output.detail, output.logical_length);
            if (options.output_prefix.has_value()) {
                write_forward_outputs(*options.output_prefix, output);
            }

            std::cerr << "lwt_execution_time_ms: " << std::fixed << std::setprecision(6) << output.execution_time_ms
                      << '\n';
            print_scheduler_telemetry("lwt", output.scheduler);
            return EXIT_SUCCESS;
        };
        if (wavelet_name == ttwv::schemes::testing::synthetic_k17::name) {
            return run_scheme.template operator()<ttwv::schemes::testing::synthetic_k17>();
        }
        return ttwv::dispatch_scheme(wavelet_name, run_scheme);
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << '\n';
        return EXIT_FAILURE;
    }
}
