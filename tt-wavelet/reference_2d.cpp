// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_wavelet/include/lifting/reference_2d.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "tt_wavelet/include/schemes/generated/registry.hpp"
#include "tt_wavelet/include/schemes/testing/synthetic_k17.hpp"

namespace {

struct Options {
    ttwv::BoundaryMode boundary_mode{ttwv::BoundaryMode::kSymmetric};
    ttwv::Fp32Arithmetic arithmetic{ttwv::Fp32Arithmetic::kIeee};
    bool benchmark{false};
    size_t repeats{1};
    size_t warmup_runs{1};
    std::string wavelet;
    size_t height{0};
    size_t width{0};
    std::filesystem::path input_path;
    bool binary_input{false};
    std::optional<std::filesystem::path> output_prefix;
};

[[nodiscard]] std::string usage() {
    return "Usage: lwt_2d_reference [--boundary-mode MODE] "
           "[--fp32-arithmetic ieee|wormhole-sfpu|blackhole-sfpu] [--binary-input] "
           "[--output-prefix PATH] [--benchmark [--repeats N] "
           "[--warmup-runs N]] WAVELET HEIGHT WIDTH INPUT_FILE";
}

[[nodiscard]] ttwv::Fp32Arithmetic parse_fp32_arithmetic(const std::string_view name) {
    if (name == "ieee") {
        return ttwv::Fp32Arithmetic::kIeee;
    }
    if (name == "wormhole-sfpu") {
        return ttwv::Fp32Arithmetic::kWormholeSfpu;
    }
    if (name == "blackhole-sfpu") {
        return ttwv::Fp32Arithmetic::kBlackholeSfpu;
    }
    throw std::runtime_error("Unsupported FP32 arithmetic model: " + std::string{name});
}

[[nodiscard]] ttwv::BoundaryMode parse_boundary_mode(const std::string_view name) {
    if (name == "zero") {
        return ttwv::BoundaryMode::kZero;
    }
    if (name == "constant") {
        return ttwv::BoundaryMode::kConstant;
    }
    if (name == "symmetric") {
        return ttwv::BoundaryMode::kSymmetric;
    }
    if (name == "periodic") {
        return ttwv::BoundaryMode::kPeriodic;
    }
    if (name == "antisymmetric") {
        return ttwv::BoundaryMode::kAntisymmetric;
    }
    if (name == "smooth") {
        return ttwv::BoundaryMode::kSmooth;
    }
    if (name == "antireflect") {
        return ttwv::BoundaryMode::kAntireflect;
    }
    if (name == "reflect") {
        return ttwv::BoundaryMode::kReflect;
    }
    throw std::runtime_error("Unsupported boundary mode: " + std::string{name});
}

[[nodiscard]] size_t parse_size(const std::string& text, const char* label) {
    if (text.empty() || text.front() == '-') {
        throw std::runtime_error(std::string{label} + " must be a positive integer");
    }
    size_t consumed = 0;
    const unsigned long long value = std::stoull(text, &consumed);
    if (consumed != text.size() || value == 0) {
        throw std::runtime_error(std::string{label} + " must be a positive integer");
    }
    if (value > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error(std::string{label} + " exceeds the platform size limit");
    }
    return static_cast<size_t>(value);
}

[[nodiscard]] size_t parse_nonnegative_size(const std::string& text, const char* label) {
    if (text.empty() || text.front() == '-') {
        throw std::runtime_error(std::string{label} + " must be a non-negative platform-sized integer");
    }
    size_t consumed = 0;
    const unsigned long long value = std::stoull(text, &consumed);
    if (consumed != text.size() || value > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error(std::string{label} + " must be a non-negative platform-sized integer");
    }
    return static_cast<size_t>(value);
}

[[nodiscard]] Options parse_options(const int argc, char** argv) {
    Options options;
    std::vector<std::string> positional;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--boundary-mode") {
            if (++index >= argc) {
                throw std::runtime_error("--boundary-mode requires a value");
            }
            options.boundary_mode = parse_boundary_mode(argv[index]);
        } else if (argument == "--fp32-arithmetic") {
            if (++index >= argc) {
                throw std::runtime_error("--fp32-arithmetic requires a value");
            }
            options.arithmetic = parse_fp32_arithmetic(argv[index]);
        } else if (argument == "--benchmark") {
            options.benchmark = true;
        } else if (argument == "--binary-input") {
            options.binary_input = true;
        } else if (argument == "--output-prefix") {
            if (++index >= argc) {
                throw std::runtime_error("--output-prefix requires a value");
            }
            options.output_prefix = std::filesystem::path{argv[index]};
        } else if (argument == "--repeats") {
            if (++index >= argc) {
                throw std::runtime_error("--repeats requires a value");
            }
            options.repeats = parse_size(argv[index], "--repeats");
        } else if (argument == "--warmup-runs") {
            if (++index >= argc) {
                throw std::runtime_error("--warmup-runs requires a value");
            }
            options.warmup_runs = parse_nonnegative_size(argv[index], "--warmup-runs");
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
    options.height = parse_size(positional[1], "HEIGHT");
    options.width = parse_size(positional[2], "WIDTH");
    options.input_path = positional[3];
    if (!options.benchmark && (options.repeats != 1 || options.warmup_runs != 1)) {
        throw std::runtime_error("--repeats and --warmup-runs require --benchmark");
    }
    if (options.benchmark && options.output_prefix.has_value()) {
        throw std::runtime_error("--output-prefix cannot be combined with --benchmark");
    }
    return options;
}

[[nodiscard]] std::vector<float> read_input(
    const std::filesystem::path& path, const size_t height, const size_t width, const bool binary_input) {
    if (height > std::numeric_limits<size_t>::max() / width) {
        throw std::runtime_error("HEIGHT x WIDTH overflows the platform size limit");
    }
    const size_t element_count = height * width;
    std::ifstream input(path, binary_input ? std::ios::binary : std::ios::in);
    if (!input.good()) {
        throw std::runtime_error("Failed to open input file: " + path.string());
    }
    std::vector<float> values(element_count);
    if (binary_input) {
        static_assert(std::endian::native == std::endian::little, "Raw FP32 reference files require little endian");
        input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(element_count * sizeof(float)));
        if (input.gcount() != static_cast<std::streamsize>(element_count * sizeof(float))) {
            throw std::runtime_error("Binary input file is shorter than HEIGHT x WIDTH FP32 values");
        }
        char trailing = 0;
        if (input.read(&trailing, 1)) {
            throw std::runtime_error("Binary input file contains trailing bytes");
        }
    } else {
        values.clear();
        values.reserve(element_count);
        for (float value = 0.0F; input >> value;) {
            values.push_back(value);
        }
        if (!input.eof()) {
            throw std::runtime_error("Input file contains a non-numeric token");
        }
    }
    if (values.size() != element_count) {
        throw std::runtime_error(
            "Input element count " + std::to_string(values.size()) + " does not match shape " + std::to_string(height) +
            "x" + std::to_string(width));
    }
    return values;
}

void write_band(const std::filesystem::path& prefix, const char* name, const std::span<const float> values) {
    std::filesystem::path path = prefix;
    path += ".";
    path += name;
    path += ".f32";
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output.good()) {
        throw std::runtime_error("Failed to open output band: " + path.string());
    }
    output.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size_bytes()));
    if (!output.good()) {
        throw std::runtime_error("Failed to write output band: " + path.string());
    }
}

void print_band(const char* name, const std::vector<float>& values, const size_t height, const size_t width) {
    std::cout << "tt-wavelet fp32 reference " << name << " (" << height << "x" << width << "): [";
    for (size_t index = 0; index < values.size(); ++index) {
        if (index > 0) {
            std::cout << ", ";
        }
        std::cout << std::scientific << std::setprecision(8) << values[index];
    }
    std::cout << std::defaultfloat << "]\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const std::vector<float> input =
            read_input(options.input_path, options.height, options.width, options.binary_input);
        const auto run = [&]<typename Scheme>() {
            if (options.benchmark) {
                double checksum = 0.0;
                for (size_t warmup = 0; warmup < options.warmup_runs; ++warmup) {
                    const ttwv::Lwt2DReferenceOutput output = ttwv::lwt_2d_fp32_reference<Scheme>(
                        input, options.height, options.width, options.boundary_mode, options.arithmetic);
                    checksum += static_cast<double>(output.ll.front()) + static_cast<double>(output.hh.back());
                }

                std::vector<double> elapsed_ms;
                elapsed_ms.reserve(options.repeats);
                for (size_t repeat = 0; repeat < options.repeats; ++repeat) {
                    const auto start = std::chrono::steady_clock::now();
                    const ttwv::Lwt2DReferenceOutput output = ttwv::lwt_2d_fp32_reference<Scheme>(
                        input, options.height, options.width, options.boundary_mode, options.arithmetic);
                    const auto stop = std::chrono::steady_clock::now();
                    elapsed_ms.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
                    checksum += static_cast<double>(output.ll.front()) + static_cast<double>(output.hh.back());
                }
                const double total = std::accumulate(elapsed_ms.begin(), elapsed_ms.end(), 0.0);
                std::cout << std::fixed << std::setprecision(9)
                          << "lwt_2d_reference_mean_time_ms: " << total / static_cast<double>(elapsed_ms.size()) << '\n'
                          << "lwt_2d_reference_min_time_ms: " << *std::min_element(elapsed_ms.begin(), elapsed_ms.end())
                          << '\n'
                          << "lwt_2d_reference_repeats: " << options.repeats << '\n'
                          << "lwt_2d_reference_warmup_runs: " << options.warmup_runs << '\n'
                          << "lwt_2d_reference_shape: " << options.height << 'x' << options.width << '\n'
                          << "lwt_2d_reference_checksum: " << checksum << '\n';
                return EXIT_SUCCESS;
            }
            const ttwv::Lwt2DReferenceOutput output = ttwv::lwt_2d_fp32_reference<Scheme>(
                input, options.height, options.width, options.boundary_mode, options.arithmetic);
            if (options.output_prefix.has_value()) {
                write_band(*options.output_prefix, "ll", output.ll);
                write_band(*options.output_prefix, "lh", output.lh);
                write_band(*options.output_prefix, "hl", output.hl);
                write_band(*options.output_prefix, "hh", output.hh);
                std::cout << "lwt_2d_reference_output_shape: " << output.band_height << 'x' << output.band_width
                          << '\n';
                return EXIT_SUCCESS;
            }
            print_band("LL", output.ll, output.band_height, output.band_width);
            print_band("LH", output.lh, output.band_height, output.band_width);
            print_band("HL", output.hl, output.band_height, output.band_width);
            print_band("HH", output.hh, output.band_height, output.band_width);
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
