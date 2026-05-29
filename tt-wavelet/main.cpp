#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
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

[[nodiscard]] std::string canonical_wavelet_name(const std::string_view raw_name) {
    const std::filesystem::path raw_path{std::string{raw_name}};
    if (raw_path.extension() == ".json") {
        return raw_path.stem().string();
    }
    return std::string{raw_name};
}

struct PadSplitBenchmarkResult {
    double mean_ms{0.0};
    double min_ms{0.0};
    double max_ms{0.0};
    size_t padded_length{0};
    size_t even_length{0};
    size_t odd_length{0};
};

template <typename Scheme>
[[nodiscard]] PadSplitBenchmarkResult run_pad_split_only(const std::vector<float>& signal, const size_t repeat_count) {
    constexpr int device_id{0};
    constexpr tt::tt_metal::CoreCoord preprocess_core{0, 0};

    auto mesh_device{tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id)};
    tt::tt_metal::distributed::MeshCommandQueue& command_queue{mesh_device->mesh_command_queue()};

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
        tt::tt_metal::distributed::MeshBuffer::create(input_replicated_config, input_local_config, mesh_device.get());
    input_desc.dram_address = input_buffer->get_backing_buffer()->address();

    // Blocking write keeps the H2D copy out of the timing window.
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(command_queue, input_buffer, signal, true);

    auto make_preprocess_bundle = [&]() {
        return ttwv::create_lifting_preprocess_program<Scheme>(
            TT_WAVELET_SOURCE_DIR, *mesh_device, preprocess_core, *(input_buffer->get_backing_buffer()), input_desc);
    };

    {
        auto warmup_bundle = make_preprocess_bundle();
        ttwv::run_preprocess(command_queue, *mesh_device, warmup_bundle);
    }

    std::vector<double> times_ms;
    times_ms.reserve(repeat_count);

    size_t padded_length = 0;
    size_t even_length = 0;
    size_t odd_length = 0;

    for (size_t i = 0; i < repeat_count; ++i) {
        auto bundle = make_preprocess_bundle();

        const auto start = std::chrono::steady_clock::now();
        ttwv::run_preprocess(command_queue, *mesh_device, bundle);
        const auto stop = std::chrono::steady_clock::now();

        const std::chrono::duration<double, std::milli> elapsed_ms = stop - start;
        times_ms.push_back(elapsed_ms.count());

        padded_length = bundle.plan.preprocess_layout.padded_length();
        even_length = bundle.plan.preprocess_layout.output.even.length;
        odd_length = bundle.plan.preprocess_layout.output.odd.length;
    }

    const double sum = std::accumulate(times_ms.begin(), times_ms.end(), 0.0);
    const auto [min_it, max_it] = std::minmax_element(times_ms.begin(), times_ms.end());

    return PadSplitBenchmarkResult{
        .mean_ms = sum / static_cast<double>(times_ms.size()),
        .min_ms = *min_it,
        .max_ms = *max_it,
        .padded_length = padded_length,
        .even_length = even_length,
        .odd_length = odd_length,
    };
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            throw std::runtime_error("Usage: lwt <scheme or scheme_path> <signal_file>");
        }
        setenv("TT_LOGGER_LEVEL", "error", 0);

        const std::string wavelet_name = canonical_wavelet_name(argv[1]);
        const std::vector<float> signal = read_signal_file(argv[2]);

        return ttwv::dispatch_scheme(wavelet_name, [&]<typename Scheme>() -> int {
            const PadSplitBenchmarkResult result = run_pad_split_only<Scheme>(signal, 3);

            std::cerr << std::fixed << std::setprecision(6);
            std::cerr << "pad_split_mean_ms: " << result.mean_ms << '\n';
            std::cerr << "pad_split_min_ms: " << result.min_ms << '\n';
            std::cerr << "pad_split_max_ms: " << result.max_ms << '\n';
            std::cerr << "padded_length: " << result.padded_length << '\n';
            std::cerr << "even_length: " << result.even_length << '\n';
            std::cerr << "odd_length: " << result.odd_length << '\n';
            return EXIT_SUCCESS;
        });
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << '\n';
        return EXIT_FAILURE;
    }
}
