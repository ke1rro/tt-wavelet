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
#ifdef TRACY_ENABLE
#include <unistd.h>  // getpid()
#endif

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt_wavelet/include/lifting/device.hpp"
#include "tt_wavelet/include/schemes/registry.hpp"

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

void print_coeffs(const char* label, const std::vector<float>& values) {
    std::cout << label << " (" << values.size() << "): [";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            std::cout << ", ";
        }

        std::cout << std::scientific << std::setprecision(8) << static_cast<double>(values[i]);
    }
    std::cout << std::defaultfloat << "]\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            throw std::runtime_error("Usage: lwt <scheme_name> <signal_file>");
        }
#ifdef TRACY_ENABLE
        std::cerr << "[Tracy] PID " << getpid() << " — connect now, then press Enter to continue...\n";
        std::cin.get();
#endif
        setenv("TT_LOGGER_LEVEL", "error", 0);

        const std::vector<float> original_signal = read_signal_file(argv[2]);
        const ttwv::LiftingScheme* scheme = ttwv::find_scheme(argv[1]);
        if (scheme == nullptr) {
            throw std::runtime_error("Unknown generated lifting scheme: " + std::string(argv[1]));
        }
        const auto start = std::chrono::steady_clock::now();

        constexpr int device_id = 0;
        constexpr tt::tt_metal::CoreCoord core{0, 0};

        auto mesh_device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
        tt::tt_metal::distributed::MeshCommandQueue& command_queue = mesh_device->mesh_command_queue();
        ttwv::WaveletProgram program(
            TT_WAVELET_SOURCE_DIR, *mesh_device, command_queue, core, *scheme, original_signal.size());
        const auto coefficients = program.execute(original_signal);
        const auto stop = std::chrono::steady_clock::now();

        print_coeffs("tt-wavelet approximation coefficients", coefficients.approximation);
        print_coeffs("tt-wavelet detail coefficients", coefficients.detail);
        std::cout << std::flush;

        const std::chrono::duration<double, std::milli> elapsed_ms = stop - start;
        std::cerr << "lwt_execution_time_ms: " << std::fixed << std::setprecision(6) << elapsed_ms.count() << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << '\n';
        return EXIT_FAILURE;
    }
}
