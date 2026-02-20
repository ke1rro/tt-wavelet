#include <cmath>
#include <iostream>
#include <cstdlib>
#include <filesystem>
#include <stdfloat>

#include "BoundaryHandlers.hpp"
#include "Benchmark.hpp"
#include "LiftingScheme.hpp"
#include "Signal.hpp"
#include "Wavelets.hpp"
#include "Configuration.hpp"


int main(int argc, char* argv[]) {
    if (argc < 2 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " <config_file> [<output_dir>] [<input_dir>]" << std::endl;
        return -1;
    }

    const std::string input_dir = (argc >= 4) ? argv[3] : "data/input";
    const std::string output_dir = (argc >= 3) ? argv[2] : "data/output";

    const std::string config_file = argv[1];
    auto config = ttwvt::config::read_config(config_file);

    for (const auto dtype : {ttwvt::benchmark::datatype::bfloat16, ttwvt::benchmark::datatype::float32}) {
        const std::string current_output_dir = output_dir + ((dtype == ttwvt::benchmark::datatype::bfloat16) ? "-bf16" : "-f32");

        for (const auto& input : config.inputs) {
            std::cout << "Processing input: " << input.name << " with data type: " << ((dtype == ttwvt::benchmark::datatype::bfloat16) ? "bfloat16" : "float32") << std::endl;

            const std::string output_path = current_output_dir + "/" + input.name;
            std::filesystem::create_directories(output_path);

            if (input.shape.size() != 1) {
                std::cerr << "Only 1D signals are supported in this benchmark." << std::endl;
                continue;
            }

            auto signal = ttwvt::config::read1D(input_dir + "/" + input.name, input.shape[0]);
            for (const auto& wavelet_entry : config.wavelets) {
                std::cout << "  Applying wavelet: " << wavelet_entry.wavelet_name << " with boundary handler: " << wavelet_entry.boundary_handler_name << std::endl;
                auto result = ttwvt::benchmark::process1D(signal, wavelet_entry.wavelet_name, wavelet_entry.boundary_handler_name, dtype);
                ttwvt::config::write1D(output_path + "/" + wavelet_entry.name + "_h_fwd", result.h);
                ttwvt::config::write1D(output_path + "/" + wavelet_entry.name + "_l_fwd", result.l);
                ttwvt::config::write1D(output_path + "/" + wavelet_entry.name + "_inv", result.reconstructed);
            }
        }
    }
}
