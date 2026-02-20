#ifndef TTWVT_CONFIGURATION_HPP
#define TTVVT_CONFIGURATION_HPP

#include <cstddef>
#include <vector>
#include <string>

namespace ttwvt::config {
    struct InputEntry {
        std::string name;
        std::vector<size_t> shape;
    };
    
    struct WaveletEntry {
        std::string name;
        std::string wavelet_name;
        std::string boundary_handler_name;
    };

    struct Config {
        std::vector<InputEntry> inputs;
        std::vector<WaveletEntry> wavelets;
    };

    std::vector<double> read1D(const std::string& filename, size_t length);
    std::vector<std::vector<double>> read2D(const std::string& filename, size_t rows, size_t cols);
    void write1D(const std::string& filename, const std::vector<double>& data);
    void write2D(const std::string& filename, const std::vector<std::vector<double>>& data);
    Config read_config(const std::string& filename);
}

#endif // TTWVT_CONFIGURATION_HPP
