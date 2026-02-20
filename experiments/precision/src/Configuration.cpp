#include <fstream>
#include <iostream>
#include <vector>
#include <toml++/toml.hpp>
#include "Configuration.hpp"

using namespace ttwvt::config;


Config ttwvt::config::read_config(const std::string& filename) {
    auto config_data = toml::parse_file(filename);
    Config config;

    if (toml::array* arr = config_data["inputs"].as_array())
    {
        arr->for_each([&config](toml::table& el)
        {
            InputEntry entry;
            entry.name = el["name"].value_or("");
            if (toml::array* shape_arr = el["shape"].as_array()) {
                shape_arr->for_each([&config, &entry](toml::value<int64_t>& dim) {
                    entry.shape.push_back(dim.value_or(0));
                });
            }
            config.inputs.push_back(std::move(entry));
        });
    }

    if (toml::array* arr = config_data["wavelets"].as_array())
    {
        arr->for_each([&config](toml::table& el)
        {
            WaveletEntry entry;
            entry.name = el["name"].value_or("");
            entry.wavelet_name = el["id"].value_or("");
            entry.boundary_handler_name = el["mode"].value_or("");
            config.wavelets.push_back(std::move(entry));
        });
    }

    return config;
}

std::vector<double> ttwvt::config::read1D(const std::string& filename, const size_t length) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    std::streamsize file_size = file.tellg();
    if (file_size < static_cast<std::streamsize>(length * sizeof(double))) {
        throw std::runtime_error("File size is smaller than expected for the given length.");
    }
    file.seekg(0, std::ios::beg);

    std::vector<double> buffer(length);
    file.read(reinterpret_cast<char*>(buffer.data()), length * sizeof(double));

    return buffer;
}

void ttwvt::config::write1D(const std::string& filename, const std::vector<double>& data) {
    std::ofstream file(filename, std::ios::binary);
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(double));
}
