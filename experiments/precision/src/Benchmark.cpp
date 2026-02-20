#include <stdexcept>
#include <vector>
#include <string>
#include <stdfloat>
#include "BoundaryHandlers.hpp"
#include "Wavelets.hpp"
#include "Benchmark.hpp"
#include "Signal.hpp"


using namespace ttwvt::benchmark;


template<typename T>
Result1D template_process1D(
    const std::vector<double>& signal,
    const std::string& wavelet_name,
    const std::string& boundary_handler_name
) {
    auto boundary_handler = ttwvt::boundary_handlers::create<T>(boundary_handler_name);
    auto wavelet = ttwvt::wavelets::create<T>(wavelet_name, boundary_handler);

    const size_t even_size = (signal.size() + 1) / 2;
    const size_t odd_size = signal.size() / 2;
    
    ttwvt::Signal<T> signal_even(even_size);
    ttwvt::Signal<T> signal_odd(odd_size);
    for (size_t i = 0; i < signal.size(); i++) {
        if (i % 2 == 0) {
            signal_even[i / 2] = static_cast<T>(signal[i]);
        } else {
            signal_odd[i / 2] = static_cast<T>(signal[i]);
        }
    }

    wavelet.forward(signal_odd, signal_even);

    Result1D result{
        std::vector<double>(signal_odd.size()),
        std::vector<double>(signal_even.size()),
        std::vector<double>(signal.size())
    };

    for (size_t i = 0; i < signal_even.size(); i++) {
        result.l[i] = static_cast<double>(signal_even[i]);
    }
    for (size_t i = 0; i < signal_odd.size(); i++) {
        result.h[i] = static_cast<double>(signal_odd[i]);
    }

    wavelet.inverse(signal_odd, signal_even);

    for (size_t i = 0; i < signal.size(); i++) {
        if (i % 2 == 0) {
            result.reconstructed[i] = static_cast<double>(signal_even[i / 2]);
        } else {
            result.reconstructed[i] = static_cast<double>(signal_odd[i / 2]);
        }
    }

    return result;
}

Result1D ttwvt::benchmark::process1D(
    const std::vector<double>& signal,
    const std::string& wavelet_name,
    const std::string& boundary_handler_name,
    datatype dtype
) {
    switch (dtype) {
        case datatype::bfloat16:
            return template_process1D<std::bfloat16_t>(signal, wavelet_name, boundary_handler_name);
        case datatype::float32:
            return template_process1D<float>(signal, wavelet_name, boundary_handler_name);
        case datatype::float64:
            return template_process1D<double>(signal, wavelet_name, boundary_handler_name);
        default:
            throw std::invalid_argument("Unsupported data type");
    }
}


// template<typename T>
// Result2D process2D(
//     const std::vector<std::vector<double>>& signal,
//     const std::string& wavelet_name,
//     const std::string& boundary_handler_name
// ) {
//     auto boundary_handler = ttwvt::boundary_handlers::create<T>(boundary_handler_name);
//     auto wavelet = ttwvt::wavelets::create<T>(wavelet_name, boundary_handler);

//     std::vector<std::vector<T>> signal_even(signal.size() / 2, std::vector<T>(signal[0].size()));
//     std::vector<std::vector<T>> signal_odd(signal.size() - signal_even.size(), std::vector<T>(signal[0].size()));
//     for (size_t i = 0; i < signal.size(); i++) {
//         for (size_t j = 0; j < signal[i].size(); j++) {
//             if (i % 2 == 0) {
//                 signal_even[i / 2][j] = static_cast<T>(signal[i][j]);
//             } else {
//                 signal_odd[i / 2][j] = static_cast<T>(signal[i][j]);
//             }
//         }
//     }

//     wavelet.forward(signal_odd, signal_even);

    

//     Result2D result{
//         std::vector<std::vector<T>>(signal.size(), std::vector<T>(signal_even[0].size())),
//         std::vector<std::vector<T>>(signal.size(), std::vector<T>(signal_odd[0].size())),
//         std::vector<std::vector<T>>(signal.size(), std::vector<T>(signal_even[0].size())),
//         std::vector<std::vector<T>>(signal.size(), std::vector<T>(signal_odd[0].size())),
//         std::vector<std::vector<T>>(signal.size(), std::vector<T>(signal[0].size()))
//     };

//     wavelet.inverse(signal_odd, signal_even);

//     for (size_t i = 0; i < signal.size(); i++) {
//         for (size_t j = 0; j < signal[i].size(); j++) {
//             if (j % 2 == 0) {
//                 result.reconstructed[i][j] = signal_even[i][j / 2];
//             } else {
//                 result.reconstructed[i][j] = signal_odd[i][j / 2];
//             }
//         }
//     }

//     return result;
// }
