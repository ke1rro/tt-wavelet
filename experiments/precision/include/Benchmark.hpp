#ifndef TTWVT_BENCHMARK_HPP
#define TTWVT_BENCHMARK_HPP

#include <vector>
#include <string>


namespace ttwvt::benchmark {
    struct Result1D {
        std::vector<double> h;
        std::vector<double> l;
        std::vector<double> reconstructed;
    };


    struct Result2D {
        std::vector<std::vector<double>> hh;
        std::vector<std::vector<double>> hl;
        std::vector<std::vector<double>> lh;
        std::vector<std::vector<double>> ll;
        std::vector<std::vector<double>> reconstructed;
    };


    enum class datatype {
        bfloat16,
        float32,
        float64
    };


    Result1D process1D(
        const std::vector<double>& signal,
        const std::string& wavelet_name,
        const std::string& boundary_handler_name,
        datatype dtype
    );

    Result2D process2D(
        const std::vector<std::vector<double>>& signal,
        const std::string& wavelet_name,
        const std::string& boundary_handler_name,
        datatype dtype
    );
}

#endif // TTWVT_BENCHMARK_HPP
