#include <iostream>
#include <vector>
#include <iomanip>

#include "tt_wavelet_baseline.hpp"

using namespace ttwv;
using namespace ttwv::baseline;

void print_vector(const std::string& name, const std::vector<float>& vec) {
    std::cout << name << " [" << vec.size() << "]: ";
    for (float v : vec) {
        std::cout << std::fixed << std::setprecision(16) << v << " ";
    }
    std::cout << "\n";
}

int main() {
    std::vector<float> signal = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

    std::cout << "original:\n";
    print_vector("X  ", signal);
    std::cout << "\n";

    std::cout << "transformed:\n";
    auto db1_res = lwt_1d<db1_tag>(signal);
    print_vector("Res", db1_res);
    std::cout << "\n";

    return 0;
}
