#include <cmath>
#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <vector>

#include "BoundaryHandlers.hpp"
#include "LiftingScheme.hpp"
#include "Signal.hpp"
#include "Wavelets.hpp"



using namespace ttwvt;

constexpr size_t PRECISION = 26;

template <typename T, typename BoundaryHandler>
void process(Signal<T>& signal_odd, Signal<T>& signal_even, const LiftingScheme<T, BoundaryHandler>& scheme) {
    if (signal_odd.size() != signal_even.size()) {
        throw std::invalid_argument("Signal odd and even parts must have the same size.");
    }

    std::cout << "Original signal coefficients:" << std::endl;
    for (size_t i = 0; i < signal_odd.size(); i++) {
        std::cout << std::setprecision(PRECISION) << signal_even[i] << ", " << signal_odd[i];
        if (i < signal_odd.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << std::endl;

    scheme.forward(signal_odd, signal_even);

    std::cout << "Transformed signal coefficients:" << std::endl;

    std::cout << "Approximation: ";
    for (size_t i = 0; i < signal_even.size(); i++) {
        std::cout << std::setprecision(PRECISION) << signal_even[i];
        if (i < signal_even.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << std::endl;

    std::cout << "Detail: ";
    for (size_t i = 0; i < signal_odd.size(); i++) {
        std::cout << std::setprecision(PRECISION) << signal_odd[i];
        if (i < signal_odd.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << std::endl;

    scheme.inverse(signal_odd, signal_even);

    std::cout << "Reconstructed signal coefficients:" << std::endl;
    for (size_t i = 0; i < signal_odd.size(); i++) {
        std::cout << std::setprecision(PRECISION) << signal_even[i] << ", " << signal_odd[i];
        if (i < signal_odd.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << std::endl;
}


int main() {
    std::vector<double> signal{
        5.0, 7.0, 3.0, 9.0,
        1.0, 4.0, 8.0, 6.0,
        2.0, 0.0, 11.0, 13.0,
        15.0, 14.0, 12.0, 10.0
    };

    std::vector<double> noise(signal.size());
    for (size_t i = 0; i < noise.size(); i++) {
        noise[i] = static_cast<double>(rand()) / RAND_MAX * 0.5 - 0.25; // Random noise in [-0.25, 0.25]
        signal[i] += noise[i];
    }

    if (signal.size() % 2 != 0) {
        std::cerr << "Signal length must be even." << std::endl;
        return EXIT_FAILURE;
    }

    Signal<double> signal_odd(signal.size() / 2);
    Signal<double> signal_even(signal.size() / 2);

    for (size_t i = 0; i < signal.size(); i++) {
        if (i % 2 == 0) {
            signal_even[i / 2] = signal[i];
        } else {
            signal_odd[i / 2] = signal[i];
        }
    }

    process(signal_odd, signal_even, create_db4_lifting_scheme<double, ZeroBoundaryHandler<double>>());

    return 0;
}
