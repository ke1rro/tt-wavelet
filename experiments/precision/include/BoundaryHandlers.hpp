#ifndef TTWVT_BOUNDARY_HANDLERS_HPP
#define TTWVT_BOUNDARY_HANDLERS_HPP
#include <cstdint>

#include "Signal.hpp"

namespace ttwvt {
    template <typename T>
    class SymmetricBoundaryHandler {
    public:
        static T eval(int64_t index, const Signal<T>& signal) {
            const int64_t length = static_cast<int64_t>(signal.size());
            const int64_t double_length = 2 * length;

            index = index % double_length;
            if (index < 0) {
                index += double_length;
            }
            if (index < length) {
                return signal[index];
            }

            return signal[double_length - index - 1];
        }
    };

    template <typename T>
    class PeriodicBoundaryHandler {
    public:
        static T eval(int64_t index, const Signal<T>& signal) {
            const int64_t length = static_cast<int64_t>(signal.size());
            index = index % length;
            if (index < 0) {
                index += length;
            }
            return signal[index];
        }
    };

    template <typename T>
    class ZeroBoundaryHandler {
    public:
        static T eval(int64_t index, const Signal<T>& signal) {
            if (index < 0 || index >= static_cast<int64_t>(signal.size())) {
                return T{};
            }
            return signal[index];
        }
    };
}

#endif // TTWVT_BOUNDARY_HANDLERS_HPP
