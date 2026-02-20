#ifndef TTWVT_BOUNDARY_HANDLERS_HPP
#define TTWVT_BOUNDARY_HANDLERS_HPP
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>

#include "Signal.hpp"

namespace ttwvt::boundary_handlers {
    template<typename T>
    using BoundaryHandler = std::function<T(int64_t, const Signal<T>&)>;

    template <typename T>
    T reflect(int64_t index, const Signal<T>& signal) {
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

    template <typename T>
    T symmetric(int64_t index, const Signal<T>& signal) {
        const int64_t length = static_cast<int64_t>(signal.size());
        const int64_t double_length = 2 * (length-1);

        if (length == 1) {
            return signal[0];
        }

        index = index % double_length;
        if (index < 0) {
            index += double_length;
        }
        if (index < length) {
            return signal[index];
        }

        return signal[double_length - index];
    }

    template <typename T>
    T periodic(int64_t index, const Signal<T>& signal) {
        const int64_t length = static_cast<int64_t>(signal.size());
        index = index % length;
        if (index < 0) {
            index += length;
        }
        return signal[index];
    }

    template <typename T>
    T zero(int64_t index, const Signal<T>& signal) {
        if (index < 0 || index >= static_cast<int64_t>(signal.size())) {
            return T{};
        }
        return signal[index];
    }

    template <typename T>
    T constant(int64_t index, const Signal<T>& signal) {
        if (index < 0) {
            return signal[0];
        }
        if (index >= static_cast<int64_t>(signal.size())) {
            return signal[signal.size() - 1];
        }
        return signal[index];
    }

    template <typename T>
    BoundaryHandler<T> create(const std::string& name) {
        if (name == "reflect") {
            return reflect<T>;
        }
        if (name == "symmetric") {
            return symmetric<T>;
        }
        if (name == "periodic") {
            return periodic<T>;
        }
        if (name == "zero") {
            return zero<T>;
        }
        if (name == "constant") {
            return constant<T>;
        }
        throw std::invalid_argument("Unknown boundary handler: " + name);
    }
}

#endif // TTWVT_BOUNDARY_HANDLERS_HPP
