#ifndef TTWVT_LIFTING_SCHEME_HPP
#define TTWVT_LIFTING_SCHEME_HPP
#include <vector>
#include <cstdint>
#include <cstddef>
#include <initializer_list>

#include "Filter.hpp"
#include "Signal.hpp"

namespace ttwvt {
    template<typename T, typename BoundaryHandler>
    class LiftingStep {
    public:
        LiftingStep(const Filter<T>& predict, const Filter<T>& update, const T scale_even = T{1.0}, const T scale_odd = T{1.0})
            : m_predict(predict), m_update(update), m_scale_even(scale_even), m_scale_odd(scale_odd) {}

        void forward(Signal<T>& signal_even, Signal<T>& signal_odd) const {
            for (size_t i = 0; i < signal_odd.size(); ++i) {
                T sum = 0;
                for (int32_t j = m_predict.start(); j < m_predict.end(); j++) {
                    const int64_t k = static_cast<int64_t>(i) + static_cast<int64_t>(j);
                    sum += m_predict[j] * BoundaryHandler::eval(k, signal_even);
                }
                signal_odd[i] += sum;
            }

            for (size_t i = 0; i < signal_even.size(); ++i) {
                T sum = 0;
                for (int32_t j = m_update.start(); j < m_update.end(); j++) {
                    const int64_t k = static_cast<int64_t>(i) + static_cast<int64_t>(j) ;
                    sum += m_update[j] * BoundaryHandler::eval(k, signal_odd);
                }
                signal_even[i] += sum;
                signal_even[i] *= m_scale_even;
            }

            for (size_t i = 0; i < signal_odd.size(); ++i) {
                signal_odd[i] *= m_scale_odd;
            }
        }

        void inverse(Signal<T>& signal_even, Signal<T>& signal_odd) const {
            for (size_t i = 0; i < signal_odd.size(); ++i) {
                signal_odd[i] /= m_scale_odd;
            }

            for (size_t i = 0; i < signal_even.size(); ++i) {
                T sum = 0;
                for (int32_t j = m_update.start(); j < m_update.end(); j++) {
                    const int64_t k = static_cast<int64_t>(i) + static_cast<int64_t>(j);
                    sum += m_update[j] * BoundaryHandler::eval(k, signal_odd);
                }
                signal_even[i] /= m_scale_even;
                signal_even[i] -= sum;
            }

            for (size_t i = 0; i < signal_odd.size(); ++i) {
                T sum = 0;
                for (int32_t j = m_predict.start(); j < m_predict.end(); j++) {
                    const int64_t k = static_cast<int64_t>(i) + static_cast<int64_t>(j);
                    sum += m_predict[j] * BoundaryHandler::eval(k, signal_even);
                }
                signal_odd[i] -= sum;
            }
        }

    private:
        Filter<T> m_predict;
        Filter<T> m_update;
        T m_scale_even;
        T m_scale_odd;
    };

    template<typename T, typename BoundaryHandler>
    class LiftingScheme {
    public:
        LiftingScheme(const std::initializer_list<LiftingStep<T, BoundaryHandler>>& steps)
            : m_steps(steps) {}

        void forward(Signal<T>& signal_odd, Signal<T>& signal_even) const {
            for (auto it = m_steps.begin(); it != m_steps.end(); ++it) {
                it->forward(signal_odd, signal_even);
            }
        }

        void inverse(Signal<T>& signal_odd, Signal<T>& signal_even) const {
            for (auto it = m_steps.rbegin(); it != m_steps.rend(); ++it) {
                it->inverse(signal_odd, signal_even);
            }
        }

    private:
        std::vector<LiftingStep<T, BoundaryHandler>> m_steps;
    };
}

#endif // TTWVT_LIFTING_SCHEME_HPP
