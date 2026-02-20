#ifndef TTWVT_LIFTING_SCHEME_HPP
#define TTWVT_LIFTING_SCHEME_HPP
#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>

#include "Filter.hpp"
#include "Signal.hpp"
#include "BoundaryHandlers.hpp"

namespace ttwvt {
    template<typename T>
    class LiftingStep {
    public:
        virtual void forward(Signal<T>& signal_even, Signal<T>& signal_odd) const = 0;
        virtual void inverse(Signal<T>& signal_even, Signal<T>& signal_odd) const = 0;

        virtual ~LiftingStep() = default;
    };

    template<typename T>
    class LiftingStepPredict : public LiftingStep<T> {
    public:
        LiftingStepPredict(const Filter<T>& predict, const boundary_handlers::BoundaryHandler<T>& boundary_handler)
            : m_filter(predict), m_boundary_handler(boundary_handler) {}

        void forward(Signal<T>& signal_even, Signal<T>& signal_odd) const override {
            for (size_t i = 0; i < signal_odd.size(); ++i) {
                T sum = 0;
                for (int32_t j = m_filter.start(); j < m_filter.end(); j++) {
                    const int64_t k = static_cast<int64_t>(i) + static_cast<int64_t>(j);
                    sum += m_filter[j] * m_boundary_handler(k, signal_even);
                }
                signal_odd[i] += sum;
            }
        }

        void inverse(Signal<T>& signal_even, Signal<T>& signal_odd) const override {
            for (size_t i = 0; i < signal_odd.size(); ++i) {
                T sum = 0;
                for (int32_t j = m_filter.start(); j < m_filter.end(); j++) {
                    const int64_t k = static_cast<int64_t>(i) + static_cast<int64_t>(j);
                    sum += m_filter[j] * m_boundary_handler(k, signal_even);
                }
                signal_odd[i] -= sum;
            }
        }

    private:
        Filter<T> m_filter;
        boundary_handlers::BoundaryHandler<T> m_boundary_handler;
    };

    template<typename T>
    class LiftingStepUpdate : public LiftingStep<T> {
    public:
        LiftingStepUpdate(const Filter<T>& update, const boundary_handlers::BoundaryHandler<T>& boundary_handler)
            : m_filter(update), m_boundary_handler(boundary_handler) {}

        void forward(Signal<T>& signal_even, Signal<T>& signal_odd) const override {
            for (size_t i = 0; i < signal_even.size(); ++i) {
                T sum = 0;
                for (int32_t j = m_filter.start(); j < m_filter.end(); j++) {
                    const int64_t k = static_cast<int64_t>(i) + static_cast<int64_t>(j) ;
                    sum += m_filter[j] * m_boundary_handler(k, signal_odd);
                }
                signal_even[i] += sum;
            }
        }

        void inverse(Signal<T>& signal_even, Signal<T>& signal_odd) const override {
            for (size_t i = 0; i < signal_even.size(); ++i) {
                T sum = 0;
                for (int32_t j = m_filter.start(); j < m_filter.end(); j++) {
                    const int64_t k = static_cast<int64_t>(i) + static_cast<int64_t>(j);
                    sum += m_filter[j] * m_boundary_handler(k, signal_odd);
                }
                signal_even[i] -= sum;
            }
        }

    private:
        Filter<T> m_filter;
        boundary_handlers::BoundaryHandler<T> m_boundary_handler;
    };

    template<typename T>
    class LiftingStepScale : public LiftingStep<T> {
    public:
        LiftingStepScale(const T& scale_even, const T& scale_odd)
            : m_scale_even(scale_even), m_scale_odd(scale_odd) {}

        void forward(Signal<T>& signal_even, Signal<T>& signal_odd) const override {
            for (size_t i = 0; i < signal_even.size(); ++i) {
                signal_even[i] *= m_scale_even;
            }
            for (size_t i = 0; i < signal_odd.size(); ++i) {
                signal_odd[i] *= m_scale_odd;
            }
        }

        void inverse(Signal<T>& signal_even, Signal<T>& signal_odd) const override {
            for (size_t i = 0; i < signal_even.size(); ++i) {
                signal_even[i] /= m_scale_even;
            }
            for (size_t i = 0; i < signal_odd.size(); ++i) {
                signal_odd[i] /= m_scale_odd;
            }
        }

    private:
        T m_scale_even;
        T m_scale_odd;
    };
    

    template<typename T>
    class LiftingScheme {
    public:
        LiftingScheme() : m_steps() {}

        void forward(Signal<T>& signal_even, Signal<T>& signal_odd) const {
            for (auto it = m_steps.begin(); it != m_steps.end(); ++it) {
                (*it)->forward(signal_even, signal_odd);
            }
        }

        void inverse(Signal<T>& signal_even, Signal<T>& signal_odd) const {
            for (auto it = m_steps.rbegin(); it != m_steps.rend(); ++it) {
                (*it)->inverse(signal_even, signal_odd);
            }
        }

        void add_step(std::unique_ptr<LiftingStep<T>> step) {
            m_steps.push_back(std::move(step));
        }

    private:
        std::vector<std::unique_ptr<LiftingStep<T>>> m_steps;
    };
}

#endif // TTWVT_LIFTING_SCHEME_HPP
