#ifndef TTWVT_FILTER_HPP
#define TTWVT_FILTER_HPP
#include <memory>
#include <initializer_list>
#include <cstdint>

namespace ttwvt {
    template <typename T>
    class Filter {
    public:
        Filter(const uint32_t length, const int32_t shift = 0)
            : m_length(length), m_shift(shift), m_coeffs(new T[length]()) {}

        Filter(const uint32_t length, const int32_t shift, const std::initializer_list<T>& coeffs)
            : m_length(length), m_shift(shift), m_coeffs(new T[length]()) {
            std::copy(coeffs.begin(), coeffs.end(), m_coeffs.get());
        }
        
        Filter(const Filter& other)
            : m_length(other.m_length), m_shift(other.m_shift), m_coeffs(new T[other.m_length]()) {
            std::copy(other.m_coeffs.get(), other.m_coeffs.get() + m_length, m_coeffs.get());
        }

        const T& operator[](int32_t index) const {
            return m_coeffs[index - m_shift];
        }

        T& operator[](int32_t index) {
            return m_coeffs[index - m_shift];
        }

        uint32_t length() const {
            return m_length;
        }

        // Inclusive start index
        int32_t start() const {
            return m_shift;
        }

        // Not inclusive end index
        int32_t end() const {
            return m_shift + m_length;
        }

    private:
        uint32_t m_length;
        int32_t m_shift;
        std::unique_ptr<T[]> m_coeffs;
    };
}

#endif // TTWVT_FILTER_HPP
