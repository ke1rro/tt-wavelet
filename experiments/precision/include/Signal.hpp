#ifndef TTWVT_SIGNAL_HPP
#define TTWVT_SIGNAL_HPP
#include <memory>

namespace ttwvt {
    template<typename T>
    class Signal {
    public:
        Signal(size_t length) : m_length(length), m_data(new T[length]()) {}
        Signal(const std::initializer_list<T>& data) : m_length(data.size()), m_data(new T[data.size()]) {
            std::copy(data.begin(), data.end(), m_data.get());
        }

        Signal(const Signal& other) : m_length(other.m_length), m_data(new T[other.m_length]()) {
            std::copy(other.m_data.get(), other.m_data.get() + m_length, m_data.get());
        }
        Signal(Signal&& other) noexcept : m_length(other.m_length), m_data(std::move(other.m_data)) {
            other.m_length = 0;
        }
        Signal& operator=(const Signal& other) {
            if (this != &other) {
                m_length = other.m_length;
                m_data.reset(new T[m_length]());
                std::copy(other.m_data.get(), other.m_data.get() + m_length, m_data.get());
            }
            return *this;
        }
        Signal& operator=(Signal&& other) noexcept {
            if (this != &other) {
                m_length = other.m_length;
                m_data = std::move(other.m_data);
                other.m_length = 0;
            }
            return *this;
        }

        const T& operator[](size_t index) const {
            return m_data[index];
        }

        T& operator[](size_t index) {
            return m_data[index];
        }

        size_t size() const {
            return m_length;
        }
    private:
        size_t m_length;
        std::unique_ptr<T[]> m_data;
    };
}

#endif // TTWVT_SIGNAL_HPP
