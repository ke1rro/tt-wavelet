#pragma once

#include <cstddef>
#include <type_traits>
#include <tuple>
#include <utility>

#include "step.hpp"

namespace ttwv {

template <typename Tag>
struct scheme_traits;

template <int Id>
struct WaveletScheme;

struct SchemeInfo {
    const char* name;
    int id;
    int tap_size;
    int delay_even;
    int delay_odd;
    int num_steps;
};

template <typename... Steps>
struct LiftingScheme {
    private:
        template <typename Visitor, std::size_t... Is>
        constexpr void for_each_step_indexed_impl(Visitor&& v, std::index_sequence<Is...>) const {
            (std::forward<Visitor>(v)(std::get<Is>(steps), std::integral_constant<std::size_t, Is>{}), ...);
        }

    public:
    static constexpr std::size_t num_steps = sizeof...(Steps);

    int tap_size;
    int delay_even;
    int delay_odd;
    std::tuple<Steps...> steps;

    template <std::size_t I>
    [[nodiscard]] constexpr const auto& step() const noexcept {
        static_assert(I < num_steps, "Step index out of range");
        return std::get<I>(steps);
    }

    template <typename Visitor>
    constexpr void for_each_step(Visitor&& v) const {
        std::apply([&v](const auto&... s) { (std::forward<Visitor>(v)(s), ...); }, steps);
    }

    template <typename Visitor>
    constexpr void for_each_step_indexed(Visitor&& v) const {
        for_each_step_indexed_impl(std::forward<Visitor>(v), std::make_index_sequence<num_steps>{});
    }
};

template <typename... Steps>
[[nodiscard]] constexpr LiftingScheme<typename std::decay<Steps>::type...> make_lifting_scheme(
    int tap_size, int delay_even, int delay_odd, Steps&&... steps) {
    return LiftingScheme<typename std::decay<Steps>::type...>{
        tap_size,
        delay_even,
        delay_odd,
        std::tuple<typename std::decay<Steps>::type...>{std::forward<Steps>(steps)...},
    };
}

template <typename Tag>
[[nodiscard]] constexpr const auto& get_scheme() noexcept {
    return scheme_traits<Tag>::scheme;
}

template <int Id>
[[nodiscard]] constexpr const auto& get_scheme_by_id() noexcept {
    return WaveletScheme<Id>::scheme;
}

}  // namespace ttwv
