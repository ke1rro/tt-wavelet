#ifndef TTWVT_WAVELETS_HPP
#define TTWVT_WAVELETS_HPP
#include <numbers>

#include "LiftingScheme.hpp"
#include "Filter.hpp"

namespace ttwvt {
    template <typename T, typename BoundaryHandler>
    LiftingScheme<T, BoundaryHandler> create_haar_lifting_scheme() {
        Filter<T> predict(1, 0, {-1.0f});
        Filter<T> update(1, 0, {0.5f});
        return LiftingScheme<T, BoundaryHandler>({LiftingStep<T, BoundaryHandler>(predict, update)});
    }

    template <typename T, typename BoundaryHandler>
    LiftingScheme<T, BoundaryHandler> create_db4_lifting_scheme() {
        constexpr T s1 = std::numbers::sqrt3_v<T>;
        constexpr T t1 = (std::numbers::sqrt3_v<T> - 2.0) / 4.0;
        constexpr T k = std::numbers::sqrt2_v<T> * std::numbers::inv_sqrt3_v<T>;

        Filter<T> predict(2, -1, {s1, s1});
        Filter<T> update(2, 0, {t1, t1});
        return LiftingScheme<T, BoundaryHandler>({LiftingStep<T, BoundaryHandler>(predict, update, k, T{1.0} / k)});
    }
}

#endif // TTWVT_WAVELETS_HPP