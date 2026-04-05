#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t kInvalidStick = 0xFFFFFFFFU;

inline __attribute__((always_inline)) uint32_t positive_mod(const int32_t value, const uint32_t modulus) {
    if (modulus == 0) {
        return 0;
    }

    const int32_t signed_modulus = static_cast<int32_t>(modulus);
    const int32_t remainder = value % signed_modulus;
    return static_cast<uint32_t>(remainder < 0 ? remainder + signed_modulus : remainder);
}

inline __attribute__((always_inline)) uint32_t symmetric_index(const int32_t index, const uint32_t length) {
    if (length <= 1) {
        return 0;
    }

    const uint64_t period = static_cast<uint64_t>(length) * 2;
    const uint32_t reflected = positive_mod(index, static_cast<uint32_t>(period > 0xFFFFFFFFU ? 0xFFFFFFFFU : period));
    return reflected < length ? reflected : static_cast<uint32_t>(period) - 1U - reflected;
}

}  // namespace

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t input_length = get_arg_val<uint32_t>(1);
    const uint32_t padded_length = get_arg_val<uint32_t>(2);
    const uint32_t left_pad = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_even = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_odd = get_compile_time_arg_val(1);
    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(2);
    constexpr uint32_t cb_id_cache = get_compile_time_arg_val(3);
    constexpr uint32_t stick_width = stick_nbytes / sizeof(float);
    constexpr auto src_args = TensorAccessorArgs<4>();
    const auto src = TensorAccessor(src_args, src_addr, stick_nbytes);

    uint32_t cached_stick_id = kInvalidStick;
    bool cache_valid = false;

    const uint32_t even_stick_count = ((padded_length + 1) / 2 + stick_width - 1) / stick_width;
    const uint32_t odd_stick_count = (padded_length / 2 + stick_width - 1) / stick_width;
    const uint32_t pair_count = padded_length / 2;

    cb_reserve_back(cb_id_even, 1);
    auto* even_ptr = reinterpret_cast<float*>(get_write_ptr(cb_id_even));
    cb_reserve_back(cb_id_odd, 1);
    auto* odd_ptr = reinterpret_cast<float*>(get_write_ptr(cb_id_odd));

    uint32_t even_count = 0;
    uint32_t odd_count = 0;
    uint32_t even_sticks_pushed = 0;
    uint32_t odd_sticks_pushed = 0;

    // Helper: read padded source value for a given output index
    auto read_val = [&](const uint32_t out_idx) -> float {
        if (input_length == 0) {
            return 0.0F;
        }
        const int32_t logical = static_cast<int32_t>(out_idx) - static_cast<int32_t>(left_pad);
        const uint32_t source_index = symmetric_index(logical, input_length);
        const uint32_t source_stick = source_index / stick_width;
        const uint32_t source_lane = source_index % stick_width;

        if (!cache_valid || source_stick != cached_stick_id) {
            if (cache_valid) {
                cb_pop_front(cb_id_cache, 1);
            }
            cb_reserve_back(cb_id_cache, 1);
            const uint32_t cache_l1_addr = get_write_ptr(cb_id_cache);
            const uint64_t src_noc_addr = src.get_noc_addr(source_stick);
            noc_async_read(src_noc_addr, cache_l1_addr, stick_nbytes);
            noc_async_read_barrier();
            cb_push_back(cb_id_cache, 1);
            cb_wait_front(cb_id_cache, 1);
            cached_stick_id = source_stick;
            cache_valid = true;
        }
        const auto* cache = reinterpret_cast<const float*>(get_read_ptr(cb_id_cache));
        return cache[source_lane];
    };

    // Helper: push a value into an output channel
    auto push_even = [&](const float val) {
        even_ptr[even_count % stick_width] = val;
        even_count++;
        if (even_count % stick_width == 0) {
            cb_push_back(cb_id_even, 1);
            even_sticks_pushed++;
            if (even_sticks_pushed < even_stick_count) {
                cb_reserve_back(cb_id_even, 1);
                even_ptr = reinterpret_cast<float*>(get_write_ptr(cb_id_even));
            }
        }
    };

    auto push_odd = [&](const float val) {
        odd_ptr[odd_count % stick_width] = val;
        odd_count++;
        if (odd_count % stick_width == 0) {
            cb_push_back(cb_id_odd, 1);
            odd_sticks_pushed++;
            if (odd_sticks_pushed < odd_stick_count) {
                cb_reserve_back(cb_id_odd, 1);
                odd_ptr = reinterpret_cast<float*>(get_write_ptr(cb_id_odd));
            }
        }
    };

    // Stride-2: each iteration produces exactly one even + one odd element
    for (uint32_t pair = 0; pair < pair_count; ++pair) {
        push_even(read_val(pair * 2));
        push_odd(read_val(pair * 2 + 1));
    }

    // If padded_length is odd, one trailing even element
    if (padded_length & 1U) {
        push_even(read_val(padded_length - 1));
    }

    // Flush partial even stick
    if (even_count % stick_width != 0) {
        const uint32_t remaining = even_count % stick_width;
        for (uint32_t i = remaining; i < stick_width; ++i) {
            even_ptr[i] = 0.0F;
        }
        cb_push_back(cb_id_even, 1);
    }

    // Flush partial odd stick
    if (odd_count % stick_width != 0) {
        const uint32_t remaining = odd_count % stick_width;
        for (uint32_t i = remaining; i < stick_width; ++i) {
            odd_ptr[i] = 0.0F;
        }
        cb_push_back(cb_id_odd, 1);
    }

    if (cache_valid) {
        cb_pop_front(cb_id_cache, 1);
    }
}
