#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t kInvalidStick = 0xFFFFFFFFU;

inline __attribute__((always_inline)) uint32_t positive_mod(int32_t value, uint32_t modulus) {
    if (modulus == 0) {
        return 0;
    }

    const int32_t signed_modulus = static_cast<int32_t>(modulus);
    const int32_t remainder = value % signed_modulus;
    return static_cast<uint32_t>(remainder < 0 ? remainder + signed_modulus : remainder);
}

inline __attribute__((always_inline)) uint32_t symmetric_index(int32_t index, uint32_t length) {
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
    const uint32_t output_length = get_arg_val<uint32_t>(2);
    const uint32_t left_pad = get_arg_val<uint32_t>(3);
    const uint32_t num_output_sticks = get_arg_val<uint32_t>(4);
    const uint32_t start_output_stick = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_cache = get_compile_time_arg_val(2);
    constexpr uint32_t stick_width = stick_nbytes / sizeof(float);
    constexpr auto src_args = TensorAccessorArgs<3>();
    const auto src = TensorAccessor(src_args, src_addr, stick_nbytes);

    uint32_t cached_stick_id = kInvalidStick;
    bool cache_valid = false;

    for (uint32_t local_stick = 0; local_stick < num_output_sticks; ++local_stick) {
        cb_reserve_back(cb_id_out, 1);
        const uint32_t out_l1_addr = get_write_ptr(cb_id_out);
        auto* out = reinterpret_cast<float*>(out_l1_addr);

        const uint32_t global_stick = start_output_stick + local_stick;
        const uint32_t base_output_index = global_stick * stick_width;

        for (uint32_t lane = 0; lane < stick_width; ++lane) {
            const uint32_t output_index = base_output_index + lane;
            if (output_index >= output_length || input_length == 0) {
                out[lane] = 0.0F;
                continue;
            }

            const int32_t logical_index = static_cast<int32_t>(output_index) - static_cast<int32_t>(left_pad);
            const uint32_t source_index = symmetric_index(logical_index, input_length);
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
            out[lane] = cache[source_lane];
        }

        cb_push_back(cb_id_out, 1);
    }

    if (cache_valid) {
        cb_pop_front(cb_id_cache, 1);
    }
}
