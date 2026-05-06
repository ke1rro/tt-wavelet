#include <array>
#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "lwt_compute_utils.hpp"

namespace {

constexpr uint32_t kArgsPerStep = 1 + ttwv::device_protocol::step_desc_word_count;
constexpr uint32_t kDstInput0 = 0;
constexpr uint32_t kDstInput1 = 1;
constexpr uint32_t kDstOutput = 2;
constexpr uint32_t kDstBase = 3;
constexpr uint32_t kDstTailOutput = 4;

template <uint32_t K>
void run_step(
    const uint32_t cb_input0,
    const uint32_t cb_input1,
    const uint32_t cb_base,
    const uint32_t cb_output,
    const uint32_t output_group_count,
    const std::array<uint32_t, K>& h_coeffs) {
    for (uint32_t group = 0; group < output_group_count; ++group) {
        tile_regs_acquire();

        cb_wait_front(cb_input0, 1);
        copy_tile_to_dst_init_short(cb_input0);
        copy_tile(cb_input0, 0, kDstInput0);
        cb_pop_front(cb_input0, 1);

        cb_wait_front(cb_input1, 1);
        copy_tile_to_dst_init_short(cb_input1);
        copy_tile(cb_input1, 0, kDstInput1);
        cb_pop_front(cb_input1, 1);

        hstencil_init();
        hstencil_tile<K>(h_coeffs, kDstInput0, kDstInput1, kDstOutput);

        add_binary_tile_init();
        cb_wait_front(cb_base, 1);
        copy_tile_to_dst_init_short(cb_base);
        copy_tile(cb_base, 0, kDstBase);
        add_binary_tile(kDstOutput, kDstBase, kDstOutput);
        cb_pop_front(cb_base, 1);

        // Only the first horizontal block of the tail tile is written by the writer.
        hstencil_tile<K>(h_coeffs, kDstInput1, kDstInput0, kDstTailOutput);

        cb_wait_front(cb_base, 1);
        copy_tile_to_dst_init_short(cb_base);
        copy_tile(cb_base, 0, kDstBase);
        add_binary_tile(kDstTailOutput, kDstBase, kDstTailOutput);
        cb_pop_front(cb_base, 1);

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_output, 2);
        pack_tile(kDstOutput, cb_output, 0);
        pack_tile(kDstTailOutput, cb_output, 1);
        cb_push_back(cb_output, 2);

        tile_regs_release();
    }
}

}  // namespace

template <uint32_t const_arg_base, uint32_t var_arg_base, uint32_t num_steps>
inline void unroll_step(
    const uint32_t cb_input0,
    const uint32_t cb_input1,
    const uint32_t cb_base,
    const uint32_t cb_output) {

    if constexpr (num_steps > 0) {
        constexpr uint32_t K = get_compile_time_arg_val(const_arg_base);

        std::array<uint32_t, K> h_coeffs{};
        #pragma unroll 17
        for (uint32_t j = 0; j < K; ++j) {
            h_coeffs[j] = get_compile_time_arg_val(const_arg_base + 1 + j);
        }

        const uint32_t output_group_count = get_arg_val<uint32_t>(var_arg_base);

        run_step<K>(cb_input0, cb_input1, cb_base, cb_output, output_group_count, h_coeffs);
        unroll_step<const_arg_base + 1 + K, var_arg_base + 1, num_steps - 1>(cb_input0, cb_input1, cb_base, cb_output);
    }
}


void kernel_main() {
    constexpr uint32_t cb_input0 = get_named_compile_time_arg_val("cb_input0");
    constexpr uint32_t cb_input1 = get_named_compile_time_arg_val("cb_input1");
    constexpr uint32_t cb_base = get_named_compile_time_arg_val("cb_base");
    constexpr uint32_t cb_output = get_named_compile_time_arg_val("cb_output");
    constexpr uint32_t num_steps = get_named_compile_time_arg_val("num_steps");

    ckernel::init_sfpu(cb_base, cb_output);

    unroll_step<0, 0, num_steps>(cb_input0, cb_input1, cb_base, cb_output);
}
