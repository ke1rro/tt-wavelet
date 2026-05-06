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
inline void run_step(
    const uint32_t cb_input0,
    const uint32_t cb_input1,
    const uint32_t cb_base,
    const uint32_t cb_output,
    const uint32_t arg_coeff_base,
    const uint32_t output_group_count) {
    std::array<uint32_t, K> h_coeffs{};
#pragma unroll 17
    for (uint32_t j = 0; j < K; ++j) {
        h_coeffs[j] = get_arg_val<uint32_t>(arg_coeff_base + j);
    }

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

        splice_ops_init();
        hstencil_spline<K>(h_coeffs, kDstInput0, kDstInput1, kDstOutput, kDstTailOutput);

        add_binary_tile_init();
        cb_wait_front(cb_base, 1);
        copy_tile_to_dst_init_short(cb_base);
        copy_tile(cb_base, 0, kDstBase);
        add_binary_tile(kDstOutput, kDstBase, kDstOutput);
        cb_pop_front(cb_base, 1);

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

void kernel_main() {
    constexpr uint32_t cb_input0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_input1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_base = get_compile_time_arg_val(2);
    constexpr uint32_t cb_output = get_compile_time_arg_val(3);
    constexpr uint8_t k_stencil_width = static_cast<uint8_t>(get_compile_time_arg_val(4));

    const uint32_t num_steps = get_arg_val<uint32_t>(0);

    ckernel::init_sfpu(cb_base, cb_output);

    for (uint32_t step = 0; step < num_steps; ++step) {
        const uint32_t step_arg_base = 1 + step * kArgsPerStep;
        const uint32_t output_group_count = get_arg_val<uint32_t>(step_arg_base);
        const uint32_t desc_arg_base = step_arg_base + 1;
        const uint32_t coeff_arg_base = desc_arg_base + ttwv::device_protocol::step_coeffs_arg_idx;

        run_step<k_stencil_width>(cb_input0, cb_input1, cb_base, cb_output, coeff_arg_base, output_group_count);
    }
}
