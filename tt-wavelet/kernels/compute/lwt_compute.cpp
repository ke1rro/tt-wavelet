#include <array>
#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include <tools/profiler/kernel_profiler.hpp>
#include "lwt_compute_utils.hpp"

namespace {

constexpr uint32_t kArgsPerStep = 1 + ttwv::device_protocol::step_desc_word_count;
constexpr uint32_t kDstInput0 = 0;
constexpr uint32_t kDstInput1 = 1;
constexpr uint32_t kDstOutput = 2;
constexpr uint32_t kDstBase = 3;

template <uint32_t K>
inline void run_step(
    const uint32_t cb_input0,
    const uint32_t cb_input1,
    const uint32_t cb_base,
    const uint32_t cb_output,
    const uint32_t arg_coeff_base,
    const uint32_t output_stick_count) {
    std::array<uint32_t, K> h_coeffs{};
#pragma unroll 17
    for (uint32_t j = 0; j < K; ++j) {
        h_coeffs[j] = get_arg_val<uint32_t>(arg_coeff_base + j);
    }

    for (uint32_t stick = 0; stick < output_stick_count; ++stick) {
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
        hstencil_row<K>(h_coeffs, kDstInput0, kDstInput1, kDstOutput);

        add_binary_tile_init();
        cb_wait_front(cb_base, 1);
        copy_tile_to_dst_init_short(cb_base);
        copy_tile(cb_base, 0, kDstBase);
        add_binary_tile(kDstOutput, kDstBase, kDstOutput);
        cb_pop_front(cb_base, 1);

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_output, 1);
        pack_tile(kDstOutput, cb_output);
        cb_push_back(cb_output, 1);

        tile_regs_release();
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_input0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_input1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_base = get_compile_time_arg_val(2);
    constexpr uint32_t cb_output = get_compile_time_arg_val(3);

    const uint32_t num_steps = get_arg_val<uint32_t>(0);

    ckernel::init_sfpu(cb_base, cb_output);

    for (uint32_t step = 0; step < num_steps; ++step) {
        DeviceZoneScopedN("compute_step");
        const uint32_t step_arg_base = 1 + step * kArgsPerStep;
        const uint32_t output_stick_count = get_arg_val<uint32_t>(step_arg_base);
        const uint32_t desc_arg_base = step_arg_base + 1;
        const uint32_t k = get_arg_val<uint32_t>(desc_arg_base + ttwv::device_protocol::step_k_arg_idx);
        const uint32_t coeff_arg_base = desc_arg_base + ttwv::device_protocol::step_coeffs_arg_idx;

        switch (k) {
            case 1: run_step<1>(cb_input0, cb_input1, cb_base, cb_output, coeff_arg_base, output_stick_count); break;
            case 2: run_step<2>(cb_input0, cb_input1, cb_base, cb_output, coeff_arg_base, output_stick_count); break;
            case 3: run_step<3>(cb_input0, cb_input1, cb_base, cb_output, coeff_arg_base, output_stick_count); break;
            case 4: run_step<4>(cb_input0, cb_input1, cb_base, cb_output, coeff_arg_base, output_stick_count); break;
            case 5: run_step<5>(cb_input0, cb_input1, cb_base, cb_output, coeff_arg_base, output_stick_count); break;
            case 6: run_step<6>(cb_input0, cb_input1, cb_base, cb_output, coeff_arg_base, output_stick_count); break;
            case 7: run_step<7>(cb_input0, cb_input1, cb_base, cb_output, coeff_arg_base, output_stick_count); break;
            case 8: run_step<8>(cb_input0, cb_input1, cb_base, cb_output, coeff_arg_base, output_stick_count); break;
            case 9: run_step<9>(cb_input0, cb_input1, cb_base, cb_output, coeff_arg_base, output_stick_count); break;
            case 10: run_step<10>(cb_input0, cb_input1, cb_base, cb_output, coeff_arg_base, output_stick_count); break;
            case 11: run_step<11>(cb_input0, cb_input1, cb_base, cb_output, coeff_arg_base, output_stick_count); break;
            case 12: run_step<12>(cb_input0, cb_input1, cb_base, cb_output, coeff_arg_base, output_stick_count); break;
            case 13: run_step<13>(cb_input0, cb_input1, cb_base, cb_output, coeff_arg_base, output_stick_count); break;
            case 14: run_step<14>(cb_input0, cb_input1, cb_base, cb_output, coeff_arg_base, output_stick_count); break;
            case 15: run_step<15>(cb_input0, cb_input1, cb_base, cb_output, coeff_arg_base, output_stick_count); break;
            case 16: run_step<16>(cb_input0, cb_input1, cb_base, cb_output, coeff_arg_base, output_stick_count); break;
            case 17: run_step<17>(cb_input0, cb_input1, cb_base, cb_output, coeff_arg_base, output_stick_count); break;
            default: run_step<1>(cb_input0, cb_input1, cb_base, cb_output, coeff_arg_base, output_stick_count); break;
        }
    }
}
