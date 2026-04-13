#include <array>
#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "stencil_sfpi.h"
#include "tt_wavelet/include/device_protocol/step_desc.hpp"

template <uint32_t K>
inline void run_stencil_step(
    const uint32_t cb_input0, const uint32_t cb_input1, const uint32_t cb_base, const uint32_t cb_output) {
    std::array<uint32_t, K> h_coeffs{};
    #pragma unroll 17
    for (uint32_t j = 0; j < K; ++j) {
        h_coeffs[j] = get_arg_val<uint32_t>(ttwv::device_protocol::step_coeffs_arg_idx + j);
    }

    constexpr uint32_t dst_input0 = 0;
    constexpr uint32_t dst_input1 = 1;
    constexpr uint32_t dst_output = 2;
    constexpr uint32_t dst_base = 3;

    tile_regs_acquire();

    cb_wait_front(cb_input0, 1);
    copy_tile_to_dst_init_short(cb_input0);
    copy_tile(cb_input0, 0, dst_input0);
    cb_pop_front(cb_input0, 1);

    cb_wait_front(cb_input1, 1);
    copy_tile_to_dst_init_short(cb_input1);
    copy_tile(cb_input1, 0, dst_input1);
    cb_pop_front(cb_input1, 1);

    hstencil_init();
    hstencil_row<K>(h_coeffs, dst_input0, dst_input1, dst_output);

    add_binary_tile_init();
    cb_wait_front(cb_base, 1);
    copy_tile_to_dst_init_short(cb_base);
    copy_tile(cb_base, 0, dst_base);
    add_binary_tile(dst_output, dst_base, dst_output);
    cb_pop_front(cb_base, 1);

    tile_regs_commit();
    tile_regs_wait();

    cb_reserve_back(cb_output, 1);
    pack_tile(dst_output, cb_output);
    cb_push_back(cb_output, 1);

    tile_regs_release();
}

void kernel_main() {
    constexpr uint32_t cb_input0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_input1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_base = get_compile_time_arg_val(2);
    constexpr uint32_t cb_output = get_compile_time_arg_val(3);
    const uint32_t k = get_arg_val<uint32_t>(ttwv::device_protocol::step_k_arg_idx);

    ckernel::init_sfpu(cb_base, cb_output);

    switch (k) {
        case 1: run_stencil_step<1>(cb_input0, cb_input1, cb_base, cb_output); break;
        case 2: run_stencil_step<2>(cb_input0, cb_input1, cb_base, cb_output); break;
        case 3: run_stencil_step<3>(cb_input0, cb_input1, cb_base, cb_output); break;
        case 4: run_stencil_step<4>(cb_input0, cb_input1, cb_base, cb_output); break;
        case 5: run_stencil_step<5>(cb_input0, cb_input1, cb_base, cb_output); break;
        case 6: run_stencil_step<6>(cb_input0, cb_input1, cb_base, cb_output); break;
        case 7: run_stencil_step<7>(cb_input0, cb_input1, cb_base, cb_output); break;
        case 8: run_stencil_step<8>(cb_input0, cb_input1, cb_base, cb_output); break;
        case 9: run_stencil_step<9>(cb_input0, cb_input1, cb_base, cb_output); break;
        case 10: run_stencil_step<10>(cb_input0, cb_input1, cb_base, cb_output); break;
        case 11: run_stencil_step<11>(cb_input0, cb_input1, cb_base, cb_output); break;
        case 12: run_stencil_step<12>(cb_input0, cb_input1, cb_base, cb_output); break;
        case 13: run_stencil_step<13>(cb_input0, cb_input1, cb_base, cb_output); break;
        case 14: run_stencil_step<14>(cb_input0, cb_input1, cb_base, cb_output); break;
        case 15: run_stencil_step<15>(cb_input0, cb_input1, cb_base, cb_output); break;
        case 16: run_stencil_step<16>(cb_input0, cb_input1, cb_base, cb_output); break;
        case 17: run_stencil_step<17>(cb_input0, cb_input1, cb_base, cb_output); break;
        default: run_stencil_step<1>(cb_input0, cb_input1, cb_base, cb_output); break;
    }
}
