#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "stencil_sfpi.h"
#include "../tt_wavelet/include/device_step_desc.hpp"

template <uint32_t K>
inline void run_stencil_step(const uint32_t cb_halo, const uint32_t cb_input, const uint32_t cb_output) {
    std::array<uint32_t, K> h_coeffs_of_step{};
    #pragma unroll 17
    for (uint32_t j = 0; j < K; ++j) {
        h_coeffs_of_step[j] = get_arg_val<uint32_t>(ttwv::device::step_coeffs_arg_idx + j);
    }

    constexpr uint32_t dst_halo = 0;
    constexpr uint32_t dst_input = 1;
    constexpr uint32_t dst_out = 2;

    tile_regs_acquire();
    cb_wait_front(cb_halo, 1);
    copy_tile_to_dst_init_short(cb_halo);
    copy_tile(cb_halo, 0, dst_halo);
    cb_pop_front(cb_halo, 1);

    cb_wait_front(cb_input, 1);
    copy_tile_to_dst_init_short(cb_input);
    copy_tile(cb_input, 0, dst_input);
    cb_pop_front(cb_input, 1);

    hstencil_init();
    hstencil_row<K>(h_coeffs_of_step, dst_halo, dst_input, dst_out);

    tile_regs_commit();
    tile_regs_wait();

    cb_reserve_back(cb_output, 1);
    pack_tile(dst_out, cb_output);
    cb_push_back(cb_output, 1);
    tile_regs_release();
}

void kernel_main() {
    constexpr uint32_t cb_halo = get_compile_time_arg_val(0);
    constexpr uint32_t cb_input = get_compile_time_arg_val(1);
    constexpr uint32_t cb_output = get_compile_time_arg_val(2);
    const uint32_t k = get_arg_val<uint32_t>(ttwv::device::step_k_arg_idx);

    ckernel::init_sfpu(cb_input, cb_output);
    switch (k) {
        case 1: run_stencil_step<1>(cb_halo, cb_input, cb_output); break;
        case 2: run_stencil_step<2>(cb_halo, cb_input, cb_output); break;
        case 3: run_stencil_step<3>(cb_halo, cb_input, cb_output); break;
        case 4: run_stencil_step<4>(cb_halo, cb_input, cb_output); break;
        case 5: run_stencil_step<5>(cb_halo, cb_input, cb_output); break;
        case 6: run_stencil_step<6>(cb_halo, cb_input, cb_output); break;
        case 7: run_stencil_step<7>(cb_halo, cb_input, cb_output); break;
        case 8: run_stencil_step<8>(cb_halo, cb_input, cb_output); break;
        case 9: run_stencil_step<9>(cb_halo, cb_input, cb_output); break;
        case 10: run_stencil_step<10>(cb_halo, cb_input, cb_output); break;
        case 11: run_stencil_step<11>(cb_halo, cb_input, cb_output); break;
        case 12: run_stencil_step<12>(cb_halo, cb_input, cb_output); break;
        case 13: run_stencil_step<13>(cb_halo, cb_input, cb_output); break;
        case 14: run_stencil_step<14>(cb_halo, cb_input, cb_output); break;
        case 15: run_stencil_step<15>(cb_halo, cb_input, cb_output); break;
        case 16: run_stencil_step<16>(cb_halo, cb_input, cb_output); break;
        case 17: run_stencil_step<17>(cb_halo, cb_input, cb_output); break;
        default: run_stencil_step<1>(cb_halo, cb_input, cb_output); break;
    }
}
