#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "stencil_sfpi.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/compile_time_arg_tmp.hpp"

void kernel_main() {
    constexpr uint32_t K = get_compile_time_arg_val(0);
    constexpr uint32_t cb_halo = get_compile_time_arg_val(1);
    constexpr uint32_t cb_input = get_compile_time_arg_val(2);
    constexpr uint32_t cb_output = get_compile_time_arg_val(3);
    constexpr auto h_coeffs_of_step = fill_array_with_next_n_args<uint32_t, 4, K>();
    (void)h_coeffs_of_step;
    // halo:  [0, 0, ..., 0, 1, 2, 3, ..., 18]
    // input: [19, 20, 21, ..., 32, 0, 0, ..., 0]
    constexpr uint dst_halo = 0;
    constexpr uint dst_input = 1;
    constexpr uint dst_out = 2;

    ckernel::init_sfpu(cb_input, cb_output);

    tile_regs_acquire();
    cb_wait_front(cb_halo, 1);
    copy_tile_to_dst_init_short(cb_halo);
    copy_tile(cb_halo, 0, dst_halo);
    cb_pop_front(cb_halo, 1);

    cb_wait_front(cb_input, 1);
    copy_tile_to_dst_init_short(cb_input);
    copy_tile(cb_input, 0, dst_input);
    cb_pop_front(cb_input, 1);

    MATH((ckernel::sfpu::calculate_stencil_init<K>()));
    MATH((ckernel::sfpu::calculate_stencil<K>(h_coeffs_of_step.data(), dst_halo, dst_input, dst_out)));

    tile_regs_commit();
    tile_regs_wait();

    cb_reserve_back(cb_output, 1);
    pack_tile(dst_out, cb_output);
    cb_push_back(cb_output, 1);

    // Temporary debug expose the copied halo input tiles  to the writer
    // cb_reserve_back(cb_output, 2);
    // pack_tile(dst_halo, cb_output);
    // pack_tile(dst_input, cb_output);
    // cb_push_back(cb_output, 2);
    tile_regs_release();
}
