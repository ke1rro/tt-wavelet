#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"

void kernel_main() {
    uint32_t n_tile_pairs = get_arg_val<uint32_t>(0);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out0 = tt::CBIndex::c_16;

    constexpr uint32_t dst_reg_low = 0;
    constexpr uint32_t dst_reg_high = 1;
    constexpr uint32_t inv_sqrt2_bits = 0x3f3504f3U;

    binary_op_init_common(cb_in0, cb_in1, cb_out0);

    for (uint32_t i = 0; i < n_tile_pairs; i++) {
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);

        tile_regs_acquire();

        // Haar lifting
        // low  = (even + odd) / sqrt(2), high = (odd - even) / sqrt(2)
        add_tiles_init(cb_in0, cb_in1);
        add_tiles(cb_in0, cb_in1, /*offset_0*/ 0, /*offset_1*/ 0, dst_reg_low);

        sub_tiles_init(cb_in1, cb_in0);
        sub_tiles(cb_in1, cb_in0, /*offset_0*/ 0, /*offset_1*/ 0, dst_reg_high);

        binop_with_scalar_tile_init();
        mul_unary_tile(dst_reg_low, inv_sqrt2_bits);
        mul_unary_tile(dst_reg_high, inv_sqrt2_bits);

        tile_regs_commit();

        tile_regs_wait();
        cb_reserve_back(cb_out0, 2);
        pack_tile(dst_reg_low, cb_out0);
        pack_tile(dst_reg_high, cb_out0);
        tile_regs_release();

        cb_push_back(cb_out0, 2);
        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);
    }
}
