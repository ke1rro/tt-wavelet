#include <cstdint>

#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_in1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t dst_reg = 0;
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);

    binary_op_init_common(cb_in0, cb_in1, cb_out);
    mul_tiles_init(cb_in0, cb_in1);

    for (uint32_t tile = 0; tile < n_tiles; ++tile) {
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);

        tile_regs_acquire();
        mul_tiles(cb_in0, cb_in1, 0, 0, dst_reg);
        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_out, 1);
        pack_tile(dst_reg, cb_out);
        cb_push_back(cb_out, 1);

        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);
        tile_regs_release();
    }
}
