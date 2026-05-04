#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_input = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output = get_compile_time_arg_val(1);
    constexpr uint32_t dst = 0;

    const uint32_t tile_page_count = get_arg_val<uint32_t>(0);
    const uint32_t scalar_packed = get_arg_val<uint32_t>(1);

    for (uint32_t page = 0; page < tile_page_count; ++page) {
        cb_wait_front(cb_input, 1);

        tile_regs_acquire();

        copy_tile_to_dst_init_short(cb_input);
        copy_tile(cb_input, 0, dst);

        binop_with_scalar_tile_init();
        mul_unary_tile(dst, scalar_packed);

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_output, 1);
        pack_tile(dst, cb_output);
        cb_push_back(cb_output, 1);

        cb_pop_front(cb_input, 1);
        tile_regs_release();
    }
}
