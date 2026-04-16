#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include <tools/profiler/kernel_profiler.hpp>

void kernel_main() {
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_in1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t dst_in0 = 0;
    constexpr uint32_t dst_in1 = 1;
    constexpr uint32_t dst_out = 2;
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);

    ckernel::init_sfpu(cb_in0, cb_out);
    mul_binary_tile_init();

    for (uint32_t tile = 0; tile < n_tiles; ++tile) {
        DeviceZoneScopedN("mul_tile");
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);

        tile_regs_acquire();

        copy_tile_to_dst_init_short(cb_in0);
        copy_tile(cb_in0, 0, dst_in0);

        copy_tile_to_dst_init_short(cb_in1);
        copy_tile(cb_in1, 0, dst_in1);

        mul_binary_tile(dst_in0, dst_in1, dst_out);

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_out, 1);
        pack_tile(dst_out, cb_out);
        cb_push_back(cb_out, 1);

        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);
        tile_regs_release();
    }
}
