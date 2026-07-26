#include <array>
#include <cstdint>

// The SFPI header uses MATH in its public wrappers, so common.h must define it
// before the relative include. Keep clang-format from sorting that include
// ahead of the compute API.
// clang-format off
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "../../kernels/sfpi/vertical_stencil_sfpi.h"
// clang-format on

void kernel_main() {
    constexpr uint32_t k = get_compile_time_arg_val(0);
    constexpr uint32_t cb_input0 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_input1 = get_compile_time_arg_val(2);
    constexpr uint32_t cb_base = get_compile_time_arg_val(3);
    constexpr uint32_t cb_output = get_compile_time_arg_val(4);
    std::array<uint32_t, k> coefficients{};
    for (uint32_t coefficient = 0; coefficient < k; ++coefficient) {
        coefficients[coefficient] = get_arg_val<uint32_t>(coefficient);
    }
    constexpr uint32_t dst_input0 = 0;
    constexpr uint32_t dst_input1 = 1;
    constexpr uint32_t dst_base = 2;
    constexpr uint32_t dst_output = 3;

    ckernel::init_sfpu(cb_input0, cb_output);
    tile_regs_acquire();

    cb_wait_front(cb_input0, 1);
    copy_tile_to_dst_init_short(cb_input0);
    copy_tile(cb_input0, 0, dst_input0);
    cb_pop_front(cb_input0, 1);

    cb_wait_front(cb_input1, 1);
    copy_tile_to_dst_init_short(cb_input1);
    copy_tile(cb_input1, 0, dst_input1);
    cb_pop_front(cb_input1, 1);

    cb_wait_front(cb_base, 1);
    copy_tile_to_dst_init_short(cb_base);
    copy_tile(cb_base, 0, dst_base);
    cb_pop_front(cb_base, 1);

    vstencil_init();
    vstencil_tile<k>(coefficients, dst_input0, dst_input1, dst_output, dst_base);

    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_output, 1);
    pack_tile(dst_output, cb_output);
    cb_push_back(cb_output, 1);
    tile_regs_release();
}
