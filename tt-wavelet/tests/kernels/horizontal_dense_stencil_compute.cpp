#include <array>
#include <cstdint>

// The SFPI header uses MATH in its public wrappers, so common.h must define it
// before the relative include. Keep clang-format from sorting that include
// ahead of the compute API.
// clang-format off
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "../../kernels/sfpi/horizontal_stencil_sfpi.h"
// clang-format on

void kernel_main() {
    constexpr uint32_t k = get_compile_time_arg_val(0);
    constexpr uint32_t cb_source0 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_source1 = get_compile_time_arg_val(2);
    constexpr uint32_t cb_base = get_compile_time_arg_val(3);
    constexpr uint32_t cb_output = get_compile_time_arg_val(4);
    std::array<uint32_t, k> coefficients{};
    for (uint32_t coefficient = 0; coefficient < k; ++coefficient) {
        coefficients[coefficient] = get_arg_val<uint32_t>(coefficient);
    }
    constexpr uint32_t dst_source0 = 0;
    constexpr uint32_t dst_source1 = 1;
    constexpr uint32_t dst_base = 2;
    constexpr uint32_t dst_output = 3;

    ckernel::init_sfpu(cb_base, cb_output);
    tile_regs_acquire();

    cb_wait_front(cb_source0, 1);
    copy_tile_to_dst_init_short(cb_source0);
    copy_tile(cb_source0, 0, dst_source0);
    cb_pop_front(cb_source0, 1);

    cb_wait_front(cb_source1, 1);
    copy_tile_to_dst_init_short(cb_source1);
    copy_tile(cb_source1, 0, dst_source1);
    cb_pop_front(cb_source1, 1);

    cb_wait_front(cb_base, 1);
    copy_tile_to_dst_init_short(cb_base);
    copy_tile(cb_base, 0, dst_base);
    cb_pop_front(cb_base, 1);

    hstencil_init();
    hstencil_dense_tile<k>(coefficients, dst_source0, dst_source1, dst_base, dst_output);

    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_output, 1);
    pack_tile(dst_output, cb_output);
    cb_push_back(cb_output, 1);
    tile_regs_release();
}
