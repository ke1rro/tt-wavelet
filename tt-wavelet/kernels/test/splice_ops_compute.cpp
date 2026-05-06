#include <cstdint>

#include "../sfpi/stencil_sfpi.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace {

constexpr uint32_t kOpShift = 0;
constexpr uint32_t kOpRecover1 = 1;
constexpr uint32_t kOpRecover2 = 2;

}  // namespace

void kernel_main() {
    const uint32_t case_count = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_input = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output = get_compile_time_arg_val(1);
    constexpr uint32_t op = get_compile_time_arg_val(2);
    constexpr uint8_t shift_k = static_cast<uint8_t>(get_compile_time_arg_val(3));

    ckernel::init_sfpu(cb_input, cb_output);

    for (uint32_t test_case = 0; test_case < case_count; ++test_case) {
        cb_wait_front(cb_input, 3);

        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_input);
        copy_tile(cb_input, 0, 0);
        copy_tile(cb_input, 1, 1);
        copy_tile(cb_input, 2, 2);

        splice_ops_init();
        if constexpr (op == kOpShift) {
            splice_shift<shift_k>(0, 1);
        } else if constexpr (op == kOpRecover1) {
            splice_recover1(0, 1, 2);
        } else if constexpr (op == kOpRecover2) {
            splice_recover2(0, 1, 2);
        }

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_output, 3);
        pack_tile(0, cb_output, 0);
        pack_tile(1, cb_output, 1);
        pack_tile(2, cb_output, 2);
        cb_push_back(cb_output, 3);

        cb_pop_front(cb_input, 3);
        tile_regs_release();
    }
}
