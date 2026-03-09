// tt_metal/programming_examples/eltwise_sfpu/kernels/compute/eltwise_sfpu.cpp
#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/tile_move_copy.h"

void kernel_main() {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);

    // Initialize the SFPU
    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_16);
    // Setup the SFPU for exponential operation
    exp_tile_init();
    for (uint32_t i = 0; i < n_tiles; i++) {
        cb_wait_front(tt::CBIndex::c_0, 1);
        // Make sure and acquire data before running the SFPU operation
        tile_regs_acquire();
        // Copy the tile from the circular buffer offset 0 to the tile registers 0
        copy_tile(tt::CBIndex::c_0, /*offset*/ 0, /*register_offset*/ 0);

        // Invoke the SFPU exponential operation on tile 0
        exp_tile(0);
        tile_regs_commit();
        tile_regs_wait();

        // Clean up and prepare for the next iteration
        cb_reserve_back(tt::CBIndex::c_16, 1);
        pack_tile(0, tt::CBIndex::c_16);  // copy tile 0 from the registers to the CB
        cb_pop_front(tt::CBIndex::c_0, 1);
        tile_regs_release();
        cb_push_back(tt::CBIndex::c_16, 1);
    }
}
