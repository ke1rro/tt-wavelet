#include <array>
#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "../include/stencil/api/stencil_mac.h"
#include "../include/stencil/policy.h"

void kernel_main() {
    constexpr uint32_t n_tile_pairs = 1;
    constexpr bool k_bypass_sfpi_for_debug = true;

    constexpr auto cb_base = tt::CBIndex::c_0;
    constexpr auto cb_in0 = tt::CBIndex::c_1;
    constexpr auto cb_in1 = tt::CBIndex::c_2;
    constexpr auto cb_out0 = tt::CBIndex::c_16;

    constexpr uint32_t dst_base = 0;
    constexpr uint32_t dst_in0 = 1;
    constexpr uint32_t dst_in1 = 2;
    constexpr uint32_t dst_out = 3;

    using Policy = ckernel::stencil::StencilMacPolicy<8>;
    constexpr std::array<uint32_t, 2> dst_input_indices = {dst_in0, dst_in1};
    constexpr std::array<float, 2> coefficients = {0.5f, -0.25f};

    init_sfpu(cb_base, cb_out0);
    copy_tile_init(cb_base);

    for (uint32_t i = 0; i < n_tile_pairs; ++i) {
        cb_wait_front(cb_base, 1);
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);

        tile_regs_acquire();

        copy_tile(cb_base, 0, dst_base);

        copy_tile_to_dst_init_short_with_dt(cb_base, cb_in0);
        copy_tile(cb_in0, 0, dst_in0);

        copy_tile_to_dst_init_short_with_dt(cb_in0, cb_in1);
        copy_tile(cb_in1, 0, dst_in1);

        if constexpr (!k_bypass_sfpi_for_debug) {
            stencil_mac_tile<false, 2, Policy>(dst_base, dst_input_indices, dst_out, coefficients);
        }

        tile_regs_commit();

        tile_regs_wait();
        cb_reserve_back(cb_out0, 1);
        constexpr uint32_t pack_src = k_bypass_sfpi_for_debug ? dst_base : dst_out;
        pack_tile(pack_src, cb_out0);
        tile_regs_release();

        cb_push_back(cb_out0, 1);
        cb_pop_front(cb_base, 1);
        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);
    }
}
