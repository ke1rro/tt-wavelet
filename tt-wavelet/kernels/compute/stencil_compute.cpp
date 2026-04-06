// stencil_compute.cpp
//
// Compute kernel for horizontal 1D stencil convolution.
//
// Dataflow:
//   CB_IN_HALO  (cb0) ──┐
//   CB_IN_CUR   (cb1) ──┼──► SFPU stencil ──► CB_OUT (cb16)
//                        │
// The stencil operates on tilized data. The reader provides two tiles:
//   - Halo tile:    previous 32 columns (face0 = even cols, face1 = odd cols)
//   - Current tile: current 32 columns
//
// The output is one tile containing the valid stencil results for the
// current 16-column half (output face0 = even, face1 = odd).
//
// Compile-time args:
//   0: K              — stencil filter length (1 < K < 18)
//   1: num_tiles      — number of tile pairs to process
//   2..2+K-1: h_packed[K] — filter coefficients as uint32_t bit-casts of float32

#include <cstdint>

#include "ckernel_stencil.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/untilize.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t K = get_compile_time_arg_val(0);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(1);

    // Unpack filter coefficients from compile-time args
    uint32_t h_packed[K];
    if constexpr (K > 0) {
        h_packed[0] = get_compile_time_arg_val(2);
    }
    if constexpr (K > 1) {
        h_packed[1] = get_compile_time_arg_val(3);
    }
    if constexpr (K > 2) {
        h_packed[2] = get_compile_time_arg_val(4);
    }
    if constexpr (K > 3) {
        h_packed[3] = get_compile_time_arg_val(5);
    }
    if constexpr (K > 4) {
        h_packed[4] = get_compile_time_arg_val(6);
    }
    if constexpr (K > 5) {
        h_packed[5] = get_compile_time_arg_val(7);
    }
    if constexpr (K > 6) {
        h_packed[6] = get_compile_time_arg_val(8);
    }
    if constexpr (K > 7) {
        h_packed[7] = get_compile_time_arg_val(9);
    }
    if constexpr (K > 8) {
        h_packed[8] = get_compile_time_arg_val(10);
    }
    if constexpr (K > 9) {
        h_packed[9] = get_compile_time_arg_val(11);
    }
    if constexpr (K > 10) {
        h_packed[10] = get_compile_time_arg_val(12);
    }
    if constexpr (K > 11) {
        h_packed[11] = get_compile_time_arg_val(13);
    }
    if constexpr (K > 12) {
        h_packed[12] = get_compile_time_arg_val(14);
    }
    if constexpr (K > 13) {
        h_packed[13] = get_compile_time_arg_val(15);
    }
    if constexpr (K > 14) {
        h_packed[14] = get_compile_time_arg_val(16);
    }
    if constexpr (K > 15) {
        h_packed[15] = get_compile_time_arg_val(17);
    }
    if constexpr (K > 16) {
        h_packed[16] = get_compile_time_arg_val(18);
    }

    constexpr auto cb_in_halo = tt::CBIndex::c_0;       // halo tile (row-major sticks)
    constexpr auto cb_in_cur = tt::CBIndex::c_1;        // current tile (row-major sticks)
    constexpr auto cb_out = tt::CBIndex::c_16;          // output tile (row-major sticks)
    constexpr auto cb_tilized_halo = tt::CBIndex::c_2;  // tilized halo
    constexpr auto cb_tilized_cur = tt::CBIndex::c_3;   // tilized current
    constexpr auto cb_tilized_out = tt::CBIndex::c_4;   // tilized output

    // DST slot assignments — each face gets its own DST tile slot
    // (the stencil addresses faces via dst_index * 64)
    constexpr uint dst_halo = 0;  // halo tile in DST
    constexpr uint dst_cur = 1;   // current tile in DST
    constexpr uint dst_out = 2;   // output tile in DST

    ckernel::sfpu::calculate_stencil_init<K>();

    for (uint32_t t = 0; t < num_tiles; t++) {
        // ── Tilize input sticks into tile format ──
        // Halo: 32 row-major sticks → 1 tile
        tilize_init_short(cb_in_halo, 1);
        cb_wait_front(cb_in_halo, 32);
        cb_reserve_back(cb_tilized_halo, 1);
        tilize_block(cb_in_halo, 1, cb_tilized_halo);
        cb_push_back(cb_tilized_halo, 1);
        cb_pop_front(cb_in_halo, 32);
        tilize_uninit(cb_in_halo);

        // Current: 32 row-major sticks → 1 tile
        tilize_init_short(cb_in_cur, 1);
        cb_wait_front(cb_in_cur, 32);
        cb_reserve_back(cb_tilized_cur, 1);
        tilize_block(cb_in_cur, 1, cb_tilized_cur);
        cb_push_back(cb_tilized_cur, 1);
        cb_pop_front(cb_in_cur, 32);
        tilize_uninit(cb_in_cur);

        // ── Copy tiles from CB to DST ──
        tile_regs_acquire();

        cb_wait_front(cb_tilized_halo, 1);
        copy_tile_to_dst_init_short(cb_tilized_halo);
        copy_tile(cb_tilized_halo, 0, dst_halo);
        cb_pop_front(cb_tilized_halo, 1);

        cb_wait_front(cb_tilized_cur, 1);
        copy_tile_to_dst_init_short(cb_tilized_cur);
        copy_tile(cb_tilized_cur, 0, dst_cur);
        cb_pop_front(cb_tilized_cur, 1);

        // ── Run SFPU stencil ──
        // The stencil operates on face-level data within each tile:
        //   Face 0 (even cols) and Face 1 (odd cols)
        // For the halo tile (DST slot 0): faces are at dst=0
        // For the current tile (DST slot 1): faces are at dst=1
        // Output to DST slot 2
        ckernel::sfpu::calculate_stencil<K, 8>(
            h_packed,
            dst_halo,  // f_e_0: halo even cols
            dst_halo,  // f_o_0: halo odd cols (same tile, different face)
            dst_cur,   // f_e_1: current even cols
            dst_cur,   // f_o_1: current odd cols
            dst_out,   // g_e: output even cols
            dst_out    // g_o: output odd cols
        );

        // ── Pack output from DST to CB ──
        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_tilized_out, 1);
        pack_tile(dst_out, cb_tilized_out);
        cb_push_back(cb_tilized_out, 1);

        tile_regs_release();

        // ── Untilize output back to row-major sticks ──
        untilize_init_short(cb_tilized_out);
        cb_wait_front(cb_tilized_out, 1);
        cb_reserve_back(cb_out, 32);
        untilize_block(cb_tilized_out, 1, cb_out);
        cb_push_back(cb_out, 32);
        cb_pop_front(cb_tilized_out, 1);
        untilize_uninit(cb_tilized_out);
    }
}
}  // namespace NAMESPACE
