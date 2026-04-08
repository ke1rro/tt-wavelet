// stencil_reader.cpp
//
// Reader dataflow kernel for the stencil compute pipeline.
//
// Reads the even (or odd) signal sticks from DRAM and pushes them as
// row-major sticks into two circular buffers:
//   CB_HALO (cb0):  the "previous" 32 columns — first tile is all zeros (17-k pad)
//   CB_CUR  (cb1):  the "current"  32 columns
//
// For the very first tile, the halo is filled with zeros.
// For subsequent tiles, the halo is the previous current tile.
//
// The 17-k stencil padding is applied by shifting the signal within the first
// tile: the first (17-k) positions of row 0 are filled with zeros, then the
// signal starts at position (17-k). This ensures the stencil convolution
// result begins at a tile-aligned offset.
//
// Compile-time args:
//   0: cb_halo        — CB index for halo output
//   1: cb_cur         — CB index for current output
//   2: stick_nbytes   — byte size of one stick (aligned)
//   3: cb_cache       — CB index for DRAM read cache
//
// Runtime args:
//   0: src_addr       — DRAM base address of signal buffer
//   1: signal_length  — number of signal samples
//   2: stencil_k      — filter length (1 < k < 18)
//   3: num_tiles      — number of tile pairs to produce

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t signal_length = get_arg_val<uint32_t>(1);
    const uint32_t stencil_k = get_arg_val<uint32_t>(2);
    const uint32_t num_tiles = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_halo = get_compile_time_arg_val(0);
    constexpr uint32_t cb_cur = get_compile_time_arg_val(1);
    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(2);
    constexpr uint32_t cb_cache = get_compile_time_arg_val(3);
    constexpr uint32_t stick_width = stick_nbytes / sizeof(float);  // 32
    constexpr auto src_args = TensorAccessorArgs<4>();
    const auto src = TensorAccessor(src_args, src_addr, stick_nbytes);

    const uint32_t pad_amount = 17 - stencil_k;  // 17-k zeros prepended

    // Track how many signal samples we've consumed
    uint32_t signal_pos = 0;

    // Helper: push a stick of zeros to a CB
    auto push_zero_stick = [&](const uint32_t cb_id) {
        cb_reserve_back(cb_id, 1);
        auto* ptr = reinterpret_cast<float*>(get_write_ptr(cb_id));
        for (uint32_t i = 0; i < stick_width; i++) {
            ptr[i] = 0.0F;
        }
        cb_push_back(cb_id, 1);
    };

    // Helper: read a stick from DRAM into the cache, return pointer
    auto read_signal_stick = [&](const uint32_t stick_idx) {
        cb_reserve_back(cb_cache, 1);
        const uint32_t cache_l1_addr = get_write_ptr(cb_cache);
        const uint64_t noc_addr = src.get_noc_addr(stick_idx);
        noc_async_read(noc_addr, cache_l1_addr, stick_nbytes);
        noc_async_read_barrier();
        cb_push_back(cb_cache, 1);
        cb_wait_front(cb_cache, 1);
        return reinterpret_cast<const float*>(get_read_ptr(cb_cache));
    };

    // Helper: push a stick of signal data (with partial fill) to CB
    auto push_signal_stick = [&](const uint32_t cb_id,
                                 const uint32_t start_col,
                                 const float* signal_data,
                                 const uint32_t signal_start,
                                 const uint32_t count) {
        cb_reserve_back(cb_id, 1);
        auto* ptr = reinterpret_cast<float*>(get_write_ptr(cb_id));
        // Zero fill
        for (uint32_t i = 0; i < stick_width; i++) {
            ptr[i] = 0.0F;
        }
        // Copy signal data
        for (uint32_t i = 0; i < count && (start_col + i) < stick_width; i++) {
            ptr[start_col + i] = signal_data[signal_start + i];
        }
        cb_push_back(cb_id, 1);
    };

    for (uint32_t tile = 0; tile < num_tiles; tile++) {
        // Each tile is composed of 32 sticks (32 rows × 32 cols).
        // For 1D signals, only row 0 has data; rows 1-31 are zero.

        if (tile == 0) {
            // ── HALO tile: all zeros (no previous data) ──
            for (uint32_t row = 0; row < 32; row++) {
                push_zero_stick(cb_halo);
            }

            // ── CURRENT tile: row 0 has pad_amount zeros then signal data ──
            // Row 0: [0...0 (pad_amount) | s0, s1, ..., s_{32-pad_amount-1}]
            {
                cb_reserve_back(cb_cur, 1);
                auto* ptr = reinterpret_cast<float*>(get_write_ptr(cb_cur));
                for (uint32_t i = 0; i < stick_width; i++) {
                    ptr[i] = 0.0F;
                }

                // Read signal data and fill after the padding
                const uint32_t samples_in_first_stick = stick_width - pad_amount;
                if (signal_length > 0) {
                    const float* sig = read_signal_stick(0);
                    const uint32_t to_copy =
                        (signal_length < samples_in_first_stick) ? signal_length : samples_in_first_stick;
                    for (uint32_t i = 0; i < to_copy; i++) {
                        ptr[pad_amount + i] = sig[i];
                    }
                    signal_pos = to_copy;
                    cb_pop_front(cb_cache, 1);
                }
                cb_push_back(cb_cur, 1);
            }

            // Rows 1-31: zeros
            for (uint32_t row = 1; row < 32; row++) {
                push_zero_stick(cb_cur);
            }
        } else {
            // ── HALO tile: previous 32 columns ──
            // For simplicity in this 1D case with short signals,
            // we repeat the pattern: row 0 has the next 32 signal samples,
            // rows 1-31 are zero.

            // Halo row 0: fill from signal
            {
                cb_reserve_back(cb_halo, 1);
                auto* ptr = reinterpret_cast<float*>(get_write_ptr(cb_halo));
                for (uint32_t i = 0; i < stick_width; i++) {
                    ptr[i] = 0.0F;
                }
                // The halo is the previous current tile's data.
                // For tile > 0, signal_pos already points past consumed data.
                // The halo data has already been pushed to compute in the
                // previous iteration's current tile.
                // For a sliding window, we'd copy the previous current here.
                // For now, re-read the relevant signal range.
                const uint32_t halo_start = signal_pos - (stick_width - (tile == 1 ? pad_amount : 0));
                const uint32_t halo_stick = halo_start / stick_width;
                if (halo_start < signal_length) {
                    const float* sig = read_signal_stick(halo_stick);
                    const uint32_t lane_start = halo_start % stick_width;
                    for (uint32_t i = 0; i < stick_width && (halo_start + i) < signal_length; i++) {
                        ptr[i] = sig[lane_start + i];
                    }
                    cb_pop_front(cb_cache, 1);
                }
                cb_push_back(cb_halo, 1);
            }
            for (uint32_t row = 1; row < 32; row++) {
                push_zero_stick(cb_halo);
            }

            // Current row 0: next 32 signal samples
            {
                cb_reserve_back(cb_cur, 1);
                auto* ptr = reinterpret_cast<float*>(get_write_ptr(cb_cur));
                for (uint32_t i = 0; i < stick_width; i++) {
                    ptr[i] = 0.0F;
                }
                if (signal_pos < signal_length) {
                    const uint32_t cur_stick = signal_pos / stick_width;
                    const float* sig = read_signal_stick(cur_stick);
                    const uint32_t lane_start = signal_pos % stick_width;
                    const uint32_t to_copy =
                        stick_width < (signal_length - signal_pos) ? stick_width : (signal_length - signal_pos);
                    for (uint32_t i = 0; i < to_copy; i++) {
                        ptr[i] = sig[lane_start + i];
                    }
                    signal_pos += to_copy;
                    cb_pop_front(cb_cache, 1);
                }
                cb_push_back(cb_cur, 1);
            }
            for (uint32_t row = 1; row < 32; row++) {
                push_zero_stick(cb_cur);
            }
        }
    }
}
