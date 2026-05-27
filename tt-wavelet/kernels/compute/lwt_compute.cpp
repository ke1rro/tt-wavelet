#include <array>
#include <cstdint>
#include <utility>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/copy_dest_values.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "lwt_compute_utils.hpp"

namespace {

constexpr uint32_t kDstEven0 = 0;
constexpr uint32_t kDstEven1 = 1;
constexpr uint32_t kDstOdd0 = 2;
constexpr uint32_t kDstOdd1 = 3;
constexpr uint32_t kDstTmp0 = 4;
constexpr uint32_t kDstTmp1 = 5;
constexpr uint32_t kDstBaseRecover = 6;
constexpr uint32_t kDstPredictionRecover = 7;

constexpr uint8_t kStepTypePredict = 0;
constexpr uint8_t kStepTypeUpdate = 1;
constexpr uint8_t kStepTypeScaleEven = 2;
constexpr uint8_t kStepTypeScaleOdd = 3;
constexpr uint8_t kStepTypeSwap = 4;

#ifndef LWT_PROGRAM_CASES
#define LWT_PROGRAM_CASES
#endif

__attribute__((noinline, optimize("no-tree-switch-conversion", "no-jump-tables"))) uint32_t
read_program_word(const uint32_t index) {
    switch (index) {
        LWT_PROGRAM_CASES
        default: return 0;
    }
}

template <uint32_t K>
inline std::array<uint32_t, K> read_coefficients(const uint32_t* words) {
    std::array<uint32_t, K> coefficients{};
    for (uint32_t index = 0; index < K; ++index) {
        coefficients[index] = words[index];
    }
    return coefficients;
}

template <uint32_t K>
inline void run_stencil(
    const std::array<uint32_t, K>& coefficients,
    const uint32_t source0,
    const uint32_t source1,
    const uint32_t output0,
    const uint32_t output1) {
    static_assert(K > 1, "K=1 predict/update is handled by the pointwise path");
    hstencil_spline<K>(coefficients, source0, source1, output0, output1);
}

template <uint32_t K>
inline void step_predict(
    const uint32_t source_even,
    const uint32_t source_odd,
    const uint32_t destination_even,
    const uint32_t destination_odd,
    const uint32_t prediction,
    const uint32_t base_recover_state,
    const uint32_t prediction_recover_state,
    int32_t& even_shift,
    int32_t& odd_shift,
    const uint32_t splice_count,
    const int32_t kernel_shift,
    const std::array<uint32_t, K>& coefficients) {
    static_assert(K > 0 && K <= 17, "K must be in the range [1, 17]");
    const int32_t convolution_shift = even_shift + kernel_shift + static_cast<int32_t>(K) - 1;
    const int32_t output_shift = convolution_shift < odd_shift ? convolution_shift : odd_shift;
    const uint32_t base_alignment = static_cast<uint32_t>(odd_shift - output_shift);
    const uint32_t prediction_alignment = static_cast<uint32_t>(convolution_shift - output_shift);

    cb_wait_front(source_even, splice_count * 2);
    cb_wait_front(source_odd, splice_count * 2);
    cb_reserve_back(destination_even, splice_count * 2);
    cb_reserve_back(destination_odd, splice_count * 2);
    if constexpr (K > 1) {
        cb_reserve_back(prediction, splice_count * 2);
    }

    for (uint32_t splice = 0; splice < splice_count; ++splice) {
        if (splice > 0 && base_alignment > 0) {
            cb_wait_front(base_recover_state, 1);
        }
        if (splice > 0 && prediction_alignment > 0) {
            cb_wait_front(prediction_recover_state, 1);
        }

        tile_regs_acquire();

        copy_tile_to_dst_init_short(source_even);
        copy_tile(source_even, splice * 2, kDstEven0);
        copy_tile(source_even, splice * 2 + 1, kDstEven1);
        copy_tile_to_dst_init_short(source_odd);
        copy_tile(source_odd, splice * 2, kDstOdd0);
        copy_tile(source_odd, splice * 2 + 1, kDstOdd1);

        if (base_alignment > 0) {
            if (splice == 0) {
                copy_dest_values_init();
                copy_dest_values(kDstBaseRecover, kDstOdd0);
            } else {
                copy_tile_to_dst_init_short(base_recover_state);
                copy_tile(base_recover_state, 0, kDstBaseRecover);
                cb_pop_front(base_recover_state, 1);
            }
            splice_ops_init();
            shift_splice(kDstOdd0, kDstOdd1, base_alignment);
            recover1_splice(kDstOdd0, kDstOdd1, kDstBaseRecover);
        }

        copy_dest_values_init();
        copy_dest_values(kDstTmp0, kDstEven0);
        copy_dest_values(kDstTmp1, kDstEven1);

        if (prediction_alignment > 0) {
            if (splice == 0) {
                copy_dest_values_init();
                copy_dest_values(kDstPredictionRecover, kDstTmp0);
            } else {
                copy_tile_to_dst_init_short(prediction_recover_state);
                copy_tile(prediction_recover_state, 0, kDstPredictionRecover);
                cb_pop_front(prediction_recover_state, 1);
            }
            splice_ops_init();
            shift_splice(kDstTmp0, kDstTmp1, prediction_alignment);
            recover1_splice(kDstTmp0, kDstTmp1, kDstPredictionRecover);
        }

        if constexpr (K == 1) {
            binop_with_scalar_tile_init();
            mul_unary_tile(kDstTmp0, coefficients[0]);
            mul_unary_tile(kDstTmp1, coefficients[0]);
        }

        if constexpr (K == 1) {
            add_binary_tile_init();
            add_binary_tile(kDstTmp0, kDstOdd0, kDstOdd0);
            add_binary_tile(kDstTmp1, kDstOdd1, kDstOdd1);
        }

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(kDstOdd0, destination_odd, splice * 2);
        pack_tile(kDstOdd1, destination_odd, splice * 2 + 1);
        pack_tile(kDstEven0, destination_even, splice * 2);
        pack_tile(kDstEven1, destination_even, splice * 2 + 1);
        if constexpr (K > 1) {
            pack_tile(kDstTmp0, prediction, splice * 2);
            pack_tile(kDstTmp1, prediction, splice * 2 + 1);
        }
        if (base_alignment > 0) {
            cb_reserve_back(base_recover_state, 1);
            pack_tile(kDstBaseRecover, base_recover_state);
            cb_push_back(base_recover_state, 1);
        }
        if (prediction_alignment > 0) {
            cb_reserve_back(prediction_recover_state, 1);
            pack_tile(kDstPredictionRecover, prediction_recover_state);
            cb_push_back(prediction_recover_state, 1);
        }
        tile_regs_release();
    }

    if (base_alignment > 0) {
        cb_wait_front(base_recover_state, 1);
        cb_pop_front(base_recover_state, 1);
    }
    if (prediction_alignment > 0) {
        cb_wait_front(prediction_recover_state, 1);
        cb_pop_front(prediction_recover_state, 1);
    }

    cb_pop_front(source_even, splice_count * 2);
    cb_pop_front(source_odd, splice_count * 2);
    cb_push_back(destination_even, splice_count * 2);
    cb_push_back(destination_odd, splice_count * 2);

    if constexpr (K > 1) {
        cb_push_back(prediction, splice_count * 2);
        cb_wait_front(prediction, splice_count * 2);
        cb_reserve_back(prediction, splice_count * 2);
        for (uint32_t splice = 0; splice < splice_count; ++splice) {
            tile_regs_acquire();
            copy_tile_to_dst_init_short(prediction);
            copy_tile(prediction, splice * 2, kDstTmp0);
            copy_tile(prediction, splice * 2 + 1, kDstTmp1);
            run_stencil<K>(coefficients, kDstTmp0, kDstTmp1, kDstEven0, kDstEven1);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(kDstEven0, prediction, splice * 2);
            pack_tile(kDstEven1, prediction, splice * 2 + 1);
            tile_regs_release();
        }
        cb_pop_front(prediction, splice_count * 2);
        cb_push_back(prediction, splice_count * 2);

        cb_wait_front(prediction, splice_count * 2);
        cb_wait_front(destination_odd, splice_count * 2);
        cb_reserve_back(destination_odd, splice_count * 2);
        for (uint32_t splice = splice_count; splice > 0; --splice) {
            if (splice < splice_count) {
                cb_wait_front(prediction_recover_state, 1);
            }

            tile_regs_acquire();
            copy_tile_to_dst_init_short(prediction);
            copy_tile(prediction, (splice - 1) * 2, kDstTmp0);
            copy_tile(prediction, (splice - 1) * 2 + 1, kDstTmp1);
            copy_tile_to_dst_init_short(destination_odd);
            copy_tile(destination_odd, (splice - 1) * 2, kDstOdd0);
            copy_tile(destination_odd, (splice - 1) * 2 + 1, kDstOdd1);
            if (splice == splice_count) {
                copy_dest_values_init();
                copy_dest_values(kDstPredictionRecover, kDstTmp1);
            } else {
                copy_tile_to_dst_init_short(prediction_recover_state);
                copy_tile(prediction_recover_state, 0, kDstPredictionRecover);
                cb_pop_front(prediction_recover_state, 1);
            }
            splice_ops_init();
            recover2_splice(kDstTmp0, kDstTmp1, kDstPredictionRecover);
            add_binary_tile_init();
            add_binary_tile(kDstTmp0, kDstOdd0, kDstOdd0);
            add_binary_tile(kDstTmp1, kDstOdd1, kDstOdd1);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(kDstOdd0, destination_odd, (splice - 1) * 2);
            pack_tile(kDstOdd1, destination_odd, (splice - 1) * 2 + 1);
            cb_reserve_back(prediction_recover_state, 1);
            pack_tile(kDstPredictionRecover, prediction_recover_state);
            cb_push_back(prediction_recover_state, 1);
            tile_regs_release();
        }
        cb_wait_front(prediction_recover_state, 1);
        cb_pop_front(prediction_recover_state, 1);
        cb_pop_front(prediction, splice_count * 2);
        cb_pop_front(destination_odd, splice_count * 2);
        cb_push_back(destination_odd, splice_count * 2);
    }

    odd_shift = output_shift;
}

inline void copy_scale_stream(
    const uint32_t source,
    const uint32_t destination,
    const uint32_t splice_count,
    const uint32_t coefficient,
    const bool apply_scale) {
    cb_wait_front(source, splice_count * 2);
    cb_reserve_back(destination, splice_count * 2);

    for (uint32_t splice = 0; splice < splice_count; ++splice) {
        tile_regs_acquire();
        copy_tile_to_dst_init_short(source);
        copy_tile(source, splice * 2, kDstEven0);
        copy_tile(source, splice * 2 + 1, kDstEven1);
        if (apply_scale) {
            binop_with_scalar_tile_init();
            mul_unary_tile(kDstEven0, coefficient);
            mul_unary_tile(kDstEven1, coefficient);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(kDstEven0, destination, splice * 2);
        pack_tile(kDstEven1, destination, splice * 2 + 1);
        tile_regs_release();
    }

    cb_pop_front(source, splice_count * 2);
    cb_push_back(destination, splice_count * 2);
}

inline void step_scale(
    const uint32_t source_even,
    const uint32_t source_odd,
    const uint32_t destination_even,
    const uint32_t destination_odd,
    const uint32_t splice_count,
    const uint32_t coefficient,
    const bool scale_even) {
    copy_scale_stream(source_even, destination_even, splice_count, coefficient, scale_even);
    copy_scale_stream(source_odd, destination_odd, splice_count, coefficient, !scale_even);
}

template <uint32_t K>
inline void run_predict_update_step(
    const bool predict,
    const uint32_t source_even,
    const uint32_t source_odd,
    const uint32_t destination_even,
    const uint32_t destination_odd,
    const uint32_t cb_prediction,
    const uint32_t cb_base_recover_state,
    const uint32_t cb_prediction_recover_state,
    int32_t& even_shift,
    int32_t& odd_shift,
    const uint32_t splice_count,
    const int32_t kernel_shift,
    const uint32_t* coefficient_words) {
    const auto coefficients = read_coefficients<K>(coefficient_words);
    if (predict) {
        step_predict<K>(
            source_even,
            source_odd,
            destination_even,
            destination_odd,
            cb_prediction,
            cb_base_recover_state,
            cb_prediction_recover_state,
            even_shift,
            odd_shift,
            splice_count,
            kernel_shift,
            coefficients);
    } else {
        step_predict<K>(
            source_odd,
            source_even,
            destination_odd,
            destination_even,
            cb_prediction,
            cb_base_recover_state,
            cb_prediction_recover_state,
            odd_shift,
            even_shift,
            splice_count,
            kernel_shift,
            coefficients);
    }
}

template <uint32_t WidthMask>
inline void dispatch_predict_update_step(
    const uint32_t width,
    const bool predict,
    const uint32_t source_even,
    const uint32_t source_odd,
    const uint32_t destination_even,
    const uint32_t destination_odd,
    const uint32_t cb_prediction,
    const uint32_t cb_base_recover_state,
    const uint32_t cb_prediction_recover_state,
    int32_t& even_shift,
    int32_t& odd_shift,
    const uint32_t splice_count,
    const int32_t kernel_shift,
    const uint32_t* coefficient_words) {
#define TTVW_DISPATCH_WIDTH(K)                                \
    case K:                                                   \
        if constexpr ((WidthMask & (1U << ((K) - 1))) != 0) { \
            run_predict_update_step<K>(                       \
                predict,                                      \
                source_even,                                  \
                source_odd,                                   \
                destination_even,                             \
                destination_odd,                              \
                cb_prediction,                                \
                cb_base_recover_state,                        \
                cb_prediction_recover_state,                  \
                even_shift,                                   \
                odd_shift,                                    \
                splice_count,                                 \
                kernel_shift,                                 \
                coefficient_words);                           \
        }                                                     \
        return

    switch (width) {
        TTVW_DISPATCH_WIDTH(1);
        TTVW_DISPATCH_WIDTH(2);
        TTVW_DISPATCH_WIDTH(3);
        TTVW_DISPATCH_WIDTH(4);
        TTVW_DISPATCH_WIDTH(5);
        TTVW_DISPATCH_WIDTH(6);
        TTVW_DISPATCH_WIDTH(7);
        TTVW_DISPATCH_WIDTH(8);
        TTVW_DISPATCH_WIDTH(9);
        TTVW_DISPATCH_WIDTH(10);
        TTVW_DISPATCH_WIDTH(11);
        TTVW_DISPATCH_WIDTH(12);
        TTVW_DISPATCH_WIDTH(13);
        TTVW_DISPATCH_WIDTH(14);
        TTVW_DISPATCH_WIDTH(15);
        TTVW_DISPATCH_WIDTH(16);
        TTVW_DISPATCH_WIDTH(17);
        default: return;
    }
#undef TTVW_DISPATCH_WIDTH
}

template <uint32_t WidthMask>
inline void execute_program(
    const uint32_t num_steps,
    const uint32_t cb_even,
    const uint32_t cb_odd,
    const uint32_t cb_even_work,
    const uint32_t cb_odd_work,
    const uint32_t cb_final_even,
    const uint32_t cb_final_odd,
    const uint32_t cb_prediction,
    const uint32_t cb_base_recover_state,
    const uint32_t cb_prediction_recover_state,
    const uint32_t cb_out,
    const uint32_t splice_count,
    int32_t even_shift,
    int32_t odd_shift) {
    uint32_t source_even = cb_even;
    uint32_t source_odd = cb_odd;
    uint32_t cursor = 0;
    bool initialized = false;

    for (uint32_t step_index = 0; step_index < num_steps; ++step_index) {
        const uint32_t meta = read_program_word(cursor++);
        const uint8_t type = static_cast<uint8_t>(meta & 0x7);
        if (type == kStepTypeSwap) {
            std::swap(source_even, source_odd);
            std::swap(even_shift, odd_shift);
            continue;
        }

        const uint32_t destination_even = initialized ? source_even : cb_even_work;
        const uint32_t destination_odd = initialized ? source_odd : cb_odd_work;
        if (type == kStepTypeScaleEven) {
            step_scale(
                source_even,
                source_odd,
                destination_even,
                destination_odd,
                splice_count,
                read_program_word(cursor++),
                true);
        } else if (type == kStepTypeScaleOdd) {
            step_scale(
                source_even,
                source_odd,
                destination_even,
                destination_odd,
                splice_count,
                read_program_word(cursor++),
                false);
        } else {
            const uint32_t width = meta >> 3;
            const int32_t kernel_shift = static_cast<int32_t>(read_program_word(cursor++));
            uint32_t coefficient_words[17];
            for (uint32_t coefficient = 0; coefficient < width; ++coefficient) {
                coefficient_words[coefficient] = read_program_word(cursor + coefficient);
            }
            dispatch_predict_update_step<WidthMask>(
                width,
                type == kStepTypePredict,
                source_even,
                source_odd,
                destination_even,
                destination_odd,
                cb_prediction,
                cb_base_recover_state,
                cb_prediction_recover_state,
                even_shift,
                odd_shift,
                splice_count,
                kernel_shift,
                coefficient_words);
            cursor += width;
        }
        source_even = destination_even;
        source_odd = destination_odd;
        initialized = true;
    }

    if (!initialized) {
        step_scale(source_even, source_odd, cb_even_work, cb_odd_work, splice_count, 0, false);
        source_even = cb_even_work;
        source_odd = cb_odd_work;
    }
    copy_scale_stream(source_even, cb_final_even, splice_count, 0, false);
    copy_scale_stream(source_odd, cb_final_odd, splice_count, 0, false);
    cb_wait_front(cb_final_even, 1);
    cb_reserve_back(cb_out, 1);
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_final_even);
    copy_tile(cb_final_even, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_even = get_named_compile_time_arg_val("cb_even");
    constexpr uint32_t cb_odd = get_named_compile_time_arg_val("cb_odd");
    constexpr uint32_t cb_even_work = get_named_compile_time_arg_val("cb_even_work");
    constexpr uint32_t cb_odd_work = get_named_compile_time_arg_val("cb_odd_work");
    constexpr uint32_t cb_final_even = get_named_compile_time_arg_val("cb_final_even");
    constexpr uint32_t cb_final_odd = get_named_compile_time_arg_val("cb_final_odd");
    constexpr uint32_t cb_prediction = get_named_compile_time_arg_val("cb_prediction");
    constexpr uint32_t cb_base_recover_state = get_named_compile_time_arg_val("cb_base_recover_state");
    constexpr uint32_t cb_prediction_recover_state = get_named_compile_time_arg_val("cb_prediction_recover_state");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr int32_t even_shift = static_cast<int32_t>(get_named_compile_time_arg_val("even_delay"));
    constexpr int32_t odd_shift = static_cast<int32_t>(get_named_compile_time_arg_val("odd_delay"));
    constexpr uint32_t num_steps = get_named_compile_time_arg_val("num_steps");
    constexpr uint32_t width_mask = get_named_compile_time_arg_val("width_mask");
    const uint32_t splice_count = get_arg_val<uint32_t>(0);

    ckernel::init_sfpu(cb_even, cb_odd);
    splice_pipeline_init();
    execute_program<width_mask>(
        num_steps,
        cb_even,
        cb_odd,
        cb_even_work,
        cb_odd_work,
        cb_final_even,
        cb_final_odd,
        cb_prediction,
        cb_base_recover_state,
        cb_prediction_recover_state,
        cb_out,
        splice_count,
        even_shift,
        odd_shift);
}
