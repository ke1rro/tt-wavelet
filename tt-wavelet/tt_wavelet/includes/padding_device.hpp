#pragma once

#include <filesystem>

#include "tt-metalium/host_api.hpp"
#include "tt_wavelet/includes/padding.hpp"

namespace ttwv {

/**
 * @brief Owns the device-side program and kernel handles for a 1D padding operation.
 *
 * Created by @ref create_symmetric_pad_1d_program and passed to
 * @ref set_pad_1d_runtime_args whenever the input/output buffer addresses or layout change.
 *
 * @note The @c program field is move-only; do not copy this struct.
 */
struct Pad1DDeviceProgram {
    tt::tt_metal::Program program;             ///< The TT-Metal program owning all CBs and kernels.
    tt::tt_metal::KernelHandle reader_kernel;  ///< Handle to the RISCV-0 reader dataflow kernel.
    tt::tt_metal::KernelHandle writer_kernel;  ///< Handle to the RISCV-1 writer dataflow kernel.
};

/**
 * @brief Creates a single-core device program that performs symmetric 1D padding.
 *
 * Allocates circular buffers, compiles reader/writer kernels from @p kernel_root, and
 * immediately calls @ref set_pad_1d_runtime_args to bind the initial layout so the
 * returned program is ready to enqueue.
 *
 * ### Restrictions (enforced via TT_FATAL)
 * - layout.config.mode must be @ref BoundaryMode::Symmetric.
 * - Both input and output buffers must be fp32 (element_size_bytes == 4).
 * - layout.input.stick_width must equal 32.
 * - Input and output stick widths must match.
 *
 * @param kernel_root   Directory that contains the dataflow/ kernel sources.
 * @param core          Tensix core to run the program on.
 * @param input_buffer  Source DRAM buffer (unpadded signal).
 * @param output_buffer Destination DRAM buffer (padded signal, must be pre-allocated).
 * @param layout        Memory layout describing input/output signals and padding config.
 * @return A ready-to-enqueue @ref Pad1DDeviceProgram.
 */
[[nodiscard]] Pad1DDeviceProgram create_symmetric_pad_1d_program(
    const std::filesystem::path& kernel_root,
    const tt::tt_metal::CoreCoord& core,
    const tt::tt_metal::Buffer& input_buffer,
    const tt::tt_metal::Buffer& output_buffer,
    const Pad1DLayout& layout);

/**
 * @brief Updates the runtime arguments of an existing @ref Pad1DDeviceProgram.
 *
 * Call this when the DRAM buffer addresses or the signal layout change between runs
 * without needing to recompile the kernels.
 *
 * The function always schedules the full output range (all sticks, starting from stick 0).
 *
 * @param program_bundle Existing program bundle whose runtime args will be updated.
 * @param core           Tensix core the kernels are bound to.
 * @param input_buffer   Source DRAM buffer.
 * @param output_buffer  Destination DRAM buffer.
 * @param layout         Updated memory layout.
 */
void set_pad_1d_runtime_args(
    const Pad1DDeviceProgram& program_bundle,
    const tt::tt_metal::CoreCoord& core,
    const tt::tt_metal::Buffer& input_buffer,
    const tt::tt_metal::Buffer& output_buffer,
    const Pad1DLayout& layout);

}  // namespace ttwv
