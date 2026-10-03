// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttwt/wavelet.hpp"

#include <tt_stl/assert.hpp>

#include "ttwt/common/wavelet_host.hpp"
#include "ttwt/device/ilwt_1d_device_operation.hpp"
#include "ttwt/device/ilwt_2d_device_operation.hpp"
#include "ttwt/device/lwt_1d_device_operation.hpp"
#include "ttwt/device/lwt_2d_device_operation.hpp"

namespace ttwt {

uint32_t dwt_coeff_len(const uint32_t input_length, const std::string_view wavelet) {
    return operations::wavelet::dwt_coefficient_length(
        input_length, operations::wavelet::scheme_id_from_string(wavelet));
}

std::tuple<ttnn::Tensor, ttnn::Tensor> dwt(
    const ttnn::Tensor& input,
    const std::string_view wavelet,
    const std::string_view boundary_mode,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<std::tuple<ttnn::Tensor, ttnn::Tensor>>& output_tensors) {
    const ttnn::MemoryConfig resolved_memory_config = memory_config.value_or(
        output_tensors.has_value() ? std::get<0>(*output_tensors).memory_config() : ttnn::MemoryConfig{});
    return prim::lwt(
        input,
        operations::wavelet::scheme_id_from_string(wavelet),
        operations::wavelet::boundary_mode_from_string(boundary_mode),
        resolved_memory_config,
        output_tensors);
}

ttnn::Tensor idwt(
    const ttnn::Tensor& approximation,
    const ttnn::Tensor& detail,
    const std::string_view wavelet,
    const uint32_t original_length,
    const std::string_view boundary_mode,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::Tensor>& output_tensor) {
    const ttnn::MemoryConfig resolved_memory_config =
        memory_config.value_or(output_tensor.has_value() ? output_tensor->memory_config() : ttnn::MemoryConfig{});
    return prim::ilwt(
        approximation,
        detail,
        operations::wavelet::scheme_id_from_string(wavelet),
        operations::wavelet::boundary_mode_from_string(boundary_mode),
        original_length,
        resolved_memory_config,
        output_tensor);
}

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> dwt_2d(
    const ttnn::Tensor& input,
    const std::string_view wavelet,
    const std::string_view boundary_mode,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<std::array<ttnn::Tensor, 4>>& output_tensors) {
    const ttnn::MemoryConfig resolved_memory_config =
        memory_config.value_or(output_tensors.has_value() ? (*output_tensors)[0].memory_config() : ttnn::MemoryConfig{});
    return prim::lwt_2d(
        input,
        operations::wavelet::scheme_id_from_string(wavelet),
        operations::wavelet::boundary_mode_from_string(boundary_mode),
        resolved_memory_config,
        output_tensors);
}

ttnn::Tensor idwt_2d(
    const ttnn::Tensor& ll,
    const ttnn::Tensor& lh,
    const ttnn::Tensor& hl,
    const ttnn::Tensor& hh,
    const std::string_view wavelet,
    const WaveletOutputShape2D& output_shape,
    const std::string_view boundary_mode,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::Tensor>& output_tensor) {
    const ttnn::MemoryConfig resolved_memory_config =
        memory_config.value_or(output_tensor.has_value() ? output_tensor->memory_config() : ttnn::MemoryConfig{});
    return prim::ilwt_2d(
        ll,
        lh,
        hl,
        hh,
        operations::wavelet::scheme_id_from_string(wavelet),
        operations::wavelet::boundary_mode_from_string(boundary_mode),
        output_shape[0],
        output_shape[1],
        resolved_memory_config,
        output_tensor);
}

}  // namespace ttwt
