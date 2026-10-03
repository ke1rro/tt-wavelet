// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>
#include <tuple>

#include "ttwt/wavelet_types.hpp"
#include "ttnn/types.hpp"

namespace ttwt {

[[nodiscard]] uint32_t dwt_coeff_len(uint32_t input_length, std::string_view wavelet);

std::tuple<ttnn::Tensor, ttnn::Tensor> dwt(
    const ttnn::Tensor& input,
    std::string_view wavelet,
    std::string_view boundary_mode = "symmetric",
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<std::tuple<ttnn::Tensor, ttnn::Tensor>>& output_tensors = std::nullopt);

ttnn::Tensor idwt(
    const ttnn::Tensor& approximation,
    const ttnn::Tensor& detail,
    std::string_view wavelet,
    uint32_t original_length,
    std::string_view boundary_mode = "symmetric",
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::Tensor>& output_tensor = std::nullopt);

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> dwt_2d(
    const ttnn::Tensor& input,
    std::string_view wavelet,
    std::string_view boundary_mode = "symmetric",
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<std::array<ttnn::Tensor, 4>>& output_tensors = std::nullopt);

ttnn::Tensor idwt_2d(
    const ttnn::Tensor& ll,
    const ttnn::Tensor& lh,
    const ttnn::Tensor& hl,
    const ttnn::Tensor& hh,
    std::string_view wavelet,
    const WaveletOutputShape2D& output_shape,
    std::string_view boundary_mode = "symmetric",
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::Tensor>& output_tensor = std::nullopt);

}  // namespace ttwt
