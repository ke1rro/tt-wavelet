// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/wavelet_types.hpp"
#include "ttnn/types.hpp"

#include <array>
#include <optional>
#include <string_view>
#include <tuple>

namespace ttnn {

std::tuple<Tensor, Tensor> lwt(
    const Tensor& input,
    std::string_view wavelet,
    std::string_view boundary_mode = "symmetric",
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<std::tuple<Tensor, Tensor>>& output_tensors = std::nullopt);

Tensor ilwt(
    const Tensor& approximation,
    const Tensor& detail,
    std::string_view wavelet,
    uint32_t original_length,
    std::string_view boundary_mode = "symmetric",
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt);

std::tuple<Tensor, Tensor, Tensor, Tensor> lwt_2d(
    const Tensor& input,
    std::string_view wavelet,
    std::string_view boundary_mode = "symmetric",
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<std::array<Tensor, 4>>& output_tensors = std::nullopt);

Tensor ilwt_2d(
    const Tensor& ll,
    const Tensor& lh,
    const Tensor& hl,
    const Tensor& hh,
    std::string_view wavelet,
    const WaveletOutputShape2D& output_shape,
    std::string_view boundary_mode = "symmetric",
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt);

}  // namespace ttnn
