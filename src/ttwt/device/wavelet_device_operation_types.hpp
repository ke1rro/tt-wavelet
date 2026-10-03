// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <tuple>

#include "ttwt/common/boundary.hpp"
#include "ttwt/generated/wavelet_schemes/scheme_catalog.hpp"
#include "ttwt/wavelet_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttwt::prim {

using Lwt1DOutputSpecs = std::tuple<tt::tt_metal::TensorSpec, tt::tt_metal::TensorSpec>;
using Lwt1DOutputs = std::tuple<ttnn::Tensor, ttnn::Tensor>;

using Lwt2DOutputSpecs = std::tuple<tt::tt_metal::TensorSpec, tt::tt_metal::TensorSpec,
                                    tt::tt_metal::TensorSpec, tt::tt_metal::TensorSpec>;
// Forward 2D subbands are ordered as (LL, LH, HL, HH).
using Lwt2DOutputs = std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, ttnn::Tensor>;

struct Lwt1DParams {
  operations::wavelet::SchemeId scheme_id;
  operations::wavelet::BoundaryMode boundary_mode;
  uint32_t available_l1_bytes;
  ttnn::MemoryConfig output_memory_config;
};

struct Lwt1DInputs {
  const ttnn::Tensor &input;
  const std::optional<Lwt1DOutputs> &preallocated_outputs;
};

struct Ilwt1DParams {
  operations::wavelet::SchemeId scheme_id;
  operations::wavelet::BoundaryMode boundary_mode;
  uint32_t original_length;
  uint32_t available_l1_bytes;
  ttnn::MemoryConfig output_memory_config;
};

struct Ilwt1DInputs {
  const ttnn::Tensor &approximation;
  const ttnn::Tensor &detail;
  const std::optional<ttnn::Tensor> &preallocated_output;
};

struct Lwt2DParams {
  operations::wavelet::SchemeId scheme_id;
  operations::wavelet::BoundaryMode boundary_mode;
  uint32_t available_l1_bytes;
  ttnn::MemoryConfig output_memory_config;
};

struct Lwt2DInputs {
  const ttnn::Tensor &input;
  const std::optional<std::array<ttnn::Tensor, 4>> &preallocated_outputs;
};

struct Ilwt2DParams {
  operations::wavelet::SchemeId scheme_id;
  operations::wavelet::BoundaryMode boundary_mode;
  uint32_t output_height;
  uint32_t output_width;
  uint32_t available_l1_bytes;
  ttnn::MemoryConfig output_memory_config;
};

struct Ilwt2DInputs {
  const ttnn::Tensor &ll;
  const ttnn::Tensor &lh;
  const ttnn::Tensor &hl;
  const ttnn::Tensor &hh;
  const std::optional<ttnn::Tensor> &preallocated_output;
};

} // namespace ttwt::prim
