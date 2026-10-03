// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttwt/device/ilwt_2d_program_factory.hpp"

namespace ttwt::prim {

struct Ilwt2DDeviceOperation {
  using operation_attributes_t = Ilwt2DParams;
  using tensor_args_t = Ilwt2DInputs;
  using spec_return_value_t = tt::tt_metal::TensorSpec;
  using tensor_return_value_t = ttnn::Tensor;
  using program_factory_t = std::variant<Ilwt2DProgramFactory>;

  static void validate_on_program_cache_miss(const operation_attributes_t &, const tensor_args_t &);
  static spec_return_value_t compute_output_specs(const operation_attributes_t &,
                                                  const tensor_args_t &);
  static tensor_return_value_t create_output_tensors(const operation_attributes_t &,
                                                     const tensor_args_t &);
};

ttnn::Tensor ilwt_2d(const ttnn::Tensor &ll, const ttnn::Tensor &lh, const ttnn::Tensor &hl,
                     const ttnn::Tensor &hh, operations::wavelet::SchemeId scheme_id,
                     operations::wavelet::BoundaryMode boundary_mode, uint32_t output_height,
                     uint32_t output_width, const ttnn::MemoryConfig &output_memory_config,
                     const std::optional<ttnn::Tensor> &preallocated_output = std::nullopt);

} // namespace ttwt::prim
