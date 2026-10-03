// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <optional>
#include <variant>

#include "ttwt/device/lwt_2d_program_factory.hpp"

namespace ttwt::prim {

struct Lwt2DDeviceOperation {
  using operation_attributes_t = Lwt2DParams;
  using tensor_args_t = Lwt2DInputs;
  using spec_return_value_t = Lwt2DOutputSpecs;
  using tensor_return_value_t = Lwt2DOutputs;
  using program_factory_t = std::variant<Lwt2DProgramFactory>;

  static void validate_on_program_cache_miss(const operation_attributes_t &, const tensor_args_t &);
  static spec_return_value_t compute_output_specs(const operation_attributes_t &,
                                                  const tensor_args_t &);
  static tensor_return_value_t create_output_tensors(const operation_attributes_t &,
                                                     const tensor_args_t &);
};

Lwt2DOutputs
lwt_2d(const ttnn::Tensor &input, operations::wavelet::SchemeId scheme_id,
       operations::wavelet::BoundaryMode boundary_mode,
       const ttnn::MemoryConfig &output_memory_config,
       const std::optional<std::array<ttnn::Tensor, 4>> &preallocated_outputs = std::nullopt);

} // namespace ttwt::prim
