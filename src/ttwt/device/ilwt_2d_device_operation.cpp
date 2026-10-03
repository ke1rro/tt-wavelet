// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttwt/device/ilwt_2d_device_operation.hpp"

#include "ttwt/device/wavelet_2d_operation_impl.hpp"
#include "ttwt/device/wavelet_l1_budget.hpp"

namespace ttwt::prim {

void Ilwt2DDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t &operation_attributes, const tensor_args_t &tensor_args) {
  detail::validate_ilwt_2d(operation_attributes, tensor_args);
}

Ilwt2DDeviceOperation::spec_return_value_t
Ilwt2DDeviceOperation::compute_output_specs(const operation_attributes_t &operation_attributes,
                                            const tensor_args_t &tensor_args) {
  return detail::compute_ilwt_2d_output_spec(operation_attributes, tensor_args);
}

Ilwt2DDeviceOperation::tensor_return_value_t
Ilwt2DDeviceOperation::create_output_tensors(const operation_attributes_t &operation_attributes,
                                             const tensor_args_t &tensor_args) {
  return detail::create_ilwt_2d_output_tensor(operation_attributes, tensor_args);
}

ttnn::Tensor ilwt_2d(const ttnn::Tensor &ll, const ttnn::Tensor &lh, const ttnn::Tensor &hl,
                     const ttnn::Tensor &hh, const operations::wavelet::SchemeId scheme_id,
                     const operations::wavelet::BoundaryMode boundary_mode,
                     const uint32_t output_height, const uint32_t output_width,
                     const ttnn::MemoryConfig &output_memory_config,
                     const std::optional<ttnn::Tensor> &preallocated_output) {
  return ttnn::device_operation::launch<Ilwt2DDeviceOperation>(
      Ilwt2DParams{
          .scheme_id = scheme_id,
          .boundary_mode = boundary_mode,
          .output_height = output_height,
          .output_width = output_width,
          .available_l1_bytes = detail::quantized_available_l1_bytes(ll.device()),
          .output_memory_config = output_memory_config,
      },
      Ilwt2DInputs{
          .ll = ll,
          .lh = lh,
          .hl = hl,
          .hh = hh,
          .preallocated_output = preallocated_output,
      });
}

} // namespace ttwt::prim
