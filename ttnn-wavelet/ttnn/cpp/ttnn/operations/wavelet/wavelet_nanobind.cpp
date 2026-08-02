// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/wavelet/wavelet_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/tuple.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/wavelet/wavelet.hpp"

namespace ttnn::operations::wavelet {

void bind_wavelet_operations(nb::module_& mod) {
    ttnn::bind_function<"dwt">(
        mod,
        R"doc(
Compute one level of the FP32 1D discrete wavelet transform.

``input`` must be a row-major INTERLEAVED FLOAT32 tensor with shape ``[W]`` or
``[B,1,1,W]`` in DRAM or L1 on one physical device. Outputs remain INTERLEAVED
DRAM tensors and preserve the optional batch dimensions.
``wavelet`` names one of the 106 discrete PyWavelets
schemes and ``boundary_mode`` is one of ``zero``, ``constant``, ``symmetric``,
``reflect``, ``periodic``, ``smooth``, ``antisymmetric``, or ``antireflect``.

Returns ``(approximation, detail)``. Both tensors have length
``(input_length + filter_length - 1) // 2``. When supplied,
``output_tensors`` must contain two non-aliasing tensors with the exact inferred
specification.
)doc",
        &ttnn::dwt,
        nb::arg("input").noconvert(),
        nb::arg("wavelet"),
        nb::kw_only(),
        nb::arg("boundary_mode") = "symmetric",
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensors") = nb::none());

    ttnn::bind_function<"idwt">(
        mod,
        R"doc(
Compute one level of the FP32 1D inverse discrete wavelet transform.

``approximation`` and ``detail`` must be non-aliasing, equal-shaped row-major
INTERLEAVED FLOAT32 tensors with shape ``[Wc]`` or ``[B,1,1,Wc]`` in DRAM or L1
on the same physical device.
Their placements may differ. The output remains an INTERLEAVED DRAM tensor.
``original_length`` restores the exact odd or even logical length and must be
consistent with the coefficient shape, wavelet, and boundary mode.

Returns one tensor matching the input rank and batch. ``output_tensor`` may provide exact-spec
preallocated storage and must not alias either input.
)doc",
        &ttnn::idwt,
        nb::arg("approximation").noconvert(),
        nb::arg("detail").noconvert(),
        nb::arg("wavelet"),
        nb::arg("original_length"),
        nb::kw_only(),
        nb::arg("boundary_mode") = "symmetric",
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());

    ttnn::bind_function<"dwt_2d">(
        mod,
        R"doc(
Compute one level of the FP32 separable 2D discrete wavelet transform.

``input`` must be a standard 32x32 tile-layout INTERLEAVED FLOAT32 tensor with
shape ``[H,W]`` or ``[B,1,H,W]`` in DRAM or L1 on one physical device. Outputs
remain INTERLEAVED DRAM tensors and preserve the optional batch dimensions. The operation preserves
the standalone vertical-first execution order and returns ``(LL, LH, HL, HH)``.
This order corresponds to ``(cA, cV, cH, cD)`` in PyWavelets terminology.

``output_tensors`` may provide four pairwise non-aliasing tensors with the exact
inferred specifications.
)doc",
        &ttnn::dwt_2d,
        nb::arg("input").noconvert(),
        nb::arg("wavelet"),
        nb::kw_only(),
        nb::arg("boundary_mode") = "symmetric",
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensors") = nb::none());

    ttnn::bind_function<"idwt_2d">(
        mod,
        R"doc(
Compute one level of the FP32 separable 2D inverse discrete wavelet transform.

``ll``, ``lh``, ``hl``, and ``hh`` must be pairwise non-aliasing, equal-shaped,
standard 32x32 tile-layout INTERLEAVED FLOAT32 tensors with shape ``[Hc,Wc]``
or ``[B,1,Hc,Wc]`` in DRAM or L1 on the same physical device. Their placements
may differ. The output remains an
INTERLEAVED DRAM tensor. ``output_shape=(height, width)`` restores the exact odd or even logical
dimensions and must be consistent with the coefficient shape.

Returns one tensor matching the input rank and batch. ``output_tensor`` may provide exact-spec
preallocated storage and must not alias an input band.
)doc",
        &ttnn::idwt_2d,
        nb::arg("ll").noconvert(),
        nb::arg("lh").noconvert(),
        nb::arg("hl").noconvert(),
        nb::arg("hh").noconvert(),
        nb::arg("wavelet"),
        nb::arg("output_shape"),
        nb::kw_only(),
        nb::arg("boundary_mode") = "symmetric",
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());
}

}  // namespace ttnn::operations::wavelet
