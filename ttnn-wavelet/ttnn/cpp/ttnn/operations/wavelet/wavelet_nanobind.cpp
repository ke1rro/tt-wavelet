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
    ttnn::bind_function<"lwt">(
        mod,
        R"doc(
Compute one level of the FP32 1D lifting wavelet transform.

``input`` must be an exact-rank-1, row-major, DRAM-interleaved FLOAT32 tensor
on one physical device. ``wavelet`` names one of the 106 discrete PyWavelets
schemes and ``boundary_mode`` is one of ``zero``, ``constant``, ``symmetric``,
``reflect``, ``periodic``, ``smooth``, ``antisymmetric``, or ``antireflect``.

Returns ``(approximation, detail)``. Both tensors have length
``(input_length + filter_length - 1) // 2``. When supplied,
``output_tensors`` must contain two non-aliasing tensors with the exact inferred
specification.
)doc",
        &ttnn::lwt,
        nb::arg("input").noconvert(),
        nb::arg("wavelet"),
        nb::kw_only(),
        nb::arg("boundary_mode") = "symmetric",
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensors") = nb::none());

    ttnn::bind_function<"ilwt">(
        mod,
        R"doc(
Compute one level of the FP32 1D inverse lifting wavelet transform.

``approximation`` and ``detail`` must be non-aliasing, equal-shaped exact-rank-1
row-major DRAM-interleaved FLOAT32 tensors on the same physical device.
``original_length`` restores the exact odd or even logical length and must be
consistent with the coefficient shape, wavelet, and boundary mode.

Returns one exact-rank-1 tensor. ``output_tensor`` may provide exact-spec
preallocated storage and must not alias either input.
)doc",
        &ttnn::ilwt,
        nb::arg("approximation").noconvert(),
        nb::arg("detail").noconvert(),
        nb::arg("wavelet"),
        nb::arg("original_length"),
        nb::kw_only(),
        nb::arg("boundary_mode") = "symmetric",
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());

    ttnn::bind_function<"lwt_2d">(
        mod,
        R"doc(
Compute one level of the FP32 separable 2D lifting wavelet transform.

``input`` must be an exact-rank-2, standard 32x32 tile-layout,
DRAM-interleaved FLOAT32 tensor on one physical device. The operation preserves
the standalone vertical-first execution order and returns ``(LL, LH, HL, HH)``.
This order corresponds to ``(cA, cV, cH, cD)`` in PyWavelets terminology.

``output_tensors`` may provide four pairwise non-aliasing tensors with the exact
inferred specifications.
)doc",
        &ttnn::lwt_2d,
        nb::arg("input").noconvert(),
        nb::arg("wavelet"),
        nb::kw_only(),
        nb::arg("boundary_mode") = "symmetric",
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensors") = nb::none());

    ttnn::bind_function<"ilwt_2d">(
        mod,
        R"doc(
Compute one level of the FP32 separable 2D inverse lifting wavelet transform.

``ll``, ``lh``, ``hl``, and ``hh`` must be pairwise non-aliasing, equal-shaped,
standard 32x32 tile-layout, DRAM-interleaved FLOAT32 tensors on the same physical
device. ``output_shape=(height, width)`` restores the exact odd or even logical
dimensions and must be consistent with the coefficient shape.

Returns one exact-rank-2 tensor. ``output_tensor`` may provide exact-spec
preallocated storage and must not alias an input band.
)doc",
        &ttnn::ilwt_2d,
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
