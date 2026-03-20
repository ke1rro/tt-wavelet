from __future__ import annotations

from typing import Any

import numpy as np
import torch
import ttnn

from .lifting_runtime import ShiftedArray, Tensor, TTNNOps
from .lifting_scheme import LiftingScheme, LiftingStepSpec, load_lifting_scheme
from .lifting_steps import (
    LiftingStep,
    LiftingStepPredict,
    LiftingStepScaleEven,
    LiftingStepScaleOdd,
    LiftingStepSwap,
    LiftingStepUpdate,
)


class LiftingWaveletTransform:
    def __init__(self, scheme: LiftingScheme, device: Any):
        self.scheme = scheme
        self.ops = TTNNOps(device)
        self.steps = self._build_steps()

    def _build_steps(self) -> list[LiftingStep]:
        built_steps: list[LiftingStep] = []
        for spec in self.scheme.steps:
            if spec.type == "predict":
                kernel = ShiftedArray.from_coefficients(spec.coefficients, spec.shift, self.ops)
                built_steps.append(LiftingStepPredict(kernel))
            elif spec.type == "update":
                kernel = ShiftedArray.from_coefficients(spec.coefficients, spec.shift, self.ops)
                built_steps.append(LiftingStepUpdate(kernel))
            elif spec.type == "scale-even":
                self._assert_scale_kernel(spec)
                built_steps.append(LiftingStepScaleEven(spec.coefficients[0]))
            elif spec.type == "scale-odd":
                self._assert_scale_kernel(spec)
                built_steps.append(LiftingStepScaleOdd(spec.coefficients[0]))
            elif spec.type == "swap":
                built_steps.append(LiftingStepSwap())
            else:
                raise ValueError(f"Unsupported lifting step type: {spec.type}")
        return built_steps

    @staticmethod
    def _assert_scale_kernel(spec: LiftingStepSpec) -> None:
        if spec.shift != 0:
            raise AssertionError("Scale steps must have zero shift")
        if len(spec.coefficients) != 1:
            raise AssertionError("Scale steps must have a single coefficient")

    def _as_ttnn_2d(self, value: Any) -> Tensor:
        if isinstance(value, torch.Tensor):
            host = value.to(dtype=self.ops.host_dtype)
        elif isinstance(value, np.ndarray):
            host = torch.from_numpy(value).to(dtype=self.ops.host_dtype)
        else:
            tensor = value
            tensor_dtype = getattr(tensor, "dtype", None)
            if tensor_dtype is not None and tensor_dtype != self.ops.dtype:
                tensor = ttnn.typecast(tensor, dtype=self.ops.dtype)
            if len(tensor.shape) == 1:
                return self.ops.reshape(tensor, (1, tensor.shape[0]))
            if len(tensor.shape) != 2:
                raise ValueError("Input must be 1D or 2D")
            return tensor

        if host.ndim == 1:
            host = host.unsqueeze(0)
        if host.ndim != 2:
            raise ValueError("Input must be 1D or 2D")
        return self.ops.to_device(host)

    @staticmethod
    def _symmetric_index(index: int, length: int) -> int:
        if length <= 1:
            return 0
        period = 2 * length
        reflected = index % period
        if reflected < length:
            return reflected
        return period - 1 - reflected

    def _pad_source_index(self, index: int, length: int) -> int | None:
        if self.scheme.mode == "zero":
            return None
        if self.scheme.mode == "constant":
            return 0 if index < 0 else length - 1
        if self.scheme.mode == "periodic":
            return index % length
        if self.scheme.mode == "symmetric":
            return self._symmetric_index(index, length)
        raise ValueError(f"Unsupported padding mode: {self.scheme.mode}")

    def _pad_segment(self, tensor: Tensor, positions: range) -> Tensor:
        batch, length = tensor.shape
        columns: list[Tensor] = []
        cache: dict[int | None, Tensor] = {}
        for index in positions:
            source_index = self._pad_source_index(index, length)
            if source_index not in cache:
                if source_index is None:
                    cache[source_index] = self.ops.full(batch, 1, 0.0)
                else:
                    cache[source_index] = self.ops.slice_columns(tensor, source_index, 1)
            columns.append(cache[source_index])
        return self.ops.concat_columns(columns, batch)

    def pad(self, tensor: Tensor, shape: int) -> Tensor:
        if shape <= 0:
            return tensor
        if tensor.shape[1] <= 0:
            return self.ops.empty(tensor.shape[0], 0)

        left = self._pad_segment(tensor, range(-shape, 0))
        right = self._pad_segment(tensor, range(tensor.shape[1], tensor.shape[1] + shape))
        return self.ops.concat_columns([left, tensor, right], tensor.shape[0])

    def _slice_columns(self, tensor: Tensor, start: int, stop: int) -> Tensor:
        return self.ops.slice_columns(tensor, start, stop - start)

    def _split_even_odd(self, tensor: Tensor) -> tuple[Tensor, Tensor]:
        _, length = tensor.shape
        return (
            self.ops.strided_slice_columns(tensor, 0, length, 2),
            self.ops.strided_slice_columns(tensor, 1, length, 2),
        )

    def forward(self, data: Any) -> dict[str, Tensor | int]:
        data_tt = self._as_ttnn_2d(data)

        L = self.scheme.tap_size
        input_len = data_tt.shape[1]
        length = (input_len + L - 1) // 2
        direct_shift = L // 2

        padded = self.pad(data_tt, L - 1)
        even_tt, odd_tt = self._split_even_odd(padded)

        even = ShiftedArray(even_tt, self.scheme.delays[0], self.ops)
        odd = ShiftedArray(odd_tt, self.scheme.delays[1], self.ops)

        for step in self.steps:
            even, odd = step.forward(even, odd)

        even_shift = even.shift - direct_shift
        odd_shift = odd.shift - direct_shift
        if even_shift > 0 or odd_shift > 0:
            raise ValueError("Coefficients have positive shift after forward lifting")

        even_data = self._slice_columns(even.data, -even_shift, -even_shift + length)
        odd_data = self._slice_columns(odd.data, -odd_shift, -odd_shift + length)

        return {
            "cA": even_data,
            "cD": odd_data,
            "delay_even": self.scheme.delays[0],
            "delay_odd": self.scheme.delays[1],
        }

    def inverse(self, even: Any, odd: Any) -> Tensor:
        even_tt = self._as_ttnn_2d(even)
        odd_tt = self._as_ttnn_2d(odd)

        if even_tt.shape[0] != odd_tt.shape[0]:
            raise ValueError("Even and odd batches must match")

        L = self.scheme.tap_size
        length = 2 * even_tt.shape[1] - L + 2
        direct_shift = L // 2

        even_sa = ShiftedArray(even_tt, 0, self.ops)
        odd_sa = ShiftedArray(odd_tt, 0, self.ops)

        for step in reversed(self.steps):
            even_sa, odd_sa = step.inverse(even_sa, odd_sa)

        even_shift = even_sa.shift + self.scheme.delays[1] - direct_shift
        odd_shift = odd_sa.shift + self.scheme.delays[0] - direct_shift
        odd_shift += 1

        half = length // 2
        even_out = self._slice_columns(even_sa.data, -even_shift, -even_shift + half)
        odd_out = self._slice_columns(odd_sa.data, -odd_shift, -odd_shift + half)

        return self.ops.interleave_columns(odd_out, even_out)
