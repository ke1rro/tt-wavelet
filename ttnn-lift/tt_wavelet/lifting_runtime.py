from __future__ import annotations

from typing import Any

import torch
import ttnn

Tensor = Any


class TTNNOps:
    def __init__(self, device: Any):
        self.device = device
        self.host_dtype = torch.float32
        self.dtype = ttnn.float32

    def to_device(self, tensor: torch.Tensor) -> Tensor:
        return ttnn.from_torch(
            tensor.to(dtype=self.host_dtype),
            dtype=self.dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )

    def from_device(self, tensor: Tensor) -> torch.Tensor:
        return ttnn.to_torch(tensor).to(dtype=torch.float32)

    def empty(self, batch: int, length: int) -> Tensor:
        width = max(int(length), 1)
        tensor = self.full(batch, width, 0.0)
        if length == width:
            return tensor
        return ttnn.slice(tensor, [0, 0], [batch, length], [1, 1])

    def full(self, batch: int, length: int, value: float) -> Tensor:
        return ttnn.full(
            [batch, length],
            fill_value=float(value),
            dtype=self.dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )

    def add(self, left: Tensor, right: Tensor) -> Tensor:
        return ttnn.add(left, right)

    def mul_scalar(self, tensor: Tensor, value: float) -> Tensor:
        if value == 1.0:
            return tensor
        if value == 0.0:
            return self.full(tensor.shape[0], tensor.shape[1], 0.0)
        return ttnn.multiply(
            tensor,
            float(value),
            dtype=self.dtype,
            use_legacy=None,
            sub_core_grids=None,
            fast_and_approximate_mode=False,
        )

    def slice_columns(self, tensor: Tensor, start: int, length: int) -> Tensor:
        if length <= 0:
            return self.empty(tensor.shape[0], 0)
        return ttnn.slice(
            tensor,
            [0, start],
            [tensor.shape[0], start + length],
            [1, 1],
        )

    def concat_columns(self, tensors: list[Tensor], batch: int) -> Tensor:
        if not tensors:
            return self.empty(batch, 0)
        if len(tensors) == 1:
            return tensors[0]
        return ttnn.concat(tensors, dim=1)

    def strided_slice_columns(self, tensor: Tensor, start: int, stop: int, step: int) -> Tensor:
        if start >= stop:
            return self.empty(tensor.shape[0], 0)
        return ttnn.slice(
            tensor,
            [0, start],
            [tensor.shape[0], stop],
            [1, step],
        )

    def reshape(self, tensor: Tensor, shape: tuple[int, ...]) -> Tensor:
        return ttnn.reshape(tensor, shape)

    def interleave_columns(self, first: Tensor, second: Tensor) -> Tensor:
        batch = first.shape[0]
        width = first.shape[1]
        if first.shape != second.shape:
            raise ValueError("Interleaved tensors must have the same shape")
        if width <= 0:
            return self.empty(batch, 0)

        first_3d = self.reshape(first, (batch, width, 1))
        second_3d = self.reshape(second, (batch, width, 1))
        interleaved = ttnn.concat([first_3d, second_3d], dim=2)
        return self.reshape(interleaved, (batch, width * 2))

    def conv_valid(
        self,
        data: Tensor,
        kernel: Tensor,
        kernel_values: tuple[float, ...] | None = None,
    ) -> Tensor:
        batch, data_len = data.shape
        kernel_len = kernel.shape[1]
        out_len = data_len - kernel_len + 1
        if out_len <= 0:
            return self.empty(batch, 0)
        if kernel_values is None:
            raise ValueError("conv_valid requires host-side kernel coefficients")

        if kernel_len == 1:
            return self.mul_scalar(data, float(kernel_values[0]))

        if kernel_len == 2:
            left = self.slice_columns(data, 0, out_len)
            right = self.slice_columns(data, 1, out_len)
            left_scaled = self.mul_scalar(left, float(kernel_values[1]))
            right_scaled = self.mul_scalar(right, float(kernel_values[0]))
            return self.add(left_scaled, right_scaled)

        accum: Tensor | None = None
        for j in range(kernel_len):
            coeff = float(kernel_values[kernel_len - 1 - j])
            if coeff == 0.0:
                continue

            data_window = self.slice_columns(data, j, out_len)
            scaled = data_window if coeff == 1.0 else self.mul_scalar(data_window, coeff)
            accum = scaled if accum is None else self.add(accum, scaled)

        if accum is None:
            return self.full(batch, out_len, 0.0)
        return accum


class ShiftedArray:
    def __init__(
        self,
        data: Tensor,
        shift: int,
        ops: TTNNOps,
        coefficients: tuple[float, ...] | None = None,
    ):
        self.data = data
        self.shift = int(shift)
        self.ops = ops
        self.coefficients = coefficients

    @property
    def end(self) -> int:
        return self.shift + int(self.data.shape[1])

    @classmethod
    def from_coefficients(
        cls,
        coefficients: list[float],
        shift: int,
        ops: TTNNOps,
    ) -> "ShiftedArray":
        coeff_tensor = torch.tensor(coefficients, dtype=ops.host_dtype).reshape(1, -1)
        return cls(
            ops.to_device(coeff_tensor),
            shift,
            ops,
            tuple(float(value) for value in coefficients),
        )

    def __mul__(self, other: float | "ShiftedArray") -> "ShiftedArray":
        if isinstance(other, (int, float)):
            coefficients = None
            if self.coefficients is not None:
                coefficients = tuple(float(other) * coeff for coeff in self.coefficients)
            return ShiftedArray(
                self.ops.mul_scalar(self.data, float(other)),
                self.shift,
                self.ops,
                coefficients,
            )

        if not isinstance(other, ShiftedArray):
            return NotImplemented

        shift_new = self.shift + other.shift + min(self.data.shape[1], other.data.shape[1]) - 1
        data_new = self.ops.conv_valid(self.data, other.data, other.coefficients)
        return ShiftedArray(data_new, shift_new, self.ops)

    def __add__(self, other: "ShiftedArray") -> "ShiftedArray":
        if not isinstance(other, ShiftedArray):
            return NotImplemented

        if self.shift == other.shift and self.data.shape[1] == other.data.shape[1]:
            return ShiftedArray(self.ops.add(self.data, other.data), self.shift, self.ops)

        shift_new = max(self.shift, other.shift)
        end_new = min(self.end, other.end)
        overlap = end_new - shift_new
        if overlap <= 0:
            return ShiftedArray(self.ops.empty(self.data.shape[0], 0), shift_new, self.ops)

        data1 = self.ops.slice_columns(self.data, shift_new - self.shift, overlap)
        data2 = self.ops.slice_columns(other.data, shift_new - other.shift, overlap)
        return ShiftedArray(self.ops.add(data1, data2), shift_new, self.ops)

    def __neg__(self) -> "ShiftedArray":
        coefficients = None
        if self.coefficients is not None:
            coefficients = tuple(-coeff for coeff in self.coefficients)
        return ShiftedArray(
            self.ops.mul_scalar(self.data, -1.0), self.shift, self.ops, coefficients
        )

    def __sub__(self, other: "ShiftedArray") -> "ShiftedArray":
        return self + (-other)
