from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import ttnn

Tensor = Any


@dataclass(frozen=True)
class LiftingStepSpec:
    type: str
    shift: int
    coefficients: list[float]


@dataclass(frozen=True)
class LiftingScheme:
    mode: str
    tap_size: int
    steps: list[LiftingStepSpec]
    delays: tuple[int, int]

    @classmethod
    def from_object(cls, obj: dict[str, Any], mode: str = "symmetric") -> "LiftingScheme":
        steps: list[LiftingStepSpec] = []
        for raw_step in obj["steps"]:
            step_type = raw_step["type"]
            coefficients = [_to_coeff_float(coeff) for coeff in raw_step.get("coefficients", [])]

            steps.append(
                LiftingStepSpec(
                    type=step_type,
                    shift=int(raw_step["shift"]),
                    coefficients=coefficients,
                )
            )

        return cls(
            mode=mode,
            tap_size=int(obj["tap_size"]),
            steps=steps,
            delays=(int(obj["delay"]["even"]), int(obj["delay"]["odd"])),
        )

    @classmethod
    def from_file(cls, path: str, mode: str = "symmetric") -> "LiftingScheme":
        with open(path, "r", encoding="utf-8") as handle:
            obj = json.load(handle)
        return cls.from_object(obj, mode=mode)


def load_lifting_scheme(path: str, mode: str = "symmetric") -> LiftingScheme:
    return LiftingScheme.from_file(path, mode=mode)


def _to_coeff_float(coeff: Any) -> float:
    if isinstance(coeff, (int, float)):
        return float(coeff)
    if isinstance(coeff, dict):
        return float(coeff["numerator"]) / float(coeff["denominator"])
    raise TypeError(f"Unsupported coefficient type: {type(coeff)}")


class TTNNOps:
    def __init__(self, device: Any):
        self.device = device

    def to_device(self, tensor: torch.Tensor) -> Tensor:
        return ttnn.from_torch(
            tensor.to(dtype=torch.float32),
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )

    def from_device(self, tensor: Tensor) -> torch.Tensor:
        return ttnn.to_torch(tensor)

    def empty(self, batch: int, length: int) -> Tensor:
        return self.to_device(torch.empty((batch, length), dtype=torch.float32))

    def full(self, batch: int, length: int, value: float) -> Tensor:
        return ttnn.full(
            [batch, length],
            fill_value=float(value),
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )

    def add(self, left: Tensor, right: Tensor) -> Tensor:
        return ttnn.add(left, right)

    def mul_scalar(self, tensor: Tensor, value: float) -> Tensor:
        return ttnn.multiply(
            tensor, float(value), dtype=ttnn.float32, use_legacy=None, sub_core_grids=None
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

    def python_slice_columns(self, tensor: Tensor, start: int, stop: int) -> Tensor:
        host = self.from_device(tensor)
        return self.to_device(host[:, start:stop])

    def conv_valid(self, data: Tensor, kernel: Tensor) -> Tensor:
        batch, data_len = data.shape
        kernel_len = kernel.shape[1]
        out_len = data_len - kernel_len + 1
        if out_len <= 0:
            return self.empty(batch, 0)

        kernel_values = self.from_device(kernel).reshape(-1).tolist()
        outputs: list[Tensor] = []

        for out_i in range(out_len):
            accum = self.full(batch, 1, 0.0)
            for j in range(kernel_len):
                data_col = self.slice_columns(data, out_i + j, 1)
                coeff = float(kernel_values[kernel_len - 1 - j])
                accum = self.add(accum, self.mul_scalar(data_col, coeff))
            outputs.append(accum)

        return self.concat_columns(outputs, batch)


class ShiftedArray:
    def __init__(self, data: Tensor, shift: int, ops: TTNNOps):
        self.data = data
        self.shift = int(shift)
        self.ops = ops

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
        coeff_tensor = torch.tensor(coefficients, dtype=torch.float32).reshape(1, -1)
        return cls(ops.to_device(coeff_tensor), shift, ops)

    def __mul__(self, other: float | "ShiftedArray") -> "ShiftedArray":
        if isinstance(other, (int, float)):
            return ShiftedArray(self.ops.mul_scalar(self.data, float(other)), self.shift, self.ops)

        if not isinstance(other, ShiftedArray):
            return NotImplemented

        shift_new = self.shift + other.shift + min(self.data.shape[1], other.data.shape[1]) - 1
        data_new = self.ops.conv_valid(self.data, other.data)
        return ShiftedArray(data_new, shift_new, self.ops)

    def __add__(self, other: "ShiftedArray") -> "ShiftedArray":
        if not isinstance(other, ShiftedArray):
            return NotImplemented

        shift_new = max(self.shift, other.shift)
        end_new = min(self.end, other.end)
        overlap = end_new - shift_new

        if overlap <= 0:
            return ShiftedArray(self.ops.empty(self.data.shape[0], 0), shift_new, self.ops)

        data1 = self.ops.slice_columns(self.data, shift_new - self.shift, overlap)
        data2 = self.ops.slice_columns(other.data, shift_new - other.shift, overlap)
        return ShiftedArray(self.ops.add(data1, data2), shift_new, self.ops)

    def __neg__(self) -> "ShiftedArray":
        return ShiftedArray(self.ops.mul_scalar(self.data, -1.0), self.shift, self.ops)

    def __sub__(self, other: "ShiftedArray") -> "ShiftedArray":
        return self + (-other)


class LiftingStep:
    def forward(
        self,
        even: ShiftedArray,
        odd: ShiftedArray,
    ) -> tuple[ShiftedArray, ShiftedArray]:
        return even, odd

    def inverse(
        self,
        even: ShiftedArray,
        odd: ShiftedArray,
    ) -> tuple[ShiftedArray, ShiftedArray]:
        return even, odd


class LiftingStepPredict(LiftingStep):
    def __init__(self, kernel: ShiftedArray):
        self.kernel = kernel

    def forward(
        self,
        even: ShiftedArray,
        odd: ShiftedArray,
    ) -> tuple[ShiftedArray, ShiftedArray]:
        pred = even * self.kernel
        return even, odd + pred

    def inverse(
        self,
        even: ShiftedArray,
        odd: ShiftedArray,
    ) -> tuple[ShiftedArray, ShiftedArray]:
        pred = even * self.kernel
        return even, odd - pred


class LiftingStepUpdate(LiftingStep):
    def __init__(self, kernel: ShiftedArray):
        self.kernel = kernel

    def forward(
        self,
        even: ShiftedArray,
        odd: ShiftedArray,
    ) -> tuple[ShiftedArray, ShiftedArray]:
        upd = odd * self.kernel
        return even + upd, odd

    def inverse(
        self,
        even: ShiftedArray,
        odd: ShiftedArray,
    ) -> tuple[ShiftedArray, ShiftedArray]:
        upd = odd * self.kernel
        return even - upd, odd


class LiftingStepScaleEven(LiftingStep):
    def __init__(self, factor: float):
        self.factor = float(factor)

    def forward(
        self,
        even: ShiftedArray,
        odd: ShiftedArray,
    ) -> tuple[ShiftedArray, ShiftedArray]:
        return even * self.factor, odd

    def inverse(
        self,
        even: ShiftedArray,
        odd: ShiftedArray,
    ) -> tuple[ShiftedArray, ShiftedArray]:
        return even * (1.0 / self.factor), odd


class LiftingStepScaleOdd(LiftingStep):
    def __init__(self, factor: float):
        self.factor = float(factor)

    def forward(
        self,
        even: ShiftedArray,
        odd: ShiftedArray,
    ) -> tuple[ShiftedArray, ShiftedArray]:
        return even, odd * self.factor

    def inverse(
        self,
        even: ShiftedArray,
        odd: ShiftedArray,
    ) -> tuple[ShiftedArray, ShiftedArray]:
        return even, odd * (1.0 / self.factor)


class LiftingStepSwap(LiftingStep):
    def forward(
        self,
        even: ShiftedArray,
        odd: ShiftedArray,
    ) -> tuple[ShiftedArray, ShiftedArray]:
        return odd, even

    def inverse(
        self,
        even: ShiftedArray,
        odd: ShiftedArray,
    ) -> tuple[ShiftedArray, ShiftedArray]:
        return odd, even


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
            host = value.to(dtype=torch.float32)
        elif isinstance(value, np.ndarray):
            host = torch.from_numpy(value).to(dtype=torch.float32)
        else:
            host = self.ops.from_device(value).to(dtype=torch.float32)

        if host.ndim == 1:
            host = host.unsqueeze(0)
        if host.ndim != 2:
            raise ValueError("Input must be 1D or 2D")
        return self.ops.to_device(host)

    def pad(self, tensor: Tensor, shape: int) -> Tensor:
        host = self.ops.from_device(tensor).detach().cpu().numpy()

        if self.scheme.mode == "symmetric":
            padded = np.pad(host, ((0, 0), (shape, shape)), mode="symmetric")
        elif self.scheme.mode == "periodic":
            padded = np.pad(host, ((0, 0), (shape, shape)), mode="wrap")
        elif self.scheme.mode == "zero":
            padded = np.pad(
                host,
                ((0, 0), (shape, shape)),
                mode="constant",
                constant_values=0,
            )
        elif self.scheme.mode == "constant":
            padded = np.pad(host, ((0, 0), (shape, shape)), mode="edge")
        else:
            raise ValueError(f"Unsupported padding mode: {self.scheme.mode}")

        return self.ops.to_device(torch.from_numpy(padded).to(dtype=torch.float32))

    def _slice_python(self, tensor: Tensor, start: int, stop: int) -> Tensor:
        return self.ops.python_slice_columns(tensor, start, stop)

    def _split_even_odd(self, tensor: Tensor) -> tuple[Tensor, Tensor]:
        batch, length = tensor.shape
        even_cols = [self.ops.slice_columns(tensor, i, 1) for i in range(0, length, 2)]
        odd_cols = [self.ops.slice_columns(tensor, i, 1) for i in range(1, length, 2)]
        return (
            self.ops.concat_columns(even_cols, batch),
            self.ops.concat_columns(odd_cols, batch),
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

        even_data = self._slice_python(even.data, -even_shift, -even_shift + length)
        odd_data = self._slice_python(odd.data, -odd_shift, -odd_shift + length)

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
        even_out = self._slice_python(even_sa.data, -even_shift, -even_shift + half)
        odd_out = self._slice_python(odd_sa.data, -odd_shift, -odd_shift + half)

        even_host = self.ops.from_device(even_out)
        odd_host = self.ops.from_device(odd_out)
        out_host = torch.empty((even_host.shape[0], length), dtype=torch.float32)
        out_host[:, 1::2] = even_host
        out_host[:, ::2] = odd_host
        return self.ops.to_device(out_host)
