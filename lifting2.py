import json
from typing import Any

import numpy as np

import dtypes


class ShiftedArray:
    shift: int
    data: np.ndarray
    dtype: dtypes.dtype

    def __init__(self, data: np.ndarray, shift: int, dtype: dtypes.dtype):
        self.data = np.array(data)
        self.shift = shift
        self.dtype = dtype

    @property
    def end(self) -> int:
        return self.shift + len(self.data)

    @classmethod
    def from_object(cls, obj: dict[str, Any], dtype: dtypes.dtype) -> "ShiftedArray":
        data = []
        for coeff in obj["coefficients"]:
            if isinstance(coeff, (int, float)):
                data.append(coeff)
            else:
                data.append(coeff["numerator"] / coeff["denominator"])

        return cls(data, obj["shift"], dtype)

    def __repr__(self):
        return f"ShiftedArray(data={self.data}, shift={self.shift})"

    def __mul__(self, other: "float | ShiftedArray") -> "ShiftedArray":
        if isinstance(other, (int, float)):
            return ShiftedArray(self.dtype.mul(self.data, other), self.shift, self.dtype)

        if not isinstance(other, ShiftedArray):
            return NotImplemented

        shift_new = self.shift + other.shift + min(len(self.data), len(other.data)) - 1
        data_new = self.dtype.conv(self.data, other.data)
        return ShiftedArray(data_new, shift_new, self.dtype)

    def __add__(self, other: "ShiftedArray") -> "ShiftedArray":
        if not isinstance(other, ShiftedArray):
            return NotImplemented

        shift_new = min(self.shift, other.shift)
        end_new = min(self.end, other.end)

        data1 = np.pad(
            self.data[: end_new - self.shift], (self.shift - shift_new, 0), mode="constant"
        )
        data2 = np.pad(
            other.data[: end_new - other.shift], (other.shift - shift_new, 0), mode="constant"
        )
        data_new = self.dtype.add(data1, data2)

        assert max(self.shift, other.shift) - shift_new < 10

        return ShiftedArray(data_new, shift_new, self.dtype)

    def __neg__(self) -> "ShiftedArray":
        return ShiftedArray(-self.data, self.shift, self.dtype)

    def __sub__(self, other: "ShiftedArray") -> "ShiftedArray":
        return self + (-other)


class LiftingStep:
    def forward(self, even: ShiftedArray, odd: ShiftedArray) -> tuple[ShiftedArray, ShiftedArray]:
        return even, odd

    def inverse(self, even: ShiftedArray, odd: ShiftedArray) -> tuple[ShiftedArray, ShiftedArray]:
        return even, odd


class LiftingStepPredict(LiftingStep):
    kernel: ShiftedArray

    def __init__(self, kernel: ShiftedArray):
        self.kernel = kernel

    def forward(self, even: ShiftedArray, odd: ShiftedArray) -> tuple[ShiftedArray, ShiftedArray]:
        pred = even * self.kernel
        return even, odd + pred

    def inverse(self, even: ShiftedArray, odd: ShiftedArray) -> tuple[ShiftedArray, ShiftedArray]:
        pred = even * self.kernel
        return even, odd - pred


class LiftingStepUpdate(LiftingStep):
    kernel: ShiftedArray

    def __init__(self, kernel: ShiftedArray):
        self.kernel = kernel

    def forward(self, even: ShiftedArray, odd: ShiftedArray) -> tuple[ShiftedArray, ShiftedArray]:
        upd = odd * self.kernel
        return even + upd, odd

    def inverse(self, even: ShiftedArray, odd: ShiftedArray) -> tuple[ShiftedArray, ShiftedArray]:
        upd = odd * self.kernel
        return even - upd, odd


class LiftingStepScaleEven(LiftingStep):
    factor: float

    def __init__(self, factor: float):
        self.factor = factor

    def forward(self, even: ShiftedArray, odd: ShiftedArray) -> tuple[ShiftedArray, ShiftedArray]:
        return even * self.factor, odd

    def inverse(self, even: ShiftedArray, odd: ShiftedArray) -> tuple[ShiftedArray, ShiftedArray]:
        return even * (1 / self.factor), odd


class LiftingStepScaleOdd(LiftingStep):
    factor: float

    def __init__(self, factor: float):
        self.factor = factor

    def forward(self, even: ShiftedArray, odd: ShiftedArray) -> tuple[ShiftedArray, ShiftedArray]:
        return even, odd * self.factor

    def inverse(self, even: ShiftedArray, odd: ShiftedArray) -> tuple[ShiftedArray, ShiftedArray]:
        return even, odd * (1 / self.factor)


class LiftingStepSwap(LiftingStep):
    def forward(self, even: ShiftedArray, odd: ShiftedArray) -> tuple[ShiftedArray, ShiftedArray]:
        return odd, even

    def inverse(self, even: ShiftedArray, odd: ShiftedArray) -> tuple[ShiftedArray, ShiftedArray]:
        return odd, even


class LiftingScheme:
    def __init__(
        self,
        mode: str,
        tap_size: int,
        steps: list[LiftingStep],
        delays: tuple[int, int],
        dtype: dtypes.dtype_,
    ):
        self.mode = mode
        self.tap_size = tap_size
        self.steps = steps
        self.delays = delays
        self.dtype = dtype

    @classmethod
    def from_object(
        cls,
        obj: dict[str, Any],
        mode: str,
        dtype: dtypes.dtype_,
    ):
        steps = []

        for step in obj["steps"]:
            kernel = ShiftedArray.from_object(step, dtype=dtype)

            if step["type"] == "predict":
                steps.append(LiftingStepPredict(kernel))
            elif step["type"] == "update":
                steps.append(LiftingStepUpdate(kernel))
            elif step["type"] == "scale-even":
                assert kernel.shift == 0, "Scale steps must have zero shift"
                assert len(kernel.data) == 1, "Scale steps must have a single coefficient"
                steps.append(LiftingStepScaleEven(kernel.data[0]))
            elif step["type"] == "scale-odd":
                assert kernel.shift == 0, "Scale steps must have zero shift"
                assert len(kernel.data) == 1, "Scale steps must have a single coefficient"
                steps.append(LiftingStepScaleOdd(kernel.data[0]))
            elif step["type"] == "swap":
                steps.append(LiftingStepSwap())
            else:
                raise ValueError(f"Unsupported lifting step type: {step['type']}")

        return cls(
            mode=mode,
            tap_size=obj["tap_size"],
            steps=steps,
            delays=(
                obj["delay"]["even"],
                obj["delay"]["odd"],
            ),
            dtype=dtype,
        )

    @classmethod
    def from_file(
        cls,
        path: str,
        mode: str,
        dtype: dtypes.dtype_,
    ):
        with open(path, "r") as f:
            obj = json.load(f)

        return cls.from_object(obj, mode=mode, dtype=dtype)

    def pad(self, data: np.ndarray, shape: int) -> np.ndarray:
        if self.mode == "symmetric":
            return np.pad(data, shape, mode="symmetric")

        if self.mode == "periodic":
            return np.pad(data, shape, mode="wrap")

        if self.mode == "zero":
            return np.pad(data, shape, mode="constant", constant_values=0)

        if self.mode == "constant":
            return np.pad(data, shape, mode="edge")

        raise ValueError(f"Unsupported padding mode: {self.mode}")

    def forward(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        L = self.tap_size
        length = (len(data) + L - 1) // 2
        direct_shift = L // 2

        data = self.pad(data, (L - 1, L - 1))

        even = data[::2]
        odd = data[1::2]

        # Shift when multiplying by kernel without
        # polyphase split
        even = ShiftedArray(even, self.delays[0], self.dtype)
        odd = ShiftedArray(odd, self.delays[1], self.dtype)

        for step in self.steps:
            even, odd = step.forward(even, odd)

        even_shift = even.shift - direct_shift
        odd_shift = odd.shift - direct_shift

        if even_shift > 0 or odd_shift > 0:
            raise ValueError("Coefficients have positive shift after forward lifting")

        even = even.data[-even_shift : -even_shift + length]
        odd = odd.data[-odd_shift : -odd_shift + length]

        return even, odd

    def inverse(self, even: np.ndarray, odd: np.ndarray) -> np.ndarray:
        L = self.tap_size
        length = 2 * len(even) - L + 2
        direct_shift = L // 2

        even = ShiftedArray(even, 0, self.dtype)
        odd = ShiftedArray(odd, 0, self.dtype)

        for step in reversed(self.steps):
            even, odd = step.inverse(even, odd)

        even_shift = even.shift + self.delays[1] - direct_shift
        odd_shift = odd.shift + self.delays[0] - direct_shift

        # Odd is shifted
        odd_shift += 1

        even = even.data[-even_shift : -even_shift + length // 2]
        odd = odd.data[-odd_shift : -odd_shift + length // 2]

        data = np.empty(length, dtype=np.float64)
        data[1::2] = even
        data[::2] = odd

        return data
