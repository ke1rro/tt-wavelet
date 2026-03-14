import json
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import torch
import ttnn


class StepType(Enum):
    PREDICT = 1
    UPDATE = 2
    SCALE_EVEN = 3
    SCALE_ODD = 4
    SWAP = 5


def map_type_step(step_type: str) -> StepType:
    if step_type == "predict":
        return StepType.PREDICT
    elif step_type == "update":
        return StepType.UPDATE
    elif step_type == "scale-even":
        return StepType.SCALE_EVEN
    elif step_type == "scale-odd":
        return StepType.SCALE_ODD
    elif step_type == "swap":
        return StepType.SWAP
    else:
        raise ValueError(f"Unknown step type: {step_type}")


@dataclass(frozen=True)
class BoundaryMode:
    name: str
    index_mapper: Callable[[int, int], int]


class BoundaryModes:
    SYMMETRIC = BoundaryMode(
        name="symmetric",
        index_mapper=lambda i, n: (
            0 if n == 1 else ((i % (2 * n)) if (i % (2 * n)) < n else (2 * n - 1 - (i % (2 * n))))
        ),
    )

    PERIODIC = BoundaryMode(
        name="periodic",
        index_mapper=lambda i, n: i % n,
    )


@dataclass(frozen=True)
class LiftingStep:
    type: StepType
    shift: int
    coefficients: list[float]


@dataclass(frozen=True)
class LiftingScheme:
    tap_size: int
    delay_odd: int
    delay_even: int
    steps: list[LiftingStep]


def load_lifting_scheme(path: str) -> LiftingScheme:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    steps = []
    for step_data in data["steps"]:
        step = LiftingStep(
            type=map_type_step(step_data["type"]),
            shift=step_data["shift"],
            coefficients=step_data["coefficients"],
        )
        steps.append(step)

    return LiftingScheme(
        tap_size=data["tap_size"],
        delay_odd=data["delay"]["odd"],
        delay_even=data["delay"]["even"],
        steps=steps,
    )


class LiftingWaveletTransform:
    scheme: LiftingScheme
    device: ttnn.device
    mode: BoundaryMode

    def __init__(self, scheme: LiftingScheme, device: ttnn.device, mode: BoundaryMode):
        self.scheme = scheme
        self.device = device
        self.mode = mode

    def _index(self, i: int, n: int) -> int:
        return self.mode.index_mapper(i, n)

    def empty(self, signal: int, length: int) -> ttnn.Tensor:
        host = torch.empty((signal, length), dtype=torch.float32)
        return self.to_device(host)

    def from_device(self, x_tt) -> torch.Tensor:
        return ttnn.to_torch(x_tt)

    def to_device(self, x: torch.Tensor):
        return ttnn.from_torch(
            x,
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )

    def add_dev(self, a: ttnn.Tensor, b: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.add(a, b)

    def mul_scalar_dev(self, x: ttnn.Tensor, scalar: float) -> ttnn.Tensor:
        return ttnn.multiply(x, float(scalar))

    def get_signal_sample(self, x: ttnn.Tensor, logic_i: int) -> ttnn.Tensor:
        signal = x.shape[0]
        mapped = self._index(logic_i, x.shape[1])
        return ttnn.slice(x, [0, mapped], [signal, mapped + 1], [1, 1])

    def get_even_sample(self, x: ttnn.Tensor, logic_i: int) -> ttnn.Tensor:
        return self.get_signal_sample(x, logic_i * 2)

    def get_odd_sample(self, x: ttnn.Tensor, logic_i: int) -> ttnn.Tensor:
        return self.get_signal_sample(x, logic_i * 2 + 1)

    def split(self, x: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        if len(x.shape) != 2:
            raise ValueError(
                "Input tensor must be 2D with shape (1 - number of signals, length of signal)"
            )

        _, length = x.shape
        n_even = (length + 1) // 2
        n_odd = length // 2
        even = [self.get_even_sample(x, k) for k in range(n_even)]
        odd = [self.get_odd_sample(x, k) for k in range(n_odd)]
        even_out = ttnn.concat(even, dim=1) if even else self.empty(x.shape[0], 0)
        odd_out = ttnn.concat(odd, dim=1) if odd else self.empty(x.shape[0], 0)
        return even_out, odd_out

    def get_splited_sample(self, x: ttnn.Tensor, logic_i) -> ttnn.Tensor:
        signal = x.shape[0]
        mapped = self._index(logic_i, x.shape[1])
        return ttnn.slice(x, [0, mapped], [signal, mapped + 1], [1, 1])

    def conv(
        self,
        dst: ttnn.Tensor,
        src: ttnn.Tensor,
        coefficients: list[float],
        shift: int,
    ) -> ttnn.Tensor:
        signal, length = dst.shape
        out = []

        for n in range(length):
            accum = ttnn.full(
                [signal, 1],
                fill_value=0.0,
                dtype=ttnn.float32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
            )
            for j, coeff in enumerate(coefficients):
                idx = n - j - shift
                src_col = self.get_splited_sample(src, idx)
                term = self.mul_scalar_dev(src_col, coeff)
                accum = self.add_dev(accum, term)

            dst_col = self.get_splited_sample(dst, n)
            out.append(self.add_dev(dst_col, accum))
        return ttnn.concat(out, dim=1)

    def apply_step(
        self,
        even: ttnn.Tensor,
        odd: ttnn.Tensor,
        step: LiftingStep,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        if step.type == StepType.PREDICT:
            odd = self.conv(odd, even, step.coefficients, step.shift)
        elif step.type == StepType.UPDATE:
            even = self.conv(even, odd, step.coefficients, step.shift)
        elif step.type == StepType.SCALE_EVEN:
            even = self.mul_scalar_dev(even, step.coefficients[0])
        elif step.type == StepType.SCALE_ODD:
            odd = self.mul_scalar_dev(odd, step.coefficients[0])
        elif step.type == StepType.SWAP:
            even, odd = odd, even
        else:
            raise ValueError(f"Unknown step type: {step.type}")

        return even, odd

    def aplpy_delay(self, x: ttnn.Tensor, delay: int) -> ttnn.Tensor:
        if delay == 0:
            return x
        signal, length = x.shape
        zero = ttnn.full(
            [signal, min(delay, length)],
            fill_value=0.0,
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )
        if delay >= length:
            return zero
        body = ttnn.slice(x, [0, 0], [signal, length - delay], [1, 1])
        return ttnn.concat([zero, body], dim=1)

    def forward(self, x) -> dict[str, ttnn.Tensor | int]:
        if isinstance(x, torch.Tensor):
            x = self.to_device(x)

        even, odd = self.split(x)

        for step in self.scheme.steps:
            even, odd = self.apply_step(even, odd, step)

        even = self.aplpy_delay(even, self.scheme.delay_even)
        odd = self.aplpy_delay(odd, self.scheme.delay_odd)

        return {
            "cA": even,
            "cD": odd,
            "delay_even": self.scheme.delay_even,
            "delay_odd": self.scheme.delay_odd,
        }
