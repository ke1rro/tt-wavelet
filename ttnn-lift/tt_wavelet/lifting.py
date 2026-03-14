import json
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
    index_mapper: callable[[int, int], int]


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

    def from_device(self, x_tt) -> torch.Tensor:
        return ttnn.to_torch(
            ttnn.to_device(x_tt, self.device),
            dtype=torch.float32,
            layout=torch.ROW_MAJOR_LAYOUT,
        )

    def to_device(self, x: torch.Tensor):
        return ttnn.to_device(
            ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT),
            self.device,
        )

    def add_dev(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        result_dev = ttnn.add(self.to_device(a), self.to_device(b))
        return self.from_device(result_dev)

    def mul_scalar_dev(self, x: torch.Tensor, scalar: float) -> torch.Tensor:
        result_dev = ttnn.mul_scalar(self.to_device(x), scalar)
        return self.from_device(result_dev)

    def get_signal_sample(self, x: torch.Tensor, logic_i: int) -> float:
        mapped = self._index(logic_i, x.shape[1])
        return x[:, mapped : mapped + 1]

    def get_even_sample(self, x: torch.Tensor, logic_i: int) -> float:
        return self.get_signal_sample(x, logic_i * 2)

    def get_odd_sample(self, x: torch.Tensor, logic_i: int) -> float:
        return self.get_signal_sample(x, logic_i * 2 + 1)

    def split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 2:
            raise ValueError(
                "Input tensor must be 2D with shape (1 - number of signals, length of signal)"
            )

        signal, length = x.shape
        n_even = (length + 1) // 2
        n_odd = length // 2
        even = torch.empty((signal, n_even), dtype=x.dtype, device=x.device)
        odd = torch.empty((signal, n_odd), dtype=x.dtype, device=x.device)

        for k in range(n_even):
            even[:, k : k + 1] = self.get_even_sample(x, k)

        for k in range(n_odd):
            odd[:, k : k + 1] = self.get_odd_sample(x, k)

        return even, odd

    def get_splited_sample(self, x: torch.Tensor, logic_i) -> torch.Tensor:
        mapped = self._index(logic_i, x.shape[1])
        return x[:, mapped : mapped + 1]

    def conv(
        self,
        dst: torch.Tensor,
        src: torch.Tensor,
        coefficients: list[float],
        shift: int,
    ) -> torch.Tensor:
        signal, length = dst.shape
        out = dst.clone()

        for n in range(length):
            accum = torch.zeros((signal, 1), dtype=dst.dtype, device=dst.device)
            for j, coeff in enumerate(coefficients):
                idx = n - j - shift
                src_col = self.get_splited_sample(src, idx)
                term = self.mul_scalar_dev(src_col, coeff)
                accum = self.add_dev(accum, term)

            out[:, n : n + 1] = self.add_dev(out[:, n : n + 1], accum)
        return out

    def apply_step(
        self,
        even: torch.Tensor,
        odd: torch.Tensor,
        step: LiftingStep,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

    def aplpy_delay(self, x: torch.Tensor, delay: int) -> torch.Tensor:
        if delay == 0:
            return x
        out = torch.zeros_like(x)
        if delay < x.shape[1]:
            out[:, delay:] = x[:, :-delay]
        return out

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
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
