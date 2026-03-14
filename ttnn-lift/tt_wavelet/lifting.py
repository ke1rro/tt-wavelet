import json
from dataclasses import dataclass
from enum import Enum

import numpy as np
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

    def __init__(self, scheme: LiftingScheme, device: ttnn.device):
        self.scheme = scheme
        self.device = device

    def pad_input(self, x: torch.Tensor, tap_size: int) -> torch.Tensor:
        pad = tap_size - 1
        if pad <= 0:
            return x

        padded = np.pad(
            x.detach().cpu().numpy(),
            ((0, 0), (pad, pad)),
            mode="symmetric",
        )
        return torch.from_numpy(padded).to(dtype=x.dtype, device=x.device)

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
        return ttnn.slice(x, [0, logic_i], [signal, logic_i + 1], [1, 1])

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
        return ttnn.slice(x, [0, logic_i], [signal, logic_i + 1], [1, 1])

    def get_core(self, x: ttnn.Tensor, start: int, length: int) -> ttnn.Tensor:
        signal = x.shape[0]
        return ttnn.slice(x, [0, start], [signal, start + length], [1, 1])

    def remap_symmetric_index(self, length: int, logic_i: int) -> int:
        if length <= 0:
            raise ValueError("Cannot read samples from an empty branch")

        period = 2 * length
        folded = logic_i % period
        if folded < length:
            return folded
        return period - 1 - folded

    def get_branch_sample(self, x: ttnn.Tensor, logic_i: int) -> ttnn.Tensor:
        mapped_i = self.remap_symmetric_index(x.shape[1], logic_i)
        return self.get_splited_sample(x, mapped_i)

    def replace_core(self, x: ttnn.Tensor, start: int, core: ttnn.Tensor) -> ttnn.Tensor:
        _, length = x.shape
        core_length = core.shape[1]
        if start == 0 and core_length == length:
            return core

        parts = []
        if start > 0:
            parts.append(self.get_core(x, 0, start))
        parts.append(core)

        right_start = start + core_length
        right_length = length - right_start
        if right_length > 0:
            parts.append(self.get_core(x, right_start, right_length))
        return ttnn.concat(parts, dim=1)

    def scale(self, x: ttnn.Tensor, start: int, length: int, scalar: float) -> ttnn.Tensor:
        if length == 0:
            return x

        core = self.get_core(x, start, length)
        scaled_core = self.mul_scalar_dev(core, scalar)
        return self.replace_core(x, start, scaled_core)

    def conv(
        self,
        dst: ttnn.Tensor,
        src: ttnn.Tensor,
        coefficients: list[float],
        shift: int,
        dst_start: int,
        dst_length: int,
        src_start: int,
    ) -> ttnn.Tensor:
        signal, length = dst.shape
        if dst_length == 0:
            return dst

        left = self.get_core(dst, 0, dst_start) if dst_start > 0 else None
        out = []

        for n in range(dst_length):
            accum = ttnn.full(
                [signal, 1],
                fill_value=0.0,
                dtype=ttnn.float32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
            )
            for j, coeff in enumerate(coefficients):
                idx = src_start + n - j - shift
                src_col = self.get_branch_sample(src, idx)
                term = self.mul_scalar_dev(src_col, coeff)
                accum = self.add_dev(accum, term)

            dst_col = self.get_splited_sample(dst, dst_start + n)
            out.append(self.add_dev(dst_col, accum))
        right_start = dst_start + dst_length
        right_length = length - right_start
        right = self.get_core(dst, right_start, right_length) if right_length > 0 else None

        parts = []
        if left is not None:
            parts.append(left)
        parts.append(ttnn.concat(out, dim=1))
        if right is not None:
            parts.append(right)
        return ttnn.concat(parts, dim=1)

    def apply_step(
        self,
        even: ttnn.Tensor,
        odd: ttnn.Tensor,
        step: LiftingStep,
        even_start: int,
        even_length: int,
        odd_start: int,
        odd_length: int,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, int, int, int, int]:
        if step.type == StepType.PREDICT:
            odd = self.conv(
                odd, even, step.coefficients, step.shift, odd_start, odd_length, even_start
            )
        elif step.type == StepType.UPDATE:
            even = self.conv(
                even, odd, step.coefficients, step.shift, even_start, even_length, odd_start
            )
        elif step.type == StepType.SCALE_EVEN:
            even = self.scale(even, even_start, even_length, step.coefficients[0])
        elif step.type == StepType.SCALE_ODD:
            odd = self.scale(odd, odd_start, odd_length, step.coefficients[0])
        elif step.type == StepType.SWAP:
            even, odd = odd, even
            even_start, odd_start = odd_start, even_start
            even_length, odd_length = odd_length, even_length
        else:
            raise ValueError(f"Unknown step type: {step.type}")

        return even, odd, even_start, even_length, odd_start, odd_length

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
        if not isinstance(x, torch.Tensor):
            x = self.from_device(x)

        x = self.to_device(x)
        even, odd = self.split(x)
        even_start = 0
        odd_start = 0
        even_length = even.shape[1]
        odd_length = odd.shape[1]

        for step in self.scheme.steps:
            even, odd, even_start, even_length, odd_start, odd_length = self.apply_step(
                even, odd, step, even_start, even_length, odd_start, odd_length
            )

        even = self.get_core(even, even_start, even_length)
        odd = self.get_core(odd, odd_start, odd_length)
        even = self.apply_delay(even, self.scheme.delay_even)
        odd = self.apply_delay(odd, self.scheme.delay_odd)

        return {
            "cA": even,
            "cD": odd,
            "delay_even": self.scheme.delay_even,
            "delay_odd": self.scheme.delay_odd,
        }
