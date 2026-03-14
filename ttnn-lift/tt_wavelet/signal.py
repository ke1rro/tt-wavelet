from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np


class SignalType(Enum):
    UNIFORM = 1
    NORMAL = 2
    ALTERNATING = 3
    IMPULSE = 4
    RAMP = 5
    SINUSOIDAL = 6
    CONSTANT = 7
    STEP = 8
    SQUARE = 9


@dataclass(frozen=True)
class InputSpec:
    kind: SignalType
    length: int
    magnitude: float
    seed: int | None = None


type SignalGenerator = Callable[[InputSpec], np.ndarray]

registry: dict[SignalType, SignalGenerator] = {}


def register(signal_type: SignalType):
    def wrapper(fn: SignalGenerator) -> SignalGenerator:
        registry[signal_type] = fn
        return fn

    return wrapper


@register(SignalType.UNIFORM)
def uniform(spec: InputSpec) -> np.ndarray:
    rng = np.random.default_rng(spec.seed)
    return rng.uniform(
        -spec.magnitude,
        spec.magnitude,
        size=spec.length,
    ).astype(np.float64)


@register(SignalType.NORMAL)
def normal(spec: InputSpec) -> np.ndarray:
    rng = np.random.default_rng(spec.seed)
    return rng.normal(
        0.0,
        spec.magnitude,
        size=spec.length,
    ).astype(np.float64)


@register(SignalType.ALTERNATING)
def alternating(spec: InputSpec) -> np.ndarray:
    x = np.ones(spec.length, dtype=np.float64) * spec.magnitude
    x[1::2] *= -1.0
    return x


@register(SignalType.IMPULSE)
def impulse(spec: InputSpec) -> np.ndarray:
    x = np.zeros(spec.length, dtype=np.float64)
    x[spec.length // 2] = spec.magnitude
    return x


@register(SignalType.RAMP)
def ramp(spec: InputSpec) -> np.ndarray:
    return np.linspace(
        -spec.magnitude,
        spec.magnitude,
        num=spec.length,
        dtype=np.float64,
    )


@register(SignalType.SINUSOIDAL)
def sinusoidal(spec: InputSpec) -> np.ndarray:
    n = np.arange(spec.length, dtype=np.float64)
    return spec.magnitude * np.sin(2.0 * np.pi * 3.0 * n / float(spec.length))


@register(SignalType.CONSTANT)
def constant(spec: InputSpec) -> np.ndarray:
    return np.full(spec.length, spec.magnitude, dtype=np.float64)


@register(SignalType.STEP)
def step(spec: InputSpec) -> np.ndarray:
    x = np.zeros(spec.length, dtype=np.float64)
    x[spec.length // 2 :] = spec.magnitude
    return x


@register(SignalType.SQUARE)
def square(spec: InputSpec) -> np.ndarray:
    n = np.arange(spec.length, dtype=np.float64)
    phase = np.sin(2.0 * np.pi * 3.0 * n / float(spec.length))
    return np.where(phase >= 0.0, spec.magnitude, -spec.magnitude).astype(np.float64)


def generate_signal(spec: InputSpec) -> np.ndarray:
    try:
        generator = registry[spec.kind]
    except KeyError as error:
        raise ValueError(f"Unsupported signal type: {spec.kind}") from error
    return generator(spec)


class Signal:
    def __init__(self, spec: InputSpec):
        self.spec = spec
        self.data = generate_signal(spec)
