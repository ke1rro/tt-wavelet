from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


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
            steps.append(
                LiftingStepSpec(
                    type=raw_step["type"],
                    shift=int(raw_step["shift"]),
                    coefficients=[
                        _to_coeff_float(coeff) for coeff in raw_step.get("coefficients", [])
                    ],
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
