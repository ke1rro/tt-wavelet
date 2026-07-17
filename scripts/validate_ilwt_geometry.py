#!/usr/bin/env python3
"""Validate 1D ILWT dependency-cone geometry for every JSON scheme.

This is an independent model of inverse_plan.hpp.  It intentionally derives
inverse requirements from each forward trace instead of planning a synthetic
"inverse forward" scheme, which would lose the original crop geometry.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Interval:
    begin: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.begin


@dataclass(frozen=True)
class Route:
    kind: str
    source_length: int
    base_length: int
    source_offset: int
    base_offset: int
    output_length: int
    coefficient_count: int


def hull(lhs: Interval, rhs: Interval) -> Interval:
    if lhs.length == 0:
        return rhs
    if rhs.length == 0:
        return lhs
    return Interval(min(lhs.begin, rhs.begin), max(lhs.end, rhs.end))


def translated(interval: Interval, offset: int, right_expansion: int = 0) -> Interval:
    if interval.length == 0:
        return interval
    return Interval(interval.begin + offset, interval.end + offset + right_expansion)


def subtract_offset(interval: Interval, offset: int) -> Interval:
    assert interval.length == 0 or interval.begin >= offset
    return (
        interval
        if interval.length == 0
        else Interval(interval.begin - offset, interval.end - offset)
    )


def contains(outer: Interval, inner: Interval) -> bool:
    return inner.length == 0 or (outer.begin <= inner.begin and inner.end <= outer.end)


def validate(interval: Interval, length: int) -> None:
    assert 0 <= interval.begin <= interval.end <= length


def make_forward_trace(
    scheme: dict, signal_length: int
) -> tuple[list[Route], tuple[int, int], tuple[int, int]]:
    tap_size = int(scheme["tap_size"])
    padded_length = signal_length + 2 * (tap_size - 1)
    even_shift = int(scheme["delay"]["even"])
    odd_shift = int(scheme["delay"]["odd"])
    even_length = (padded_length + 1) // 2
    odd_length = padded_length // 2
    initial_lengths = (even_length, odd_length)
    routes: list[Route] = []

    for step in scheme["steps"]:
        kind = step["type"]
        if kind in ("predict", "update"):
            k = len(step["coefficients"])
            if kind == "predict":
                source_shift, source_length = even_shift, even_length
                base_shift, base_length = odd_shift, odd_length
            else:
                source_shift, source_length = odd_shift, odd_length
                base_shift, base_length = even_shift, even_length

            convolution_shift = (
                source_shift + int(step["shift"]) + min(source_length, k) - 1
            )
            convolution_length = max(source_length - k + 1, 0)
            output_shift = max(base_shift, convolution_shift)
            output_end = min(
                base_shift + base_length, convolution_shift + convolution_length
            )
            output_length = max(output_end - output_shift, 0)
            source_offset = output_shift - convolution_shift
            base_offset = output_shift - base_shift
            routes.append(
                Route(
                    kind,
                    source_length,
                    base_length,
                    source_offset,
                    base_offset,
                    output_length,
                    k,
                )
            )
            if kind == "predict":
                odd_shift, odd_length = output_shift, output_length
            else:
                even_shift, even_length = output_shift, output_length
        elif kind in ("scale-even", "scale-odd"):
            source_length = even_length if kind == "scale-even" else odd_length
            routes.append(
                Route(kind, source_length, source_length, 0, 0, source_length, 1)
            )
        elif kind == "swap":
            routes.append(Route(kind, even_length, odd_length, 0, 0, 0, 0))
            even_shift, odd_shift = odd_shift, even_shift
            even_length, odd_length = odd_length, even_length
        else:
            raise AssertionError(f"unknown step type: {kind}")

    return routes, initial_lengths, (even_shift, odd_shift)


def validate_chunk(scheme: dict, signal_length: int, output: Interval) -> float:
    routes, initial_lengths, final_shifts = make_forward_trace(scheme, signal_length)
    pad = int(scheme["tap_size"]) - 1
    coefficient_length = (signal_length + int(scheme["tap_size"]) - 1) // 2
    target_even = Interval((output.begin + pad + 1) // 2, (output.end + pad + 1) // 2)
    target_odd = Interval((output.begin + pad) // 2, (output.end + pad) // 2)
    required: list[tuple[Interval, Interval]] = [(target_even, target_odd)]
    even_length, odd_length = initial_lengths

    for route in routes:
        before_even, before_odd = required[-1]
        validate(before_even, even_length)
        validate(before_odd, odd_length)
        if route.kind == "predict":
            assert (
                even_length == route.source_length and odd_length == route.base_length
            )
            assert contains(
                Interval(route.base_offset, route.base_offset + route.output_length),
                before_odd,
            )
            target = subtract_offset(before_odd, route.base_offset)
            after = (
                hull(
                    before_even,
                    translated(
                        target, route.source_offset, route.coefficient_count - 1
                    ),
                ),
                target,
            )
            even_length, odd_length = route.source_length, route.output_length
        elif route.kind == "update":
            assert (
                odd_length == route.source_length and even_length == route.base_length
            )
            assert contains(
                Interval(route.base_offset, route.base_offset + route.output_length),
                before_even,
            )
            target = subtract_offset(before_even, route.base_offset)
            after = (
                target,
                hull(
                    before_odd,
                    translated(
                        target, route.source_offset, route.coefficient_count - 1
                    ),
                ),
            )
            even_length, odd_length = route.output_length, route.source_length
        elif route.kind.startswith("scale-"):
            after = (before_even, before_odd)
        else:
            after = (before_odd, before_even)
            even_length, odd_length = odd_length, even_length
        validate(after[0], even_length)
        validate(after[1], odd_length)
        required.append(after)

    canonical_start = int(scheme["tap_size"]) // 2
    canonical = []
    for interval, shift in zip(required[-1], final_shifts):
        mapped = Interval(
            interval.begin + shift - canonical_start,
            interval.end + shift - canonical_start,
        )
        validate(mapped, coefficient_length)
        canonical.append(mapped)

    active_even, active_odd = required[-1]
    for route, before in zip(reversed(routes), reversed(required[:-1])):
        if route.kind == "swap":
            active_even, active_odd = active_odd, active_even
        elif route.kind in ("predict", "update"):
            output_interval = before[1] if route.kind == "predict" else before[0]
            target = subtract_offset(output_interval, route.base_offset)
            source_required = translated(
                target, route.source_offset, route.coefficient_count - 1
            )
            source = active_even if route.kind == "predict" else active_odd
            base = active_odd if route.kind == "predict" else active_even
            assert contains(source, source_required) and contains(base, target)
            if route.kind == "predict":
                active_odd = output_interval
            else:
                active_even = output_interval
        elif route.kind == "scale-even":
            assert contains(active_even, before[0])
            active_even = before[0]
        else:
            assert route.kind == "scale-odd" and contains(active_odd, before[1])
            active_odd = before[1]

    assert contains(active_even, target_even) and contains(active_odd, target_odd)
    dependencies = canonical[0].length + canonical[1].length
    return max(dependencies - output.length, 0) / output.length


def intervals_for_length(length: int, exhaustive: bool) -> list[Interval]:
    intervals = [Interval(0, length)]
    for begin in range(0, length, 3072):
        intervals.append(Interval(begin, min(begin + 3072, length)))
    if exhaustive:
        intervals.extend(Interval(index, index + 1) for index in range(length))
    else:
        for index in {0, length // 2, length - 1}:
            intervals.append(Interval(index, index + 1))
    return list(dict.fromkeys(intervals))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wavelets",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "wavelets",
    )
    parser.add_argument("--exhaustive", action="store_true")
    args = parser.parse_args()

    lengths = [*range(1, 130), 255, 256, 257, 1023, 1024, 1025, 3071, 3072, 3073, 10000]
    scheme_count = 0
    case_count = 0
    max_overhead = (0.0, "", 0, Interval(0, 1))
    for path in sorted(args.wavelets.glob("*.json")):
        scheme = json.loads(path.read_text())
        scheme_count += 1
        for length in lengths:
            for interval in intervals_for_length(length, args.exhaustive):
                overhead = validate_chunk(scheme, length, interval)
                case_count += 1
                if overhead > max_overhead[0]:
                    max_overhead = (overhead, path.stem, length, interval)

    overhead, name, length, interval = max_overhead
    print(f"validated_schemes: {scheme_count}")
    print(f"validated_chunks: {case_count}")
    print(
        "max_dependency_overhead: "
        f"{overhead:.6f} ({name}, N={length}, output=[{interval.begin},{interval.end}))"
    )


if __name__ == "__main__":
    main()
