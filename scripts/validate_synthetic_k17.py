#!/usr/bin/env python3
"""Validate the production synthetic K=17 LWT/ILWT round trip on hardware."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import numpy as np

from runtime_checks import (
    check_consistent_architecture,
    parse_runtime_architecture,
    run_ncrisc_elf_gate,
)


def parse_signal(output: str) -> np.ndarray:
    for line in output.splitlines():
        if line.startswith("tt-wavelet reconstructed signal"):
            text = line.split("[", 1)[1].rsplit("]", 1)[0]
            return np.fromstring(text, sep=",", dtype=np.float32)
    raise RuntimeError("device output did not contain the reconstructed signal")


def parse_roundtrip_error(output: str) -> float:
    prefix = "ilwt_roundtrip_max_abs_error:"
    for line in output.splitlines():
        if line.startswith(prefix):
            return float(line.removeprefix(prefix).strip())
    raise RuntimeError("device output did not contain the round-trip error")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=root / "build" / "lwt")
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=[1, 2, 3, 17, 31, 32, 33, 3071, 3072, 3073],
    )
    parser.add_argument(
        "--layouts",
        nargs="+",
        choices=["row-major", "tile-native"],
        default=["row-major", "tile-native"],
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Maximum deterministic FP32 round-trip error (default: exact zero).",
    )
    args = parser.parse_args()

    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f"binary not found: {binary}")
    if any(length <= 0 for length in args.lengths):
        parser.error("all lengths must be positive")
    if args.tolerance < 0.0:
        parser.error("--tolerance must be non-negative")

    architecture: str | None = None
    failures: list[str] = []
    case_count = 0
    max_abs_error = 0.0
    reference_by_length: dict[int, np.ndarray] = {}
    for layout in args.layouts:
        for length in args.lengths:
            environment = os.environ.copy()
            environment["TT_LOGGER_LEVEL"] = "FATAL"
            environment["TT_WAVELET_LWT_WORKSPACE_LAYOUT"] = layout
            result = subprocess.run(
                [
                    str(binary),
                    "--inverse",
                    "--length",
                    str(length),
                    "synthetic-k17",
                ],
                text=True,
                capture_output=True,
                env=environment,
                check=False,
            )
            if result.returncode != 0:
                failures.append(
                    f"layout={layout} N={length}: runtime failure\n"
                    f"{result.stdout}{result.stderr}"
                )
                continue
            output = result.stdout + result.stderr
            architecture = check_consistent_architecture(
                architecture, parse_runtime_architecture(output)
            )
            reconstructed = parse_signal(result.stdout)
            error = parse_roundtrip_error(output)
            layout_identical = True
            if length in reference_by_length:
                layout_identical = np.array_equal(
                    reconstructed.view(np.uint32),
                    reference_by_length[length].view(np.uint32),
                )
            else:
                reference_by_length[length] = reconstructed.copy()
            max_abs_error = max(max_abs_error, error)
            case_count += 1
            print(
                f"synthetic-k17 N={length:4d} layout={layout:11s} "
                f"roundtrip_zero={int(error == 0.0)} "
                f"layout_bit_identical={int(layout_identical)} max_abs={error:.8e}"
            )
            if not np.isfinite(error) or error > args.tolerance:
                failures.append(
                    f"layout={layout} N={length}: round-trip error {error:.8e} "
                    f"exceeds {args.tolerance:.8e}"
                )
            if not layout_identical:
                failures.append(
                    f"layout={layout} N={length}: output differs from the first layout"
                )

    print(f"validated_device_cases: {case_count}")
    print(f"max_abs_roundtrip_error: {max_abs_error:.8e}")
    print(f"roundtrip_tolerance: {args.tolerance:.8e}")
    if failures:
        raise SystemExit("\n".join(["Synthetic K=17 validation failed:", *failures]))
    if architecture is None:
        raise SystemExit("Synthetic K=17 validation did not execute a device case")
    run_ncrisc_elf_gate(root, architecture)


if __name__ == "__main__":
    main()
