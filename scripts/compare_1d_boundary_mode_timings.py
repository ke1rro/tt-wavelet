#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Compare 1D LWT/ILWT latency with PyWavelets for all boundary modes."""

from __future__ import annotations

import argparse
import csv
import re
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pywt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from compare_timings import tt_benchmark_env  # noqa: E402

LWT = PROJECT_ROOT / "build" / "lwt"
SET_ENV = PROJECT_ROOT / "scripts" / "set_env.sh"
MODES = (
    "zero",
    "constant",
    "symmetric",
    "reflect",
    "periodic",
    "smooth",
    "antisymmetric",
    "antireflect",
)
CSV_FIELDS = (
    "transform",
    "boundary_mode",
    "wavelet",
    "signal_length",
    "backend",
    "architecture",
    "active_core_count",
    "sample_count",
    "median_latency_ms",
    "timing_scope",
    "status",
    "error",
)
MEDIAN_PATTERN = re.compile(
    r"(?P<transform>i?lwt)_median_time_ms:\s*(?P<latency>[0-9eE+.\-]+)"
)
ARCH_PATTERN = re.compile(r"(?:i?lwt)_architecture:\s*(\S+)")
ACTIVE_CORES_PATTERN = re.compile(r"(?:i?lwt)_active_core_count:\s*(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wavelet", default="db7")
    parser.add_argument("--length-start", type=int, default=100_000)
    parser.add_argument("--length-stop", type=int, default=1_000_000)
    parser.add_argument("--length-step", type=int, default=10_000)
    parser.add_argument("--tt-repeats", type=int, default=3)
    parser.add_argument("--tt-warmup-runs", type=int, default=1)
    parser.add_argument("--pywt-repeats", type=int, default=3)
    parser.add_argument("--pywt-warmup-runs", type=int, default=1)
    parser.add_argument(
        "--timing-scope",
        choices=("device-only",),
        default="device-only",
        help="TT measures prepared device execution; PyWavelets measures only dwt/idwt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "docs" / "db7_1d_boundary_timings",
    )
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> list[int]:
    positive = {
        "--length-start": args.length_start,
        "--length-stop": args.length_stop,
        "--length-step": args.length_step,
        "--tt-repeats": args.tt_repeats,
        "--pywt-repeats": args.pywt_repeats,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.tt_warmup_runs < 0 or args.pywt_warmup_runs < 0:
        raise ValueError("warmup counts must be non-negative")
    if args.length_start > args.length_stop:
        raise ValueError("--length-start cannot exceed --length-stop")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")
    if not LWT.is_file():
        raise FileNotFoundError(f"missing production binary: {LWT}")
    pywt.Wavelet(args.wavelet)
    return list(range(args.length_start, args.length_stop + 1, args.length_step))


def run_device(command: list[str], timeout: float) -> str:
    quoted = " ".join(shlex.quote(value) for value in command)
    shell = f"source {shlex.quote(str(SET_ENV))} && exec {quoted}"
    completed = subprocess.run(
        ["bash", "-lc", shell],
        cwd=PROJECT_ROOT,
        env=tt_benchmark_env(),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(
            f"device command failed ({completed.returncode}): {' '.join(command)}\n{output}"
        )
    return output


def timed_samples(operation: Callable[[], Any], repeats: int, warmup_runs: int) -> list[float]:
    for _ in range(warmup_runs):
        operation()
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        operation()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def make_signal(length: int) -> np.ndarray:
    return np.linspace(-1.0, 1.0, num=length, dtype=np.float32)


def tt_command(args: argparse.Namespace, transform: str, mode: str, length: int) -> list[str]:
    step = 0.0 if length == 1 else 2.0 / (length - 1)
    return [
        str(LWT),
        *(["--inverse"] if transform == "ilwt" else []),
        "--benchmark",
        "--repeats",
        str(args.tt_repeats),
        "--warmup-runs",
        str(args.tt_warmup_runs),
        "--boundary-mode",
        mode,
        "--length",
        str(length),
        "--signal-start",
        "-1.0",
        "--signal-step",
        f"{step:.17g}",
        args.wavelet,
    ]


def parse_tt_output(output: str, transform: str) -> tuple[float, str, int]:
    medians = {
        match.group("transform"): float(match.group("latency"))
        for match in MEDIAN_PATTERN.finditer(output)
    }
    if transform not in medians:
        raise RuntimeError(f"runtime did not report {transform}_median_time_ms")
    architecture_match = ARCH_PATTERN.search(output)
    active_cores_match = ACTIVE_CORES_PATTERN.search(output)
    if architecture_match is None or active_cores_match is None:
        raise RuntimeError("runtime did not report architecture and active core count")
    return medians[transform], architecture_match.group(1), int(active_cores_match.group(1))


def result_row(
    args: argparse.Namespace,
    transform: str,
    mode: str,
    length: int,
    backend: str,
    median_ms: float,
    sample_count: int,
    architecture: str,
    active_cores: int,
) -> dict[str, Any]:
    return {
        "transform": transform,
        "boundary_mode": mode,
        "wavelet": args.wavelet,
        "signal_length": length,
        "backend": backend,
        "architecture": architecture,
        "active_core_count": active_cores,
        "sample_count": sample_count,
        "median_latency_ms": f"{median_ms:.9f}",
        "timing_scope": args.timing_scope,
        "status": "pass",
        "error": "",
    }


def failure_row(
    args: argparse.Namespace,
    transform: str,
    mode: str,
    length: int,
    backend: str,
    error: Exception,
) -> dict[str, Any]:
    return {
        "transform": transform,
        "boundary_mode": mode,
        "wavelet": args.wavelet,
        "signal_length": length,
        "backend": backend,
        "architecture": "",
        "active_core_count": 0,
        "sample_count": 0,
        "median_latency_ms": "",
        "timing_scope": args.timing_scope,
        "status": "error",
        "error": str(error),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_plots(
    output_dir: Path,
    rows: list[dict[str, Any]],
    architecture: str,
    wavelet: str,
) -> None:
    import pandas as pd
    df = pd.DataFrame(rows)
    if df.empty or "status" not in df.columns:
        return
    df = df[df["status"] == "pass"]

    for transform in ("lwt", "ilwt"):
        for mode in MODES:
            sub_df = df[(df["transform"] == transform) & (df["boundary_mode"] == mode)]
            if sub_df.empty:
                continue

            plt.figure(figsize=(7, 4.5))
            for backend, label in (("pywavelets", "PyWavelets"), ("tt-wavelet", "tt-wavelet"), ("ttnn", "ttnn-wavelet")):
                b_df = sub_df[sub_df["backend"] == backend]
                if not b_df.empty:
                    b_df = b_df.sort_values("signal_length")
                    plt.plot(
                        b_df["signal_length"].astype(int),
                        b_df["median_latency_ms"].astype(float),
                        label=label,
                    )

            plt.yscale("log")
            plt.xlabel("Signal length")
            plt.ylabel("Runtime (ms, log scale)")
            plt.title(f"1D {wavelet} {transform.upper()} runtime vs signal length ({mode})")
            plt.grid(True, which="both", linestyle=":")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / f"{transform}_{mode}.png", dpi=200)
            plt.close()


def main() -> int:
    args = parse_args()
    lengths = validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    architecture = ""
    total = len(lengths) * len(MODES)
    completed = 0

    for mode in MODES:
        for length in lengths:
            signal = make_signal(length)
            try:
                coefficients = pywt.dwt(signal, args.wavelet, mode=mode)
                pywt_samples = {
                    "lwt": timed_samples(
                        lambda: pywt.dwt(signal, args.wavelet, mode=mode),
                        args.pywt_repeats,
                        args.pywt_warmup_runs,
                    ),
                    "ilwt": timed_samples(
                        lambda: pywt.idwt(
                            coefficients[0], coefficients[1], args.wavelet, mode=mode
                        ),
                        args.pywt_repeats,
                        args.pywt_warmup_runs,
                    ),
                }
                for transform, samples in pywt_samples.items():
                    rows.append(
                        result_row(
                            args,
                            transform,
                            mode,
                            length,
                            "pywavelets",
                            statistics.median(samples),
                            len(samples),
                            "host-cpu",
                            0,
                        )
                    )
            except Exception as error:  # noqa: BLE001
                rows.append(failure_row(args, "lwt", mode, length, "pywavelets", error))
                rows.append(failure_row(args, "ilwt", mode, length, "pywavelets", error))

            for transform in ("lwt", "ilwt"):
                try:
                    output = run_device(
                        tt_command(args, transform, mode, length), args.timeout_seconds
                    )
                    median_ms, case_architecture, active_cores = parse_tt_output(
                        output, transform
                    )
                    if architecture and architecture != case_architecture:
                        raise RuntimeError(
                            f"architecture changed from {architecture} to {case_architecture}"
                        )
                    architecture = case_architecture
                    rows.append(
                        result_row(
                            args,
                            transform,
                            mode,
                            length,
                            "tt-wavelet",
                            median_ms,
                            args.tt_repeats,
                            case_architecture,
                            active_cores,
                        )
                    )
                except Exception as error:  # noqa: BLE001
                    rows.append(failure_row(args, transform, mode, length, "tt-wavelet", error))

            completed += 1
            print(f"[{completed}/{total}] {mode} N={length}", flush=True)
            write_csv(args.output_dir / "timings.csv", rows)

    write_plots(args.output_dir, rows, architecture, args.wavelet)
    failures = sum(row["status"] != "pass" for row in rows)
    print(
        f"wrote {args.output_dir / 'timings.csv'} and 16 plots; "
        f"architecture={architecture or 'unknown'} failures={failures}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
