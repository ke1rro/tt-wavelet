#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Benchmark scalar and fused-tiled 2D split stages with device telemetry."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVICE_BINARY = PROJECT_ROOT / "build" / "lwt_2d"
SET_ENV = PROJECT_ROOT / "scripts" / "set_env.sh"
REQUIRED_SHAPES = (
    (64, 64),
    (128, 128),
    (256, 256),
    (512, 512),
    (1000, 100),
    (1000, 200),
    (1024, 1024),
)
METRIC_PATTERN = re.compile(
    r"^(lwt_2d_[a-z0-9_]+):\s+([-+0-9.eE]+)\s*$", re.MULTILINE
)


def parse_shapes(value: str) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    for item in value.split(","):
        try:
            height_text, width_text = item.lower().split("x", 1)
            height, width = int(height_text), int(width_text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid shape {item!r}; expected HEIGHTxWIDTH"
            ) from exc
        if height <= 0 or width <= 0:
            raise argparse.ArgumentTypeError("shape dimensions must be positive")
        shapes.append((height, width))
    return shapes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wavelet", default="db7")
    parser.add_argument(
        "--implementations",
        choices=("scalar", "tiled", "both"),
        default="both",
    )
    parser.add_argument(
        "--shapes",
        type=parse_shapes,
        default=list(REQUIRED_SHAPES),
        help="Comma-separated HEIGHTxWIDTH list (default: required corpus).",
    )
    parser.add_argument("--cores", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("lwt_2d_split_timings.csv"),
    )
    parser.add_argument("--result-json", type=Path)
    return parser.parse_args()


def run_case(
    args: argparse.Namespace,
    implementation: str,
    height: int,
    width: int,
    input_path: Path,
) -> dict[str, int | float | str]:
    command = [
        str(DEVICE_BINARY),
        "--binary-input",
        "--benchmark",
        "--quiet",
        "--split-metrics",
        "--split-implementation",
        implementation,
        "--cores",
        str(args.cores),
        "--repeats",
        str(args.repeats),
        "--warmup-runs",
        str(args.warmup_runs),
        args.wavelet,
        str(height),
        str(width),
        str(input_path),
    ]
    shell_command = (
        f"source {shlex.quote(str(SET_ENV))} >/dev/null && "
        + " ".join(shlex.quote(item) for item in command)
    )
    completed = subprocess.run(
        ["bash", "-lc", shell_command],
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
        timeout=args.timeout_seconds,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"device command failed ({completed.returncode})\n"
            f"{completed.stdout}{completed.stderr}"
        )
    metrics = {
        name: float(value)
        for name, value in METRIC_PATTERN.findall(
            completed.stdout + "\n" + completed.stderr
        )
    }
    required = (
        "lwt_2d_execution_time_ms",
        "lwt_2d_min_execution_time_ms",
        "lwt_2d_split_max_core_cycles",
        "lwt_2d_split_time_ms_at_1ghz",
        "lwt_2d_split_input_elements_per_second_at_1ghz",
        "lwt_2d_split_raw_input_bytes",
        "lwt_2d_split_local_output_bytes",
        "lwt_2d_split_noc_read_calls",
        "lwt_2d_split_noc_read_barriers",
        "lwt_2d_split_interior_macro_tiles",
        "lwt_2d_split_boundary_macro_tiles",
        "lwt_2d_active_core_count",
        "lwt_2d_split_max_macro_tiles_per_core",
    )
    missing = [name for name in required if name not in metrics]
    if missing:
        raise RuntimeError(f"missing telemetry: {', '.join(missing)}")
    return {
        "wavelet": args.wavelet,
        "height": height,
        "width": width,
        "implementation": implementation,
        "active_cores": int(metrics["lwt_2d_active_core_count"]),
        "split_latency_ms": metrics["lwt_2d_split_time_ms_at_1ghz"],
        "split_max_core_cycles": int(
            metrics["lwt_2d_split_max_core_cycles"]
        ),
        "input_elements_per_second": metrics[
            "lwt_2d_split_input_elements_per_second_at_1ghz"
        ],
        "raw_input_bytes": int(metrics["lwt_2d_split_raw_input_bytes"]),
        "local_output_bytes": int(
            metrics["lwt_2d_split_local_output_bytes"]
        ),
        "noc_read_calls": int(metrics["lwt_2d_split_noc_read_calls"]),
        "noc_read_barriers": int(
            metrics["lwt_2d_split_noc_read_barriers"]
        ),
        "interior_macro_tiles": int(
            metrics["lwt_2d_split_interior_macro_tiles"]
        ),
        "boundary_macro_tiles": int(
            metrics["lwt_2d_split_boundary_macro_tiles"]
        ),
        "max_macro_tiles_per_core": int(
            metrics["lwt_2d_split_max_macro_tiles_per_core"]
        ),
        "complete_lwt_ms": metrics["lwt_2d_execution_time_ms"],
        "complete_lwt_min_ms": metrics["lwt_2d_min_execution_time_ms"],
    }


def print_results(results: list[dict[str, int | float | str]]) -> None:
    by_shape: dict[tuple[int, int], dict[str, dict[str, int | float | str]]] = {}
    for result in results:
        key = (int(result["height"]), int(result["width"]))
        by_shape.setdefault(key, {})[str(result["implementation"])] = result
    print(
        "shape       implementation split_ms  split_speedup "
        "NoC_calls barriers full_db7_ms"
    )
    for height, width in sorted(by_shape):
        scalar = by_shape[(height, width)].get("scalar")
        for implementation in ("scalar", "tiled"):
            result = by_shape[(height, width)].get(implementation)
            if result is None:
                continue
            speedup = (
                float(scalar["split_latency_ms"])
                / float(result["split_latency_ms"])
                if scalar is not None
                else 1.0
            )
            print(
                f"{height}x{width:<6} {implementation:<14} "
                f"{float(result['split_latency_ms']):8.4f} "
                f"{speedup:13.3f} "
                f"{int(result['noc_read_calls']):9d} "
                f"{int(result['noc_read_barriers']):8d} "
                f"{float(result['complete_lwt_ms']):11.4f}"
            )


def main() -> int:
    args = parse_args()
    if args.cores <= 0 or args.repeats <= 0 or args.warmup_runs < 0:
        raise ValueError("cores/repeats must be positive and warmups non-negative")
    implementations = (
        ("scalar", "tiled")
        if args.implementations == "both"
        else (args.implementations,)
    )
    max_elements = max(height * width for height, width in args.shapes)
    with tempfile.TemporaryDirectory(prefix="ttwv-split-benchmark-") as directory:
        input_path = Path(directory) / "input.f32"
        np.linspace(
            -1.0, 1.0, num=max_elements, dtype=np.float32
        ).tofile(input_path)
        results = [
            run_case(args, implementation, height, width, input_path)
            for height, width in args.shapes
            for implementation in implementations
        ]

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    if args.result_json:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8"
        )
    print_results(results)
    print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
