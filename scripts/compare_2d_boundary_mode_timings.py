#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Compare db7 2D LWT/ILWT latency with PyWavelets for all boundary modes."""

from __future__ import annotations

import argparse
import csv
import math
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pywt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from compare_timings import tt_benchmark_env  # noqa: E402

LWT = PROJECT_ROOT / "build" / "lwt_2d"
ILWT = PROJECT_ROOT / "build" / "ilwt_2d"
SET_ENV = PROJECT_ROOT / "scripts" / "set_env.sh"
BANDS = ("LL", "LH", "HL", "HH")
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
    "height",
    "width",
    "backend",
    "architecture",
    "core_count",
    "repeat",
    "latency_ms",
    "timing_scope",
    "status",
    "error",
)
REPEAT_PATTERN = re.compile(
    r"(?P<transform>i?lwt)_2d_repeat_time_ms\[(?P<repeat>\d+)\]:\s*(?P<latency>[0-9eE+.\-]+)"
)
ARCH_PATTERN = re.compile(r"(?:i?lwt)_2d_architecture:\s*(\S+)")
AVAILABLE_CORES_PATTERN = re.compile(r"(?:i?lwt)_2d_available_worker_core_count:\s*(\d+)")
ACTIVE_CORES_PATTERN = re.compile(r"(?:i?lwt)_2d_active_core_count:\s*(\d+)")
CHUNK_COUNT_PATTERN = re.compile(r"(?:i?lwt)_2d_chunk_count:\s*(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wavelet", default="db7")
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--width-start", type=int, default=100)
    parser.add_argument("--width-stop", type=int, default=1000)
    parser.add_argument("--width-step", type=int, default=10)
    parser.add_argument("--tt-repeats", type=int, default=3)
    parser.add_argument("--tt-warmup-runs", type=int, default=1)
    parser.add_argument("--pywt-repeats", type=int, default=3)
    parser.add_argument("--pywt-warmup-runs", type=int, default=1)
    parser.add_argument(
        "--cores",
        type=int,
        default=64,
        help="TT core limit for the boundary-mode comparison (default: %(default)s).",
    )
    parser.add_argument(
        "--timing-scope",
        choices=("device-only",),
        default="device-only",
        help="TT scope is prepared-workload execution; PyWavelets measures only dwt2/idwt2.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "docs" / "db7_2d_boundary_timings",
    )
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--core-sweep",
        action="store_true",
        help="Measure both transforms at 1000x100, 1000x500, and 1000x1000.",
    )
    parser.add_argument(
        "--core-sweep-only",
        action="store_true",
        help="Skip the 16-plot boundary sweep and write only core_scaling.csv.",
    )
    parser.add_argument(
        "--core-sweep-repeats",
        type=int,
        default=3,
        help="Prepared TT repeats per core-sweep point (default: %(default)s).",
    )
    parser.add_argument(
        "--backend",
        choices=("all", "tt-wavelet", "pywavelets", "ttnn"),
        default="all",
        help="Select backends to include: all, tt-wavelet, pywavelets, ttnn (default: %(default)s).",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> list[int]:
    positive = {
        "--height": args.height,
        "--width-start": args.width_start,
        "--width-stop": args.width_stop,
        "--width-step": args.width_step,
        "--tt-repeats": args.tt_repeats,
        "--pywt-repeats": args.pywt_repeats,
        "--cores": args.cores,
        "--core-sweep-repeats": args.core_sweep_repeats,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.tt_warmup_runs < 0 or args.pywt_warmup_runs < 0:
        raise ValueError("warmup counts must be non-negative")
    if args.width_start > args.width_stop:
        raise ValueError("--width-start cannot exceed --width-stop")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")
    for binary in (LWT, ILWT):
        if not binary.is_file():
            raise FileNotFoundError(f"missing production binary: {binary}")
    return list(range(args.width_start, args.width_stop + 1, args.width_step))


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


def make_signal(height: int, width: int) -> np.ndarray:
    values = np.arange(height * width, dtype=np.float32)
    values = ((values % 4096.0) - 2048.0) / 2048.0
    return values.reshape(height, width)


def prepare_tt_bands(
    args: argparse.Namespace,
    mode: str,
    height: int,
    width: int,
    signal_path: Path,
    prefix: Path,
) -> list[Path]:
    run_device(
        [
            str(LWT),
            "--binary-input",
            "--quiet",
            "--boundary-mode",
            mode,
            "--cores",
            str(args.cores),
            "--output-prefix",
            str(prefix),
            args.wavelet,
            str(height),
            str(width),
            str(signal_path),
        ],
        args.timeout_seconds,
    )
    return [Path(f"{prefix}_{band}.f32") for band in BANDS]


def tt_command(
    args: argparse.Namespace,
    transform: str,
    mode: str,
    height: int,
    width: int,
    cores: int,
    repeats: int,
    signal_path: Path,
    band_paths: list[Path],
    output_path: Path,
    output_prefix: Path | None = None,
) -> list[str]:
    common = [
        "--boundary-mode",
        mode,
        "--cores",
        str(cores),
        "--repeats",
        str(repeats),
        "--warmup-runs",
        str(args.tt_warmup_runs),
    ]
    if transform == "lwt":
        return [
            str(LWT),
            "--binary-input",
            "--benchmark",
            *(["--output-prefix", str(output_prefix)] if output_prefix is not None else []),
            *common,
            args.wavelet,
            str(height),
            str(width),
            str(signal_path),
        ]
    return [
        str(ILWT),
        *common,
        "--output",
        str(output_path),
        args.wavelet,
        str(height),
        str(width),
        *(str(path) for path in band_paths),
    ]


def parse_tt_output(output: str, expected_repeats: int) -> tuple[list[float], dict[str, int | str]]:
    samples = [float(match.group("latency")) for match in REPEAT_PATTERN.finditer(output)]
    if len(samples) != expected_repeats:
        raise RuntimeError(
            f"TT output contained {len(samples)} repeat timings, expected {expected_repeats}"
        )
    architecture = ARCH_PATTERN.search(output)
    available = AVAILABLE_CORES_PATTERN.search(output)
    active = ACTIVE_CORES_PATTERN.search(output)
    chunks = CHUNK_COUNT_PATTERN.search(output)
    if architecture is None or active is None or chunks is None:
        raise RuntimeError("TT output omitted architecture or scheduler topology")
    return samples, {
        "architecture": architecture.group(1),
        "available_cores": int(available.group(1)) if available else 0,
        "active_cores": int(active.group(1)),
        "chunk_count": int(chunks.group(1)),
    }


def append_rows(
    rows: list[dict[str, Any]],
    *,
    transform: str,
    mode: str,
    wavelet: str,
    height: int,
    width: int,
    backend: str,
    architecture: str,
    core_count: int,
    samples: list[float],
    timing_scope: str,
) -> None:
    for repeat, latency in enumerate(samples):
        rows.append(
            {
                "transform": transform,
                "boundary_mode": mode,
                "wavelet": wavelet,
                "height": height,
                "width": width,
                "backend": backend,
                "architecture": architecture,
                "core_count": core_count,
                "repeat": repeat,
                "latency_ms": f"{latency:.9f}",
                "timing_scope": timing_scope,
                "status": "pass",
                "error": "",
            }
        )


def append_failure(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    transform: str,
    mode: str,
    width: int,
    backend: str,
    error: Exception,
) -> None:
    rows.append(
        {
            "transform": transform,
            "boundary_mode": mode,
            "wavelet": args.wavelet,
            "height": args.height,
            "width": width,
            "backend": backend,
            "architecture": "",
            "core_count": args.cores if backend == "tt-wavelet" else 0,
            "repeat": -1,
            "latency_ms": "",
            "timing_scope": args.timing_scope,
            "status": "error",
            "error": str(error),
        }
    )


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
                    b_df = b_df.sort_values("width")
                    plt.plot(
                        b_df["width"].astype(int),
                        b_df["latency_ms"].astype(float),
                        label=label,
                    )

            plt.yscale("log")
            plt.xlabel("Signal width")
            plt.ylabel("Runtime (ms, log scale)")
            plt.title(f"2D {wavelet} {transform.upper()} runtime vs signal width ({mode})")
            plt.grid(True, which="both", linestyle=":")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / f"{transform}_{mode}.png", dpi=200)
            plt.close()


def run_core_sweep(
    args: argparse.Namespace,
    root: Path,
    available_cores: int,
) -> list[dict[str, Any]]:
    requested = [1, 2, 4, 8, 16, 32, 64, available_cores]
    core_counts = sorted({core for core in requested if 0 < core <= available_cores})
    rows: list[dict[str, Any]] = []
    mode = "symmetric"
    wavelet = pywt.Wavelet(args.wavelet)
    for width in (100, 500, 1000):
        signal = make_signal(args.height, width)
        signal_path = root / f"core_{args.height}x{width}.f32"
        signal.tofile(signal_path)
        prefix = root / f"core_{args.height}x{width}"
        band_paths = prepare_tt_bands(args, mode, args.height, width, signal_path, prefix)
        output_path = root / f"core_{args.height}x{width}_ilwt.f32"
        band_height = pywt.dwt_coeff_len(args.height, wavelet.dec_len, mode)
        band_width = pywt.dwt_coeff_len(width, wavelet.dec_len, mode)
        band_tiles = math.ceil(band_height / 32) * math.ceil(band_width / 32)
        for transform in ("lwt", "ilwt"):
            baseline: float | None = None
            for cores in core_counts:
                output = run_device(
                    tt_command(
                        args,
                        transform,
                        mode,
                        args.height,
                        width,
                        cores,
                        args.core_sweep_repeats,
                        signal_path,
                        band_paths,
                        output_path,
                        None,
                    ),
                    args.timeout_seconds,
                )
                samples, telemetry = parse_tt_output(output, args.core_sweep_repeats)
                latency = statistics.median(samples)
                if baseline is None:
                    baseline = latency
                speedup = baseline / latency
                active = int(telemetry["active_cores"])
                rows.append(
                    {
                        "transform": transform,
                        "height": args.height,
                        "width": width,
                        "requested_core_count": cores,
                        "active_core_count": active,
                        "available_worker_core_count": available_cores,
                        "chunk_count": telemetry["chunk_count"],
                        "average_chunk_band_tiles": band_tiles / int(telemetry["chunk_count"]),
                        "median_latency_ms": latency,
                        "speedup_vs_one_core": speedup,
                        "parallel_efficiency": speedup / active,
                        "architecture": telemetry["architecture"],
                    }
                )
    return rows


def write_core_sweep(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "transform",
        "height",
        "width",
        "requested_core_count",
        "active_core_count",
        "available_worker_core_count",
        "chunk_count",
        "average_chunk_band_tiles",
        "median_latency_ms",
        "speedup_vs_one_core",
        "parallel_efficiency",
        "architecture",
    )
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def discover_available_cores(args: argparse.Namespace, root: Path) -> tuple[int, str]:
    signal = make_signal(args.height, 100)
    signal_path = root / "core_discovery.f32"
    signal.tofile(signal_path)
    prefix = root / "core_discovery"
    output = run_device(
        tt_command(
            args,
            "lwt",
            "symmetric",
            args.height,
            100,
            args.cores,
            1,
            signal_path,
            [Path(f"{prefix}_{band}.f32") for band in BANDS],
            root / "core_discovery_ilwt.f32",
            prefix,
        ),
        args.timeout_seconds,
    )
    _, telemetry = parse_tt_output(output, 1)
    available = int(telemetry["available_cores"])
    if available <= 0:
        raise RuntimeError("runtime did not report a positive worker-core count")
    return available, str(telemetry["architecture"])


def main() -> int:
    args = parse_args()
    if args.core_sweep_only:
        args.core_sweep = True
    widths = validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    architecture = ""
    available_cores = 0

    with tempfile.TemporaryDirectory(prefix="ttwv-2d-timings-") as temporary:
        root = Path(temporary)
        if not args.core_sweep_only:
            total = len(widths) * len(MODES)
            completed = 0
            for mode in MODES:
                for width in widths:
                    signal = make_signal(args.height, width)
                    signal_path = root / f"{mode}_{args.height}x{width}.f32"
                    signal.tofile(signal_path)
                    prefix = root / f"{mode}_{args.height}x{width}"
                    output_path = root / f"{mode}_{args.height}x{width}_ilwt.f32"
                    if args.backend in ("all", "pywavelets"):
                        try:
                            coefficients = pywt.dwt2(signal, args.wavelet, mode=mode)
                            pywt_lwt = timed_samples(
                                lambda: pywt.dwt2(signal, args.wavelet, mode=mode),
                                args.pywt_repeats,
                                args.pywt_warmup_runs,
                            )
                            pywt_ilwt = timed_samples(
                                lambda: pywt.idwt2(coefficients, args.wavelet, mode=mode),
                                args.pywt_repeats,
                                args.pywt_warmup_runs,
                            )
                            append_rows(
                                rows,
                                transform="lwt",
                                mode=mode,
                                wavelet=args.wavelet,
                                height=args.height,
                                width=width,
                                backend="pywavelets",
                                architecture="host-cpu",
                                core_count=0,
                                samples=pywt_lwt,
                                timing_scope=args.timing_scope,
                            )
                            append_rows(
                                rows,
                                transform="ilwt",
                                mode=mode,
                                wavelet=args.wavelet,
                                height=args.height,
                                width=width,
                                backend="pywavelets",
                                architecture="host-cpu",
                                core_count=0,
                                samples=pywt_ilwt,
                                timing_scope=args.timing_scope,
                            )
                        except Exception as error:  # noqa: BLE001
                            append_failure(rows, args, "lwt", mode, width, "pywavelets", error)
                            append_failure(rows, args, "ilwt", mode, width, "pywavelets", error)

                    if args.backend in ("all", "tt-wavelet"):
                        try:
                            for transform in ("lwt", "ilwt"):
                                band_paths = [Path(f"{prefix}_{band}.f32") for band in BANDS]
                                output = run_device(
                                    tt_command(
                                        args,
                                        transform,
                                        mode,
                                        args.height,
                                        width,
                                        args.cores,
                                        args.tt_repeats,
                                        signal_path,
                                        band_paths,
                                        output_path,
                                        prefix if transform == "lwt" else None,
                                    ),
                                    args.timeout_seconds,
                                )
                                samples, telemetry = parse_tt_output(output, args.tt_repeats)
                                architecture = str(telemetry["architecture"])
                                available_cores = max(
                                    available_cores, int(telemetry["available_cores"])
                                )
                                append_rows(
                                    rows,
                                    transform=transform,
                                    mode=mode,
                                    wavelet=args.wavelet,
                                    height=args.height,
                                    width=width,
                                    backend="tt-wavelet",
                                    architecture=architecture,
                                    core_count=int(telemetry["active_cores"]),
                                    samples=samples,
                                    timing_scope=args.timing_scope,
                                )
                        except Exception as error:  # noqa: BLE001
                            append_failure(rows, args, "lwt", mode, width, "tt-wavelet", error)
                            append_failure(rows, args, "ilwt", mode, width, "tt-wavelet", error)

                    if args.backend in ("all", "ttnn"):
                        try:
                            import torch
                            import ttnn
                            dev = ttnn.open_device(device_id=0)
                            try:
                                sig_2d = torch.from_numpy(signal)
                                inp_2d = ttnn.from_torch(sig_2d, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
                                
                                # Warmup JIT
                                b_t = ttnn.dwt_2d(inp_2d, args.wavelet, boundary_mode=mode)
                                r_t = ttnn.idwt_2d(*b_t, args.wavelet, [args.height, width], boundary_mode=mode)
                                ttnn.synchronize_device(dev)

                                # 2D DWT Trace Capture for pure device hardware execution time
                                trace_lwt = ttnn.begin_trace_capture(dev)
                                b_t = ttnn.dwt_2d(inp_2d, args.wavelet, boundary_mode=mode)
                                ttnn.end_trace_capture(dev, trace_lwt)
                                ttnn.synchronize_device(dev)

                                ttnn_lwt_samples = []
                                for _ in range(args.tt_repeats):
                                    t0 = time.perf_counter()
                                    ttnn.execute_trace(dev, trace_lwt)
                                    ttnn.synchronize_device(dev)
                                    ttnn_lwt_samples.append((time.perf_counter() - t0) * 1000.0)
                                ttnn.release_trace(dev, trace_lwt)

                                # 2D ILWT Trace Capture
                                trace_ilwt = ttnn.begin_trace_capture(dev)
                                r_t = ttnn.idwt_2d(*b_t, args.wavelet, [args.height, width], boundary_mode=mode)
                                ttnn.end_trace_capture(dev, trace_ilwt)
                                ttnn.synchronize_device(dev)

                                ttnn_ilwt_samples = []
                                for _ in range(args.tt_repeats):
                                    t0 = time.perf_counter()
                                    ttnn.execute_trace(dev, trace_ilwt)
                                    ttnn.synchronize_device(dev)
                                    ttnn_ilwt_samples.append((time.perf_counter() - t0) * 1000.0)
                                ttnn.release_trace(dev, trace_ilwt)
                            finally:
                                ttnn.close_device(dev)

                            append_rows(
                                rows,
                                transform="lwt",
                                mode=mode,
                                wavelet=args.wavelet,
                                height=args.height,
                                width=width,
                                backend="ttnn",
                                architecture=architecture or "blackhole",
                                core_count=64,
                                samples=ttnn_lwt_samples,
                                timing_scope=args.timing_scope,
                            )
                            append_rows(
                                rows,
                                transform="ilwt",
                                mode=mode,
                                wavelet=args.wavelet,
                                height=args.height,
                                width=width,
                                backend="ttnn",
                                architecture=architecture or "blackhole",
                                core_count=64,
                                samples=ttnn_ilwt_samples,
                                timing_scope=args.timing_scope,
                            )
                        except Exception as error:  # noqa: BLE001
                            append_failure(rows, args, "lwt", mode, width, "ttnn", error)
                            append_failure(rows, args, "ilwt", mode, width, "ttnn", error)

                    completed += 1
                    print(f"[{completed}/{total}] {mode} {args.height}x{width}", flush=True)
                    write_csv(args.output_dir / "timings.csv", rows)

            write_plots(args.output_dir, rows, architecture, args.wavelet)

        if args.core_sweep:
            if available_cores <= 0:
                available_cores, architecture = discover_available_cores(args, root)
            sweep_rows = run_core_sweep(args, root, available_cores)
            write_core_sweep(args.output_dir / "core_scaling.csv", sweep_rows)

    failures = sum(row["status"] != "pass" for row in rows)
    if args.core_sweep_only:
        print(
            f"wrote {args.output_dir / 'core_scaling.csv'}; "
            f"architecture={architecture} failures={failures}"
        )
    else:
        print(
            f"wrote {args.output_dir / 'timings.csv'} and 16 plots; "
            f"architecture={architecture or 'unknown'} failures={failures}"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
