#!/usr/bin/env python3
import argparse
import csv
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parent
TT_WAVELET_BINARY = PROJECT_ROOT / "build" / "lwt"
TT_WAVELET_2D_BINARY = PROJECT_ROOT / "build" / "lwt_2d"
TT_WAVELET_2D_REFERENCE_BINARY = PROJECT_ROOT / "build" / "lwt_2d_reference"
TT_WAVELET_ENV = PROJECT_ROOT / "scripts" / "set_env.sh"
DEFAULT_SCHEMES_DIR = PROJECT_ROOT / "wavelets"
TT_PREFIX = r"(?:lwt|ilwt)"
TT_TIME_PATTERN = re.compile(rf"{TT_PREFIX}_execution_time_ms:\s*([0-9eE+.\-]+)")
TT_MIN_TIME_PATTERN = re.compile(rf"{TT_PREFIX}_min_time_ms:\s*([0-9eE+.\-]+)")
TT_MEDIAN_TIME_PATTERN = re.compile(rf"{TT_PREFIX}_median_time_ms:\s*([0-9eE+.\-]+)")
TT_P10_TIME_PATTERN = re.compile(rf"{TT_PREFIX}_p10_time_ms:\s*([0-9eE+.\-]+)")
TT_P90_TIME_PATTERN = re.compile(rf"{TT_PREFIX}_p90_time_ms:\s*([0-9eE+.\-]+)")
TT_STDDEV_TIME_PATTERN = re.compile(rf"{TT_PREFIX}_stddev_time_ms:\s*([0-9eE+.\-]+)")
TT_ARCHITECTURE_PATTERN = re.compile(rf"{TT_PREFIX}_architecture:\s*(\S+)")
TT_LAYOUT_PATTERN = re.compile(rf"{TT_PREFIX}_layout:\s*(\S+)")
TT_MAX_GROUP_COUNT_PATTERN = re.compile(rf"{TT_PREFIX}_max_group_count:\s*(\d+)")
TT_ACTIVE_CORE_COUNT_PATTERN = re.compile(rf"{TT_PREFIX}_active_core_count:\s*(\d+)")
TT_CHUNK_COUNT_PATTERN = re.compile(rf"{TT_PREFIX}_chunk_count:\s*(\d+)")
TT_GROUPS_PER_CHUNK_PATTERN = re.compile(rf"{TT_PREFIX}_groups_per_chunk:\s*(\d+)")
TT_WORKSPACE_ELEMENTS_PATTERN = re.compile(rf"{TT_PREFIX}_workspace_elements:\s*(\d+)")
TT_MAX_WORKSPACE_ELEMENTS_PATTERN = re.compile(
    rf"{TT_PREFIX}_max_workspace_elements:\s*(\d+)"
)
TT_MAX_DEPENDENCY_OVERHEAD_PATTERN = re.compile(
    rf"{TT_PREFIX}_max_dependency_overhead:\s*([0-9eE+.\-]+)"
)
TT_TERMINAL_SCALE_INLINE_PATTERN = re.compile(
    rf"{TT_PREFIX}_terminal_scale_inline:\s*(\d+)"
)
TT_INVERSE_SCALE_INLINE_PATTERN = re.compile(
    rf"{TT_PREFIX}_inverse_scale_inline:\s*(\d+)"
)
TT_INVERSE_FINAL_INTERLEAVE_DIRECT_PATTERN = re.compile(
    rf"{TT_PREFIX}_inverse_final_interleave_direct:\s*(\d+)"
)
TT_L1_TOTAL_BYTES_PATTERN = re.compile(rf"{TT_PREFIX}_l1_total_bytes:\s*(\d+)")
TT_L1_CAPACITY_BYTES_PATTERN = re.compile(rf"{TT_PREFIX}_l1_capacity_bytes:\s*(\d+)")
TT_L1_HEADROOM_BYTES_PATTERN = re.compile(rf"{TT_PREFIX}_l1_headroom_bytes:\s*(\d+)")
REFERENCE_2D_MEAN_TIME_PATTERN = re.compile(
    r"lwt_2d_reference_mean_time_ms:\s*([0-9eE+.\-]+)"
)
REFERENCE_2D_MIN_TIME_PATTERN = re.compile(
    r"lwt_2d_reference_min_time_ms:\s*([0-9eE+.\-]+)"
)
TT_2D_MEAN_TIME_PATTERN = re.compile(
    r"lwt_2d_execution_time_ms:\s*([0-9eE+.\-]+)"
)
TT_2D_MIN_TIME_PATTERN = re.compile(
    r"lwt_2d_min_execution_time_ms:\s*([0-9eE+.\-]+)"
)
TT_2D_MEDIAN_TIME_PATTERN = re.compile(
    r"lwt_2d_median_time_ms:\s*([0-9eE+.\-]+)"
)
TT_2D_P10_TIME_PATTERN = re.compile(
    r"lwt_2d_p10_time_ms:\s*([0-9eE+.\-]+)"
)
TT_2D_P90_TIME_PATTERN = re.compile(
    r"lwt_2d_p90_time_ms:\s*([0-9eE+.\-]+)"
)
TT_2D_STDDEV_TIME_PATTERN = re.compile(
    r"lwt_2d_stddev_time_ms:\s*([0-9eE+.\-]+)"
)
TT_2D_ACTIVE_CORE_PATTERN = re.compile(
    r"lwt_2d_active_core_count:\s*(\d+)"
)
TT_2D_CHUNK_COUNT_PATTERN = re.compile(r"lwt_2d_chunk_count:\s*(\d+)")
TT_2D_CHUNK_TILES_PATTERN = re.compile(r"lwt_2d_chunk_tiles:\s*(\d+x\d+)")
TT_2D_ROUTE_COUNT_PATTERN = re.compile(r"lwt_2d_route_count:\s*(\d+)")
TT_2D_EXECUTABLE_ROUTE_COUNT_PATTERN = re.compile(
    r"lwt_2d_executable_route_count:\s*(\d+)"
)
TT_2D_SCALE_ROUTES_REMOVED_PATTERN = re.compile(
    r"lwt_2d_scale_routes_removed:\s*(\d+)"
)
DEFAULT_LOG_CANDIDATES = [
    PROJECT_ROOT / "wavelets.log",
    PROJECT_ROOT / "wavelets (1).log",
]
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python3"
TimingKey = tuple[str, str, int, float, float, str, str]


@dataclass(frozen=True)
class TTTimingResult:
    mean_s: float
    min_s: float
    architecture: str = ""
    layout: str = ""
    median_s: float | None = None
    p10_s: float | None = None
    p90_s: float | None = None
    stddev_s: float | None = None
    max_group_count: int | None = None
    active_core_count: int | None = None
    chunk_count: int | None = None
    groups_per_chunk: int | None = None
    workspace_elements: int | None = None
    max_workspace_elements: int | None = None
    max_dependency_overhead: float | None = None
    terminal_scale_inline: int | None = None
    inverse_scale_inline: int | None = None
    inverse_final_interleave_direct: int | None = None
    l1_total_bytes: int | None = None
    l1_capacity_bytes: int | None = None
    l1_headroom_bytes: int | None = None


def ensure_runtime_packages(require_pywt: bool) -> None:
    try:
        if require_pywt:
            import pywt  # noqa: F401
        from tqdm import tqdm  # noqa: F401
    except ModuleNotFoundError as exc:
        if VENV_PYTHON.exists() and Path(sys.executable) != VENV_PYTHON:
            os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), __file__, *sys.argv[1:]])
        raise ModuleNotFoundError(
            "Missing runtime packages. Install PyWavelets and tqdm, e.g. "
            "`pip install PyWavelets tqdm`."
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark TT-wavelet 1D device timings or PyWavelets vs the "
            "vertical-first 2D FP32 scalar oracle."
        )
    )
    parser.add_argument(
        "--wavelets-log",
        type=Path,
        help="Path to wavelets.log listing runnable wavelets (default: auto-detect).",
    )
    parser.add_argument(
        "--schemes-dir",
        type=Path,
        default=DEFAULT_SCHEMES_DIR,
        help="Directory with lifting-scheme JSON files (default: %(default)s).",
    )
    parser.add_argument(
        "--wavelets",
        nargs="*",
        help="Optional list of wavelet names (override wavelets.log).",
    )
    parser.add_argument(
        "--length-start",
        type=int,
        default=100000,
        help="Signal length start (default: %(default)s).",
    )
    parser.add_argument(
        "--length-stop",
        type=int,
        default=1000000,
        help="Signal length stop, inclusive (default: %(default)s).",
    )
    parser.add_argument(
        "--length-step",
        type=int,
        default=10000,
        help="Signal length step (default: %(default)s).",
    )
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        help="Explicit signal lengths; overrides --length-start/--length-stop/--length-step.",
    )
    parser.add_argument(
        "--shapes",
        nargs="+",
        metavar="HEIGHTxWIDTH",
        help=(
            "Run 2D timing mode for explicit shapes such as 512x512 1024x768. "
            "TT-wavelet measures the fused device program; fp32-reference "
            "measures the scalar correctness oracle."
        ),
    )
    parser.add_argument(
        "--signal-start",
        type=float,
        default=1.0,
        help="Start value for generated signal ramp (default: %(default)s).",
    )
    parser.add_argument(
        "--signal-step",
        type=float,
        default=1.0,
        help="Step value for generated signal ramp (default: %(default)s).",
    )
    parser.add_argument(
        "--pywt-mode",
        default="symmetric",
        help="PyWavelets extension mode (default: %(default)s).",
    )
    parser.add_argument(
        "--backend",
        choices=["both", "tt-wavelet", "pywt", "fp32-reference"],
        default="both",
        help=(
            "Benchmark both available backends or one backend. fp32-reference "
            "is valid only with --shapes (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--transform",
        choices=["lwt", "ilwt", "both"],
        default="lwt",
        help=(
            "Benchmark forward LWT, inverse LWT, or both. ILWT coefficients are "
            "prepared outside the timed interval (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--tt-mode",
        choices=["benchmark", "legacy"],
        default="benchmark",
        help=(
            "TT-wavelet timing mode. 'benchmark' runs warmups/repeats inside one C++ "
            "process with program cache and no coefficient readback. 'legacy' launches "
            "one lwt process per repeat (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--tt-boundary-mode",
        choices=[
            "symmetric",
            "zero",
            "constant",
            "periodic",
            "antisymmetric",
            "smooth",
            "reflect",
            "antireflect",
        ],
        default="symmetric",
        help="TT-wavelet signal extension mode (default: %(default)s).",
    )
    parser.add_argument(
        "--pywt-repeats",
        type=int,
        default=1,
        help="Number of timing runs for PyWavelets (default: %(default)s).",
    )
    parser.add_argument(
        "--tt-repeats",
        type=int,
        default=1,
        help="Number of timing runs for TT-wavelet (default: %(default)s).",
    )
    parser.add_argument(
        "--tt-cores",
        type=int,
        default=64,
        help="Maximum worker cores used by the fused 2D TT backend (default: %(default)s).",
    )
    parser.add_argument(
        "--tt-split-implementation",
        choices=["scalar", "tiled"],
        default="tiled",
        help=(
            "Initial 2D input-to-polyphase split implementation used by the fused "
            "TT backend (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--tt-route-staging",
        choices=["scalar", "optimized"],
        default="optimized",
        help="2D route source/base staging implementation (default: %(default)s).",
    )
    parser.add_argument(
        "--tt-route-persistence",
        choices=["scalar", "full-tile"],
        default="full-tile",
        help="2D compute-output persistence implementation (default: %(default)s).",
    )
    parser.add_argument(
        "--tt-terminal-writes",
        choices=["fragmented", "tiled"],
        default="tiled",
        help="2D terminal-band writer implementation (default: %(default)s).",
    )
    parser.add_argument(
        "--tt-scale-policy",
        choices=["explicit", "fused"],
        default="fused",
        help="2D terminal scale policy (default: %(default)s).",
    )
    parser.add_argument(
        "--tt-planner",
        choices=["max-cores", "latency"],
        default="latency",
        help="2D chunk-planner selection policy (default: %(default)s).",
    )
    parser.add_argument(
        "--tt-route-config",
        choices=("per-route", "preloaded"),
        default="preloaded",
        help="2D route descriptor loading policy (default: %(default)s).",
    )
    parser.add_argument(
        "--tt-exact-transfer",
        choices=("local-noc", "l1-copy"),
        default="local-noc",
        help="Same-core exact-tile transfer mechanism (default: %(default)s).",
    )
    parser.add_argument(
        "--tt-transport-metrics",
        action="store_true",
        help="Enable compile-time-optional 2D route-transport telemetry.",
    )
    parser.add_argument(
        "--tt-alignment-csv-dir",
        type=Path,
        help="Write host-derived per-tile route-alignment CSVs to this directory.",
    )
    parser.add_argument(
        "--tt-warmup-runs",
        type=int,
        default=1,
        help="Warmup runs to discard before timing TT-wavelet (default: %(default)s).",
    )
    parser.add_argument(
        "--tt-warmup-scope",
        choices=["wavelet", "global", "length"],
        default="length",
        help=(
            "Legacy TT mode only: granularity at which external TT-wavelet warmup "
            "processes are issued. In benchmark mode, --tt-warmup-runs is handled "
            "inside the C++ process for every (transform, wavelet, length) tuple."
        ),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help=(
            "Output CSV path (default: tt_wavelet_timings.csv for 1D and "
            "tt_wavelet_timings_2d.csv for --shapes)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite --csv instead of merging new backend columns into existing rows.",
    )
    return parser.parse_args()


def resolve_wavelets_log(log_path: Path | None) -> Path:
    if log_path is not None:
        return log_path
    for candidate in DEFAULT_LOG_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "wavelets.log not found. Pass --wavelets-log or supply --wavelets explicitly."
    )


def load_wavelets_from_log(log_path: Path) -> list[str]:
    pattern = re.compile(r"\b([A-Za-z0-9_.-]+)\.json\b")
    wavelets: list[str] = []
    seen = set()
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if "OK" not in line:
            continue
        match = pattern.search(line)
        if match is None:
            continue
        name = match.group(1)
        if name not in seen:
            wavelets.append(name)
            seen.add(name)
    if not wavelets:
        raise ValueError(f"No wavelets found in {log_path}")
    return wavelets


def generate_signal(length: int, start: float, step: float) -> list[float]:
    if length <= 0:
        raise ValueError("Signal length must be positive.")
    return [start + i * step for i in range(length)]


def write_signal_file(path: Path, signal: list[float]) -> None:
    path.write_text(" ".join(repr(value) for value in signal), encoding="utf-8")


def sh_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def build_tt_command(
    args: argparse.Namespace,
    transform: str,
    wavelet: str,
    length: int,
    signal_file: Path,
) -> str:
    command_args = [
        str(TT_WAVELET_BINARY),
    ]
    if transform == "ilwt":
        command_args.append("--inverse")
    command_args.extend(
        [
            "--boundary-mode",
            args.tt_boundary_mode,
        ]
    )
    if args.tt_mode == "benchmark":
        command_args.extend(
            [
                "--benchmark",
                "--repeats",
                str(args.tt_repeats),
                "--warmup-runs",
                str(args.tt_warmup_runs),
                "--length",
                str(length),
                "--signal-start",
                repr(args.signal_start),
                "--signal-step",
                repr(args.signal_step),
                wavelet,
            ]
        )
    else:
        command_args.extend([wavelet, str(signal_file)])

    command = " ".join(sh_quote(arg) for arg in command_args)
    return f"source {sh_quote(str(TT_WAVELET_ENV))} && {command}"


def tt_benchmark_env() -> dict[str, str]:
    env = os.environ.copy()
    env["TT_LOGGER_LEVEL"] = "FATAL"
    env["TT_METAL_INSPECTOR_RPC"] = "0"
    for name in (
        "TT_METAL_DPRINT_CORES",
        "TT_METAL_WATCHER",
        "TT_METAL_SLOW_DISPATCH_MODE",
        "TT_METAL_DEVICE_PROFILER",
        "TT_METAL_DEVICE_PROFILER_DISPATCH",
        "TT_METAL_DISPATCH_DATA_COLLECTION",
        "TTNN_CONFIG_OVERRIDES",
    ):
        env.pop(name, None)
    return env


def optional_pattern_float(
    pattern: re.Pattern[str], text: str, scale: float = 1.0
) -> float | None:
    match = pattern.search(text)
    return float(match.group(1)) * scale if match is not None else None


def optional_pattern_int(pattern: re.Pattern[str], text: str) -> int | None:
    match = pattern.search(text)
    return int(match.group(1)) if match is not None else None


def optional_pattern_string(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text)
    return match.group(1) if match is not None else ""


def run_tt_wavelet(
    command: str, environment_overrides: dict[str, str] | None = None
) -> TTTimingResult:
    environment = tt_benchmark_env()
    if environment_overrides is not None:
        environment.update(environment_overrides)
    completed = subprocess.run(
        ["bash", "-lc", command],
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"TT-wavelet run failed with exit code {completed.returncode}.\n{completed.stderr}"
        )
    match = TT_TIME_PATTERN.search(completed.stderr)
    if match is None:
        raise RuntimeError(
            "TT-wavelet output did not include an LWT/ILWT execution time."
        )
    min_match = TT_MIN_TIME_PATTERN.search(completed.stderr)
    mean_s = float(match.group(1)) / 1000.0
    min_s = float(min_match.group(1)) / 1000.0 if min_match is not None else mean_s
    return TTTimingResult(
        mean_s=mean_s,
        min_s=min_s,
        architecture=optional_pattern_string(TT_ARCHITECTURE_PATTERN, completed.stderr),
        layout=optional_pattern_string(TT_LAYOUT_PATTERN, completed.stderr),
        median_s=optional_pattern_float(
            TT_MEDIAN_TIME_PATTERN, completed.stderr, 0.001
        ),
        p10_s=optional_pattern_float(TT_P10_TIME_PATTERN, completed.stderr, 0.001),
        p90_s=optional_pattern_float(TT_P90_TIME_PATTERN, completed.stderr, 0.001),
        stddev_s=optional_pattern_float(
            TT_STDDEV_TIME_PATTERN, completed.stderr, 0.001
        ),
        max_group_count=optional_pattern_int(
            TT_MAX_GROUP_COUNT_PATTERN, completed.stderr
        ),
        active_core_count=optional_pattern_int(
            TT_ACTIVE_CORE_COUNT_PATTERN, completed.stderr
        ),
        chunk_count=optional_pattern_int(TT_CHUNK_COUNT_PATTERN, completed.stderr),
        groups_per_chunk=optional_pattern_int(
            TT_GROUPS_PER_CHUNK_PATTERN, completed.stderr
        ),
        workspace_elements=optional_pattern_int(
            TT_WORKSPACE_ELEMENTS_PATTERN, completed.stderr
        ),
        max_workspace_elements=optional_pattern_int(
            TT_MAX_WORKSPACE_ELEMENTS_PATTERN, completed.stderr
        ),
        max_dependency_overhead=optional_pattern_float(
            TT_MAX_DEPENDENCY_OVERHEAD_PATTERN, completed.stderr
        ),
        terminal_scale_inline=optional_pattern_int(
            TT_TERMINAL_SCALE_INLINE_PATTERN, completed.stderr
        ),
        inverse_scale_inline=optional_pattern_int(
            TT_INVERSE_SCALE_INLINE_PATTERN, completed.stderr
        ),
        inverse_final_interleave_direct=optional_pattern_int(
            TT_INVERSE_FINAL_INTERLEAVE_DIRECT_PATTERN, completed.stderr
        ),
        l1_total_bytes=optional_pattern_int(
            TT_L1_TOTAL_BYTES_PATTERN, completed.stderr
        ),
        l1_capacity_bytes=optional_pattern_int(
            TT_L1_CAPACITY_BYTES_PATTERN, completed.stderr
        ),
        l1_headroom_bytes=optional_pattern_int(
            TT_L1_HEADROOM_BYTES_PATTERN, completed.stderr
        ),
    )


def time_repeats(
    run_once: Callable[[], None], repeats: int
) -> tuple[float | None, float | None]:
    if repeats <= 0:
        return None, None
    times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        run_once()
        times.append(time.perf_counter() - start)
    mean = sum(times) / len(times)
    return mean, min(times)


def value_repeats(
    run_once: Callable[[], float], repeats: int
) -> tuple[float | None, float | None]:
    if repeats <= 0:
        return None, None
    times = [run_once() for _ in range(repeats)]
    return sum(times) / len(times), min(times)


def should_warmup(
    scope: str,
    transform: str,
    wavelet: str,
    length: int,
    warmed: set[tuple[str, str]],
    warmed_global: list[bool],
    warmed_pairs: set[tuple[str, str, int]],
) -> bool:
    if scope == "global":
        if warmed_global[0]:
            return False
        warmed_global[0] = True
        return True
    if scope == "length":
        key = (transform, wavelet, length)
        if key in warmed_pairs:
            return False
        warmed_pairs.add(key)
        return True
    key = (transform, wavelet)
    if key in warmed:
        return False
    warmed.add(key)
    return True


def row_key(row: dict[str, object]) -> TimingKey:
    return (
        str(row["transform"]),
        str(row["wavelet"]),
        int(row["signal_length"]),
        float(row["signal_start"]),
        float(row["signal_step"]),
        str(row["pywt_mode"]),
        str(row["lwt_boundary_mode"]),
    )


def read_existing_rows(
    csv_path: Path, fieldnames: list[str]
) -> tuple[dict[TimingKey, dict[str, str]], list[TimingKey]]:
    rows: dict[TimingKey, dict[str, str]] = {}
    order: list[TimingKey] = []
    if not csv_path.exists():
        return rows, order

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            row = {name: raw_row.get(name, "") for name in fieldnames}
            # CSV files created before ILWT support contain only forward LWT rows.
            row["transform"] = row["transform"] or "lwt"
            # CSV files created before boundary-mode support used symmetric.
            row["lwt_boundary_mode"] = row["lwt_boundary_mode"] or "symmetric"
            try:
                key = row_key(row)
            except (KeyError, TypeError, ValueError):
                continue
            if key not in rows:
                order.append(key)
            rows[key] = row
    return rows, order


def base_row(
    args: argparse.Namespace,
    transform: str,
    wavelet: str,
    length: int,
    fieldnames: list[str],
) -> dict[str, object]:
    row: dict[str, object] = {name: "" for name in fieldnames}
    row.update(
        {
            "transform": transform,
            "wavelet": wavelet,
            "signal_length": length,
            "signal_start": args.signal_start,
            "signal_step": args.signal_step,
            "pywt_mode": args.pywt_mode,
            "lwt_boundary_mode": args.tt_boundary_mode,
            "pywt_runs": 0,
            "tt_wavelet_runs": 0,
            "status": "pending",
            "error": "",
        }
    )
    return row


def optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def refresh_speedup(row: dict[str, object]) -> None:
    pywt_mean = optional_float(row.get("pywt_mean_s"))
    tt_mean = optional_float(row.get("tt_wavelet_mean_s"))
    row["speedup_pywt_over_tt"] = (
        pywt_mean / tt_mean
        if pywt_mean is not None and tt_mean is not None and tt_mean > 0
        else ""
    )


def write_rows(
    csv_path: Path,
    fieldnames: list[str],
    rows: dict[TimingKey, dict[str, object]],
    order: list[TimingKey],
) -> None:
    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key in order:
            writer.writerow({name: rows[key].get(name, "") for name in fieldnames})
    tmp_path.replace(csv_path)


def parse_2d_shapes(raw_shapes: list[str]) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for raw_shape in raw_shapes:
        parts = raw_shape.lower().split("x")
        if len(parts) != 2:
            raise ValueError(f"Invalid 2D shape '{raw_shape}'; expected HEIGHTxWIDTH")
        try:
            height, width = (int(part) for part in parts)
        except ValueError as exc:
            raise ValueError(
                f"Invalid 2D shape '{raw_shape}'; expected integer HEIGHTxWIDTH"
            ) from exc
        if height <= 0 or width <= 0:
            raise ValueError(f"2D shape '{raw_shape}' must be positive")
        shape = (height, width)
        if shape not in seen:
            shapes.append(shape)
            seen.add(shape)
    return shapes


def run_fp32_reference_2d_benchmark(
    args: argparse.Namespace,
    wavelet: str,
    height: int,
    width: int,
    signal_file: Path,
) -> tuple[float, float]:
    command = [
        str(TT_WAVELET_2D_REFERENCE_BINARY),
        "--boundary-mode",
        args.tt_boundary_mode,
        "--benchmark",
        "--repeats",
        str(args.tt_repeats),
        "--warmup-runs",
        str(args.tt_warmup_runs),
        wavelet,
        str(height),
        str(width),
        str(signal_file),
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(
            "2D FP32 reference benchmark failed with exit code "
            f"{completed.returncode}.\n{output}"
        )
    mean_match = REFERENCE_2D_MEAN_TIME_PATTERN.search(output)
    min_match = REFERENCE_2D_MIN_TIME_PATTERN.search(output)
    if mean_match is None or min_match is None:
        raise RuntimeError(
            "2D FP32 reference output did not include mean and minimum timings"
        )
    return float(mean_match.group(1)) / 1000.0, float(min_match.group(1)) / 1000.0


def run_tt_wavelet_2d_benchmark(
    args: argparse.Namespace,
    wavelet: str,
    height: int,
    width: int,
    signal_file: Path,
) -> tuple[float, float, float, float, float, float, int, int, str, int, int, int]:
    alignment_option = ""
    if args.tt_alignment_csv_dir is not None:
        args.tt_alignment_csv_dir.mkdir(parents=True, exist_ok=True)
        alignment_path = (
            args.tt_alignment_csv_dir / f"{wavelet}_{height}x{width}.csv"
        )
        alignment_option = f"--alignment-csv {sh_quote(str(alignment_path))} "
    command = (
        f"source {sh_quote(str(TT_WAVELET_ENV))} "
        f"&& {sh_quote(str(TT_WAVELET_2D_BINARY))} "
        f"--boundary-mode symmetric --benchmark --cores {args.tt_cores} "
        f"--split-implementation {args.tt_split_implementation} "
        f"--route-staging {args.tt_route_staging} "
        f"--route-persistence {args.tt_route_persistence} "
        f"--terminal-writes {args.tt_terminal_writes} "
        f"--scale-policy {args.tt_scale_policy} "
        f"--planner {args.tt_planner} "
        f"--route-config {args.tt_route_config} "
        f"--exact-transfer {args.tt_exact_transfer} "
        f"{alignment_option}"
        f"{'--transport-metrics ' if args.tt_transport_metrics else ''}"
        f"--repeats {args.tt_repeats} "
        f"--warmup-runs {args.tt_warmup_runs} "
        f"{sh_quote(wavelet)} {height} {width} "
        f"{sh_quote(str(signal_file))}"
    )
    completed = subprocess.run(
        ["bash", "-lc", command],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(
            "Fused 2D TT-wavelet benchmark failed with exit code "
            f"{completed.returncode}.\n{output}"
        )
    if args.tt_transport_metrics:
        print(output, file=sys.stderr, end="")
    mean_match = TT_2D_MEAN_TIME_PATTERN.search(output)
    min_match = TT_2D_MIN_TIME_PATTERN.search(output)
    median_match = TT_2D_MEDIAN_TIME_PATTERN.search(output)
    p10_match = TT_2D_P10_TIME_PATTERN.search(output)
    p90_match = TT_2D_P90_TIME_PATTERN.search(output)
    stddev_match = TT_2D_STDDEV_TIME_PATTERN.search(output)
    core_match = TT_2D_ACTIVE_CORE_PATTERN.search(output)
    chunk_match = TT_2D_CHUNK_COUNT_PATTERN.search(output)
    chunk_tiles_match = TT_2D_CHUNK_TILES_PATTERN.search(output)
    route_count_match = TT_2D_ROUTE_COUNT_PATTERN.search(output)
    executable_route_count_match = TT_2D_EXECUTABLE_ROUTE_COUNT_PATTERN.search(output)
    scale_routes_removed_match = TT_2D_SCALE_ROUTES_REMOVED_PATTERN.search(output)
    if (
        mean_match is None
        or min_match is None
        or median_match is None
        or p10_match is None
        or p90_match is None
        or stddev_match is None
        or core_match is None
        or chunk_match is None
        or chunk_tiles_match is None
        or route_count_match is None
        or executable_route_count_match is None
        or scale_routes_removed_match is None
    ):
        raise RuntimeError(
            "Fused 2D TT-wavelet output omitted timing or scheduler telemetry"
        )
    return (
        float(mean_match.group(1)) / 1000.0,
        float(min_match.group(1)) / 1000.0,
        float(median_match.group(1)) / 1000.0,
        float(p10_match.group(1)) / 1000.0,
        float(p90_match.group(1)) / 1000.0,
        float(stddev_match.group(1)) / 1000.0,
        int(core_match.group(1)),
        int(chunk_match.group(1)),
        chunk_tiles_match.group(1),
        int(route_count_match.group(1)),
        int(executable_route_count_match.group(1)),
        int(scale_routes_removed_match.group(1)),
    )


def run_2d_benchmarks(args: argparse.Namespace) -> int:
    if args.transform != "lwt":
        raise ValueError("2D timing mode currently supports forward LWT only")

    needs_pywt = args.backend in {"both", "pywt"}
    needs_tt = args.backend in {"both", "tt-wavelet"}
    needs_reference = args.backend == "fp32-reference"
    ensure_runtime_packages(require_pywt=needs_pywt)
    if needs_pywt:
        import numpy as np
        import pywt
    from tqdm import tqdm

    shapes = parse_2d_shapes(args.shapes)
    if args.pywt_repeats < 0:
        raise ValueError("--pywt-repeats cannot be negative")
    if args.tt_repeats <= 0:
        raise ValueError("--tt-repeats must be positive")
    if args.tt_warmup_runs < 0:
        raise ValueError("--tt-warmup-runs cannot be negative")
    if args.tt_cores <= 0:
        raise ValueError("--tt-cores must be positive")
    if needs_tt and args.tt_boundary_mode != "symmetric":
        raise ValueError(
            "The fused 2D TT-wavelet backend currently supports symmetric boundary mode only"
        )
    if any(min(height, width) <= 1 for height, width in shapes) and (
        (needs_pywt and args.pywt_mode in {"reflect", "antireflect"})
        or (needs_reference and args.tt_boundary_mode in {"reflect", "antireflect"})
    ):
        raise ValueError(
            "2D reflect and antireflect modes require both dimensions to exceed one"
        )
    if needs_reference and not TT_WAVELET_2D_REFERENCE_BINARY.exists():
        raise FileNotFoundError(
            "2D FP32 reference binary not found at "
            f"{TT_WAVELET_2D_REFERENCE_BINARY}. Rebuild with "
            "./update.sh Release lwt_2d_reference"
        )
    if needs_tt and not TT_WAVELET_2D_BINARY.exists():
        raise FileNotFoundError(
            f"Fused 2D TT-wavelet binary not found at {TT_WAVELET_2D_BINARY}. "
            "Rebuild with ./update.sh Release lwt_2d"
        )

    if args.wavelets:
        wavelets = args.wavelets
    else:
        wavelets = load_wavelets_from_log(resolve_wavelets_log(args.wavelets_log))
    if (needs_reference or needs_tt) and not args.schemes_dir.exists():
        raise FileNotFoundError(f"Schemes directory not found: {args.schemes_dir}")

    csv_path = args.csv or PROJECT_ROOT / "tt_wavelet_timings_2d.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "transform",
        "wavelet",
        "signal_height",
        "signal_width",
        "signal_elements",
        "signal_start",
        "signal_step",
        "pywt_mode",
        "pywt_mean_s",
        "pywt_min_s",
        "pywt_runs",
        "tt_boundary_mode",
        "tt_mean_s",
        "tt_min_s",
        "tt_median_s",
        "tt_p10_s",
        "tt_p90_s",
        "tt_stddev_s",
        "tt_runs",
        "tt_active_core_count",
        "tt_chunk_count",
        "tt_chunk_tiles",
        "tt_route_count",
        "tt_executable_route_count",
        "tt_scale_routes_removed",
        "fp32_reference_boundary_mode",
        "fp32_reference_mean_s",
        "fp32_reference_min_s",
        "fp32_reference_runs",
        "speedup_pywt_over_fp32_reference",
        "speedup_pywt_over_tt",
        "status",
        "error",
    ]
    rows: dict[tuple[object, ...], dict[str, object]] = {}
    row_order: list[tuple[object, ...]] = []
    if csv_path.exists() and not args.overwrite:
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            for raw_row in csv.DictReader(handle):
                try:
                    key = (
                        raw_row["wavelet"],
                        int(raw_row["signal_height"]),
                        int(raw_row["signal_width"]),
                        float(raw_row["signal_start"]),
                        float(raw_row["signal_step"]),
                        raw_row["pywt_mode"],
                        raw_row["fp32_reference_boundary_mode"],
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                if key not in rows:
                    row_order.append(key)
                rows[key] = {name: raw_row.get(name, "") for name in fieldnames}

    signal_file = csv_path.with_suffix(".signal.txt")
    total_runs = len(shapes) * len(wavelets)
    with tqdm(total=total_runs, desc="Benchmarking 2D", unit="run") as progress:
        for height, width in shapes:
            signal_list = generate_signal(
                height * width, args.signal_start, args.signal_step
            )
            if needs_reference or needs_tt:
                write_signal_file(signal_file, signal_list)
            matrix = (
                np.asarray(signal_list, dtype=np.float64).reshape(height, width)
                if needs_pywt
                else None
            )
            for wavelet in wavelets:
                key = (
                    wavelet,
                    height,
                    width,
                    args.signal_start,
                    args.signal_step,
                    args.pywt_mode,
                    args.tt_boundary_mode,
                )
                if key not in rows:
                    rows[key] = {name: "" for name in fieldnames}
                    row_order.append(key)
                row = rows[key]
                row.update(
                    {
                        "transform": "lwt2d",
                        "wavelet": wavelet,
                        "signal_height": height,
                        "signal_width": width,
                        "signal_elements": height * width,
                        "signal_start": args.signal_start,
                        "signal_step": args.signal_step,
                        "pywt_mode": args.pywt_mode,
                        "tt_boundary_mode": args.tt_boundary_mode,
                        "fp32_reference_boundary_mode": args.tt_boundary_mode,
                        "status": "pending",
                        "error": "",
                    }
                )

                status = "ok"
                error_message = ""
                try:
                    if needs_pywt:
                        pywt_run = lambda: pywt.dwt2(
                            matrix, wavelet, mode=args.pywt_mode
                        )
                        pywt_mean, pywt_min = time_repeats(pywt_run, args.pywt_repeats)
                        row["pywt_mean_s"] = pywt_mean if pywt_mean is not None else ""
                        row["pywt_min_s"] = pywt_min if pywt_min is not None else ""
                        row["pywt_runs"] = args.pywt_repeats

                    if needs_reference:
                        scheme_path = args.schemes_dir / f"{wavelet}.json"
                        if not scheme_path.exists() and wavelet != "synthetic-k17":
                            raise FileNotFoundError(f"Scheme not found: {scheme_path}")
                        reference_mean, reference_min = run_fp32_reference_2d_benchmark(
                            args, wavelet, height, width, signal_file
                        )
                        row["fp32_reference_mean_s"] = reference_mean
                        row["fp32_reference_min_s"] = reference_min
                        row["fp32_reference_runs"] = args.tt_repeats

                    if needs_tt:
                        scheme_path = args.schemes_dir / f"{wavelet}.json"
                        if not scheme_path.exists() and wavelet != "synthetic-k17":
                            raise FileNotFoundError(
                                f"Scheme not found: {scheme_path}"
                            )
                        (
                            tt_mean,
                            tt_min,
                            tt_median,
                            tt_p10,
                            tt_p90,
                            tt_stddev,
                            active_cores,
                            chunk_count,
                            chunk_tiles,
                            route_count,
                            executable_route_count,
                            scale_routes_removed,
                        ) = run_tt_wavelet_2d_benchmark(
                            args, wavelet, height, width, signal_file
                        )
                        row["tt_mean_s"] = tt_mean
                        row["tt_min_s"] = tt_min
                        row["tt_median_s"] = tt_median
                        row["tt_p10_s"] = tt_p10
                        row["tt_p90_s"] = tt_p90
                        row["tt_stddev_s"] = tt_stddev
                        row["tt_runs"] = args.tt_repeats
                        row["tt_active_core_count"] = active_cores
                        row["tt_chunk_count"] = chunk_count
                        row["tt_chunk_tiles"] = chunk_tiles
                        row["tt_route_count"] = route_count
                        row["tt_executable_route_count"] = executable_route_count
                        row["tt_scale_routes_removed"] = scale_routes_removed

                    pywt_mean = optional_float(row.get("pywt_mean_s"))
                    reference_mean = optional_float(row.get("fp32_reference_mean_s"))
                    tt_mean = optional_float(row.get("tt_mean_s"))
                    row["speedup_pywt_over_fp32_reference"] = (
                        pywt_mean / reference_mean
                        if pywt_mean is not None
                        and reference_mean is not None
                        and reference_mean > 0
                        else ""
                    )
                    row["speedup_pywt_over_tt"] = (
                        pywt_mean / tt_mean
                        if pywt_mean is not None
                        and tt_mean is not None
                        and tt_mean > 0
                        else ""
                    )
                except Exception as exc:  # noqa: BLE001
                    status = "error"
                    error_message = str(exc)
                row["status"] = status
                row["error"] = error_message

                tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
                with tmp_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    for row_key_2d in row_order:
                        writer.writerow(
                            {
                                name: rows[row_key_2d].get(name, "")
                                for name in fieldnames
                            }
                        )
                tmp_path.replace(csv_path)
                progress.update(1)
    return 0


def main() -> int:
    args = parse_args()
    if args.shapes is not None:
        return run_2d_benchmarks(args)
    if args.backend == "fp32-reference":
        raise ValueError("--backend fp32-reference requires --shapes")

    needs_pywt = args.backend in {"both", "pywt"}
    needs_tt = args.backend in {"both", "tt-wavelet"}

    ensure_runtime_packages(require_pywt=needs_pywt)
    if needs_pywt:
        import pywt  # noqa: E402
    from tqdm import tqdm  # noqa: E402

    if args.lengths is not None:
        if any(length <= 0 for length in args.lengths):
            raise ValueError("Signal lengths must be positive.")
    else:
        if args.length_step <= 0:
            raise ValueError("--length-step must be positive.")
        if args.length_start <= 0 or args.length_stop <= 0:
            raise ValueError("Signal lengths must be positive.")
        if args.length_start > args.length_stop:
            raise ValueError("--length-start cannot exceed --length-stop.")

    if needs_tt and not TT_WAVELET_BINARY.exists():
        raise FileNotFoundError(
            f"TT-wavelet binary not found at {TT_WAVELET_BINARY}. Rebuild with ./update.sh Release lwt"
        )
    if needs_tt and not TT_WAVELET_ENV.exists():
        raise FileNotFoundError(f"TT-wavelet env script not found at {TT_WAVELET_ENV}")
    if args.tt_repeats <= 0:
        raise ValueError("--tt-repeats must be positive.")
    if args.tt_warmup_runs < 0:
        raise ValueError("--tt-warmup-runs cannot be negative.")
    if args.pywt_repeats < 0:
        raise ValueError("--pywt-repeats cannot be negative.")

    transforms = ["lwt", "ilwt"] if args.transform == "both" else [args.transform]
    if args.wavelets:
        wavelets = args.wavelets
    else:
        log_path = resolve_wavelets_log(args.wavelets_log)
        wavelets = load_wavelets_from_log(log_path)

    if not args.schemes_dir.exists():
        raise FileNotFoundError(f"Schemes directory not found: {args.schemes_dir}")

    lengths = (
        args.lengths
        if args.lengths is not None
        else list(range(args.length_start, args.length_stop + 1, args.length_step))
    )
    if 1 in lengths:
        if needs_tt and args.tt_boundary_mode in {"reflect", "antireflect"}:
            raise ValueError(
                "TT reflect and antireflect modes require signal lengths greater than one."
            )
        if needs_pywt and args.pywt_mode in {"reflect", "antireflect"}:
            raise ValueError(
                "PyWavelets reflect and antireflect modes require signal lengths greater than one."
            )

    csv_path = args.csv or PROJECT_ROOT / "tt_wavelet_timings.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "transform",
        "wavelet",
        "signal_length",
        "signal_start",
        "signal_step",
        "pywt_mode",
        "pywt_mean_s",
        "pywt_min_s",
        "pywt_runs",
        "tt_wavelet_mean_s",
        "tt_wavelet_min_s",
        "tt_wavelet_median_s",
        "tt_wavelet_p10_s",
        "tt_wavelet_p90_s",
        "tt_wavelet_stddev_s",
        "tt_wavelet_runs",
        "lwt_boundary_mode",
        "lwt_architecture",
        "lwt_layout",
        "lwt_max_group_count",
        "lwt_active_core_count",
        "lwt_chunk_count",
        "lwt_groups_per_chunk",
        "lwt_workspace_elements",
        "lwt_max_workspace_elements",
        "lwt_max_dependency_overhead",
        "lwt_terminal_scale_inline",
        "lwt_inverse_scale_inline",
        "lwt_inverse_final_interleave_direct",
        "lwt_l1_total_bytes",
        "lwt_l1_capacity_bytes",
        "lwt_l1_headroom_bytes",
        "speedup_pywt_over_tt",
        "status",
        "error",
    ]

    warmed_wavelets: set[tuple[str, str]] = set()
    warmed_global = [False]
    warmed_pairs: set[tuple[str, str, int]] = set()
    signal_file = csv_path.with_suffix(".signal.txt")
    rows, row_order = (
        ({}, []) if args.overwrite else read_existing_rows(csv_path, fieldnames)
    )

    total_runs = len(lengths) * len(wavelets) * len(transforms)
    with tqdm(total=total_runs, desc="Benchmarking", unit="run") as progress:
        for length in lengths:
            signal = None
            if needs_pywt or (needs_tt and args.tt_mode == "legacy"):
                signal_list = generate_signal(
                    length, args.signal_start, args.signal_step
                )
                if needs_tt and args.tt_mode == "legacy":
                    write_signal_file(signal_file, signal_list)
                if needs_pywt:
                    try:
                        import numpy as np

                        signal = np.array(signal_list, dtype=np.float64)
                    except ImportError:
                        signal = signal_list
            for wavelet in wavelets:
                scheme_path = args.schemes_dir / f"{wavelet}.json"
                for transform in transforms:
                    key = row_key(
                        base_row(args, transform, wavelet, length, fieldnames)
                    )
                    if key not in rows:
                        rows[key] = base_row(
                            args, transform, wavelet, length, fieldnames
                        )
                        row_order.append(key)
                    row = rows[key]

                    if needs_tt and not scheme_path.exists():
                        tqdm.write(
                            f"Skipping {wavelet}: scheme not found at {scheme_path}"
                        )
                        row["status"] = "missing_scheme"
                        row["error"] = f"Scheme not found: {scheme_path}"
                        if needs_pywt:
                            row["pywt_runs"] = args.pywt_repeats
                        if needs_tt:
                            row["tt_wavelet_runs"] = args.tt_repeats
                        refresh_speedup(row)
                        write_rows(csv_path, fieldnames, rows, row_order)
                        progress.update(1)
                        continue

                    command = (
                        build_tt_command(args, transform, wavelet, length, signal_file)
                        if needs_tt
                        else ""
                    )
                    status = "ok"
                    error_message = ""

                    try:
                        if needs_pywt:
                            if transform == "lwt":
                                pywt_run = lambda: pywt.dwt(
                                    signal, wavelet, mode=args.pywt_mode
                                )
                            else:
                                # Match the TT ILWT timing boundary: coefficient
                                # preparation is intentionally not timed.
                                approximation, detail = pywt.dwt(
                                    signal, wavelet, mode=args.pywt_mode
                                )
                                pywt_run = lambda: pywt.idwt(
                                    approximation,
                                    detail,
                                    wavelet,
                                    mode=args.pywt_mode,
                                )
                            pywt_mean, pywt_min = time_repeats(
                                pywt_run,
                                args.pywt_repeats,
                            )
                            row["pywt_mean_s"] = (
                                pywt_mean if pywt_mean is not None else ""
                            )
                            row["pywt_min_s"] = pywt_min if pywt_min is not None else ""
                            row["pywt_runs"] = args.pywt_repeats

                        if needs_tt:
                            if args.tt_mode == "benchmark":
                                tt_result = run_tt_wavelet(command)
                                tt_mean = tt_result.mean_s
                                tt_min = tt_result.min_s
                                row["tt_wavelet_median_s"] = tt_result.median_s or ""
                                row["tt_wavelet_p10_s"] = tt_result.p10_s or ""
                                row["tt_wavelet_p90_s"] = tt_result.p90_s or ""
                                row["tt_wavelet_stddev_s"] = tt_result.stddev_s or ""
                                row["lwt_architecture"] = tt_result.architecture
                                row["lwt_layout"] = tt_result.layout
                                row["lwt_max_group_count"] = (
                                    tt_result.max_group_count
                                    if tt_result.max_group_count is not None
                                    else ""
                                )
                                row["lwt_active_core_count"] = (
                                    tt_result.active_core_count
                                    if tt_result.active_core_count is not None
                                    else ""
                                )
                                row["lwt_chunk_count"] = (
                                    tt_result.chunk_count
                                    if tt_result.chunk_count is not None
                                    else ""
                                )
                                row["lwt_groups_per_chunk"] = (
                                    tt_result.groups_per_chunk
                                    if tt_result.groups_per_chunk is not None
                                    else ""
                                )
                                row["lwt_workspace_elements"] = (
                                    tt_result.workspace_elements
                                    if tt_result.workspace_elements is not None
                                    else ""
                                )
                                row["lwt_max_workspace_elements"] = (
                                    tt_result.max_workspace_elements
                                    if tt_result.max_workspace_elements is not None
                                    else ""
                                )
                                row["lwt_max_dependency_overhead"] = (
                                    tt_result.max_dependency_overhead
                                    if tt_result.max_dependency_overhead is not None
                                    else ""
                                )
                                row["lwt_terminal_scale_inline"] = (
                                    tt_result.terminal_scale_inline
                                    if tt_result.terminal_scale_inline is not None
                                    else ""
                                )
                                row["lwt_inverse_scale_inline"] = (
                                    tt_result.inverse_scale_inline
                                    if tt_result.inverse_scale_inline is not None
                                    else ""
                                )
                                row["lwt_inverse_final_interleave_direct"] = (
                                    tt_result.inverse_final_interleave_direct
                                    if tt_result.inverse_final_interleave_direct
                                    is not None
                                    else ""
                                )
                                row["lwt_l1_total_bytes"] = (
                                    tt_result.l1_total_bytes
                                    if tt_result.l1_total_bytes is not None
                                    else ""
                                )
                                row["lwt_l1_capacity_bytes"] = (
                                    tt_result.l1_capacity_bytes
                                    if tt_result.l1_capacity_bytes is not None
                                    else ""
                                )
                                row["lwt_l1_headroom_bytes"] = (
                                    tt_result.l1_headroom_bytes
                                    if tt_result.l1_headroom_bytes is not None
                                    else ""
                                )
                            else:
                                if args.tt_warmup_runs > 0 and should_warmup(
                                    args.tt_warmup_scope,
                                    transform,
                                    wavelet,
                                    length,
                                    warmed_wavelets,
                                    warmed_global,
                                    warmed_pairs,
                                ):
                                    for _ in range(args.tt_warmup_runs):
                                        run_tt_wavelet(command)

                                tt_mean, tt_min = value_repeats(
                                    lambda: run_tt_wavelet(command).mean_s,
                                    args.tt_repeats,
                                )
                            row["tt_wavelet_mean_s"] = (
                                tt_mean if tt_mean is not None else ""
                            )
                            row["tt_wavelet_min_s"] = (
                                tt_min if tt_min is not None else ""
                            )
                            row["tt_wavelet_runs"] = args.tt_repeats
                    except Exception as exc:  # noqa: BLE001
                        status = "error"
                        error_message = str(exc)

                    row["status"] = status
                    row["error"] = error_message
                    refresh_speedup(row)
                    write_rows(csv_path, fieldnames, rows, row_order)
                    progress.update(1)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
