#!/usr/bin/env python3
import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parent
TT_WAVELET_BINARY = PROJECT_ROOT / "build" / "lwt"
TT_WAVELET_ENV = PROJECT_ROOT / "scripts" / "set_env.sh"
DEFAULT_SCHEMES_DIR = PROJECT_ROOT / "ttnn-wavelet" / "lifting_schemes"
TT_TIME_PATTERN = re.compile(r"lwt_execution_time_ms:\s*([0-9eE+.\-]+)")
TT_MIN_TIME_PATTERN = re.compile(r"lwt_min_time_ms:\s*([0-9eE+.\-]+)")
DEFAULT_LOG_CANDIDATES = [
    PROJECT_ROOT / "wavelets.log",
    PROJECT_ROOT / "wavelets (1).log",
]
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python3"
TimingKey = tuple[str, int, float, float, str]


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
            "`pip install -r ttnn-wavelet/requirements.txt`."
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark TT-wavelet (device) vs PyWavelets timings over multiple lengths."
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
        choices=["both", "tt-wavelet", "pywt"],
        default="both",
        help="Benchmark both backends or only one backend (default: %(default)s).",
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
            "inside the C++ process for every (wavelet, length) pair."
        ),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT / "tt_wavelet_timings.csv",
        help="Output CSV path (default: %(default)s).",
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


def build_tt_command(args: argparse.Namespace, wavelet: str, length: int, signal_file: Path) -> str:
    command_args = [str(TT_WAVELET_BINARY)]
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


def run_tt_wavelet(command: str) -> tuple[float, float]:
    completed = subprocess.run(
        ["bash", "-lc", command],
        cwd=PROJECT_ROOT,
        env=tt_benchmark_env(),
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
        raise RuntimeError("TT-wavelet output did not include lwt_execution_time_ms.")
    min_match = TT_MIN_TIME_PATTERN.search(completed.stderr)
    mean_s = float(match.group(1)) / 1000.0
    min_s = float(min_match.group(1)) / 1000.0 if min_match is not None else mean_s
    return mean_s, min_s


def time_repeats(run_once: Callable[[], None], repeats: int) -> tuple[float | None, float | None]:
    if repeats <= 0:
        return None, None
    times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        run_once()
        times.append(time.perf_counter() - start)
    mean = sum(times) / len(times)
    return mean, min(times)


def value_repeats(run_once: Callable[[], float], repeats: int) -> tuple[float | None, float | None]:
    if repeats <= 0:
        return None, None
    times = [run_once() for _ in range(repeats)]
    return sum(times) / len(times), min(times)


def should_warmup(
    scope: str,
    wavelet: str,
    length: int,
    warmed: set[str],
    warmed_global: list[bool],
    warmed_pairs: set[tuple[str, int]],
) -> bool:
    if scope == "global":
        if warmed_global[0]:
            return False
        warmed_global[0] = True
        return True
    if scope == "length":
        # Each lwt call is a new process; kernel loading happens per (wavelet, length) pair.
        key = (wavelet, length)
        if key in warmed_pairs:
            return False
        warmed_pairs.add(key)
        return True
    # scope == "wavelet": warm once per wavelet name (first length only)
    if wavelet in warmed:
        return False
    warmed.add(wavelet)
    return True


def row_key(row: dict[str, object]) -> TimingKey:
    return (
        str(row["wavelet"]),
        int(row["signal_length"]),
        float(row["signal_start"]),
        float(row["signal_step"]),
        str(row["pywt_mode"]),
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
            try:
                key = row_key(row)
            except (KeyError, TypeError, ValueError):
                continue
            if key not in rows:
                order.append(key)
            rows[key] = row
    return rows, order


def base_row(
    args: argparse.Namespace, wavelet: str, length: int, fieldnames: list[str]
) -> dict[str, object]:
    row: dict[str, object] = {name: "" for name in fieldnames}
    row.update(
        {
            "wavelet": wavelet,
            "signal_length": length,
            "signal_start": args.signal_start,
            "signal_step": args.signal_step,
            "pywt_mode": args.pywt_mode,
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
        pywt_mean / tt_mean if pywt_mean is not None and tt_mean is not None and tt_mean > 0 else ""
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


def main() -> int:
    args = parse_args()
    needs_pywt = args.backend in {"both", "pywt"}
    needs_tt = args.backend in {"both", "tt-wavelet"}

    ensure_runtime_packages(require_pywt=needs_pywt)
    if needs_pywt:
        import pywt  # noqa: E402
    from tqdm import tqdm  # noqa: E402

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

    if args.wavelets:
        wavelets = args.wavelets
    else:
        log_path = resolve_wavelets_log(args.wavelets_log)
        wavelets = load_wavelets_from_log(log_path)

    if not args.schemes_dir.exists():
        raise FileNotFoundError(f"Schemes directory not found: {args.schemes_dir}")

    lengths = list(range(args.length_start, args.length_stop + 1, args.length_step))

    csv_path = args.csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
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
        "tt_wavelet_runs",
        "speedup_pywt_over_tt",
        "status",
        "error",
    ]

    warmed_wavelets: set[str] = set()
    warmed_global = [False]
    warmed_pairs: set[tuple[str, int]] = set()
    signal_file = csv_path.with_suffix(".signal.txt")
    rows, row_order = ({}, []) if args.overwrite else read_existing_rows(csv_path, fieldnames)

    total_runs = len(lengths) * len(wavelets)
    with tqdm(total=total_runs, desc="Benchmarking", unit="run") as progress:
        for length in lengths:
            signal = None
            if needs_pywt or (needs_tt and args.tt_mode == "legacy"):
                signal_list = generate_signal(length, args.signal_start, args.signal_step)
                if needs_tt and args.tt_mode == "legacy":
                    write_signal_file(signal_file, signal_list)
                if needs_pywt:
                    try:
                        import numpy as np

                        signal = np.array(signal_list, dtype=np.float64)
                    except ImportError:
                        signal = signal_list
            for wavelet in wavelets:
                key = row_key(base_row(args, wavelet, length, fieldnames))
                if key not in rows:
                    rows[key] = base_row(args, wavelet, length, fieldnames)
                    row_order.append(key)
                row = rows[key]

                scheme_path = args.schemes_dir / f"{wavelet}.json"
                if needs_tt and not scheme_path.exists():
                    tqdm.write(f"Skipping {wavelet}: scheme not found at {scheme_path}")
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

                command = build_tt_command(args, wavelet, length, signal_file) if needs_tt else ""

                status = "ok"
                error_message = ""

                try:
                    if needs_pywt:
                        pywt_mean, pywt_min = time_repeats(
                            lambda: pywt.dwt(signal, wavelet, mode=args.pywt_mode),
                            args.pywt_repeats,
                        )
                        row["pywt_mean_s"] = pywt_mean if pywt_mean is not None else ""
                        row["pywt_min_s"] = pywt_min if pywt_min is not None else ""
                        row["pywt_runs"] = args.pywt_repeats

                    if needs_tt:
                        if args.tt_mode == "benchmark":
                            tt_mean, tt_min = run_tt_wavelet(command)
                        else:
                            if args.tt_warmup_runs > 0 and should_warmup(
                                args.tt_warmup_scope,
                                wavelet,
                                length,
                                warmed_wavelets,
                                warmed_global,
                                warmed_pairs,
                            ):
                                for _ in range(args.tt_warmup_runs):
                                    run_tt_wavelet(command)

                            tt_mean, tt_min = value_repeats(
                                lambda: run_tt_wavelet(command)[0],
                                args.tt_repeats,
                            )
                        row["tt_wavelet_mean_s"] = tt_mean if tt_mean is not None else ""
                        row["tt_wavelet_min_s"] = tt_min if tt_min is not None else ""
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
