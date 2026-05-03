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
TT_WAVELET_BINARY = PROJECT_ROOT / "build" / "tt-wavelet" / "lwt"
TT_WAVELET_ENV = PROJECT_ROOT / "scripts" / "set_env.sh"
DEFAULT_SCHEMES_DIR = PROJECT_ROOT / "ttnn-wavelet" / "lifting_schemes"
DEFAULT_LOG_CANDIDATES = [
    PROJECT_ROOT / "wavelets.log",
    PROJECT_ROOT / "wavelets (1).log",
]
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python3"


def ensure_runtime_packages() -> None:
    try:
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
            "Granularity at which TT-wavelet warmup runs are issued. "
            "'length' (default) warms up once per (wavelet, length) pair, which is correct "
            "because each lwt invocation is a fresh process that reloads kernels. "
            "'wavelet' only warms the first length for each wavelet name. "
            "'global' warms only once for the entire benchmark."
        ),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT / "tt_wavelet_timings.csv",
        help="Output CSV path (default: %(default)s).",
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


def sh_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def build_tt_command(scheme_path: Path, length: int, start: float, step: float) -> str:
    args = [
        str(TT_WAVELET_BINARY),
        "--quiet",
        "--signal-length",
        str(length),
        "--signal-start",
        repr(start),
        "--signal-step",
        repr(step),
        str(scheme_path),
    ]
    command = " ".join(sh_quote(arg) for arg in args)
    return f"source {sh_quote(str(TT_WAVELET_ENV))} && {command}"


def run_tt_wavelet(command: str) -> None:
    completed = subprocess.run(
        ["bash", "-lc", command],
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"TT-wavelet run failed with exit code {completed.returncode}.\n{completed.stderr}"
        )


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


def should_warmup(
    scope: str,
    wavelet: str,
    length: int,
    warmed: set[str],
    warmed_lengths: set[int],
    warmed_global: list[bool],
    warmed_pairs: set[tuple],
) -> bool:
    if scope == "global":
        if warmed_global[0]:
            return False
        warmed_global[0] = True
        return True
    if scope == "length":
        # Each lwt call is a new process — kernel loading happens per (wavelet, length) pair.
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


def main() -> int:
    ensure_runtime_packages()
    import pywt  # noqa: E402
    from tqdm import tqdm  # noqa: E402

    args = parse_args()

    if args.length_step <= 0:
        raise ValueError("--length-step must be positive.")
    if args.length_start <= 0 or args.length_stop <= 0:
        raise ValueError("Signal lengths must be positive.")
    if args.length_start > args.length_stop:
        raise ValueError("--length-start cannot exceed --length-stop.")

    if not TT_WAVELET_BINARY.exists():
        raise FileNotFoundError(
            f"TT-wavelet binary not found at {TT_WAVELET_BINARY}. Rebuild with ./update.sh Release lwt"
        )
    if not TT_WAVELET_ENV.exists():
        raise FileNotFoundError(f"TT-wavelet env script not found at {TT_WAVELET_ENV}")

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
    warmed_lengths: set[int] = set()
    warmed_global = [False]
    warmed_pairs: set[tuple] = set()

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        total_runs = len(lengths) * len(wavelets)
        with tqdm(total=total_runs, desc="Benchmarking", unit="run") as progress:
            for length in lengths:
                signal_list = generate_signal(length, args.signal_start, args.signal_step)
                try:
                    import numpy as np

                    signal = np.array(signal_list, dtype=np.float64)
                except ImportError:
                    signal = signal_list
                for wavelet in wavelets:
                    scheme_path = args.schemes_dir / f"{wavelet}.json"
                    if not scheme_path.exists():
                        tqdm.write(f"Skipping {wavelet}: scheme not found at {scheme_path}")
                        writer.writerow(
                            {
                                "wavelet": wavelet,
                                "signal_length": length,
                                "signal_start": args.signal_start,
                                "signal_step": args.signal_step,
                                "pywt_mode": args.pywt_mode,
                                "pywt_mean_s": None,
                                "pywt_min_s": None,
                                "pywt_runs": args.pywt_repeats,
                                "tt_wavelet_mean_s": None,
                                "tt_wavelet_min_s": None,
                                "tt_wavelet_runs": args.tt_repeats,
                                "speedup_pywt_over_tt": None,
                                "status": "missing_scheme",
                                "error": f"Scheme not found: {scheme_path}",
                            }
                        )
                        handle.flush()
                        progress.update(1)
                        continue

                    command = build_tt_command(
                        scheme_path, length, args.signal_start, args.signal_step
                    )

                    status = "ok"
                    error_message = ""
                    pywt_mean = None
                    pywt_min = None
                    tt_mean = None
                    tt_min = None

                    try:
                        if args.tt_warmup_runs > 0 and should_warmup(
                            args.tt_warmup_scope,
                            wavelet,
                            length,
                            warmed_wavelets,
                            warmed_lengths,
                            warmed_global,
                            warmed_pairs,
                        ):
                            for _ in range(args.tt_warmup_runs):
                                run_tt_wavelet(command)

                        pywt_mean, pywt_min = time_repeats(
                            lambda: pywt.dwt(signal, wavelet, mode=args.pywt_mode),
                            args.pywt_repeats,
                        )
                        tt_mean, tt_min = time_repeats(
                            lambda: run_tt_wavelet(command),
                            args.tt_repeats,
                        )
                    except Exception as exc:  # noqa: BLE001
                        status = "error"
                        error_message = str(exc)

                    speedup = None
                    if pywt_mean is not None and tt_mean is not None and tt_mean > 0:
                        speedup = pywt_mean / tt_mean

                    writer.writerow(
                        {
                            "wavelet": wavelet,
                            "signal_length": length,
                            "signal_start": args.signal_start,
                            "signal_step": args.signal_step,
                            "pywt_mode": args.pywt_mode,
                            "pywt_mean_s": pywt_mean,
                            "pywt_min_s": pywt_min,
                            "pywt_runs": args.pywt_repeats,
                            "tt_wavelet_mean_s": tt_mean,
                            "tt_wavelet_min_s": tt_min,
                            "tt_wavelet_runs": args.tt_repeats,
                            "speedup_pywt_over_tt": speedup,
                            "status": status,
                            "error": error_message,
                        }
                    )
                    handle.flush()
                    progress.update(1)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
