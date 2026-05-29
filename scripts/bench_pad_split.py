#!/usr/bin/env python3
import argparse
import csv
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRIC_KEYS = [
    "pad_split_mean_ms",
    "pad_split_min_ms",
    "pad_split_max_ms",
    "padded_length",
    "even_length",
    "odd_length",
]
FLOAT_KEYS = {
    "pad_split_mean_ms",
    "pad_split_min_ms",
    "pad_split_max_ms",
}
INT_KEYS = {
    "padded_length",
    "even_length",
    "odd_length",
}

CSV_COLUMNS = [
    "signal_length",
    "run_idx",
    "pad_split_mean_ms",
    "pad_split_min_ms",
    "pad_split_max_ms",
    "padded_length",
    "even_length",
    "odd_length",
    "samples_per_ms_min",
    "ms_per_million_samples_min",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark pad_split preprocess timing.")
    parser.add_argument("--bin", default="./build/lwt", help="Path to the lwt binary.")
    parser.add_argument("--wavelet", default="bior1.3", help="Wavelet name.")
    parser.add_argument("--start", type=int, default=1024, help="First signal length.")
    parser.add_argument("--step", type=int, default=250000, help="Step for subsequent lengths.")
    parser.add_argument("--end", type=int, default=10000000, help="Maximum signal length.")
    parser.add_argument("--runs", type=int, default=3, help="Runs per signal length.")
    parser.add_argument(
        "--signal-file",
        default="/tmp/tt_wavelet_pad_signal.txt",
        help="Temporary signal file path.",
    )
    parser.add_argument("--csv", default="pad_split_bench.csv", help="CSV output path.")
    parser.add_argument("--plot", default="pad_split_bench.png", help="Plot output path.")
    return parser.parse_args()


def build_lengths(start: int, step: int, end: int) -> list[int]:
    if start <= 0 or step <= 0 or end <= 0:
        raise ValueError("start, step, and end must be positive")
    if start > end:
        raise ValueError("start must be <= end")

    lengths: list[int] = []
    seen: set[int] = set()

    def add_length(value: int) -> None:
        if value not in seen:
            lengths.append(value)
            seen.add(value)

    add_length(start)
    max_multiplier = end // step
    for multiplier in range(1, max_multiplier + 1):
        add_length(step * multiplier)
    return lengths


def write_signal_file(path: str, length: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    line = "1.0\n"
    chunk_size = 100000
    remaining = length
    with open(path, "w", encoding="ascii", newline="\n") as handle:
        while remaining > 0:
            count = chunk_size if remaining > chunk_size else remaining
            handle.write(line * count)
            remaining -= count


def parse_metrics(stderr_text: str) -> dict[str, float | int]:
    values: dict[str, str] = {}
    for raw_line in stderr_text.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in METRIC_KEYS:
            values[key] = value.strip()

    missing = [key for key in METRIC_KEYS if key not in values]
    if missing:
        raise ValueError(f"Missing metrics: {', '.join(missing)}")

    parsed: dict[str, float | int] = {}
    for key in METRIC_KEYS:
        raw_value = values[key]
        try:
            if key in FLOAT_KEYS:
                parsed[key] = float(raw_value)
            else:
                parsed[key] = int(raw_value)
        except ValueError as exc:
            raise ValueError(f"Failed to parse {key} from '{raw_value}'") from exc

    return parsed


def run_once(bin_path: str, wavelet: str, signal_file: str) -> dict[str, float | int]:
    result = subprocess.run(
        [bin_path, wavelet, signal_file],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        sys.stderr.write("Binary failed with non-zero exit code.\n")
        sys.stderr.write(result.stderr)
        raise RuntimeError("Benchmark binary failed")

    try:
        return parse_metrics(result.stderr)
    except ValueError as exc:
        sys.stderr.write("Failed to parse stderr. Full stderr follows:\n")
        sys.stderr.write(result.stderr)
        raise RuntimeError("Failed to parse benchmark metrics") from exc


def write_plot(csv_path: str, plot_path: str) -> None:
    grouped: dict[int, list[float]] = defaultdict(list)
    with open(csv_path, "r", encoding="ascii", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            length = int(row["signal_length"])
            min_ms = float(row["pad_split_min_ms"])
            grouped[length].append(min_ms)

    if not grouped:
        raise RuntimeError("No data found in CSV for plotting")

    lengths = sorted(grouped.keys())
    medians = [statistics.median(grouped[length]) for length in lengths]
    bests = [min(grouped[length]) for length in lengths]

    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(lengths, medians, marker="o", label="median min ms")
    plt.plot(lengths, bests, marker="o", label="best min ms")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.title("Pad split min timing")
    plt.xlabel("signal_length")
    plt.ylabel("pad_split_min_ms")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def main() -> int:
    args = parse_args()

    if not os.path.exists(args.bin):
        raise FileNotFoundError(f"Binary not found: {args.bin}")
    if args.runs <= 0:
        raise ValueError("runs must be positive")

    lengths = build_lengths(args.start, args.step, args.end)

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for length_idx, length in enumerate(lengths):
            print(f"signal_length={length} ({length_idx + 1}/{len(lengths)})")
            write_signal_file(args.signal_file, length)

            for run_idx in range(args.runs):
                print(f"  run {run_idx + 1}/{args.runs}")
                metrics = run_once(args.bin, args.wavelet, args.signal_file)

                min_ms = float(metrics["pad_split_min_ms"])
                samples_per_ms = float("inf") if min_ms <= 0 else length / min_ms
                ms_per_million_samples = (
                    float("inf") if length <= 0 else min_ms * 1_000_000.0 / length
                )

                row = {
                    "signal_length": length,
                    "run_idx": run_idx,
                    "pad_split_mean_ms": metrics["pad_split_mean_ms"],
                    "pad_split_min_ms": metrics["pad_split_min_ms"],
                    "pad_split_max_ms": metrics["pad_split_max_ms"],
                    "padded_length": metrics["padded_length"],
                    "even_length": metrics["even_length"],
                    "odd_length": metrics["odd_length"],
                    "samples_per_ms_min": samples_per_ms,
                    "ms_per_million_samples_min": ms_per_million_samples,
                }
                writer.writerow(row)
                handle.flush()

    write_plot(args.csv, args.plot)
    print(f"Wrote CSV: {args.csv}")
    print(f"Wrote plot: {args.plot}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        raise
