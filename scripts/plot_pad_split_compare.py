#!/usr/bin/env python3
import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare pad_split benchmark CSVs.")
    parser.add_argument("--old-csv", required=True, help="Path to the old CSV.")
    parser.add_argument("--new-csv", required=True, help="Path to the new CSV.")
    parser.add_argument("--out", default="pad_split_compare.png", help="Output plot path.")
    return parser.parse_args()


def load_medians(csv_path: str) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    with open(csv_path, "r", encoding="ascii", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            length = int(row["signal_length"])
            min_ms = float(row["pad_split_min_ms"])
            grouped[length].append(min_ms)

    if not grouped:
        raise RuntimeError(f"No data found in CSV: {csv_path}")

    return {length: statistics.median(values) for length, values in grouped.items()}


def print_summary(old_medians: dict[int, float], new_medians: dict[int, float]) -> None:
    common = sorted(set(old_medians.keys()) & set(new_medians.keys()))
    if not common:
        print("No common signal lengths found.")
        return

    ratios = []
    improvements = []
    best_length = None
    worst_length = None
    best_ratio = None
    worst_ratio = None

    for length in common:
        old_ms = old_medians[length]
        new_ms = new_medians[length]
        if old_ms <= 0 or new_ms <= 0:
            continue
        ratio = old_ms / new_ms
        improvement = (old_ms - new_ms) / old_ms * 100.0
        ratios.append(ratio)
        improvements.append(improvement)

        if best_ratio is None or ratio > best_ratio:
            best_ratio = ratio
            best_length = length
        if worst_ratio is None or ratio < worst_ratio:
            worst_ratio = ratio
            worst_length = length

    if not ratios:
        print("No valid speedup ratios to report.")
        return

    median_ratio = statistics.median(ratios)
    median_improvement = statistics.median(improvements)

    print(f"Common signal lengths: {len(common)}")
    print(f"Median speedup ratio (old/new): {median_ratio:.4f}")
    print(f"Median improvement: {median_improvement:.2f}%")

    if best_length is not None and best_ratio is not None:
        print(
            "Best speedup at length "
            f"{best_length}: {best_ratio:.4f}x "
            f"(old {old_medians[best_length]:.6f} ms, "
            f"new {new_medians[best_length]:.6f} ms)"
        )
    if worst_length is not None and worst_ratio is not None:
        print(
            "Worst speedup at length "
            f"{worst_length}: {worst_ratio:.4f}x "
            f"(old {old_medians[worst_length]:.6f} ms, "
            f"new {new_medians[worst_length]:.6f} ms)"
        )


def write_plot(old_medians: dict[int, float], new_medians: dict[int, float], out_path: str) -> None:
    old_lengths = sorted(old_medians.keys())
    new_lengths = sorted(new_medians.keys())

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(
        old_lengths,
        [old_medians[length] for length in old_lengths],
        marker="o",
        label="old",
    )
    plt.plot(
        new_lengths,
        [new_medians[length] for length in new_lengths],
        marker="o",
        label="new",
    )
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.title("Pad split min timing comparison")
    plt.xlabel("signal_length")
    plt.ylabel("pad_split_min_ms")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> int:
    args = parse_args()
    old_medians = load_medians(args.old_csv)
    new_medians = load_medians(args.new_csv)

    write_plot(old_medians, new_medians, args.out)
    print_summary(old_medians, new_medians)
    print(f"Wrote plot: {args.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        raise
