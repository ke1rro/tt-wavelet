#!/usr/bin/env python3
"""Summarize Blackhole forced-layout oracle and automatic planner quality."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--raw", type=Path)
    parser.add_argument(
        "--blackhole-calibrated-policy",
        action="store_true",
        help="Evaluate the final metadata rule against forced-layout medians.",
    )
    parser.add_argument("--meaningful-percent", type=float, default=3.0)
    return parser.parse_args()


def as_float(row: dict[str, str], field: str) -> float:
    return float(row[field])


def main() -> int:
    args = parse_args()
    with args.summary.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))

    metadata: dict[tuple[str, str, int], tuple[int, int]] = {}
    if args.blackhole_calibrated_policy:
        raw_path = args.raw
        if raw_path is None:
            raw_path = Path(str(args.summary).replace("_summary.csv", "_raw.csv"))
        with raw_path.open(newline="", encoding="utf-8") as source:
            for raw_row in csv.DictReader(source):
                if raw_row["layout"] != "auto" or not raw_row["status"].startswith("ok"):
                    continue
                key = (
                    raw_row["transform"],
                    raw_row["wavelet"],
                    int(raw_row["signal_length"]),
                )
                metadata[key] = (
                    int(raw_row["route_count"]),
                    int(raw_row["groups_per_chunk"]),
                )

    statistics: dict[str, object] = {
        "summary": str(args.summary),
        "calibration_points": len(rows),
        "architecture": sorted({row["architecture"] for row in rows}),
        "exact_winner_matches": 0,
        "within_1_percent": 0,
        "within_3_percent": 0,
        "meaningful_misses": [],
        "planner_evaluation": (
            "blackhole-calibrated-metadata-rule"
            if args.blackhole_calibrated_policy
            else "recorded-auto"
        ),
    }
    winner_counts: Counter[str] = Counter()
    decisions: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)

    for row in rows:
        winner = row["winner"]
        selected = row["auto_selected_layout"]
        winner_counts[f"{row['transform']}:{winner}"] += 1
        decisions[(row["transform"], row["wavelet"])].append(
            (int(row["signal_length"]), winner)
        )

        row_ms = as_float(row, "row_major_median_ms")
        tile_ms = as_float(row, "tile_native_median_ms")
        auto_ms = as_float(row, "auto_median_ms")
        if args.blackhole_calibrated_policy:
            route_count, groups_per_chunk = metadata[
                (row["transform"], row["wavelet"], int(row["signal_length"]))
            ]
            if row["transform"] == "lwt":
                selected = "tile-native"
            else:
                row_major_crossover = (
                    groups_per_chunk >= 2 and route_count <= 2
                ) or (groups_per_chunk >= 3 and route_count <= 3)
                selected = "row-major" if row_major_crossover else "tile-native"
            auto_ms = row_ms if selected == "row-major" else tile_ms
        oracle_layout = "row-major" if row_ms < tile_ms else "tile-native"
        oracle_ms = min(row_ms, tile_ms)
        regression = 100.0 * (auto_ms / oracle_ms - 1.0)
        exact = winner == "tie" or selected == oracle_layout
        statistics["exact_winner_matches"] += int(exact)
        statistics["within_1_percent"] += int(regression <= 1.0)
        statistics["within_3_percent"] += int(regression <= 3.0)
        if regression > args.meaningful_percent:
            statistics["meaningful_misses"].append(
                {
                    "transform": row["transform"],
                    "scheme": row["wavelet"],
                    "N": int(row["signal_length"]),
                    "auto_choice": selected,
                    "oracle_choice": oracle_layout,
                    "auto_median_ms": auto_ms,
                    "oracle_median_ms": oracle_ms,
                    "relative_regression_percent": regression,
                }
            )

    winner_changes = []
    for (transform, wavelet), points in sorted(decisions.items()):
        ordered = sorted(points)
        concrete = [(size, winner) for size, winner in ordered if winner != "tie"]
        if len({winner for _, winner in concrete}) > 1:
            winner_changes.append(
                {"transform": transform, "scheme": wavelet, "points": ordered}
            )
    statistics["winner_counts"] = dict(sorted(winner_counts.items()))
    statistics["winner_changes"] = winner_changes
    statistics["meaningful_misses"].sort(
        key=lambda miss: miss["relative_regression_percent"], reverse=True
    )

    rendered = json.dumps(statistics, indent=2) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
