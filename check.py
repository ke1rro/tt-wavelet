import json
import sys
from collections import Counter
from pathlib import Path

STEP_TYPES = ["predict", "update", "scale-even", "scale-odd", "swap"]


def load_step_types(path: Path):
    with path.open("r", encoding="utf-8") as f:
        scheme = json.load(f)

    return [step.get("type", "unknown") for step in scheme.get("steps", [])]


def main():
    if len(sys.argv) != 2:
        print("Usage: python step_stats.py <lifting_schemes_dir>")
        sys.exit(1)

    root = Path(sys.argv[1])

    if not root.exists():
        print(f"Directory does not exist: {root}")
        sys.exit(1)

    json_files = sorted(root.rglob("*.json"))

    if not json_files:
        print(f"No JSON files found in: {root}")
        sys.exit(1)

    total_counts = Counter()
    per_file_counts = []

    for path in json_files:
        step_types = load_step_types(path)
        counts = Counter(step_types)

        total_counts.update(counts)
        per_file_counts.append((path, counts, len(step_types)))

    total_files = len(json_files)
    total_steps = sum(total_counts.values())

    print(f"Checked files: {total_files}")
    print(f"Total steps: {total_steps}")

    print("\nStep type totals:")
    for step_type in STEP_TYPES:
        count = total_counts[step_type]
        percent = (count / total_steps * 100.0) if total_steps else 0.0
        avg_per_filter = count / total_files if total_files else 0.0
        print(f"  {step_type:10s}: {count:4d} | {percent:6.2f}% | avg/filter: {avg_per_filter:.3f}")

    unknown_count = sum(
        count for step_type, count in total_counts.items() if step_type not in STEP_TYPES
    )

    if unknown_count:
        print("\nUnknown step types:")
        for step_type, count in total_counts.items():
            if step_type not in STEP_TYPES:
                percent = count / total_steps * 100.0
                avg_per_filter = count / total_files
                print(
                    f"  {step_type:10s}: {count:4d} | {percent:6.2f}% | avg/filter: {avg_per_filter:.3f}"
                )

    predict_update_total = total_counts["predict"] + total_counts["update"]
    scale_total = total_counts["scale-even"] + total_counts["scale-odd"]

    print("\nGrouped totals:")
    print(
        f"  predict/update : {predict_update_total:4d} | "
        f"{predict_update_total / total_steps * 100.0:6.2f}% | "
        f"avg/filter: {predict_update_total / total_files:.3f}"
    )
    print(
        f"  scale total    : {scale_total:4d} | "
        f"{scale_total / total_steps * 100.0:6.2f}% | "
        f"avg/filter: {scale_total / total_files:.3f}"
    )

    print("\nScale details:")
    print(f"  avg scale-even/filter: {total_counts['scale-even'] / total_files:.3f}")
    print(f"  avg scale-odd/filter : {total_counts['scale-odd'] / total_files:.3f}")
    print(f"  avg total scale/filter: {scale_total / total_files:.3f}")

    print("\nPer-filter step count summary:")
    step_counts = [n for _, _, n in per_file_counts]

    min_steps = min(step_counts)
    max_steps = max(step_counts)
    avg_steps = sum(step_counts) / total_files

    print(f"  min steps/filter: {min_steps}")
    print(f"  max steps/filter: {max_steps}")
    print(f"  avg steps/filter: {avg_steps:.3f}")

    no_scale = [
        path
        for path, counts, _ in per_file_counts
        if counts["scale-even"] + counts["scale-odd"] == 0
    ]

    if no_scale:
        print("\nFiles without scale steps:")
        for path in no_scale:
            print(f"  {path}")


if __name__ == "__main__":
    main()
