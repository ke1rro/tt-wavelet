#!/usr/bin/env python3
"""
Generate Performance Benchmark Line Plots (Log-Scale PNG and SVG Files).

Creates individual log-scale plots comparing:
- PyWavelets (CPU)
- tt-wavelet (Standalone C++)
- ttnn-wavelet (TTNN Device)

For each (dimension, transform, wavelet, boundary_mode) combination.
Strictly validates that data for all 3 backends is present.
"""

import argparse
import os
import sys
from pathlib import Path

import site
user_site = site.getusersitepackages()
if user_site not in sys.path and os.path.exists(user_site):
    sys.path.insert(0, user_site)

try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    print(f"Error importing required package for plotting: {exc}")
    sys.exit(1)


def parse_float_col(series: pd.Series) -> np.ndarray:
    """Parse timing column, converting to float ms (scaling seconds to ms if needed)."""
    vals = pd.to_numeric(series, errors="coerce").values
    # If maximum value is < 100, assume values are in seconds and convert to ms
    valid_vals = vals[~np.isnan(vals)]
    if len(valid_vals) > 0 and np.nanmax(valid_vals) < 100.0:
        vals = vals * 1000.0
    return vals


def generate_plots(tsv_path: Path, output_dir: Path) -> int:
    if not tsv_path.exists():
        print(f"Benchmark summary file not found at {tsv_path}, skipping plot generation.")
        return 0

    # Determine separator by sniffing first line (tab vs comma)
    with open(tsv_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
    sep = "\t" if "\t" in first_line else ","
    df = pd.read_csv(tsv_path, sep=sep)

    # Determine dimension
    is_2d = "signal_width" in df.columns or "signal_height" in df.columns or "2d" in tsv_path.stem.lower()
    dim = "2d" if is_2d else "1d"
    dim_title = "2D" if is_2d else "1D"

    # Identify boundary mode column
    mode_col = "boundary_mode"
    if "lwt_boundary_mode" in df.columns:
        mode_col = "lwt_boundary_mode"
    elif "tt_boundary_mode" in df.columns:
        mode_col = "tt_boundary_mode"

    # Identify X axis column (signal length for 1D, signal width for 2D)
    x_col = "signal_width" if is_2d and "signal_width" in df.columns else ("length" if "length" in df.columns else "signal_length")

    # Identify backend timing columns
    pywt_col = [c for c in ["pywt_mean_s", "pywt_dwt_ms", "pywt_idwt_ms"] if c in df.columns]
    tt_col = [c for c in ["tt_wavelet_mean_s", "tt_mean_s", "standalone_dwt_ms", "standalone_idwt_ms"] if c in df.columns]
    ttnn_col = [c for c in ["ttnn_mean_s", "ttnn_dwt_ms", "ttnn_idwt_ms"] if c in df.columns]

    if not pywt_col or not tt_col or not ttnn_col:
        print(f"Missing one or more required backend timing columns in {tsv_path}")
        return 0

    pywt_c = pywt_col[0]
    tt_c = tt_col[0]
    ttnn_c = ttnn_col[0]

    wavelets = df["wavelet"].unique()
    modes = df[mode_col].unique()
    transforms = df["transform"].unique() if "transform" in df.columns else ["lwt"]

    created_count = 0
    missing_data_warnings = 0

    for wavelet in wavelets:
        w_df = df[df["wavelet"] == wavelet]
        plot_dir = output_dir / dim / wavelet.replace(".", "")
        plot_dir.mkdir(parents=True, exist_ok=True)

        for transform in transforms:
            t_df = w_df[w_df["transform"] == transform] if "transform" in w_df.columns else w_df
            for mode in modes:
                m_df = t_df[t_df[mode_col] == mode]
                if m_df.empty:
                    continue

                m_df = m_df.sort_values(x_col)
                x_vals = m_df[x_col].values

                pywt_ms = parse_float_col(m_df[pywt_c])
                tt_ms = parse_float_col(m_df[tt_c])
                ttnn_ms = parse_float_col(m_df[ttnn_c])

                # Check data presence for all 3 backends
                has_pywt = not np.all(np.isnan(pywt_ms)) and np.nanmax(pywt_ms) > 0
                has_tt = not np.all(np.isnan(tt_ms)) and np.nanmax(tt_ms) > 0
                has_ttnn = not np.all(np.isnan(ttnn_ms)) and np.nanmax(ttnn_ms) > 0

                if not (has_pywt and has_tt and has_ttnn):
                    missing_data_warnings += 1
                    missing = []
                    if not has_pywt: missing.append("PyWavelets")
                    if not has_tt: missing.append("tt-wavelet")
                    if not has_ttnn: missing.append("ttnn-wavelet")
                    print(f"Warning: Data missing for backend(s) {missing} on {dim_title} {transform} {wavelet} {mode}")

                plt.figure(figsize=(7, 4.5))

                if has_pywt:
                    plt.plot(x_vals, pywt_ms, label="PyWavelets", color="#1f77b4", linewidth=2.0, marker="o", markersize=3)
                if has_tt:
                    plt.plot(x_vals, tt_ms, label="tt-wavelet", color="#ff7f0e", linewidth=2.0, marker="s", markersize=3)
                if has_ttnn:
                    plt.plot(x_vals, ttnn_ms, label="ttnn-wavelet", color="#2ca02c", linewidth=2.0, marker="^", markersize=3)

                plt.yscale("log")
                x_label = "Matrix width (height = 1000)" if is_2d else "Signal length"
                plt.xlabel(x_label, fontsize=11, fontweight="bold")
                plt.ylabel("Runtime (ms, log scale)", fontsize=11, fontweight="bold")
                
                title_trans = transform.upper()
                title_suffix = " — 1000×W" if is_2d else ""
                plt.title(f"{dim_title} {title_trans} — {wavelet} — {mode}{title_suffix}", fontsize=12, fontweight="bold")

                plt.grid(True, which="both", linestyle=":", alpha=0.7)
                plt.legend(frameon=True, facecolor="white", edgecolor="none")
                plt.tight_layout()

                base_filename = f"{dim}_{transform}_{wavelet}_{mode}"
                png_path = plot_dir / f"{base_filename}.png"
                svg_path = plot_dir / f"{base_filename}.svg"

                plt.savefig(png_path, dpi=200)
                plt.savefig(svg_path, format="svg", dpi=200)
                plt.close()
                created_count += 1

    print(f"[Plot Generator Complete] Generated {created_count} PNG/SVG benchmark plots in {output_dir}")
    if missing_data_warnings > 0:
        print(f"[Plot Generator Warning] {missing_data_warnings} plot configurations had missing backend data.")
    return created_count


def main():
    parser = argparse.ArgumentParser(description="Generate individual benchmark log-scale line plots.")
    parser.add_argument("--summary-file", type=Path, help="Path to summary CSV/TSV.")
    parser.add_argument("--summary-tsv", type=Path, help="Path to summary TSV (legacy flag).")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/performance/plots"), help="Output plots directory.")
    args = parser.parse_args()

    summary_file = args.summary_file or args.summary_tsv
    if not summary_file:
        # Default fallback check
        for candidate in [Path("benchmarks/performance/1d/summary_1d.tsv"), Path("benchmarks/performance/2d/summary_2d.tsv"), Path("tt_wavelet_timings.csv"), Path("tt_wavelet_timings_2d.csv")]:
            if candidate.exists():
                generate_plots(candidate, args.output_dir)
    else:
        generate_plots(summary_file, args.output_dir)


if __name__ == "__main__":
    main()
