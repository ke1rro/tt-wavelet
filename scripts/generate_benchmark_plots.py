#!/usr/bin/env python3
"""
Generate Performance Benchmark Line Plots (Log-Scale SVG Files).

Creates individual SVG plots for each wavelet, dimension, transform (DWT/ILWT), and boundary mode:
- Output path structure: benchmarks/performance/plots/{wavelet}/{1d|2d}/{lwt|ilwt}_{boundary_mode}.svg
- Format: SVG (Scalable Vector Graphics), log-scale axes, PyWT vs Standalone vs TTNN.
"""

import argparse
from pathlib import Path
import sys

try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    print(f"Error importing required package for plotting: {exc}")
    sys.exit(1)


def generate_plots(tsv_path: Path, output_dir: Path):
    if not tsv_path.exists():
        print(f"Summary TSV not found at {tsv_path}, skipping plot generation.")
        return

    dim = "2d" if "2d" in tsv_path.stem.lower() else "1d"
    dim_title = "2D" if dim == "2d" else "1D"
    df = pd.read_csv(tsv_path, sep="\t")

    wavelets = df["wavelet"].unique()
    modes = df["boundary_mode"].unique()
    lengths = sorted(df["length"].unique())

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    created_count = 0
    for wavelet in wavelets:
        w_df = df[df["wavelet"] == wavelet]
        w_dir = output_dir / wavelet.replace(".", "") / dim
        w_dir.mkdir(parents=True, exist_ok=True)

        for mode in modes:
            m_df = w_df[w_df["boundary_mode"] == mode]
            if m_df.empty:
                continue

            # Sort by signal length / dimension
            m_df = m_df.sort_values("length")
            lens = m_df["length"].values

            # 1. DWT Plot
            fig, ax = plt.subplots(figsize=(7.5, 5.0))
            
            py_dwt = m_df["pywt_dwt_ms"].values
            std_dwt = m_df["standalone_dwt_ms"].values
            tt_dwt = m_df["ttnn_dwt_ms"].values

            if any(v > 0 for v in py_dwt):
                ax.plot(lens, [v if v > 0 else np.nan for v in py_dwt], "o-", label="PyWavelets (CPU)", color="#4c72b0", linewidth=2, markersize=6)
            if any(v > 0 for v in std_dwt):
                ax.plot(lens, [v if v > 0 else np.nan for v in std_dwt], "s--", label="Standalone C++", color="#dd8452", linewidth=2, markersize=6)
            if any(v > 0 for v in tt_dwt):
                ax.plot(lens, [v if v > 0 else np.nan for v in tt_dwt], "^-", label="TTNN (Device)", color="#2ca02c", linewidth=2.5, markersize=7)

            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.set_title(f"{dim_title} DWT {wavelet} — Mode: {mode} (Log-Scale)", fontsize=12, fontweight="bold")
            ax.set_xlabel("Signal Length N / Dimension (log)", fontsize=10)
            ax.set_ylabel("Median Latency (ms, log scale)", fontsize=10)
            
            ax.set_xticks(lens)
            ax.set_xticklabels([f"{l//1000}k" if l >= 1000 else str(l) for l in lens])
            ax.legend(fontsize=9)
            ax.grid(True, which="both", linestyle="--", alpha=0.5)

            fig.tight_layout()
            dwt_svg_path = w_dir / f"lwt_{mode}.svg"
            fig.savefig(dwt_svg_path, format="svg", dpi=300)
            plt.close(fig)
            created_count += 1

            # 2. ILWT Plot
            fig, ax = plt.subplots(figsize=(7.5, 5.0))
            
            py_idwt = m_df["pywt_idwt_ms"].values
            std_idwt = m_df["standalone_idwt_ms"].values
            tt_idwt = m_df["ttnn_idwt_ms"].values

            if any(v > 0 for v in py_idwt):
                ax.plot(lens, [v if v > 0 else np.nan for v in py_idwt], "o-", label="PyWavelets (CPU)", color="#4c72b0", linewidth=2, markersize=6)
            if any(v > 0 for v in std_idwt):
                ax.plot(lens, [v if v > 0 else np.nan for v in std_idwt], "s--", label="Standalone C++", color="#dd8452", linewidth=2, markersize=6)
            if any(v > 0 for v in tt_idwt):
                ax.plot(lens, [v if v > 0 else np.nan for v in tt_idwt], "^-", label="TTNN (Device)", color="#2ca02c", linewidth=2.5, markersize=7)

            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.set_title(f"{dim_title} ILWT {wavelet} — Mode: {mode} (Log-Scale)", fontsize=12, fontweight="bold")
            ax.set_xlabel("Signal Length N / Dimension (log)", fontsize=10)
            ax.set_ylabel("Median Latency (ms, log scale)", fontsize=10)
            
            ax.set_xticks(lens)
            ax.set_xticklabels([f"{l//1000}k" if l >= 1000 else str(l) for l in lens])
            ax.legend(fontsize=9)
            ax.grid(True, which="both", linestyle="--", alpha=0.5)

            fig.tight_layout()
            idwt_svg_path = w_dir / f"ilwt_{mode}.svg"
            fig.savefig(idwt_svg_path, format="svg", dpi=300)
            plt.close(fig)
            created_count += 1

    print(f"[SVG Plots Generated] Created {created_count} SVG plots in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate individual SVG benchmark log-scale line plots.")
    parser.add_argument("--summary-tsv", type=Path, default=Path("benchmarks/performance/summary_1d.tsv"), help="Path to summary TSV.")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/performance/plots"), help="Output plots directory.")
    args = parser.parse_args()

    generate_plots(args.summary_tsv, args.output_dir)


if __name__ == "__main__":
    main()
