#!/usr/bin/env python3
"""
Generate Performance Benchmark Line Plots (Log-Scale SVG Files).

Creates individual SVG plots for each wavelet, dimension, transform (DWT/ILWT), and boundary mode:
- Output path structure: benchmarks/performance/plots/{wavelet}/{1d|2d}/{lwt|ilwt}_{boundary_mode}.svg
- Format: SVG (Scalable Vector Graphics), log-scale axes, PyWT vs Standalone vs TTNN.
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
            plt.figure(figsize=(7, 4.5))
            
            py_dwt = m_df["pywt_dwt_ms"].values
            std_dwt = m_df["standalone_dwt_ms"].values
            tt_dwt = m_df["ttnn_dwt_ms"].values

            if any(v > 0 for v in py_dwt):
                plt.plot(lens, [v if v > 0 else np.nan for v in py_dwt], label="PyWavelets")
            if any(v > 0 for v in std_dwt):
                plt.plot(lens, [v if v > 0 else np.nan for v in std_dwt], label="tt-wavelet")
            if any(v > 0 for v in tt_dwt):
                plt.plot(lens, [v if v > 0 else np.nan for v in tt_dwt], label="ttnn-wavelet")

            plt.yscale("log")
            plt.xlabel("Signal width" if dim == "2d" else "Signal length")
            plt.ylabel("Runtime (ms, log scale)")
            plt.title(f"{dim_title} {wavelet} DWT runtime vs {'signal width' if dim == '2d' else 'signal length'}")

            plt.grid(True, which="both", linestyle=":")
            plt.legend()
            plt.tight_layout()

            dwt_png_path = w_dir / f"lwt_{mode}.png"
            plt.savefig(dwt_png_path, dpi=200)
            dwt_svg_path = w_dir / f"lwt_{mode}.svg"
            plt.savefig(dwt_svg_path, format="svg", dpi=200)
            plt.close()
            created_count += 1

            # 2. ILWT Plot
            plt.figure(figsize=(7, 4.5))
            
            py_idwt = m_df["pywt_idwt_ms"].values
            std_idwt = m_df["standalone_idwt_ms"].values
            tt_idwt = m_df["ttnn_idwt_ms"].values

            if any(v > 0 for v in py_idwt):
                plt.plot(lens, [v if v > 0 else np.nan for v in py_idwt], label="PyWavelets")
            if any(v > 0 for v in std_idwt):
                plt.plot(lens, [v if v > 0 else np.nan for v in std_idwt], label="tt-wavelet")
            if any(v > 0 for v in tt_idwt):
                plt.plot(lens, [v if v > 0 else np.nan for v in tt_idwt], label="ttnn-wavelet")

            plt.yscale("log")
            plt.xlabel("Signal width" if dim == "2d" else "Signal length")
            plt.ylabel("Runtime (ms, log scale)")
            plt.title(f"{dim_title} {wavelet} ILWT runtime vs {'signal width' if dim == '2d' else 'signal length'}")

            plt.grid(True, which="both", linestyle=":")
            plt.legend()
            plt.tight_layout()

            idwt_png_path = w_dir / f"ilwt_{mode}.png"
            plt.savefig(idwt_png_path, dpi=200)
            idwt_svg_path = w_dir / f"ilwt_{mode}.svg"
            plt.savefig(idwt_svg_path, format="svg", dpi=200)
            plt.close()
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
