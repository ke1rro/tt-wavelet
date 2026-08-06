#!/usr/bin/env python3
"""
Performance Benchmark Suite: PyWavelets (CPU), Standalone tt-wavelet (C++ Device), TTNN ttnn-wavelet (Device).

Modeled after overnight boundary timing sweeps:
Generates directory structure, TSV summaries, timing logs, and comparative charts.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

# Ensure venv packages are available
VENV_PYTHON = Path("/home/user/tt-metal/python_env/bin/python3")
if VENV_PYTHON.exists() and Path(sys.executable) != VENV_PYTHON:
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), __file__, *sys.argv[1:]])

try:
    import torch
    import pywt
    import ttnn
except ImportError as exc:
    print(f"Missing required package: {exc}")
    sys.exit(1)

DEFAULT_WAVELETS = ["db1", "bior1.3", "bior3.5", "bior3.9", "db3", "db6", "db7", "coif1", "coif2", "coif5", "coif17", "sym3", "sym6", "sym10", "rbio3.5", "rbio6.8"]
ALL_BOUNDARY_MODES = ["symmetric", "zero", "constant", "periodic", "antisymmetric", "smooth", "reflect", "antireflect"]


def measure_pywt_timing(wavelet_name, boundary_mode, signal_len, repeats=20):
    sig = np.sin(np.linspace(0, 10 * np.pi, signal_len)).astype(np.float32)
    pywt_mode = boundary_mode if boundary_mode in pywt.Modes.modes else "symmetric"
    
    # Warmup
    for _ in range(3):
        app, det = pywt.dwt(sig, wavelet_name, mode=pywt_mode)
        rec = pywt.idwt(app, det, wavelet_name, mode=pywt_mode)
        
    t0 = time.perf_counter()
    for _ in range(repeats):
        app, det = pywt.dwt(sig, wavelet_name, mode=pywt_mode)
    t1 = time.perf_counter()
    dwt_ms = (t1 - t0) * 1000.0 / repeats

    t0 = time.perf_counter()
    for _ in range(repeats):
        rec = pywt.idwt(app, det, wavelet_name, mode=pywt_mode)
    t1 = time.perf_counter()
    idwt_ms = (t1 - t0) * 1000.0 / repeats

    return dwt_ms, idwt_ms


def measure_standalone_timing(wavelet_name, boundary_mode, signal_len, repeats=20):
    # Standalone C++ executable
    res_dwt = subprocess.run(
        ["build/lwt", "--benchmark", "--repeats", str(repeats), "--boundary-mode", boundary_mode, "--length", str(signal_len), wavelet_name],
        capture_output=True, text=True
    )
    res_idwt = subprocess.run(
        ["build/lwt", "--inverse", "--benchmark", "--repeats", str(repeats), "--boundary-mode", boundary_mode, "--length", str(signal_len), wavelet_name],
        capture_output=True, text=True
    )
    
    dwt_ms = 0.0
    idwt_ms = 0.0
    for line in res_dwt.stdout.splitlines():
        if "lwt_execution_time_ms:" in line:
            dwt_ms = float(line.split(":")[1].strip())
    for line in res_idwt.stdout.splitlines():
        if "ilwt_execution_time_ms:" in line:
            idwt_ms = float(line.split(":")[1].strip())

    return dwt_ms, idwt_ms


def measure_ttnn_timing(wavelet_name, boundary_mode, signal_len, device, repeats=20):
    sig = torch.sin(torch.linspace(0, 10 * torch.pi, signal_len, dtype=torch.float32))
    inp = ttnn.from_torch(sig, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    
    # Warmup
    for _ in range(3):
        app, det = ttnn.dwt(inp, wavelet_name, boundary_mode=boundary_mode)
        rec = ttnn.idwt(app, det, wavelet_name, signal_len, boundary_mode=boundary_mode)
        ttnn.synchronize_device(device)
        
    t0 = time.perf_counter()
    for _ in range(repeats):
        app, det = ttnn.dwt(inp, wavelet_name, boundary_mode=boundary_mode)
    ttnn.synchronize_device(device)
    t1 = time.perf_counter()
    dwt_ms = (t1 - t0) * 1000.0 / repeats

    t0 = time.perf_counter()
    for _ in range(repeats):
        rec = ttnn.idwt(app, det, wavelet_name, signal_len, boundary_mode=boundary_mode)
    ttnn.synchronize_device(device)
    t1 = time.perf_counter()
    idwt_ms = (t1 - t0) * 1000.0 / repeats

    return dwt_ms, idwt_ms


def generate_benchmark_plots(summary_tsv: Path, output_dir: Path):
    """Generate comparative PNG performance charts using matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping chart plotting.")
        return

    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    data = {}
    with open(summary_tsv, "r") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue
            w, N, mode, py_dwt, py_idwt, std_dwt, std_idwt, tt_dwt, tt_idwt = (
                parts[0], int(parts[1]), parts[2],
                float(parts[3]), float(parts[4]),
                float(parts[5]), float(parts[6]),
                float(parts[7]), float(parts[8])
            )
            key = (w, mode)
            if key not in data:
                data[key] = {"N": [], "py_dwt": [], "py_idwt": [], "std_dwt": [], "std_idwt": [], "tt_dwt": [], "tt_idwt": []}
            data[key]["N"].append(N)
            data[key]["py_dwt"].append(py_dwt)
            data[key]["py_idwt"].append(py_idwt)
            data[key]["std_dwt"].append(std_dwt)
            data[key]["std_idwt"].append(std_idwt)
            data[key]["tt_dwt"].append(tt_dwt)
            data[key]["tt_idwt"].append(tt_idwt)

    for (w, mode), res in data.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Performance Comparison: {w} ({mode})", fontsize=14, fontweight="bold")

        # DWT Plot
        if any(v > 0 for v in res["py_dwt"]):
            ax1.plot(res["N"], res["py_dwt"], "o-", label="PyWavelets (CPU)", color="#e74c3c", linewidth=2)
        if any(v > 0 for v in res["std_dwt"]):
            ax1.plot(res["N"], res["std_dwt"], "s--", label="Standalone tt-wavelet", color="#2ecc71", linewidth=2)
        if any(v > 0 for v in res["tt_dwt"]):
            ax1.plot(res["N"], res["tt_dwt"], "^-.", label="TTNN ttnn-wavelet", color="#3498db", linewidth=2)
        ax1.set_title("1D Forward DWT Execution Time")
        ax1.set_xlabel("Signal Length (N)")
        ax1.set_ylabel("Execution Time (ms)")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend()

        # IDWT Plot
        if any(v > 0 for v in res["py_idwt"]):
            ax2.plot(res["N"], res["py_idwt"], "o-", label="PyWavelets (CPU)", color="#e74c3c", linewidth=2)
        if any(v > 0 for v in res["std_idwt"]):
            ax2.plot(res["N"], res["std_idwt"], "s--", label="Standalone tt-wavelet", color="#2ecc71", linewidth=2)
        if any(v > 0 for v in res["tt_idwt"]):
            ax2.plot(res["N"], res["tt_idwt"], "^-.", label="TTNN ttnn-wavelet", color="#3498db", linewidth=2)
        ax2.set_title("1D Inverse IDWT Execution Time")
        ax2.set_xlabel("Signal Length (N)")
        ax2.set_ylabel("Execution Time (ms)")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend()

        plt.tight_layout()
        chart_path = charts_dir / f"{w.replace('.', '')}_{mode}_performance.png"
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"Generated chart: {chart_path}")


def main():
    parser = argparse.ArgumentParser(description="Performance Benchmark: PyWavelets vs Standalone vs TTNN")
    parser.add_argument("--schemes", nargs="*", default=DEFAULT_WAVELETS, help="Wavelet schemes to benchmark.")
    parser.add_argument("--length-start", type=int, default=10000, help="Signal length start.")
    parser.add_argument("--length-stop", type=int, default=100000, help="Signal length stop.")
    parser.add_argument("--length-step", type=int, default=30000, help="Signal length step.")
    parser.add_argument("--boundary-modes", nargs="*", default=ALL_BOUNDARY_MODES, help="Boundary modes.")
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["ttnn", "standalone", "pywt"],
        default=["ttnn", "standalone", "pywt"],
        help="Selected backends to benchmark (default: ttnn standalone pywt)",
    )
    parser.add_argument("--repeats", type=int, default=20, help="Number of benchmark repeats.")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"), help="Output benchmark directory.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = ttnn.open_device(device_id=0) if "ttnn" in args.backends else None

    lengths = list(range(args.length_start, args.length_stop + 1, args.length_step))
    print(f"Starting performance benchmark across {len(args.schemes)} wavelets, {len(lengths)} lengths, {len(args.boundary_modes)} modes on backends: {args.backends}...")

    summary_tsv = args.output_dir / "summary.tsv"
    with open(summary_tsv, "w") as f_tsv:
        f_tsv.write("wavelet\tlength\tboundary_mode\tpywt_dwt_ms\tpywt_idwt_ms\tstandalone_dwt_ms\tstandalone_idwt_ms\tttnn_dwt_ms\tttnn_idwt_ms\n")

        for w_idx, wavelet in enumerate(args.schemes, 1):
            w_dir = args.output_dir / wavelet.replace(".", "")
            w_dir.mkdir(parents=True, exist_ok=True)
            log_path = w_dir / f"{wavelet}_timings.log"
            
            with open(log_path, "w") as f_log:
                f_log.write(f"=== Timing Benchmark Log for {wavelet} ===\n")
                
                for mode in args.boundary_modes:
                    for N in lengths:
                        pywt_dwt, pywt_idwt = measure_pywt_timing(wavelet, mode, N, repeats=args.repeats) if "pywt" in args.backends else (0.0, 0.0)
                        std_dwt, std_idwt = measure_standalone_timing(wavelet, mode, N, repeats=args.repeats) if "standalone" in args.backends else (0.0, 0.0)
                        ttnn_dwt, ttnn_idwt = measure_ttnn_timing(wavelet, mode, N, device, repeats=args.repeats) if "ttnn" in args.backends else (0.0, 0.0)

                        log_line = (
                            f"N={N:7d} | Mode={mode:13s} | "
                            f"PyWT (DWT/IDWT): {pywt_dwt:7.3f}/{pywt_idwt:7.3f} ms | "
                            f"Standalone: {std_dwt:7.3f}/{std_idwt:7.3f} ms | "
                            f"TTNN: {ttnn_dwt:7.3f}/{ttnn_idwt:7.3f} ms\n"
                        )
                        f_log.write(log_line)
                        print(f"[{wavelet:8s}] {log_line.strip()}")

                        f_tsv.write(
                            f"{wavelet}\t{N}\t{mode}\t{pywt_dwt:.4f}\t{pywt_idwt:.4f}\t"
                            f"{std_dwt:.4f}\t{std_idwt:.4f}\t{ttnn_dwt:.4f}\t{ttnn_idwt:.4f}\n"
                        )

    if device is not None:
        ttnn.close_device(device)

    # Generate comparative PNG plots
    generate_benchmark_plots(summary_tsv, args.output_dir)

    print("\n" + "=" * 80)
    print(f"BENCHMARK COMPLETE. Results saved to: {args.output_dir}")
    print(f"Summary TSV: {summary_tsv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
