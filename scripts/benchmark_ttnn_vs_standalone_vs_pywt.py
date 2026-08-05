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


def main():
    parser = argparse.ArgumentParser(description="Performance Benchmark: PyWavelets vs Standalone vs TTNN")
    parser.add_argument("--schemes", nargs="*", default=DEFAULT_WAVELETS, help="Wavelet schemes to benchmark.")
    parser.add_argument("--length-start", type=int, default=10000, help="Signal length start.")
    parser.add_argument("--length-stop", type=int, default=100000, help="Signal length stop.")
    parser.add_argument("--length-step", type=int, default=30000, help="Signal length step.")
    parser.add_argument("--boundary-modes", nargs="*", default=ALL_BOUNDARY_MODES, help="Boundary modes.")
    parser.add_argument("--repeats", type=int, default=20, help="Number of benchmark repeats.")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/benchmark_results"), help="Output benchmark directory.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = ttnn.open_device(device_id=0)

    lengths = list(range(args.length_start, args.length_stop + 1, args.length_step))
    print(f"Starting performance benchmark across {len(args.schemes)} wavelets, {len(lengths)} lengths, {len(args.boundary_modes)} modes...")

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
                        pywt_dwt, pywt_idwt = measure_pywt_timing(wavelet, mode, N, repeats=args.repeats)
                        std_dwt, std_idwt = measure_standalone_timing(wavelet, mode, N, repeats=args.repeats)
                        ttnn_dwt, ttnn_idwt = measure_ttnn_timing(wavelet, mode, N, device, repeats=args.repeats)

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

    ttnn.close_device(device)
    print("\n" + "=" * 80)
    print(f"BENCHMARK COMPLETE. Results saved to: {args.output_dir}")
    print(f"Summary TSV: {summary_tsv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
