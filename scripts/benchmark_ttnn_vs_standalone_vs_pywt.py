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
import tempfile
import time
from pathlib import Path
import numpy as np


import site
user_site = site.getusersitepackages()
if user_site not in sys.path and os.path.exists(user_site):
    sys.path.insert(0, user_site)

# Ensure venv packages are available
if "VENV_PYTHON" in os.environ:
    venv_python = Path(os.environ["VENV_PYTHON"])
    if venv_python.exists() and Path(sys.executable) != venv_python:
        os.execv(str(venv_python), [str(venv_python), __file__, *sys.argv[1:]])

try:
    import pywt
except ImportError as exc:
    print(f"Missing required package: {exc}")
    sys.exit(1)

from tqdm import tqdm

DEFAULT_WAVELETS = ["db1", "bior1.3", "bior3.5", "bior3.9", "db3", "db6", "db7", "coif1", "coif2", "coif5", "coif17", "sym3", "sym6", "sym10", "rbio3.5", "rbio6.8"]
ALL_BOUNDARY_MODES = ["symmetric", "zero", "constant", "periodic", "antisymmetric", "smooth", "reflect", "antireflect"]


def measure_pywt_timing(wavelet_name, boundary_mode, signal_len, repeats=20):
    sig = np.sin(np.linspace(0, 10 * np.pi, signal_len)).astype(np.float32)
    pywt_mode = boundary_mode if boundary_mode in pywt.Modes.modes else "symmetric"
    
    # Warmup
    for _ in range(2):
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
    res_dwt = subprocess.run(
        ["scripts/set_env.sh", "build/lwt", "--benchmark", "--repeats", str(repeats), "--boundary-mode", boundary_mode, "--length", str(signal_len), wavelet_name],
        capture_output=True, text=True, env=os.environ
    )
    res_idwt = subprocess.run(
        ["scripts/set_env.sh", "build/lwt", "--inverse", "--benchmark", "--repeats", str(repeats), "--boundary-mode", boundary_mode, "--length", str(signal_len), wavelet_name],
        capture_output=True, text=True, env=os.environ
    )
    
    dwt_ms = 0.0
    idwt_ms = 0.0
    for line in (res_dwt.stdout + "\n" + res_dwt.stderr).splitlines():
        if "lwt_execution_time_ms:" in line:
            dwt_ms = float(line.split(":")[1].strip())
            break
    for line in (res_idwt.stdout + "\n" + res_idwt.stderr).splitlines():
        if "lwt_execution_time_ms:" in line or "ilwt_execution_time_ms:" in line:
            idwt_ms = float(line.split(":")[1].strip())
            break

    return dwt_ms, idwt_ms


def measure_ttnn_timing(wavelet_name, boundary_mode, signal_len, device, repeats=20):
    try:
        import torch
        import ttnn
        import ttnn._ttnn as _ttnn
        s_sticks = (signal_len + 31) // 32
        padded_len = s_sticks * 32
        sig = torch.sin(torch.linspace(0, 10 * torch.pi, padded_len, dtype=torch.float32)).reshape(s_sticks, 32)
        inp = _ttnn.tensor.Tensor(sig, _ttnn.tensor.DataType.FLOAT32).to(_ttnn.tensor.Layout.ROW_MAJOR).to(device)
        
        # Warmup JIT
        app, det = _ttnn.operations.dwt(inp, wavelet_name, boundary_mode=boundary_mode)
        rec = _ttnn.operations.idwt(app, det, wavelet_name, padded_len, boundary_mode=boundary_mode)
        _ttnn.device.synchronize_device(device)

        # 1D DWT Trace Capture for pure device hardware execution time
        trace_lwt = _ttnn.operations.begin_trace_capture(device)
        app, det = _ttnn.operations.dwt(inp, wavelet_name, boundary_mode=boundary_mode)
        _ttnn.operations.end_trace_capture(device, trace_lwt)
        _ttnn.device.synchronize_device(device)

        t0 = time.perf_counter()
        for _ in range(repeats):
            _ttnn.operations.execute_trace(device, trace_lwt)
        _ttnn.device.synchronize_device(device)
        t1 = time.perf_counter()
        dwt_ms = (t1 - t0) * 1000.0 / repeats
        _ttnn.operations.release_trace(device, trace_lwt)

        # 1D ILWT Trace Capture
        trace_ilwt = _ttnn.operations.begin_trace_capture(device)
        rec = _ttnn.operations.idwt(app, det, wavelet_name, padded_len, boundary_mode=boundary_mode)
        _ttnn.operations.end_trace_capture(device, trace_ilwt)
        _ttnn.device.synchronize_device(device)

        t0 = time.perf_counter()
        for _ in range(repeats):
            _ttnn.operations.execute_trace(device, trace_ilwt)
        _ttnn.device.synchronize_device(device)
        t1 = time.perf_counter()
        idwt_ms = (t1 - t0) * 1000.0 / repeats
        _ttnn.operations.release_trace(device, trace_ilwt)

        return dwt_ms, idwt_ms
    except Exception as exc:
        print(f"[TTNN 1D Warn] {wavelet_name} {boundary_mode} N={signal_len}: {exc}")
        return 0.0, 0.0


def measure_pywt_timing_2d(wavelet_name, boundary_mode, h, w, repeats=20):
    sig_2d = np.sin(np.linspace(0, 10 * np.pi, h * w)).reshape(h, w).astype(np.float32)
    pywt_mode = boundary_mode if boundary_mode in pywt.Modes.modes else "symmetric"
    
    for _ in range(2):
        coeffs = pywt.dwt2(sig_2d, wavelet_name, mode=pywt_mode)
        rec = pywt.idwt2(coeffs, wavelet_name, mode=pywt_mode)
        
    t0 = time.perf_counter()
    for _ in range(repeats):
        coeffs = pywt.dwt2(sig_2d, wavelet_name, mode=pywt_mode)
    t1 = time.perf_counter()
    dwt_ms = (t1 - t0) * 1000.0 / repeats

    t0 = time.perf_counter()
    for _ in range(repeats):
        rec = pywt.idwt2(coeffs, wavelet_name, mode=pywt_mode)
    t1 = time.perf_counter()
    idwt_ms = (t1 - t0) * 1000.0 / repeats

    return dwt_ms, idwt_ms


def measure_standalone_timing_2d(wavelet_name, boundary_mode, h, w, repeats=20):
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as tmp:
        input_path = tmp.name
        np.savetxt(input_path, np.sin(np.linspace(0, 10 * np.pi, h * w, dtype=np.float32)))

    try:
        res_dwt = subprocess.run(
            [
                "scripts/set_env.sh",
                "build/lwt_2d",
                "--boundary-mode", boundary_mode,
                "--cores", "64",
                "--benchmark",
                "--repeats", str(repeats),
                "--warmup-runs", "1",
                wavelet_name,
                str(h),
                str(w),
                input_path,
            ],
            capture_output=True, text=True, env=os.environ
        )
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

    dwt_ms = 0.0
    for line in (res_dwt.stdout + "\n" + res_dwt.stderr).splitlines():
        if "lwt_2d_execution_time_ms:" in line or "execution_time_ms:" in line:
            dwt_ms = float(line.split(":")[1].strip())
            break

    return dwt_ms, dwt_ms


def measure_ttnn_timing_2d(wavelet_name, boundary_mode, h, w, device, repeats=20):
    try:
        import torch
        import ttnn
        sig_2d = torch.sin(torch.linspace(0, 10 * torch.pi, h * w, dtype=torch.float32)).reshape(h, w)
        inp = ttnn.from_torch(sig_2d, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        
        # Warmup JIT
        res = ttnn.dwt_2d(inp, wavelet_name, boundary_mode=boundary_mode)
        rec = ttnn.idwt_2d(res[0], res[1], res[2], res[3], wavelet_name, [h, w], boundary_mode=boundary_mode)
        ttnn.synchronize_device(device)
        
        # 2D DWT Trace Capture for pure device hardware execution time
        trace_lwt = ttnn.begin_trace_capture(device)
        res = ttnn.dwt_2d(inp, wavelet_name, boundary_mode=boundary_mode)
        ttnn.end_trace_capture(device, trace_lwt)
        ttnn.synchronize_device(device)

        t0 = time.perf_counter()
        for _ in range(repeats):
            ttnn.execute_trace(device, trace_lwt)
        ttnn.synchronize_device(device)
        t1 = time.perf_counter()
        dwt_ms = (t1 - t0) * 1000.0 / repeats
        ttnn.release_trace(device, trace_lwt)

        # 2D ILWT Trace Capture
        trace_ilwt = ttnn.begin_trace_capture(device)
        rec = ttnn.idwt_2d(res[0], res[1], res[2], res[3], wavelet_name, [h, w], boundary_mode=boundary_mode)
        ttnn.end_trace_capture(device, trace_ilwt)
        ttnn.synchronize_device(device)

        t0 = time.perf_counter()
        for _ in range(repeats):
            ttnn.execute_trace(device, trace_ilwt)
        ttnn.synchronize_device(device)
        t1 = time.perf_counter()
        idwt_ms = (t1 - t0) * 1000.0 / repeats
        ttnn.release_trace(device, trace_ilwt)

        return dwt_ms, idwt_ms
    except Exception as exc:
        print(f"[TTNN 2D Warn] {wavelet_name} {boundary_mode} {h}x{w}: {exc}")
        return 0.0, 0.0


def main():
    parser = argparse.ArgumentParser(description="Performance Benchmark: PyWavelets vs Standalone vs TTNN")
    parser.add_argument("--dim", choices=["1d", "2d"], default="1d", help="Dimension of transforms to benchmark.")
    parser.add_argument("--backends", nargs="*", default=["ttnn", "standalone", "pywt"], help="Backends to benchmark.")
    parser.add_argument("--schemes", nargs="*", default=DEFAULT_WAVELETS, help="Wavelet schemes to benchmark.")
    parser.add_argument("--length-start", type=int, default=10000, help="Signal length start.")
    parser.add_argument("--length-stop", type=int, default=100000, help="Signal length stop.")
    parser.add_argument("--length-step", type=int, default=30000, help="Signal length step.")
    parser.add_argument("--boundary-modes", nargs="*", default=ALL_BOUNDARY_MODES, help="Boundary modes.")
    parser.add_argument("--repeats", type=int, default=20, help="Number of benchmark repeats.")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/performance"), help="Output benchmark directory.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    backends = set(args.backends)

    lengths = list(range(args.length_start, args.length_stop + 1, args.length_step))
    print(f"Starting performance benchmark ({args.dim.upper()}) across {len(args.schemes)} wavelets, {len(lengths)} lengths, {len(args.boundary_modes)} modes...")

    pywt_res = {}
    std_res = {}
    ttnn_res = {}

    if args.dim == "2d":
        pywt_fn = lambda w, m, n: measure_pywt_timing_2d(w, m, n, n, repeats=args.repeats)
        std_fn = lambda w, m, n: measure_standalone_timing_2d(w, m, n, n, repeats=args.repeats)
        ttnn_fn = lambda w, m, n, dev: measure_ttnn_timing_2d(w, m, n, n, dev, repeats=args.repeats)
    else:
        pywt_fn = lambda w, m, n: measure_pywt_timing(w, m, n, repeats=args.repeats)
        std_fn = lambda w, m, n: measure_standalone_timing(w, m, n, repeats=args.repeats)
        ttnn_fn = lambda w, m, n, dev: measure_ttnn_timing(w, m, n, dev, repeats=args.repeats)

    total_runs = len(args.schemes) * len(args.boundary_modes) * len(lengths)

    # Phase 1: PyWT Benchmark (CPU)
    if "pywt" in backends:
        print(f"\n--- Running PyWavelets (CPU) {args.dim.upper()} Benchmarks ---")
        pbar = tqdm(total=total_runs, desc=f"PyWT ({args.dim.upper()})")
        for wavelet in args.schemes:
            for mode in args.boundary_modes:
                for N in lengths:
                    pywt_res[(wavelet, mode, N)] = pywt_fn(wavelet, mode, N)
                    pbar.update(1)
        pbar.close()

    # Phase 2: Standalone C++ Benchmark (Device process, run without active TTNN context)
    if "standalone" in backends:
        print(f"\n--- Running Standalone C++ {args.dim.upper()} Benchmarks ---")
        pbar = tqdm(total=total_runs, desc=f"Standalone ({args.dim.upper()})")
        for wavelet in args.schemes:
            for mode in args.boundary_modes:
                for N in lengths:
                    std_res[(wavelet, mode, N)] = std_fn(wavelet, mode, N)
                    pbar.update(1)
        pbar.close()

    # Phase 3: TTNN Benchmark (Single open_device context)
    if "ttnn" in backends:
        import ttnn._ttnn as _ttnn
        device = _ttnn.device.open_device(device_id=0)
        pbar = tqdm(total=total_runs, desc=f"TTNN Device ({args.dim.upper()})")
        try:
            for wavelet in args.schemes:
                for mode in args.boundary_modes:
                    for N in lengths:
                        ttnn_res[(wavelet, mode, N)] = ttnn_fn(wavelet, mode, N, device)
                        pbar.update(1)
        finally:
            pbar.close()
            _ttnn.device.close_device(device)

    # Phase 4: Write summaries and logs
    summary_name = f"summary_{args.dim}.tsv"
    summary_tsv = args.output_dir / summary_name
    with open(summary_tsv, "w") as f_tsv:
        f_tsv.write("wavelet\tlength\tboundary_mode\tpywt_dwt_ms\tpywt_idwt_ms\tstandalone_dwt_ms\tstandalone_idwt_ms\tttnn_dwt_ms\tttnn_idwt_ms\n")

        for wavelet in args.schemes:
            w_dir = args.output_dir / wavelet.replace(".", "")
            w_dir.mkdir(parents=True, exist_ok=True)
            log_path = w_dir / f"{wavelet}_timings_{args.dim}.log"

            with open(log_path, "w") as f_log:
                f_log.write(f"=== Timing Benchmark Log for {wavelet} ({args.dim.upper()}) ===\n")
                for mode in args.boundary_modes:
                    for N in lengths:
                        pywt_dwt, pywt_idwt = pywt_res.get((wavelet, mode, N), (0.0, 0.0))
                        std_dwt, std_idwt = std_res.get((wavelet, mode, N), (0.0, 0.0))
                        ttnn_dwt, ttnn_idwt = ttnn_res.get((wavelet, mode, N), (0.0, 0.0))

                        log_line = (
                            f"N/Dim={N:7d} | Mode={mode:13s} | "
                            f"PyWT (DWT/IDWT): {pywt_dwt:7.3f}/{pywt_idwt:7.3f} ms | "
                            f"Standalone: {std_dwt:7.3f}/{std_idwt:7.3f} ms | "
                            f"TTNN: {ttnn_dwt:7.3f}/{ttnn_idwt:7.3f} ms\n"
                        )
                        f_log.write(log_line)
                        print(f"[{wavelet:8s} {args.dim.upper()}] {log_line.strip()}")

                        f_tsv.write(
                            f"{wavelet}\t{N}\t{mode}\t{pywt_dwt:.4f}\t{pywt_idwt:.4f}\t"
                            f"{std_dwt:.4f}\t{std_idwt:.4f}\t{ttnn_dwt:.4f}\t{ttnn_idwt:.4f}\n"
                        )

    print("\n" + "=" * 80)
    print(f"BENCHMARK ({args.dim.upper()}) COMPLETE. Results saved to: {args.output_dir}")
    print(f"Summary TSV: {summary_tsv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
