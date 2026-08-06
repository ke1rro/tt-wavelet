#!/usr/bin/env python3
"""
Architecture-Independent Bringup, Precision Validation, and Performance Benchmark Suite.

Features:
- Auto-detects device architecture (Wormhole, Blackhole, etc.).
- Phase 1 (Precision): Tests all 106 schemes x 8 boundary modes on small signals (N=1024).
  Outputs saved to benchmarks/precision/.
- Phase 2 (Performance): Randomly selects 1 Small (Compact), 1 Medium, 1 Large scheme,
  runs all 8 boundary modes for signal lengths 100k to 1M (100k, 400k, 700k, 1M).
  Outputs saved to benchmarks/performance/ and charts in benchmarks/performance/plots/.
- Uses tqdm for real-time progress tracking across all loops.
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

# Ensure venv packages are available
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if "VENV_PYTHON" in os.environ:
    venv_python = Path(os.environ["VENV_PYTHON"])
    if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
        os.execv(str(venv_python), [str(venv_python), __file__, *sys.argv[1:]])

try:
    import torch
    import pywt
    import ttnn
    from tqdm import tqdm
    import matplotlib.pyplot as plt
except ImportError as exc:
    print(f"Error importing required package: {exc}")
    sys.exit(1)


# All 106 Wavelet Metadata Mapping
WAVELET_CATEGORIES = {
    "bior1.1": {"category": "Compact", "tap_size": 2, "num_steps": 5, "max_abs_coeff": 1.41421},
    "bior1.3": {"category": "Compact", "tap_size": 6, "num_steps": 4, "max_abs_coeff": 1.41421},
    "bior1.5": {"category": "Compact", "tap_size": 10, "num_steps": 4, "max_abs_coeff": 1.41421},
    "bior2.2": {"category": "Compact", "tap_size": 6, "num_steps": 5, "max_abs_coeff": 1.41421},
    "bior2.4": {"category": "Compact", "tap_size": 10, "num_steps": 5, "max_abs_coeff": 1.41421},
    "bior2.6": {"category": "Large / Sensitive", "tap_size": 14, "num_steps": 5, "max_abs_coeff": 1.41421},
    "bior2.8": {"category": "Large / Sensitive", "tap_size": 18, "num_steps": 5, "max_abs_coeff": 1.41421},
    "bior3.1": {"category": "Compact", "tap_size": 4, "num_steps": 5, "max_abs_coeff": 1.125},
    "bior3.3": {"category": "Compact", "tap_size": 8, "num_steps": 6, "max_abs_coeff": 2.12132},
    "bior3.5": {"category": "Compact", "tap_size": 12, "num_steps": 6, "max_abs_coeff": 2.12132},
    "bior3.7": {"category": "Large / Sensitive", "tap_size": 16, "num_steps": 6, "max_abs_coeff": 2.12132},
    "bior3.9": {"category": "Large / Sensitive", "tap_size": 20, "num_steps": 6, "max_abs_coeff": 2.12132},
    "bior4.4": {"category": "Compact", "tap_size": 10, "num_steps": 7, "max_abs_coeff": 1.58613},
    "bior5.5": {"category": "Compact", "tap_size": 12, "num_steps": 7, "max_abs_coeff": 5.58579},
    "bior6.8": {"category": "Large / Sensitive", "tap_size": 18, "num_steps": 9, "max_abs_coeff": 1.15131},
    "coif1": {"category": "Compact", "tap_size": 6, "num_steps": 7, "max_abs_coeff": 91.8006},
    "coif10": {"category": "Large / Sensitive", "tap_size": 60, "num_steps": 33, "max_abs_coeff": 2437.12},
    "coif11": {"category": "Large / Sensitive", "tap_size": 66, "num_steps": 37, "max_abs_coeff": 16170.8},
    "coif12": {"category": "Large / Sensitive", "tap_size": 72, "num_steps": 39, "max_abs_coeff": 2116.57},
    "coif13": {"category": "Large / Sensitive", "tap_size": 78, "num_steps": 43, "max_abs_coeff": 23489.6},
    "coif14": {"category": "Large / Sensitive", "tap_size": 84, "num_steps": 45, "max_abs_coeff": 3417.27},
    "coif15": {"category": "Large / Sensitive", "tap_size": 90, "num_steps": 49, "max_abs_coeff": 36169.8},
    "coif16": {"category": "Large / Sensitive", "tap_size": 96, "num_steps": 51, "max_abs_coeff": 4375.36},
    "coif17": {"category": "Large / Sensitive", "tap_size": 102, "num_steps": 55, "max_abs_coeff": 34912.6},
    "coif2": {"category": "Compact", "tap_size": 12, "num_steps": 9, "max_abs_coeff": 3.57823},
    "coif3": {"category": "Medium", "tap_size": 18, "num_steps": 13, "max_abs_coeff": 42.7863},
    "coif4": {"category": "Medium", "tap_size": 24, "num_steps": 15, "max_abs_coeff": 14.799},
    "coif5": {"category": "Large / Sensitive", "tap_size": 30, "num_steps": 19, "max_abs_coeff": 283.731},
    "coif6": {"category": "Large / Sensitive", "tap_size": 36, "num_steps": 21, "max_abs_coeff": 93.4704},
    "coif7": {"category": "Large / Sensitive", "tap_size": 42, "num_steps": 25, "max_abs_coeff": 1724.88},
    "coif8": {"category": "Large / Sensitive", "tap_size": 48, "num_steps": 27, "max_abs_coeff": 828.314},
    "coif9": {"category": "Large / Sensitive", "tap_size": 54, "num_steps": 31, "max_abs_coeff": 306199},
    "db1": {"category": "Compact", "tap_size": 2, "num_steps": 5, "max_abs_coeff": 1.41421},
    "db10": {"category": "Medium", "tap_size": 20, "num_steps": 13, "max_abs_coeff": 21.6996},
    "db11": {"category": "Medium", "tap_size": 22, "num_steps": 15, "max_abs_coeff": 1039.05},
    "db12": {"category": "Medium", "tap_size": 24, "num_steps": 15, "max_abs_coeff": 43.1261},
    "db13": {"category": "Large / Sensitive", "tap_size": 26, "num_steps": 17, "max_abs_coeff": 2596.18},
    "db14": {"category": "Large / Sensitive", "tap_size": 28, "num_steps": 17, "max_abs_coeff": 85.6274},
    "db15": {"category": "Large / Sensitive", "tap_size": 30, "num_steps": 19, "max_abs_coeff": 6283.63},
    "db16": {"category": "Large / Sensitive", "tap_size": 32, "num_steps": 19, "max_abs_coeff": 170.158},
    "db17": {"category": "Large / Sensitive", "tap_size": 34, "num_steps": 21, "max_abs_coeff": 15893.3},
    "db18": {"category": "Large / Sensitive", "tap_size": 36, "num_steps": 21, "max_abs_coeff": 487.219},
    "db19": {"category": "Large / Sensitive", "tap_size": 38, "num_steps": 23, "max_abs_coeff": 105101},
    "db2": {"category": "Compact", "tap_size": 4, "num_steps": 5, "max_abs_coeff": 1.11536},
    "db20": {"category": "Large / Sensitive", "tap_size": 40, "num_steps": 23, "max_abs_coeff": 609.252},
    "db21": {"category": "Large / Sensitive", "tap_size": 42, "num_steps": 25, "max_abs_coeff": 2710.08},
    "db22": {"category": "Large / Sensitive", "tap_size": 44, "num_steps": 25, "max_abs_coeff": 15693.6},
    "db23": {"category": "Large / Sensitive", "tap_size": 46, "num_steps": 27, "max_abs_coeff": 1207.55},
    "db24": {"category": "Large / Sensitive", "tap_size": 48, "num_steps": 27, "max_abs_coeff": 840.317},
    "db25": {"category": "Large / Sensitive", "tap_size": 50, "num_steps": 29, "max_abs_coeff": 6606.6},
    "db26": {"category": "Large / Sensitive", "tap_size": 52, "num_steps": 29, "max_abs_coeff": 55897.7},
    "db27": {"category": "Large / Sensitive", "tap_size": 54, "num_steps": 31, "max_abs_coeff": 285776},
    "db28": {"category": "Large / Sensitive", "tap_size": 56, "num_steps": 31, "max_abs_coeff": 13871.4},
    "db29": {"category": "Large / Sensitive", "tap_size": 58, "num_steps": 33, "max_abs_coeff": 2.02289e06},
    "db3": {"category": "Compact", "tap_size": 6, "num_steps": 7, "max_abs_coeff": 14.932},
    "db30": {"category": "Large / Sensitive", "tap_size": 60, "num_steps": 33, "max_abs_coeff": 132646},
    "db31": {"category": "Large / Sensitive", "tap_size": 62, "num_steps": 35, "max_abs_coeff": 57852.8},
    "db32": {"category": "Large / Sensitive", "tap_size": 64, "num_steps": 35, "max_abs_coeff": 2473.4},
    "db33": {"category": "Large / Sensitive", "tap_size": 66, "num_steps": 37, "max_abs_coeff": 9.28142e06},
    "db34": {"category": "Large / Sensitive", "tap_size": 68, "num_steps": 37, "max_abs_coeff": 2.2496e06},
    "db35": {"category": "Large / Sensitive", "tap_size": 70, "num_steps": 39, "max_abs_coeff": 7.97156e06},
    "db36": {"category": "Large / Sensitive", "tap_size": 72, "num_steps": 39, "max_abs_coeff": 54462.6},
    "db37": {"category": "Large / Sensitive", "tap_size": 74, "num_steps": 41, "max_abs_coeff": 2.55517e07},
    "db38": {"category": "Large / Sensitive", "tap_size": 76, "num_steps": 41, "max_abs_coeff": 170662},
    "db4": {"category": "Compact", "tap_size": 8, "num_steps": 7, "max_abs_coeff": 2.63378},
    "db5": {"category": "Compact", "tap_size": 10, "num_steps": 9, "max_abs_coeff": 46.0839},
    "db6": {"category": "Compact", "tap_size": 12, "num_steps": 9, "max_abs_coeff": 5.42251},
    "db7": {"category": "Large / Sensitive", "tap_size": 14, "num_steps": 11, "max_abs_coeff": 143.003},
    "db8": {"category": "Large / Sensitive", "tap_size": 16, "num_steps": 11, "max_abs_coeff": 10.8886},
    "db9": {"category": "Medium", "tap_size": 18, "num_steps": 13, "max_abs_coeff": 398.18},
    "dmey": {"category": "Large / Sensitive", "tap_size": 62, "num_steps": 33, "max_abs_coeff": 28.028},
    "haar": {"category": "Compact", "tap_size": 2, "num_steps": 5, "max_abs_coeff": 1.41421},
    "rbio1.1": {"category": "Compact", "tap_size": 2, "num_steps": 5, "max_abs_coeff": 1.41421},
    "rbio1.3": {"category": "Compact", "tap_size": 6, "num_steps": 5, "max_abs_coeff": 1.41421},
    "rbio1.5": {"category": "Compact", "tap_size": 10, "num_steps": 5, "max_abs_coeff": 1.41421},
    "rbio2.2": {"category": "Compact", "tap_size": 6, "num_steps": 5, "max_abs_coeff": 1.41421},
    "rbio2.4": {"category": "Compact", "tap_size": 10, "num_steps": 5, "max_abs_coeff": 1.41421},
    "rbio2.6": {"category": "Large / Sensitive", "tap_size": 14, "num_steps": 5, "max_abs_coeff": 1.41421},
    "rbio2.8": {"category": "Large / Sensitive", "tap_size": 18, "num_steps": 5, "max_abs_coeff": 1.41421},
    "rbio3.1": {"category": "Compact", "tap_size": 4, "num_steps": 5, "max_abs_coeff": 2.12132},
    "rbio3.3": {"category": "Compact", "tap_size": 8, "num_steps": 5, "max_abs_coeff": 2.12132},
    "rbio3.5": {"category": "Compact", "tap_size": 12, "num_steps": 5, "max_abs_coeff": 2.12132},
    "rbio3.7": {"category": "Large / Sensitive", "tap_size": 16, "num_steps": 5, "max_abs_coeff": 2.12132},
    "rbio3.9": {"category": "Large / Sensitive", "tap_size": 20, "num_steps": 5, "max_abs_coeff": 2.12132},
    "rbio4.4": {"category": "Compact", "tap_size": 10, "num_steps": 7, "max_abs_coeff": 1.58613},
    "rbio5.5": {"category": "Compact", "tap_size": 12, "num_steps": 7, "max_abs_coeff": 5.58579},
    "rbio6.8": {"category": "Large / Sensitive", "tap_size": 18, "num_steps": 9, "max_abs_coeff": 1.15131},
    "sym10": {"category": "Medium", "tap_size": 20, "num_steps": 13, "max_abs_coeff": 26.3969},
    "sym11": {"category": "Medium", "tap_size": 22, "num_steps": 15, "max_abs_coeff": 94.636},
    "sym12": {"category": "Medium", "tap_size": 24, "num_steps": 15, "max_abs_coeff": 32.8267},
    "sym13": {"category": "Large / Sensitive", "tap_size": 26, "num_steps": 17, "max_abs_coeff": 4.89679e06},
    "sym14": {"category": "Large / Sensitive", "tap_size": 28, "num_steps": 17, "max_abs_coeff": 2592.45},
    "sym15": {"category": "Large / Sensitive", "tap_size": 30, "num_steps": 19, "max_abs_coeff": 68.4772},
    "sym16": {"category": "Large / Sensitive", "tap_size": 32, "num_steps": 19, "max_abs_coeff": 19.8683},
    "sym17": {"category": "Large / Sensitive", "tap_size": 34, "num_steps": 21, "max_abs_coeff": 1070.37},
    "sym18": {"category": "Large / Sensitive", "tap_size": 36, "num_steps": 21, "max_abs_coeff": 28.6658},
    "sym19": {"category": "Large / Sensitive", "tap_size": 38, "num_steps": 23, "max_abs_coeff": 1849.14},
    "sym2": {"category": "Compact", "tap_size": 4, "num_steps": 5, "max_abs_coeff": 1.11536},
    "sym20": {"category": "Large / Sensitive", "tap_size": 40, "num_steps": 23, "max_abs_coeff": 16728.7},
    "sym3": {"category": "Compact", "tap_size": 6, "num_steps": 7, "max_abs_coeff": 14.932},
    "sym4": {"category": "Compact", "tap_size": 8, "num_steps": 7, "max_abs_coeff": 5.87693},
    "sym5": {"category": "Compact", "tap_size": 10, "num_steps": 9, "max_abs_coeff": 3.32658},
    "sym6": {"category": "Compact", "tap_size": 12, "num_steps": 9, "max_abs_coeff": 13.9887},
    "sym7": {"category": "Large / Sensitive", "tap_size": 14, "num_steps": 11, "max_abs_coeff": 17127.2},
    "sym8": {"category": "Large / Sensitive", "tap_size": 16, "num_steps": 11, "max_abs_coeff": 19.0019},
    "sym9": {"category": "Medium", "tap_size": 18, "num_steps": 13, "max_abs_coeff": 309.77},
}

ALL_BOUNDARY_MODES = [
    "symmetric",
    "zero",
    "constant",
    "periodic",
    "antisymmetric",
    "smooth",
    "reflect",
    "antireflect",
]


def calculate_metrics(a: np.ndarray, b: np.ndarray):
    min_len = min(a.size, b.size)
    a_flat = a.flatten()[:min_len].astype(np.float64)
    b_flat = b.flatten()[:min_len].astype(np.float64)
    
    max_abs = float(np.max(np.abs(a_flat - b_flat)))
    mean_abs = float(np.mean(np.abs(a_flat - b_flat)))
    
    norm_a = np.linalg.norm(a_flat - np.mean(a_flat))
    norm_b = np.linalg.norm(b_flat - np.mean(b_flat))
    if norm_a > 1e-12 and norm_b > 1e-12:
        pcc = float(np.corrcoef(a_flat, b_flat)[0, 1])
    else:
        pcc = 1.0 if max_abs < 1e-4 else 0.0
        
    return {"max_abs": max_abs, "mean_abs": mean_abs, "pcc": pcc}


def test_precision_single(wavelet_name, boundary_mode, signal_len, device):
    meta = WAVELET_CATEGORIES.get(wavelet_name, {"category": "Unknown", "max_abs_coeff": 1.0})
    cat = meta["category"]
    max_coeff = meta["max_abs_coeff"]
    
    if cat == "Compact":
        abs_tol = 1e-4 * max(1.0, max_coeff)
    elif cat == "Medium":
        abs_tol = 1e-3 * max(1.0, max_coeff)
    else:
        abs_tol = 5e-3 * max(1.0, max_coeff)

    s_sticks = (signal_len + 31) // 32
    padded_len = s_sticks * 32
    x = np.sin(np.linspace(0, 10 * np.pi, padded_len, dtype=np.float32))
    
    # TTNN DWT + IDWT
    inp_tensor = ttnn.from_torch(
        torch.from_numpy(x.reshape(s_sticks, 32)), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    ttnn_app, ttnn_det = ttnn.dwt(inp_tensor, wavelet_name, boundary_mode=boundary_mode)
    ttnn_rec = ttnn.idwt(ttnn_app, ttnn_det, wavelet_name, padded_len, boundary_mode=boundary_mode)
    ttnn_rec_np = ttnn.to_torch(ttnn_rec).numpy().flatten()[:signal_len]
    
    rec_metrics = calculate_metrics(x[:signal_len], ttnn_rec_np)
    passed = rec_metrics["max_abs"] <= abs_tol or rec_metrics["pcc"] >= 0.999
    
    return {
        "wavelet": wavelet_name,
        "boundary_mode": boundary_mode,
        "category": cat,
        "max_coeff": max_coeff,
        "abs_tol": abs_tol,
        "rec_max_abs": rec_metrics["max_abs"],
        "rec_pcc": rec_metrics["pcc"],
        "passed": passed,
    }


def measure_pywt_timing(wavelet_name, boundary_mode, signal_len, repeats=20):
    sig = np.sin(np.linspace(0, 10 * np.pi, signal_len)).astype(np.float32)
    pywt_mode = boundary_mode if boundary_mode in pywt.Modes.modes else "symmetric"
    
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
    s_sticks = (signal_len + 31) // 32
    padded_len = s_sticks * 32
    sig = torch.sin(torch.linspace(0, 10 * torch.pi, padded_len, dtype=torch.float32)).reshape(s_sticks, 32)
    inp = ttnn.from_torch(sig, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    
    for _ in range(2):
        app, det = ttnn.dwt(inp, wavelet_name, boundary_mode=boundary_mode)
        rec = ttnn.idwt(app, det, wavelet_name, padded_len, boundary_mode=boundary_mode)
        ttnn.synchronize_device(device)
        
    t0 = time.perf_counter()
    for _ in range(repeats):
        app, det = ttnn.dwt(inp, wavelet_name, boundary_mode=boundary_mode)
    ttnn.synchronize_device(device)
    t1 = time.perf_counter()
    dwt_ms = (t1 - t0) * 1000.0 / repeats

    t0 = time.perf_counter()
    for _ in range(repeats):
        rec = ttnn.idwt(app, det, wavelet_name, padded_len, boundary_mode=boundary_mode)
    ttnn.synchronize_device(device)
    t1 = time.perf_counter()
    idwt_ms = (t1 - t0) * 1000.0 / repeats

    return dwt_ms, idwt_ms


def plot_performance_results(perf_data, output_dir, selected_wavelets, lengths):
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    for wavelet in selected_wavelets:
        w_dir = plots_dir / wavelet.replace(".", "")
        w_dir.mkdir(parents=True, exist_ok=True)

        for mode in ALL_BOUNDARY_MODES:
            # 1. DWT Plot
            plt.figure(figsize=(7, 4.5))

            py_dwts = [perf_data.get((wavelet, mode, N), {}).get("pywt_dwt", 0.0) for N in lengths]
            std_dwts = [perf_data.get((wavelet, mode, N), {}).get("std_dwt", 0.0) for N in lengths]
            ttnn_dwts = [perf_data.get((wavelet, mode, N), {}).get("ttnn_dwt", 0.0) for N in lengths]

            if any(v > 0 for v in py_dwts):
                plt.plot(lengths, py_dwts, label="PyWavelets")
            if any(v > 0 for v in std_dwts):
                plt.plot(lengths, std_dwts, label="tt-wavelet")
            if any(v > 0 for v in ttnn_dwts):
                plt.plot(lengths, ttnn_dwts, label="ttnn-wavelet")

            plt.yscale("log")
            plt.xlabel("Signal length")
            plt.ylabel("Runtime (ms, log scale)")
            plt.title(f"1D {wavelet} DWT runtime vs signal length ({mode})")

            plt.grid(True, which="both", linestyle=":")
            plt.legend()
            plt.tight_layout()

            plt.savefig(w_dir / f"lwt_{mode}.png", dpi=200)
            plt.close()

            # 2. ILWT Plot
            plt.figure(figsize=(7, 4.5))

            py_idwts = [perf_data.get((wavelet, mode, N), {}).get("pywt_idwt", 0.0) for N in lengths]
            std_idwts = [perf_data.get((wavelet, mode, N), {}).get("std_idwt", 0.0) for N in lengths]
            ttnn_idwts = [perf_data.get((wavelet, mode, N), {}).get("ttnn_idwt", 0.0) for N in lengths]

            if any(v > 0 for v in py_idwts):
                plt.plot(lengths, py_idwts, label="PyWavelets")
            if any(v > 0 for v in std_idwts):
                plt.plot(lengths, std_idwts, label="tt-wavelet")
            if any(v > 0 for v in ttnn_idwts):
                plt.plot(lengths, ttnn_idwts, label="ttnn-wavelet")

            plt.yscale("log")
            plt.xlabel("Signal length")
            plt.ylabel("Runtime (ms, log scale)")
            plt.title(f"1D {wavelet} ILWT runtime vs signal length ({mode})")

            plt.grid(True, which="both", linestyle=":")
            plt.legend()
            plt.tight_layout()

            plt.savefig(w_dir / f"ilwt_{mode}.png", dpi=200)
            plt.close()


def main():
    parser = argparse.ArgumentParser(description="Full Architecture-Independent Bringup & Verification Suite.")
    parser.add_argument("--test-run", action="store_true", help="Quick test run with small subset for verification.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for performance wavelet selection.")
    parser.add_argument("--output-base", type=Path, default=Path("benchmarks"), help="Base output directory.")
    args = parser.parse_args()

    random.seed(args.seed)
    
    precision_dir = args.output_base / "precision"
    performance_dir = args.output_base / "performance"
    precision_dir.mkdir(parents=True, exist_ok=True)
    performance_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" TENSTORRENT WAVELET AUTOMATED BRINGUP & BENCHMARK SUITE ")
    print("=" * 80)

    # 1. Device Architecture Auto-Detection
    open_device_fn = getattr(ttnn, "open_device", getattr(getattr(ttnn, "_ttnn", None), "device", None).open_device if hasattr(ttnn, "_ttnn") else None)
    close_device_fn = getattr(ttnn, "close_device", getattr(getattr(ttnn, "_ttnn", None), "device", None).close_device if hasattr(ttnn, "_ttnn") else None)
    device = open_device_fn(device_id=0)
    arch_str = str(device.arch()).replace("Arch.", "").lower()
    print(f"[Device Init] Detected Architecture: {device.arch()} ({arch_str})")

    # Determine wavelets and parameters based on test-run
    if args.test_run:
        print("[Mode] DRY-RUN / TEST MODE: Running subset for quick validation.")
        precision_wavelets = ["db1", "bior3.9"]
        precision_modes = ["symmetric", "zero"]
        
        selected_compact = "db1"
        selected_medium = "bior3.9"
        selected_large = "coif17"
        perf_lengths = [100000, 400000]
        perf_repeats = 5
        perf_modes = ["symmetric", "zero"]
    else:
        print("[Mode] FULL BRINGUP SWEEP MODE")
        precision_wavelets = list(WAVELET_CATEGORIES.keys())
        precision_modes = ALL_BOUNDARY_MODES
        
        # Categorize wavelets for selection
        compact_list = [w for w, meta in WAVELET_CATEGORIES.items() if meta["category"] == "Compact"]
        medium_list = [w for w, meta in WAVELET_CATEGORIES.items() if meta["category"] == "Medium"]
        large_list = [w for w, meta in WAVELET_CATEGORIES.items() if meta["category"] == "Large / Sensitive"]
        
        selected_compact = random.choice(compact_list)
        selected_medium = random.choice(medium_list)
        selected_large = random.choice(large_list)
        
        perf_lengths = [100000, 400000, 700000, 1000000]
        perf_repeats = 20
        perf_modes = ALL_BOUNDARY_MODES

    selected_perf_wavelets = [selected_compact, selected_medium, selected_large]
    print(f"[Selection] Performance Wavelets Selected: Compact='{selected_compact}', Medium='{selected_medium}', Large='{selected_large}'")
    print(f"[Selection] Signal Lengths: {perf_lengths}")

    # PHASE 1: PRECISION VALIDATION
    print("\n" + "=" * 80)
    print(f" PHASE 1: PRECISION VALIDATION ({len(precision_wavelets)} schemes x {len(precision_modes)} boundary modes)")
    print("=" * 80)
    
    precision_results = []
    passed_count = 0
    total_count = 0

    pbar_prec = tqdm(total=len(precision_wavelets) * len(precision_modes), desc="Precision Validation Suite")
    for wavelet in precision_wavelets:
        for mode in precision_modes:
            total_count += 1
            res = test_precision_single(wavelet, mode, signal_len=1024, device=device)
            res["arch"] = arch_str
            precision_results.append(res)
            if res["passed"]:
                passed_count += 1
            
            pbar_prec.set_postfix(wavelet=wavelet, mode=mode, rec_err=f"{res['rec_max_abs']:.2e}", pcc=f"{res['rec_pcc']:.4f}")
            pbar_prec.update(1)
    pbar_prec.close()

    # Save Precision Results
    prec_json_path = precision_dir / "precision_results.json"
    prec_tsv_path = precision_dir / "precision_results.tsv"

    with open(prec_json_path, "w") as f:
        json.dump(precision_results, f, indent=2)

    with open(prec_tsv_path, "w") as f:
        f.write("wavelet\tboundary_mode\tcategory\tmax_coeff\tabs_tol\trec_max_abs\trec_pcc\tpassed\n")
        for r in precision_results:
            f.write(
                f"{r['wavelet']}\t{r['boundary_mode']}\t{r['category']}\t{r['max_coeff']}\t"
                f"{r['abs_tol']}\t{r['rec_max_abs']:.6e}\t{r['rec_pcc']:.6f}\t{r['passed']}\n"
            )

    print(f"[Precision Complete] {passed_count}/{total_count} passed. Saved to {prec_tsv_path}")

    # Close device before running standalone subprocesses to avoid locking
    if device is not None and close_device_fn is not None:
        close_device_fn(device)
    device = None

    # PHASE 2: PERFORMANCE BENCHMARK SUITE
    print("\n" + "=" * 80)
    print(f" PHASE 2: PERFORMANCE BENCHMARK (3 selected schemes x {len(perf_modes)} modes x {len(perf_lengths)} lengths)")
    print("=" * 80)

    perf_total_runs = len(selected_perf_wavelets) * len(perf_modes) * len(perf_lengths)
    pywt_perf = {}
    std_perf = {}
    ttnn_perf = {}

    # 1. PyWT CPU Phase
    print("\n[Perf Phase 1/3] PyWavelets (CPU)...")
    pbar_pywt = tqdm(total=perf_total_runs, desc="PyWT CPU Benchmark")
    for wavelet in selected_perf_wavelets:
        for mode in perf_modes:
            for N in perf_lengths:
                pywt_dwt, pywt_idwt = measure_pywt_timing(wavelet, mode, N, repeats=perf_repeats)
                pywt_perf[(wavelet, mode, N)] = (pywt_dwt, pywt_idwt)
                pbar_pywt.update(1)
    pbar_pywt.close()

    # 2. Standalone C++ Phase
    print("\n[Perf Phase 2/3] Standalone tt-wavelet (Device)...")
    pbar_std = tqdm(total=perf_total_runs, desc="Standalone C++ Benchmark")
    for wavelet in selected_perf_wavelets:
        for mode in perf_modes:
            for N in perf_lengths:
                std_dwt, std_idwt = measure_standalone_timing(wavelet, mode, N, repeats=perf_repeats)
                std_perf[(wavelet, mode, N)] = (std_dwt, std_idwt)
                pbar_std.update(1)
    pbar_std.close()

    # 3. TTNN Phase
    print("\n[Perf Phase 3/3] TTNN ttnn-wavelet (Device)...")
    device = open_device_fn(device_id=0)
    pbar_ttnn = tqdm(total=perf_total_runs, desc="TTNN Operations Benchmark")
    try:
        for wavelet in selected_perf_wavelets:
            for mode in perf_modes:
                for N in perf_lengths:
                    ttnn_dwt, ttnn_idwt = measure_ttnn_timing(wavelet, mode, N, device, repeats=perf_repeats)
                    ttnn_perf[(wavelet, mode, N)] = (ttnn_dwt, ttnn_idwt)
                    pbar_ttnn.update(1)
    finally:
        pbar_ttnn.close()
        if close_device_fn is not None:
            close_device_fn(device)

    # Save Performance Results & Plots
    perf_tsv_path = performance_dir / "performance_results.tsv"
    perf_json_path = performance_dir / "performance_results.json"
    
    perf_structured_data = {}
    perf_json_list = []

    with open(perf_tsv_path, "w") as f_tsv:
        f_tsv.write("wavelet\tlength\tboundary_mode\tpywt_dwt_ms\tpywt_idwt_ms\tstandalone_dwt_ms\tstandalone_idwt_ms\tttnn_dwt_ms\tttnn_idwt_ms\n")
        
        for wavelet in selected_perf_wavelets:
            for mode in perf_modes:
                for N in perf_lengths:
                    pywt_dwt, pywt_idwt = pywt_perf.get((wavelet, mode, N), (0.0, 0.0))
                    std_dwt, std_idwt = std_perf.get((wavelet, mode, N), (0.0, 0.0))
                    ttnn_dwt, ttnn_idwt = ttnn_perf.get((wavelet, mode, N), (0.0, 0.0))

                    perf_structured_data[(wavelet, mode, N)] = {
                        "pywt_dwt": pywt_dwt, "pywt_idwt": pywt_idwt,
                        "std_dwt": std_dwt, "std_idwt": std_idwt,
                        "ttnn_dwt": ttnn_dwt, "ttnn_idwt": ttnn_idwt
                    }
                    
                    perf_json_list.append({
                        "wavelet": wavelet,
                        "length": N,
                        "boundary_mode": mode,
                        "pywt_dwt_ms": pywt_dwt, "pywt_idwt_ms": pywt_idwt,
                        "standalone_dwt_ms": std_dwt, "standalone_idwt_ms": std_idwt,
                        "ttnn_dwt_ms": ttnn_dwt, "ttnn_idwt_ms": ttnn_idwt
                    })

                    f_tsv.write(
                        f"{wavelet}\t{N}\t{mode}\t{pywt_dwt:.4f}\t{pywt_idwt:.4f}\t"
                        f"{std_dwt:.4f}\t{std_idwt:.4f}\t{ttnn_dwt:.4f}\t{ttnn_idwt:.4f}\n"
                    )

    with open(perf_json_path, "w") as f_json:
        json.dump(perf_json_list, f_json, indent=2)

    print("\n[Plot Generation] Creating benchmark comparison plots...")
    plot_performance_results(perf_structured_data, performance_dir, selected_perf_wavelets, perf_lengths)

    print("\n" + "=" * 80)
    print(" AUTOMATED SUITE COMPLETE ")
    print(f" Architecture: {arch_str.upper()}")
    print(f" Precision TSV: {prec_tsv_path}")
    print(f" Performance TSV: {perf_tsv_path}")
    print(f" Plots Saved To: {performance_dir / 'plots'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
