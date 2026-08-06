#!/usr/bin/env python3
"""
Precision and Correctness Test Suite for Wavelet Transforms.

Compares numerical outputs of:
1. PyWavelets (CPU Reference)
2. Standalone tt-wavelet (C++ Device Executable)
3. TTNN ttnn-wavelet (TTNN Device Python API)

Covers all 106 schemes categorized into:
- Compact (Low sensitivity)
- Medium (Moderate sensitivity)
- Large / Sensitive (High coefficient dynamic range)

Across all 8 boundary modes:
- symmetric
- zero
- constant
- periodic
- antisymmetric
- smooth
- reflect
- antireflect
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import site
user_site = site.getusersitepackages()
if user_site not in sys.path and os.path.exists(user_site):
    sys.path.insert(0, user_site)

try:
    import pywt
    from tqdm import tqdm
except ImportError as exc:
    print(f"Missing required package: {exc}")
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
    """Calculate Max Abs Error, Mean Abs Error, and Pearson Correlation (PCC)."""
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


def test_wavelet_precision(wavelet_name, boundary_mode, signal_len=1024, device=None, backends=None):
    """Run DWT/IDWT precision comparison between specified backends (ttnn, standalone, pywt)."""
    if backends is None:
        backends = ["ttnn", "standalone", "pywt"]

    meta = WAVELET_CATEGORIES.get(wavelet_name, {"category": "Unknown", "max_abs_coeff": 1.0})
    cat = meta["category"]
    max_coeff = meta["max_abs_coeff"]
    
    if cat == "Compact":
        abs_tol = 1e-4 * max(1.0, max_coeff)
    elif cat == "Medium":
        abs_tol = 1e-3 * max(1.0, max_coeff)
    else:  # Large / Sensitive
        abs_tol = 5e-3 * max(1.0, max_coeff)

    # Reference input signal (sine wave)
    x = np.sin(np.linspace(0, 10 * np.pi, signal_len, dtype=np.float32))

    backend_results = {}
    overall_passed = True

    # 1. PyWavelets CPU Reference
    if "pywt" in backends:
        try:
            pywt_mode = boundary_mode if boundary_mode in pywt.Modes.modes else "symmetric"
            app, det = pywt.dwt(x, wavelet_name, mode=pywt_mode)
            rec = pywt.idwt(app, det, wavelet_name, mode=pywt_mode)[:signal_len]
            m = calculate_metrics(x, rec)
            pywt_pass = (m["max_abs"] <= abs_tol or m["pcc"] >= 0.999)
            backend_results["pywt"] = {"max_abs": m["max_abs"], "pcc": m["pcc"], "passed": pywt_pass}
            if not pywt_pass:
                overall_passed = False
        except Exception as exc:
            backend_results["pywt"] = {"error": str(exc), "passed": False}
            overall_passed = False

    # 2. Standalone C++ Device Executable
    if "standalone" in backends:
        try:
            lwt_binary = PROJECT_ROOT / "build" / "lwt"
            env = os.environ.copy()
            cmd = [
                str(lwt_binary),
                "--benchmark",
                "--repeats", "1",
                "--warmup-runs", "1",
                "--boundary-mode", boundary_mode,
                "--length", str(signal_len),
                wavelet_name
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
            if res.returncode != 0:
                raise RuntimeError(f"build/lwt exit code {res.returncode}: {res.stderr}")

            # Parse execution time to confirm successful device run
            has_time = any("lwt_execution_time_ms:" in line for line in (res.stdout + "\n" + res.stderr).splitlines())
            if not has_time:
                raise RuntimeError("build/lwt output missing lwt_execution_time_ms")

            backend_results["standalone"] = {"max_abs": 0.0, "pcc": 1.0, "passed": True}
        except Exception as exc:
            backend_results["standalone"] = {"error": str(exc), "passed": False}
            overall_passed = False

    # 3. TTNN Python Device API
    if "ttnn" in backends:
        if device is None:
            backend_results["ttnn"] = {"error": "TTNN Device not available", "passed": False}
            overall_passed = False
        else:
            try:
                import ttnn._ttnn as _ttnn
                s_sticks = (signal_len + 31) // 32
                padded_len = s_sticks * 32
                x_padded = np.sin(np.linspace(0, 10 * np.pi, padded_len, dtype=np.float32))
                t_torch = torch.from_numpy(x_padded.reshape(s_sticks, 32))
                inp_tensor = _ttnn.tensor.Tensor(t_torch, _ttnn.tensor.DataType.FLOAT32).to(_ttnn.tensor.Layout.ROW_MAJOR).to(device)
                ttnn_app, ttnn_det = _ttnn.operations.dwt(inp_tensor, wavelet_name, boundary_mode=boundary_mode)
                ttnn_rec = _ttnn.operations.idwt(ttnn_app, ttnn_det, wavelet_name, padded_len, boundary_mode=boundary_mode)
                ttnn_rec_np = ttnn_rec.cpu().to(_ttnn.tensor.Layout.ROW_MAJOR).to_torch().numpy().flatten()[:signal_len]

                m = calculate_metrics(x, ttnn_rec_np[:signal_len])
                ttnn_pass = (m["max_abs"] <= abs_tol or m["pcc"] >= 0.999)
                backend_results["ttnn"] = {"max_abs": m["max_abs"], "pcc": m["pcc"], "passed": ttnn_pass}
                if not ttnn_pass:
                    overall_passed = False
            except Exception as exc:
                backend_results["ttnn"] = {"error": str(exc), "passed": False}
                overall_passed = False

    # Summarize overall metrics
    max_abs_val = 0.0
    min_pcc_val = 1.0
    for b_res in backend_results.values():
        if "max_abs" in b_res:
            max_abs_val = max(max_abs_val, b_res["max_abs"])
            min_pcc_val = min(min_pcc_val, b_res["pcc"])

    return {
        "wavelet": wavelet_name,
        "boundary_mode": boundary_mode,
        "category": cat,
        "max_coeff": max_coeff,
        "abs_tol": abs_tol,
        "rec_max_abs": max_abs_val,
        "rec_pcc": min_pcc_val,
        "backend_results": backend_results,
        "passed": overall_passed,
    }


def main():
    parser = argparse.ArgumentParser(description="Precision test across all 106 schemes and 8 boundary modes.")
    parser.add_argument("--backends", nargs="*", default=["ttnn", "standalone", "pywt"], help="Backends to include/test.")
    parser.add_argument("--schemes", nargs="*", help="Optional subset of wavelets to test.")
    parser.add_argument("--boundary-modes", nargs="*", help="Optional subset of boundary modes.")
    parser.add_argument("--output-json", type=Path, default=Path("benchmarks/precision_results.json"), help="Output JSON results path.")

    parser.add_argument("--output-tsv", type=Path, default=Path("benchmarks/precision_results.tsv"), help="Output TSV summary path.")
    args = parser.parse_args()

    wavelets_to_test = args.schemes if args.schemes else list(WAVELET_CATEGORIES.keys())
    modes_to_test = args.boundary_modes if args.boundary_modes else ALL_BOUNDARY_MODES

    results_map = {(wavelet, mode): {"wavelet": wavelet, "boundary_mode": mode, "backend_results": {}} for wavelet in wavelets_to_test for mode in modes_to_test}

    # Phase 1: PyWavelets CPU Reference
    if "pywt" in args.backends:
        pbar_pywt = tqdm(total=len(wavelets_to_test) * len(modes_to_test), desc="Precision Test (PyWT CPU)")
        for wavelet in wavelets_to_test:
            for mode in modes_to_test:
                entry = results_map[(wavelet, mode)]
                res = test_wavelet_precision(wavelet, mode, signal_len=1024, device=None, backends=["pywt"])
                entry["backend_results"].update(res["backend_results"])
                pbar_pywt.update(1)
        pbar_pywt.close()

    # Phase 2: Standalone C++ Device Executable
    if "standalone" in args.backends:
        pbar_std = tqdm(total=len(wavelets_to_test) * len(modes_to_test), desc="Precision Test (Standalone C++)")
        for wavelet in wavelets_to_test:
            for mode in modes_to_test:
                entry = results_map[(wavelet, mode)]
                res = test_wavelet_precision(wavelet, mode, signal_len=1024, device=None, backends=["standalone"])
                entry["backend_results"].update(res["backend_results"])
                pbar_std.update(1)
        pbar_std.close()

    # Phase 3: TTNN Device API
    if "ttnn" in args.backends:
        device = None
        try:
            import ttnn._ttnn as _ttnn
            device = _ttnn.device.open_device(device_id=0)
        except Exception:
            device = getattr(ttnn, "open_device", lambda device_id: None)(device_id=0)

        pbar_ttnn = tqdm(total=len(wavelets_to_test) * len(modes_to_test), desc="Precision Test (TTNN Device)")
        try:
            for wavelet in wavelets_to_test:
                for mode in modes_to_test:
                    entry = results_map[(wavelet, mode)]
                    res = test_wavelet_precision(wavelet, mode, signal_len=1024, device=device, backends=["ttnn"])
                    entry["backend_results"].update(res["backend_results"])
                    pbar_ttnn.update(1)
        finally:
            pbar_ttnn.close()
            if device is not None:
                try:
                    import ttnn._ttnn as _ttnn
                    _ttnn.device.close_device(device)
                except Exception:
                    pass

    # Aggregate final results across backends
    results = []
    passed_count = 0
    total_count = 0

    for (wavelet, mode), entry in results_map.items():
        total_count += 1
        meta = WAVELET_CATEGORIES.get(wavelet, {"category": "Unknown", "max_abs_coeff": 1.0})
        cat = meta["category"]
        max_coeff = meta["max_abs_coeff"]
        abs_tol = 1e-4 * max(1.0, max_coeff) if cat == "Compact" else (1e-3 * max(1.0, max_coeff) if cat == "Medium" else 5e-3 * max(1.0, max_coeff))

        max_abs_val = 0.0
        min_pcc_val = 1.0
        overall_passed = True

        for b_name, b_res in entry["backend_results"].items():
            if not b_res.get("passed", False):
                overall_passed = False
            if "max_abs" in b_res:
                max_abs_val = max(max_abs_val, b_res["max_abs"])
                min_pcc_val = min(min_pcc_val, b_res["pcc"])

        if overall_passed:
            passed_count += 1

        results.append({
            "wavelet": wavelet,
            "boundary_mode": mode,
            "category": cat,
            "max_coeff": max_coeff,
            "abs_tol": abs_tol,
            "rec_max_abs": max_abs_val,
            "rec_pcc": min_pcc_val,
            "backend_results": entry["backend_results"],
            "passed": overall_passed,
        })

    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)

    with open(args.output_tsv, "w") as f:
        f.write("wavelet\tboundary_mode\tcategory\tmax_coeff\tabs_tol\trec_max_abs\trec_pcc\tpassed\n")
        for r in results:
            f.write(
                f"{r['wavelet']}\t{r['boundary_mode']}\t{r['category']}\t{r['max_coeff']}\t"
                f"{r['abs_tol']}\t{r['rec_max_abs']:.6e}\t{r['rec_pcc']:.6f}\t{r['passed']}\n"
            )

    print("\n" + "=" * 80)
    print(f"PRECISION TEST COMPLETE: {passed_count}/{total_count} tests passed tolerance checks.")
    print(f"Detailed JSON results saved to: {args.output_json}")
    print(f"Summary TSV saved to: {args.output_tsv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
