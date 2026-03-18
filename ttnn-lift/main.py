import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tt-metal", "ttnn"))

import numpy as np
import pywt
import torch
import ttnn
from tt_wavelet.lifting import LiftingWaveletTransform, load_lifting_scheme


def mse(a: np.ndarray, b: np.ndarray) -> float:
    err = a - b
    return float(np.mean(err * err))


def print_cmp(name: str, a: np.ndarray, b: np.ndarray) -> None:
    err_mse = mse(a, b)
    label = f" {name.upper()}:"
    padding = " " * max(1, 17 - len(label))

    if err_mse < 5e-6:
        color = "\033[32m"
    elif err_mse < 1e-2:
        color = "\033[33m"
    else:
        color = "\033[31m"

    print(f"{label}{padding}{color}{err_mse:.2e}\033[0m")


def main() -> None:
    np.random.seed(0)

    root = Path(__file__).resolve().parent
    schemes_dir = root / "lifting_schemes"
    scheme_paths = sorted(schemes_dir.glob("*.json"))

    tested = 0
    skipped = 0

    device = ttnn.open_device(device_id=0)
    try:
        for scheme_path in scheme_paths:
            wavelet_name = scheme_path.stem.removesuffix("-fp64")

            try:
                wavelet = pywt.Wavelet(wavelet_name)
            except Exception:
                skipped += 1
                continue

            print(f"WAVELET: {wavelet_name}")
            scheme = load_lifting_scheme(str(scheme_path), mode="symmetric")
            transform = LiftingWaveletTransform(scheme, device)

            signal = np.random.normal(scale=1.0, size=(1001,)).astype(np.float32)
            signal_torch = torch.from_numpy(signal).unsqueeze(0)

            out = transform.forward(signal_torch)
            approx_ttnn = ttnn.to_torch(out["cA"]).cpu().numpy().reshape(-1)
            details_ttnn = ttnn.to_torch(out["cD"]).cpu().numpy().reshape(-1)

            approx_pywt, details_pywt = pywt.dwt(signal, wavelet=wavelet, mode="symmetric")

            print_cmp("approximation", approx_pywt, approx_ttnn)
            print_cmp("details", details_pywt, details_ttnn)

            recon_ttnn = transform.inverse(out["cA"], out["cD"])
            recon = ttnn.to_torch(recon_ttnn).cpu().numpy().reshape(-1)
            recon = recon[: len(signal)]

            print_cmp("reconstruction", signal.astype(np.float64), recon.astype(np.float64))
            tested += 1

        print(f"Tested {tested} wavelets")
        print(f"Skipped {skipped} schemes (no matching pywt wavelet)")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
