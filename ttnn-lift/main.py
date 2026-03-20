import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tt-metal", "ttnn"))

import numpy as np
import pywt
import torch
import ttnn
from tt_wavelet.lifting import LiftingWaveletTransform, load_lifting_scheme
from tt_wavelet.signal import InputSpec, Signal, SignalType

TEST_LENGTHS = [8, 15, 17, 33, 64, 65]
TEST_SIGNAL_TYPES = [
    SignalType.CONSTANT,
    SignalType.IMPULSE,
    SignalType.STEP,
    SignalType.RAMP,
    SignalType.SINUSOIDAL,
    SignalType.ALTERNATING,
    SignalType.NORMAL,
]
RANDOM_SEEDS = [2]
RANDOM_SIGNAL_TYPES = {SignalType.NORMAL}
EXCLUDED_WAVELETS = {f"coif{i}" for i in range(10, 18)}


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


def seed_values_for_signal(signal_type: SignalType) -> list[int]:
    if signal_type in RANDOM_SIGNAL_TYPES:
        return RANDOM_SEEDS
    return [0]


def iter_input_specs() -> list[InputSpec]:
    specs: list[InputSpec] = []
    for length in TEST_LENGTHS:
        for signal_type in TEST_SIGNAL_TYPES:
            for seed in seed_values_for_signal(signal_type):
                specs.append(
                    InputSpec(
                        kind=signal_type,
                        length=length,
                        magnitude=1.0,
                        seed=seed,
                    )
                )
    return specs


def group_input_specs_by_length(
    input_specs: list[InputSpec],
) -> dict[int, list[InputSpec]]:
    grouped: dict[int, list[InputSpec]] = defaultdict(list)
    for spec in input_specs:
        grouped[spec.length].append(spec)
    return dict(grouped)


def main() -> None:
    root = Path(__file__).resolve().parent
    schemes_dir = root / "lifting_schemes"
    scheme_paths = sorted(path for path in schemes_dir.rglob("*.json") if path.is_file())
    csv_dir = root / "results"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "sfpu_errors.csv"

    input_specs = iter_input_specs()
    grouped_input_specs = group_input_specs_by_length(input_specs)
    rows: list[dict[str, object]] = []

    tested = 0
    skipped = 0
    excluded = 0
    failures = 0

    device = ttnn.open_device(device_id=0)
    try:
        for scheme_path in scheme_paths:
            wavelet_name = scheme_path.stem.removesuffix("-fp64")
            if wavelet_name in EXCLUDED_WAVELETS:
                excluded += 1
                print(f"SKIP: {scheme_path.name} (excluded)")
                continue

            try:
                wavelet = pywt.Wavelet(wavelet_name)
            except Exception:
                skipped += 1
                print(f"SKIP: {scheme_path.name} (no matching pywt wavelet)")
                continue

            print(f"WAVELET: {wavelet_name} | FILE: {scheme_path.relative_to(schemes_dir)}")
            scheme = load_lifting_scheme(str(scheme_path), mode="symmetric")
            transform = LiftingWaveletTransform(scheme, device)

            for _, specs_for_length in sorted(grouped_input_specs.items()):
                signals = [Signal(spec).data.astype(np.float32) for spec in specs_for_length]
                signal_batch = np.stack(signals, axis=0)
                signal_torch = torch.from_numpy(signal_batch)

                base_rows = [
                    {
                        "wavelet": wavelet_name,
                        "file": str(scheme_path.relative_to(schemes_dir)),
                        "signal_type": spec.kind.name,
                        "length": spec.length,
                        "seed": int(spec.seed or 0),
                    }
                    for spec in specs_for_length
                ]

                try:
                    out = transform.forward(signal_torch)
                    approx_ttnn_batch = ttnn.to_torch(out["cA"]).cpu().numpy()
                    details_ttnn_batch = ttnn.to_torch(out["cD"]).cpu().numpy()

                    recon_ttnn = transform.inverse(out["cA"], out["cD"])
                    recon_batch = ttnn.to_torch(recon_ttnn).cpu().numpy()

                    for index, spec in enumerate(specs_for_length):
                        signal = signals[index]
                        approx_ttnn = approx_ttnn_batch[index].reshape(-1)
                        details_ttnn = details_ttnn_batch[index].reshape(-1)
                        recon = recon_batch[index].reshape(-1)[: len(signal)]
                        approx_pywt, details_pywt = pywt.dwt(
                            signal, wavelet=wavelet, mode="symmetric"
                        )

                        mse_approx = mse(approx_pywt, approx_ttnn)
                        mse_detail = mse(details_pywt, details_ttnn)
                        mse_inverse = mse(signal.astype(np.float64), recon.astype(np.float64))

                        print(
                            f"CASE: type={spec.kind.name}, len={spec.length}, seed={int(spec.seed or 0)}"
                        )
                        print_cmp("approximation", approx_pywt, approx_ttnn)
                        print_cmp("details", details_pywt, details_ttnn)
                        print_cmp(
                            "reconstruction",
                            signal.astype(np.float64),
                            recon.astype(np.float64),
                        )

                        row = dict(base_rows[index])
                        row.update(
                            {
                                "status": "ok",
                                "error": "",
                                "mse_approx": mse_approx,
                                "mse_detail": mse_detail,
                                "mse_inverse": mse_inverse,
                            }
                        )
                        rows.append(row)
                except Exception as error:
                    for index, spec in enumerate(specs_for_length):
                        failures += 1
                        row = dict(base_rows[index])
                        row.update(
                            {
                                "status": "error",
                                "error": str(error),
                                "mse_approx": float("nan"),
                                "mse_detail": float("nan"),
                                "mse_inverse": float("nan"),
                            }
                        )
                        rows.append(row)
                        print(
                            f"\033[31m ERROR: type={spec.kind.name}, len={spec.length}, "
                            f"seed={int(spec.seed or 0)}: {error}\033[0m"
                        )

            tested += 1

        fieldnames = [
            "wavelet",
            "file",
            "signal_type",
            "length",
            "seed",
            "status",
            "error",
            "mse_approx",
            "mse_detail",
            "mse_inverse",
        ]

        with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        print(f"Discovered {len(scheme_paths)} JSON schemes")
        print(f"Tested {tested} wavelets")
        print(f"Excluded {excluded} wavelets (coif10-coif17)")
        print(f"Skipped {skipped} schemes (no matching pywt wavelet)")
        print(f"Failed cases: {failures}")
        print(f"CSV saved: {csv_path}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
