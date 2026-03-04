import tomllib
from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class ComparisonResult:
    abs_max: float
    abs_mean: float
    rel_max: float
    rel_mean: float


def compare(
    cmp: np.ndarray,
    ref: np.ndarray,
) -> ComparisonResult:
    if cmp.shape != ref.shape:
        raise ValueError(f"Shape mismatch: {cmp.shape} vs {ref.shape}")

    diff = cmp - ref
    mask = diff > 1e-12
    rel = np.abs(diff[mask] / ref[mask])
    abs_ = np.abs(diff[~mask])

    return ComparisonResult(
        abs_max=np.max(abs_) if abs_.size > 0 else 0,
        abs_mean=np.mean(abs_) if abs_.size > 0 else 0,
        rel_max=np.max(rel) if rel.size > 0 else 0,
        rel_mean=np.mean(rel) if rel.size > 0 else 0,
    )


def main(
    config_path: str = "config.toml",
    output_path: str = "benchmark.csv",
    input_path: str = "data/input",
    cross_check: list[tuple[tuple[str, str], tuple[str, str]]] = [
        (("data/output-f32", "f32"), ("data/output-pywt", "pywt")),
        (("data/output-bf16", "bf16"), ("data/output-f32", "f32")),
    ],
    inv_check: list[tuple[str, str]] = [
        ("data/output-pywt", "pywt"),
        ("data/output-f32", "f32"),
        ("data/output-bf16", "bf16"),
    ],
):
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    results = []
    for input_descriptor in config["inputs"]:
        input_name = input_descriptor["name"]

        for wavelet in config["wavelets"]:
            wavelet_name = wavelet["name"]

            print(f"Comparing {wavelet_name} for input {input_name}...")
            for suffix, type_ in [("l_fwd", "approximation"), ("h_fwd", "details")]:
                for (cmp, cmp_name), (ref, ref_name) in cross_check:
                    with open(
                        f"{cmp}/{input_name}/{wavelet_name}_{suffix}",
                        "rb",
                    ) as f:
                        data_cmp = np.load(f, allow_pickle=True)

                    with open(
                        f"{ref}/{input_name}/{wavelet_name}_{suffix}",
                        "rb",
                    ) as f:
                        data_ref = np.load(f, allow_pickle=True)

                    result = compare(data_cmp, data_ref)
                    results.append(
                        {
                            "input": input_name,
                            "wavelet": wavelet_name,
                            "type": type_,
                            "cmp": cmp_name,
                            "ref": ref_name,
                            "abs_max": result.abs_max,
                            "rel_max": result.rel_max,
                            "rel_mean": result.rel_mean,
                            "abs_mean": result.abs_mean,
                        }
                    )

            with open(
                f"{input_path}/{input_name}",
                "rb",
            ) as f:
                data_ref = (
                    np.frombuffer(f.read(), dtype="<f8")
                    .astype(np.float64)
                    .reshape(input_descriptor["shape"])
                )

            for cmp, cmp_name in inv_check:
                with open(
                    f"{cmp}/{input_name}/{wavelet_name}_inv",
                    "rb",
                ) as f:
                    data_cmp = np.load(f, allow_pickle=True)

                # Fix for odd-length signals
                if data_cmp.shape[0] - data_ref.shape[0] == 1:
                    data_cmp = data_cmp[:-1]

                result = compare(data_cmp, data_ref)
                results.append(
                    {
                        "input": input_name,
                        "wavelet": wavelet_name,
                        "type": "reconstruction",
                        "cmp": cmp_name,
                        "ref": "signal",
                        "abs_max": result.abs_max,
                        "rel_max": result.rel_max,
                        "rel_mean": result.rel_mean,
                        "abs_mean": result.abs_mean,
                    }
                )

    pl.DataFrame(results).write_csv(output_path)


if __name__ == "__main__":
    main()
