import tomllib

import numpy as np


def compare(
    data_cmp: np.ndarray,
    data_ref: np.ndarray,  # Reference
    suffix: str,
):
    if data_cmp.shape != data_ref.shape:
        print(f"Shape mismatch for {suffix}: {data_cmp.shape} vs {data_ref.shape}")
        return

    diff = data_cmp - data_ref
    mask = diff != 0
    rel = np.abs(diff[mask] / data_ref[mask])
    abs_ = np.abs(diff[~mask])
    print(
        f"  {suffix}: max absolute error (zero elements only) = {np.max(abs_) if abs_.size > 0 else 0}"
    )
    print(f"  {suffix}: max relative error = {np.max(rel) if rel.size > 0 else 0}")
    print(f"  {suffix}: mean relative error = {np.mean(rel) if rel.size > 0 else 0}")


def main(
    config_path: str = "config.toml",
    # input_path: str = "data/input",
    # output_paths: str = [
    #     "data/output-pywt",
    #     "data/output-f32",
    #     "data/output-bf16",
    # ],
    cross_check: tuple[str, str] = [
        ("data/output-f32", "data/output-pywt"),
        ("data/output-bf16", "data/output-f32"),
    ],
):
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    for cmp, ref in cross_check:
        print(f"Comparing {cmp} against {ref}...")
        for input_descriptor in config["inputs"]:
            print(
                f"Input: {input_descriptor['name']} with shape {input_descriptor['shape']} and magnitude {input_descriptor['magnitude']}"
            )
            input_name = input_descriptor["name"]
            shape = input_descriptor["shape"]

            if len(shape) == 1:
                suffixes = [
                    ("l_fwd", ((shape[0] + 1) // 2,)),
                    ("h_fwd", ((shape[0]) // 2,)),
                    ("inv", shape),
                ]
            elif len(shape) == 2:
                print(
                    f"2D input {input_name} with shape {shape} is not supported in this benchmark, skipping."
                )
                continue
                # suffixes = ["ll_fwd", "lh_fwd", "hl_fwd", "hh_fwd", "inv"]
            else:
                print(f"Unsupported shape {shape} for input {input_name}, skipping.")
                continue

            for wavelet in config["wavelets"]:
                wavelet_name = wavelet["name"]

                print(f"Comparing {wavelet_name} for input {input_name}...")
                for suffix, subshape in suffixes:
                    try:
                        try:
                            with open(
                                f"{cmp}/{input_name}/{wavelet_name}_{suffix}",
                                "rb",
                            ) as f:
                                data_cmp = (
                                    np.frombuffer(f.read(), dtype="<f8")
                                    .astype(np.float64)
                                    .reshape(subshape)
                                )
                        except FileNotFoundError:
                            print(
                                f"File not found: {cmp}/{input_name}/{wavelet_name}_{suffix}, skipping."
                            )
                            continue

                        try:
                            with open(
                                f"{ref}/{input_name}/{wavelet_name}_{suffix}",
                                "rb",
                            ) as f:
                                data_ref = (
                                    np.frombuffer(f.read(), dtype="<f8")
                                    .astype(np.float64)
                                    .reshape(subshape)
                                )
                        except FileNotFoundError:
                            print(
                                f"File not found: {ref}/{input_name}/{wavelet_name}_{suffix}, skipping."
                            )
                            continue

                        compare(data_cmp, data_ref, suffix)
                    except Exception as e:
                        print(
                            f"Error comparing {cmp} and {ref} for {input_name} with wavelet {wavelet_name} and suffix {suffix}: {e}"
                        )


if __name__ == "__main__":
    main()
