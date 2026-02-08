# Autotests manual

## Config

`config.json` is the file used to describe test parameters.

- `"name"`: Unique test name (for convenience)
- `"kernel_path"`: Path to the C++ kernel to be executed.
- `"wavelet"`: Wavelet type:
    - `haar` for Haar
    - `brior2.2` for CDF 5/3
    - ...
- `"dim"`: Signal dimensions:
    - `nd`, where `n` stands for dimensionality
- shape
- `"atol"`: Absolute tolerance:
    - For standard case, **0.01 (1%)** is usually sufficient to account for rounding errors in `bfloat16`.

## ...