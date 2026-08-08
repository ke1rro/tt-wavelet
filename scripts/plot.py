import matplotlib.pyplot as plt
import pandas as pd

WAVELET = "db7"
input_file = "tt_wavelet_timings.csv"
output_file = f"{WAVELET}_log_scale_runtime_plot.png"

df = pd.read_csv(input_file)
if "wavelet" in df.columns:
    df = df[df["wavelet"] == WAVELET]

col_x = "signal_length" if "signal_length" in df.columns else ("signal_width" if "signal_width" in df.columns else df.columns[0])
df = df.sort_values(col_x)

plt.figure(figsize=(7, 4.5))

if "pywt_mean_s" in df.columns:
    plt.plot(
        df[col_x],
        df["pywt_mean_s"],
        label="PyWavelets",
    )

tt_col = "tt_wavelet_mean_s" if "tt_wavelet_mean_s" in df.columns else "tt_mean_s"
if tt_col in df.columns:
    plt.plot(
        df[col_x],
        df[tt_col],
        label="tt-wavelet",
    )

if "ttnn_mean_s" in df.columns:
    plt.plot(
        df[col_x],
        df["ttnn_mean_s"],
        label="ttnn-wavelet",
    )

plt.yscale("log")

plt.xlabel(col_x.replace("_", " ").capitalize())
plt.ylabel("Runtime (s, log scale)")
plt.title(f"{WAVELET} runtime vs {col_x.replace('_', ' ')}")

plt.grid(True, which="both", linestyle=":")
plt.legend()
plt.tight_layout()

plt.savefig(output_file, dpi=200)
plt.show()
