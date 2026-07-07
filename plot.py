import matplotlib.pyplot as plt
import pandas as pd

WAVELET = "db7"
df = pd.read_csv("tt_wavelet_timings.csv")
# df = pd.read_csv("tests_optimization/tt_wavelet_timings_scale_opt.csv")
df = df[df["wavelet"] == WAVELET]
df = df.sort_values("signal_length")

plt.figure(figsize=(8, 5))

plt.plot(
    df["signal_length"],
    df["tt_wavelet_mean_s"],
    marker="o",
    label="TT-wavelet",
)

plt.plot(
    df["signal_length"],
    df["pywt_mean_s"],
    marker="o",
    label="PyWavelets",
)

plt.yscale("log")

plt.xlabel("Signal length")
plt.ylabel("Mean time (s), log scale")
plt.title(f"TT-wavelet vs PyWavelets for {WAVELET}")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()

plt.savefig(f"{WAVELET}_tt_vs_pywt_log_y_new.svg", format="svg")
