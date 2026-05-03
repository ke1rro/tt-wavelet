import matplotlib.pyplot as plt
import pandas as pd

WAVELET = "coif2"
df = pd.read_csv("tt_wavelet_timings.csv")
df = df[df["status"] == "ok"]
# df = df[df["wavelet"] == "db7"]
df = df.sort_values("signal_length")

plt.figure(figsize=(8, 5))
plt.plot(df["signal_length"], df["tt_wavelet_mean_s"], marker="o", label="TT-wavelet")
plt.plot(df["signal_length"], df["pywt_mean_s"], marker="o", label="PyWavelets")

plt.xlabel("Signal length")
plt.ylabel("Mean time (s)")
plt.title("TT-wavelet vs PyWavelets for db7")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"{WAVELET}_tt_vs_pywt.svg", format="svg")
