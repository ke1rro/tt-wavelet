# [Feature Request]: Implement Fast Wavelet Transform (FWT) via Lifting Scheme for DSP

## Is your feature request related to a problem? Please describe

While work on Fast Fourier Transform (FFT) is currently in progress within the tt-metal ecosystem ([#21412](https://github.com/tenstorrent/tt-metal/issues/21412)), FFT alone is insufficient for modern signal processing tasks dealing with **non-stationary signals**. FFT provides frequency information but completely loses temporal localization - it cannot tell _when_ a particular frequency event occurred.

The Fast Wavelet Transform applications are vast and critical for many domains:

- **Advanced DSP**: Processing audio, biosignals, radar, and seismic data where _when_ a frequency occurs is as important as _which_ frequency. Short-Time Fourier Transform (STFT) attempts to solve this but suffers from fixed resolution (Heisenberg uncertainty principle) and higher computational cost.
- **Next-Gen AI Models**: Emerging architectures like **WaveKAN** (Wavelet Kolmogorov-Arnold Networks) replace standard activation functions with wavelet bases to improve convergence, interpretability, and function approximation. Currently, there are no native ops to accelerate these models on Wormhole/Blackhole.
- **Signal Compression**: Wavelet transforms are the foundation of JPEG 2000 and modern audio codecs, offering superior compression without block artifacts compared to DCT-based methods.
- **Denoising**: Wavelet thresholding is one of the most effective techniques for noise removal across domains - from medical imaging to financial time series.

## Describe the solution you'd like

We propose implementing **Fast Wavelet Transform (FWT)** and **Inverse FWT** as a logical extension to the DSP capabilities being built around FFT.

Our team has been exploring the **Lifting Scheme** (Sweldens, 1996) as the implementation approach. We believe this is the right fit for Tenstorrent hardware for the following reasons:

### Why Lifting Scheme?

| Property | Convolution-based DWT | Lifting Scheme |
|---|---|---|
| **Complexity** | O(N) | O(N) with smaller constants |
| **Memory** | Requires auxiliary buffers | **In-place** computation |
| **DRAM bandwidth** | Higher | **Minimal** - critical for L1/SRAM utilization |
| **Reversibility** | Separate inverse implementation | Trivially invertible (reverse the steps) |
| **Customizability** | Fixed filter banks | Easily extensible to new wavelet types |

The in-place property is especially valuable for TT-Metal architecture where L1 memory is the most precious resource and minimizing DMA transfers is key to performance.

### Proposed Roadmap

**Phase 1 - Core 2D Implementation (Lifting Scheme)**

- Implement the core **2D Discrete Wavelet Transform (DWT)** and **Inverse DWT (IDWT)** using the Lifting Scheme, optimized for Tenstorrent's 2D torus mesh topology
- The 2D core is designed to work efficiently for **any dimensionality** (1D, 2D, 3D) — the API will expose all variants, but the underlying engine targets 2D as the primary compute path
- Support for standard wavelet families:
  - **Haar** (simplest, good baseline and building block)
  - **Daubechies** (db2, db4 - widely used in signal processing and compression)
- The Lifting Scheme decomposes the transform into three simple steps:
  1. **Split**: Separate even/odd indexed samples
  2. **Predict**: Predict odd samples from even (generates detail coefficients)
  3. **Update**: Update even samples (generates approximation coefficients)
- Multi-level decomposition support
- Public API covering 1D, 2D, and N-D transforms built on top of the 2D core

**Phase 2 - Optimization & Benchmarking**

- Optimize sharding strategies to maximize L1 memory utilization on Tenstorrent chips
- Benchmark against CPU and GPU implementations
- Profile and minimize DMA transfers using in-place Lifting Scheme properties

## Describe alternatives you've considered

- **Waiting for FFT completion and using STFT**: FFT is excellent for stationary signals, but using Short-Time Fourier Transform (STFT) to approximate wavelet-like analysis is computationally more expensive, has fixed time-frequency resolution trade-offs, and fundamentally cannot match the multi-resolution capability of wavelets.

- **CPU Fallback**: Performing FWT on the host CPU creates a massive bottleneck due to PCIe data movement, making it unusable for real-time DSP pipelines or high-throughput WaveKAN training/inference.

- **Convolution-based DWT**: While functionally equivalent, it requires additional memory buffers and more complex memory management compared to the Lifting Scheme approach, which is suboptimal for TT-Metal's memory hierarchy.

### Computational Advantage

FWT has **O(N) complexity** compared to FFT's O(N log N). For large-scale signal processing on hardware accelerators optimized for throughput, this translates to significant performance gains.

### References

- Sweldens, W. (1996). "The Lifting Scheme: A Custom-Design Construction of Biorthogonal Wavelets" — [Paper](https://doi.org/10.1006/acha.1996.0015)
- Bhatnagar et al. (2024). "WaveKAN: Wavelet Kolmogorov-Arnold Networks" — [arXiv:2405.12832](https://arxiv.org/abs/2405.12832)
- FFT Feature Request on tt-metal — [#21412](https://github.com/tenstorrent/tt-metal/issues/21412)
- SIGGRAPH Course: "Wavelets in Computer Graphics"

Our team is interested in working on this as part of the bounty program. We have already started prototyping the Lifting Scheme approach.
