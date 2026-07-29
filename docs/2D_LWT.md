# 2D LWT architecture and implementation status

This document specifies the vertical-first, separable 2D LWT architecture for
Tenstorrent hardware. It assumes the reader already understands the
[1D LWT](./LWT.md) implementation.

The exact dependency-cone planner, mandatory tile-padding contract, dense
full-tile horizontal stencil, `K=14..17` vertical stencil, and fused
reader/compute/writer device program are implemented. The current device path
is a five-plane implementation:
intermediate vertical and horizontal routes remain core-local, and only
LL/LH/HL/HH are written to DRAM.

## Note on tensors

Persistent 2D workspace uses full FP32 `32x32` tiles. Row-major arrays below
are mathematical pseudocode only.

## Main idea

The production contract is vertical-first: apply the 1D transform to each
column and then apply it to every row of the two vertical bands. This produces
LL, LH, HL, and HH. The first letter names the vertical result and the second
letter names the horizontal result.

For this we should do the following steps (for original matrix $D$ of the shape $2M \times 2N$):

```python
def lwt2d(D: np.ndarray):
    M = D.shape[0] // 2 # Assume height is even
    N = D.shape[1] // 2 # Assume width is even

    L = np.zeros((M, 2*N))
    H = np.zeros((M, 2*N))

    for c in range(2*N):
        column = D[:, c] # 1D column of shape 2M
        even, odd = split(column)
        approx, details = lwt(even, odd) # Same lwt as in 1D
        L[:, c] = approx
        H[:, c] = details

    LL = np.zeros((M, N))
    LH = np.zeros((M, N))
    HL = np.zeros((M, N))
    HH = np.zeros((M, N))
    for r in range(M):
        row = L[r, :]
        even, odd = split1d(row)
        approx, details = lwt1d(even, odd)
        LL[r, :] = approx
        LH[r, :] = details

        row = H[r, :]
        even, odd = split1d(row)
        approx, details = lwt1d(even, odd)
        HL[r, :] = approx
        HH[r, :] = details

    return LL, LH, HL, HH
```


The above approach relies on `lwt1d` program, which is described in [LWT](./LWT.md) for 1D tensors. However since we use SFPU as our core engine in Tenstorrent hardware, we can implement `lwt_extended_horizontal` and `lwt_extended_vertical` which acts like `lwt` on each row/column respectively, but does it in optimized way. For `lwt_extended_vertical`, refer to **Vertical LWT** section of this document and for `lwt_extended_horizontal`, refer to **Horizontal LWT**.

Also additional functions defined:

```python
def split_horizontal(D: np.ndarray):
    M = D.shape[0]
    N = D.shape[1] // 2 # Assume width is even

    even = np.zeros((M, N))
    odd = np.zeros((M, N))

    for r in range(M):
        for c in range(N):
            even[r, c] = D[r, 2*c]
            odd[r, c] = D[r, 2*c+1]

    return even, odd


def split_vertical(D: np.ndarray):
    M = D.shape[0] // 2 # Assume height is even
    N = D.shape[1]

    even = np.zeros((M, N))
    odd = np.zeros((M, N))

    for r in range(M):
        for c in range(N):
            even[r, c] = D[2*r, c]
            odd[r, c] = D[2*r+1, c]

    return even, odd

```

This allows us to rewrite code as:


```python
def lwt2d(D: np.ndarray):
    M = D.shape[0] // 2 # Assume height is even
    N = D.shape[1] // 2 # Assume width is even

    E, O = split_vertical(D) # This gives two Mx2N matrices

    L, H = lwt_vertical_extended(E, O)

    LE, LO = split_horizontal(L) # This gives two MxN matrices
    HE, HO = split_horizontal(H) # This gives two MxN matrices

    LL, LH = lwt_horizontal_extended(LE, LO)
    HL, HH = lwt_horizontal_extended(HE, HO)

    return LL, LH, HL, HH
```


But with this approach there are separate splits, which require separate programs. Thus we decided to move all the splits to the beginning of the wavelet transformation, so we can do them as a single fused program. Rewritten code is given as:

```python
def split2d(D: np.ndarray):
    M = D.shape[0] // 2 # Assume height is even
    N = D.shape[1] // 2 # Assume width is even
    EE = np.zeros((M, N))
    EO = np.zeros((M, N))
    OE = np.zeros((M, N))
    OO = np.zeros((M, N))

    for r in range(M):
        for c in range(N):
            EE[r, c] = D[2*r, 2*c]
            EO[r, c] = D[2*r, 2*c+1]
            OE[r, c] = D[2*r+1, 2*c]
            OO[r, c] = D[2*r+1, 2*c+1]

    return EE, EO, OE, OO

def lwt2d(D: np.ndarray):
    M = D.shape[0] // 2 # Assume height is even
    N = D.shape[1] // 2 # Assume width is even

    """
    E, O = split_vertical(D) # This gives two Mx2N matrices
    EE, EO = split_horizontal(D) # This gives two MxN matrices
    OE, OO = split_horizontal(D) # This gives two MxN matrices
    """
    # Above are actually fused in split2d
    EE, EO, OE, OO = split2d(D)

    LE, HE = lwt_vertical_extended(EE, OE)
    LO, HO = lwt_vertical_extended(EO, OO)

    LL, LH = lwt_horizontal_extended(LE, LO)
    HL, HH = lwt_horizontal_extended(HE, HO)

    return LL, LH, HL, HH
```


## Horizontal LWT

The implementation is similar to the 1D LWT, but we drop the idea of Splice and process tiles without pre-processing. 

## Exact dependency cones

Every lifting route expands dependencies along one axis only. For requested
output interval \(I=[a,b)\), a route with source offset \(s\), base offset
\(t\), and \(K\) coefficients requires:

$$
I_{\text{source}}=[a+s,b+s+K-1), \qquad
I_{\text{base}}=[a+t,b+t).
$$

Run the existing backward interval planner independently for the final band
intervals \(Q_y\) and \(Q_x\):

$$
(E_y,O_y)=B_y(Q_y,Q_y), \qquad
(E_x,O_x)=B_x(Q_x,Q_x).
$$

The exact initial polyphase rectangles are:

$$
R_{EE}=E_y\times E_x,\quad R_{EO}=E_y\times O_x,\quad
R_{OE}=O_y\times E_x,\quad R_{OO}=O_y\times O_x.
$$

[`plan_2d.hpp`](../tt-wavelet/tt_wavelet/include/lifting/plan_2d.hpp)
implements this product construction, retains per-route axis requirements,
assigns every vertical and horizontal route explicit source/base/output
rectangles and plane slots, searches rectangular band-tile chunk shapes, and
rejects candidates that exceed the configured L1 budget.

## Ownership and local workspace

One worker owns one rectangular output region in all four final bands. It
loads the exact raw-input cone, applies the boundary operator in original
image coordinates, splits locally to EE/EO/OE/OO, executes both axis
transforms in its own L1, and writes only LL/LH/HL/HH to DRAM. No worker reads
another worker's intermediate state.

The production workspace is currently four active full-tile planes plus one
scratch plane. The four-plane in-place policy remains a later optimization.
The dependency-local design removes global shift materialization but does not
remove route offsets, \(K-1\) halos, boundary mapping, or local gather/scatter
alignment.

The conservative per-core model currently accounts:

```text
workspace = sum(per-slot phase-aware tile span * 4 bytes)
circular buffers = 8 * 32 * 32 * 4
metadata = 4 * 96
synchronization and scalar NoC staging = 32 + 128
```

The planner uses these exact allocations and rejects a chunk shape when the
uniform per-core allocation would exceed the selected L1 budget.

## Current fused device pipeline

One transform is one Metalium `Program` with three kernels:

```text
reader: padded raw input -> symmetric cone gather -> EE/EO/OE/OO in L1
compute: two vertical route chains -> two horizontal route chains
writer: persist every intermediate route in local L1 -> LL/LH/HL/HH in DRAM
```

There is no device-program launch between the vertical and horizontal axes,
and no intermediate band is looped through DRAM.

The benchmark executable creates and prepares one device program per
wavelet/shape case. Warmups and repeats enqueue that same `MeshWorkload`
again. Thus `--warmup-runs 1 --repeats 3` performs four launches of one
prepared fused program, not four program constructions. Each launch is one
complete reader/compute/writer transform. Benchmark mode also enables the
program cache.

## Performance measurement

[`compare_2d_boundary_mode_timings.py`](../scripts/compare_2d_boundary_mode_timings.py)
measures prepared-workload device latency against PyWavelets for all eight
boundary modes and can also sweep the runtime-discovered worker-core count.
Input upload, coefficient preparation, and result readback remain outside the
device-only timed interval.

## FP32 contract

- workspace and circular buffers are FP32;
- destination accumulation enables `fp32_dest_acc_en`;
- taps execute in scheme order with the generated FP32 bit patterns;
- scale routes remain at their defined semantic positions;
- vertical-first axis order is fixed;
- no algebraically equivalent long filter bank may replace the lifting route
  sequence.

Horizontal-first and vertical-first transforms are algebraically equivalent,
but their FP32 results need not be bit-identical.

## Validation

Build the four production executables and run the Python validators:

```bash
source scripts/set_env.sh
cmake --build build --target lwt ilwt lwt_2d ilwt_2d -j
.venv/bin/python scripts/validate_lwt_2d_extension_modes.py \
  --schemes db7 --shape 64x64
.venv/bin/python scripts/validate_ilwt_2d_device.py
```

The extension-mode validator is also the retained 106-scheme by eight-mode
matrix driver. It compares production device output directly with
PyWavelets and applies the configured FP32 tolerance without a C++ reference
executable.
