# 2D LWT implementation proposal

This document describes implementation details of proposed 2D LWT algorithm on Tenstorrent hardware. It assumes the reader to already understand [1D LWT](./LWT.md) implementation.

## Note on tensors

All tensors we discuss in this document are considered to be two-dimensional (in tile layout) or one-dimensional (in row-major layout), but latter is used only for demonstrations, and final code will work solely with 2D tensors in tile layout. For detailed documentation refer to [Tenstorrent documentation]().

## Main idea

We will not explain why, but 2D wavelet transform is similar to first applying 1D wavelet transform along horizontal axis (for each row) and then on each of the bands (produced by vertical transformation: approximation and details) we apply 1D wavelet transform along vertical axis (for each column in each band). This produces 4 2D matrices/tensors: LL, LH, HL, HH (for each axis respectively: **h**igh frequencies - details, **l**ow frequencies - approximations).

For this we should do the following steps (for original matrix $D$ of the shape $2M \times 2N$):

```python
def lwt2d(D: np.ndarray):
    M = D.shape[0] // 2 # Assume height is even
    N = D.shape[1] // 2 # Assume width is even

    L = np.zeros(M, 2*N) # create empty matrix Mx2N
    H = np.zeros(M, 2*N) # create empty matrix Mx2N

    for c in range(2*N):
        row = D[:, c] # Row is now 1D matrix of shape 2Mx1
        even, odd = split(row) # Same split as in 1D LWT -> even and odd are of shape Mx1
        approx, details = lwt(even, odd) # Same lwt as in 1D
        L[:, c] = approx
        H[:, c] = odd

    LL = np.zeros(M, N)
    LH = np.zeros(M, N)
    HL = np.zeros(M, N)
    HH = np.zeros(M, N)
    for r in range(M):
        column = L[r, :] # Column is now 1D matrix of shape 1x2N
        even, odd = split1d(column.T) # Do same split on transpose of column
        approx, details = lwt1d(even, odd)
        LL[r, :] = approx
        HL[r, :] = details

        column = H[r, :] # Column is now 1D matrix of shape 1x2N
        even, odd = split1d(column.T) # Do same split on transpose of column
        approx, details = lwt1d(even, odd)
        LH[r, :] = approx
        HH[r, :] = details
```


The above approach relies on `lwt1d` program, which is described in [LWT](./LWT.md) for 1D tensors. However since we use SFPU as our core engine in Tenstorrent hardware, we can implement `lwt_extended_horizontal` and `lwt_extended_vertical` which acts like `lwt` on each row/column respectively, but does it in optimized way. For `lwt_extended_vertical`, refer to **Vertical LWT** section of this document and for `lwt_extended_horizontal`, refer to **Horizontal LWT**.

Also additional functions defined:

```python
def split_horizontal(D: np.ndarray):
    M = D.shape[0]
    N = d.shape[1] // 2 # Assume width is even

    even = zeros(M, N)
    odd = zeros(M, N)

    for r in range(M):
        for c in range(N):
            even[r, c] = D[r, 2*c]
            odd[r, c] = D[r, 2*c+1]

    return even, odd


def split_vertical(D: np.ndarray):
    M = D.shape[0] // 2 # Assume height is even
    N = d.shape[1]

    even = np.zeros(M, N)
    odd = np.zeros(M, N)

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
    EE = np.zeros(M, N)
    EO = np.zeros(M, N)
    OE = np.zeros(M, N)
    OO = np.zeros(M, N)

    for r in range(M):whole tiles
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

