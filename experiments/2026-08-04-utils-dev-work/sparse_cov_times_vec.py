

"""
ctypes bindings for sparse_cov_times_vec.cu

out = N*vec + s*( Del @ (Del^T vec) + Sig @ (Sig^T vec) ),   s = -1 if isinv else +1

Flat/ragged layout matching the diffuse_bar kernel:
    noise    (n_row,)          float32   -- caller passes N^-1 when isinv=True
    diffuse  (n_row, n_eig)    float32   C-contiguous
    source   (n_row, n_src)    float32   C-contiguous
    vec      (n_row,)          float32
    edges    (nb+1,)           int32
"""

import os
import ctypes
import numpy as np
import cupy as cp

# ---------------------------------------------------------------------------
# library load
# ---------------------------------------------------------------------------
_LIB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "sparse_cov_times_vec.so"
)
_lib = ctypes.CDLL(_LIB_PATH)

_lib.launch_sparse_cov_times_vec.restype = None
_lib.launch_sparse_cov_times_vec.argtypes = [
    ctypes.c_void_p,   # noise
    ctypes.c_void_p,   # diffuse
    ctypes.c_void_p,   # source
    ctypes.c_void_p,   # vec
    ctypes.c_void_p,   # out
    ctypes.c_void_p,   # del_tmp workspace
    ctypes.c_void_p,   # sig_tmp workspace
    ctypes.c_void_p,   # edges (int32)
    ctypes.c_int,      # nb
    ctypes.c_int,      # n_src
    ctypes.c_int,      # n_eig
    ctypes.c_int,      # is_inv
    ctypes.c_void_p,   # cudaStream_t
]

MAX_N_EIG = 20
MAX_N_SRC = 20


def _ptr(arr):
    """Raw device pointer from a CuPy array."""
    return ctypes.c_void_p(int(arr.data.ptr))


# ---------------------------------------------------------------------------
# workspace
# ---------------------------------------------------------------------------
class MatvecWorkspace:
    """
    Preallocated scratch for sparse_cov_times_vec.

    Instantiate once before the CG loop -- nb, n_eig and n_src are fixed across
    iterations, so nothing here needs to be reallocated.

    del_tmp is fully overwritten every call and is cp.empty-safe.
    sig_tmp is an atomic accumulator; it is zeroed inside the C launcher, so it
    does NOT need to be zeroed from Python.
    """

    def __init__(self, nb, n_eig, n_src, out_size=None):
        self.nb = int(nb)
        self.n_eig = int(n_eig)
        self.n_src = int(n_src)

        self.del_tmp = cp.empty(self.nb * self.n_eig, dtype=cp.float32)
        self.sig_tmp = cp.empty(self.n_src, dtype=cp.float32)

        # optional persistent output buffer for the CG loop
        self.out = (
            cp.empty(int(out_size), dtype=cp.float32)
            if out_size is not None
            else None
        )


# ---------------------------------------------------------------------------
# wrapper
# ---------------------------------------------------------------------------
def sparse_cov_times_vec(
    noise,
    diffuse,
    source,
    vec,
    edges,
    ws,
    isinv=True,
    out=None,
    check=True,
):
    """
    Apply the sparse covariance (or its inverse) to a vector.

    Parameters
    ----------
    noise    : (n_row,) float32 cupy array.  Pass N^-1 when isinv=True.
    diffuse  : (n_row, n_eig) float32, C-contiguous.  Pass Del_bar when isinv=True.
    source   : (n_row, n_src) float32, C-contiguous.  Pass Sig_bar when isinv=True.
    vec      : (n_row,) float32
    edges    : (nb+1,) int32
    ws       : MatvecWorkspace sized for this (nb, n_eig, n_src)
    isinv    : sign convention.  True  -> N*vec - Del@.. - Sig@..
                                 False -> N*vec + Del@.. + Sig@..
    out      : optional preallocated (n_row,) float32 output
    check    : run the shape/dtype assertions (disable in the hot loop)

    Returns
    -------
    (n_row,) float32 cupy array
    """
    n_row = vec.shape[0]
    nb = edges.shape[0] - 1
    n_eig = diffuse.shape[1]
    n_src = source.shape[1]

    if check:
        for name, a in (("noise", noise), ("diffuse", diffuse),
                        ("source", source), ("vec", vec)):
            assert a.dtype == cp.float32, f"{name} must be float32, got {a.dtype}"
            assert a.flags.c_contiguous, f"{name} must be C-contiguous"
        assert edges.dtype == cp.int32, f"edges must be int32, got {edges.dtype}"
        assert noise.shape[0] == n_row, "noise/vec length mismatch"
        assert diffuse.shape[0] == n_row, "diffuse row count mismatch"
        assert source.shape[0] == n_row, "source row count mismatch"
        assert vec.ndim == 1, "vec must be flat; ravel a (nb, m) array first"
        assert 1 <= n_eig <= MAX_N_EIG, f"n_eig={n_eig} outside compiled grid"
        assert 1 <= n_src <= MAX_N_SRC, f"n_src={n_src} outside compiled grid"
        assert (ws.nb, ws.n_eig, ws.n_src) == (nb, n_eig, n_src), (
            f"workspace is sized ({ws.nb}, {ws.n_eig}, {ws.n_src}), "
            f"call is ({nb}, {n_eig}, {n_src})"
        )
        assert int(edges[-1]) == n_row, (
            f"edges[-1]={int(edges[-1])} does not match n_row={n_row}"
        )

    if out is None:
        out = ws.out if ws.out is not None else cp.empty(n_row, dtype=cp.float32)

    stream = cp.cuda.get_current_stream()

    _lib.launch_sparse_cov_times_vec(
        _ptr(noise),
        _ptr(diffuse),
        _ptr(source),
        _ptr(vec),
        _ptr(out),
        _ptr(ws.del_tmp),
        _ptr(ws.sig_tmp),
        _ptr(edges),
        ctypes.c_int(nb),
        ctypes.c_int(n_src),
        ctypes.c_int(n_eig),
        ctypes.c_int(1 if isinv else 0),
        ctypes.c_void_p(stream.ptr),
    )

    return out


# ---------------------------------------------------------------------------
# CuPy reference (flat layout, for validation)
# ---------------------------------------------------------------------------
def sparse_cov_times_vec_ref(noise, diffuse, source, vec, edges, isinv=True):
    """Straight CuPy implementation in the flat/ragged layout."""
    sig_tmp = source.T @ vec                      # (n_src,)
    sig_term = source @ sig_tmp                   # (n_row,)

    del_term = cp.empty_like(vec)
    edges_h = cp.asnumpy(edges)
    for b in range(len(edges_h) - 1):
        s, e = int(edges_h[b]), int(edges_h[b + 1])
        D = diffuse[s:e]
        del_term[s:e] = D @ (D.T @ vec[s:e])

    sgn = -1.0 if isinv else 1.0
    return noise * vec + sgn * (del_term + sig_term)


# ---------------------------------------------------------------------------
# test data
# ---------------------------------------------------------------------------
def make_test_data(nb=64, rows_per_block=96, n_eig=8, n_src=5, seed=0, ragged=True):
    rng = np.random.default_rng(seed)

    if ragged:
        counts = rng.integers(rows_per_block // 2, rows_per_block * 2, size=nb)
    else:
        counts = np.full(nb, rows_per_block)
    edges_h = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    n_row = int(edges_h[-1])

    noise = cp.asarray(rng.uniform(0.5, 2.0, n_row), dtype=cp.float32)
    diffuse = cp.asarray(rng.normal(size=(n_row, n_eig)), dtype=cp.float32)
    source = cp.asarray(rng.normal(size=(n_row, n_src)), dtype=cp.float32)
    vec = cp.asarray(rng.normal(size=n_row), dtype=cp.float32)
    edges = cp.asarray(edges_h, dtype=cp.int32)

    return noise, diffuse, source, vec, edges, nb, n_row


# ---------------------------------------------------------------------------
# __main__ switchboard
# ---------------------------------------------------------------------------
def _validate(nb=64, n_eig=8, n_src=5, ragged=True, verbose=True):
    noise, diffuse, source, vec, edges, nb, n_row = make_test_data(
        nb=nb, n_eig=n_eig, n_src=n_src, ragged=ragged
    )
    ws = MatvecWorkspace(nb, n_eig, n_src)

    ok = True
    for isinv in (True, False):
        ref = sparse_cov_times_vec_ref(noise, diffuse, source, vec, edges, isinv)
        cp.cuda.Stream.null.synchronize()

        got = sparse_cov_times_vec(
            noise, diffuse, source, vec, edges, ws, isinv=isinv
        )
        cp.cuda.Stream.null.synchronize()

        scale = float(cp.abs(ref).max())
        maxdiff = float(cp.abs(got - ref).max())
        rel = maxdiff / scale
        passed = rel < 1e-4
        ok &= passed

        if verbose:
            print(
                f"  isinv={str(isinv):5s}  n_eig={n_eig:2d} n_src={n_src:2d} "
                f"n_row={n_row:6d}  max|d|={maxdiff:.3e}  rel={rel:.3e}  "
                f"{'PASS' if passed else 'FAIL'}"
            )
    return ok


def _sweep():
    """Correctness across the template grid, both sign conventions."""
    print("sweeping n_eig x n_src ...")
    all_ok = True
    for n_eig in (1, 2, 4, 8, 12, 16, 20):
        for n_src in (1, 3, 8, 20):
            all_ok &= _validate(n_eig=n_eig, n_src=n_src, verbose=True)
    print("ALL PASS" if all_ok else "FAILURES PRESENT")


def _sentinel():
    """Fill out with a sentinel to catch rows the kernel never writes."""
    noise, diffuse, source, vec, edges, nb, n_row = make_test_data()
    ws = MatvecWorkspace(nb, 8, 5)
    out = cp.full(n_row, cp.nan, dtype=cp.float32)
    sparse_cov_times_vec(noise, diffuse, source, vec, edges, ws, out=out)
    cp.cuda.Stream.null.synchronize()
    n_unwritten = int(cp.isnan(out).sum())
    print(f"unwritten rows: {n_unwritten} / {n_row}")


def _bench(nb=512, rows_per_block=256, n_eig=3, n_src=5):
    from cupyx.profiler import benchmark

    noise, diffuse, source, vec, edges, nb, n_row = make_test_data(
        nb=nb, rows_per_block=rows_per_block, n_eig=n_eig, n_src=n_src
    )
    ws = MatvecWorkspace(nb, n_eig, n_src)
    out = cp.empty(n_row, dtype=cp.float32)

    k = benchmark(
        lambda: sparse_cov_times_vec(
            noise, diffuse, source, vec, edges, ws, out=out, check=False
        ),
        n_repeat=100,
    )
    r = benchmark(
        lambda: sparse_cov_times_vec_ref(noise, diffuse, source, vec, edges),
        n_repeat=20,
    )
    kt = float(np.mean(k.gpu_times)) * 1e3
    rt = float(np.mean(r.gpu_times)) * 1e3
    print(f"n_row={n_row}  kernel {kt:.4f} ms   cupy-ref {rt:.4f} ms   {rt/kt:.2f}x")


if __name__ == "__main__":
    import sys
    mode = "sweep"
    # mode = sys.argv[1] if len(sys.argv) > 1 else "bench"

    # mode == "validate"

    if mode == "validate":
        _validate()
    elif mode == "sweep":
        _sweep()
    elif mode == "sentinel":
        _sentinel()
    elif mode == "bench":
        _bench()
    else:
        print(f"unknown mode '{mode}'; use validate | sweep | sentinel | bench")