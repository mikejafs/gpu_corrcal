"""
Shared machinery for the gpu_corrcal accuracy suite.

Every accuracy test compares a GPU kernel (float32) against the CPU CorrCal
implementation (r-pascua/corrcal, float64) on bit-identical inputs:

    1. Host data is generated in NumPy, then rounded to float32.
    2. The GPU receives those float32 arrays.
    3. CorrCal receives the same float32 arrays upcast to float64.

So the measured difference is kernel arithmetic error only, never input
rounding. CPU results are cached on disk (CACHE_DIR) because at realistic
sizes the CPU reference dominates runtime and never changes for fixed inputs.
"""

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np

from gpu_corrcal.utils.gridding import grid_redundancy_edges

TESTS_DIR = Path(__file__).resolve().parent
CACHE_DIR = TESTS_DIR / ".accuracy_cache"

# Bump this whenever the data generator or a CPU reference function changes,
# so stale cache entries are never reused.
REF_VERSION = 1


# ---------------------------------------------------------------------------
# test data
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def grid_edges(rows, cols):
    """Re/Im-split edges for a rows x cols CHORD-like grid (int64, host)."""
    edges = grid_redundancy_edges(rows, cols, precision="int64")[0]
    return 2 * edges


def make_host_data(rc, n_eig, n_src, seed=12):
    """
    Same distributions as SimCorrcalParams.sim_data, generated on the host
    so the inputs (and therefore the cached CPU reference) are reproducible
    without touching the GPU.

    Returns float32 arrays plus int64 edges.
    """
    rows, cols = rc
    edges = grid_edges(rows, cols)
    n_row = int(edges[-1])

    rng = np.random.default_rng(seed)
    return {
        "noise": (rng.random(n_row) + 1e-6).astype(np.float32),
        "diffuse": rng.random((n_row, n_eig)).astype(np.float32),
        "source": rng.random((n_row, n_src)).astype(np.float32),
        "vec": rng.random(n_row).astype(np.float32),
        "edges": edges,
    }


def to_gpu(d):
    """Upload host data in the layout/dtypes the kernels expect."""
    import cupy as cp

    return {
        "noise": cp.asarray(d["noise"]),
        "diffuse": cp.ascontiguousarray(cp.asarray(d["diffuse"])),
        "source": cp.ascontiguousarray(cp.asarray(d["source"])),
        "vec": cp.asarray(d["vec"]),
        "edges": cp.asarray(d["edges"], dtype=cp.int32),
    }


def f64(a):
    """Device or host array -> float64 NumPy."""
    if hasattr(a, "get"):
        a = a.get()
    return np.asarray(a, dtype=np.float64)


# ---------------------------------------------------------------------------
# disk cache for CPU references
# ---------------------------------------------------------------------------
def cached(key_parts, compute, refresh=False):
    """
    Return the dict of arrays produced by compute(), loading it from
    CACHE_DIR/<hash>.npz when present. key_parts must identify the inputs
    uniquely (kernel name, rc, n_eig, n_src, seed, ...).
    """
    key = "|".join(str(p) for p in (REF_VERSION, *key_parts))
    path = CACHE_DIR / (hashlib.sha1(key.encode()).hexdigest()[:16] + ".npz")

    if path.exists() and not refresh:
        with np.load(path) as f:
            return {k: f[k] for k in f.files}

    out = compute()
    CACHE_DIR.mkdir(exist_ok=True)
    np.savez(path, **out)
    return out


# ---------------------------------------------------------------------------
# error metrics
# ---------------------------------------------------------------------------
def block_rel_err(got, ref, edges, scale=None):
    """
    Per-block relative error ||got_b - ref_b|| / ||scale_b|| (Frobenius over
    the rows of block b and all columns), with scale = ref by default.
    Returns an (nb,) float64 array.
    """
    got, ref = f64(got), f64(ref)
    scale = ref if scale is None else f64(scale)
    diff2 = (got - ref) ** 2
    ref2 = scale ** 2
    if got.ndim == 2:
        diff2 = diff2.sum(axis=1)
        ref2 = ref2.sum(axis=1)
    starts = np.asarray(edges[:-1], dtype=np.int64)
    num = np.add.reduceat(diff2, starts)
    den = np.add.reduceat(ref2, starts)
    return np.sqrt(num) / np.maximum(np.sqrt(den), np.finfo(float).tiny)


def summarize(label, errs, tol):
    """
    One-line summary of a per-block error array. Returns (passed, line).
    The worst block index is what tells you whether failures cluster at
    particular block sizes or at the ragged tail.
    """
    errs = np.atleast_1d(np.asarray(errs, dtype=np.float64))
    worst = int(np.argmax(np.where(np.isfinite(errs), errs, np.inf)))
    n_bad = int((~(errs <= tol)).sum())       # NaN/inf count as failures
    passed = n_bad == 0
    where = f"(block {worst:>5d})" if errs.size > 1 else " " * 13
    line = (
        f"{label:<22s} max={errs[worst]:.2e} {where}  "
        f"median={np.median(errs):.2e}  over_tol={n_bad}/{errs.size}  "
        f"tol={tol:.0e}  {'PASS' if passed else 'FAIL'}"
    )
    return passed, line