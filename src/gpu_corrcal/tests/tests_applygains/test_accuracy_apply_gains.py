"""
Accuracy: apply gains to a Re/Im-split matrix (kernels/apply_gains.cu)
against corrcal.utils.apply_gains_to_mat in float64.

    test_apply_gains_diffuse   mat = diffuse, n_col = n_eig
    test_apply_gains_source    mat = source,  n_col = n_src

Metric: per-block ||got - ref|| / ||ref|| over the redundant groups in edges.
The gain multiplication is a complex scaling of each Re/Im row pair, so the
block norm of ref is |g|-weighted ||mat_b|| and never cancels; plain relative
error is the right scale. The output buffer is pre-filled with NaN, so rows
the kernel never writes fail.
"""

import cupy as cp
import numpy as np

from accuracy_helpers import block_rel_err, make_host_data, summarize
from cpu_reference import ref_apply_gains
from gpu_corrcal.linalg import gpu_apply_gains

# One complex multiply per element: a NumPy float32 mirror of the kernel
# measures <= 1.2e-7 max relative error.
TOL = 1e-5


def _gains_inputs(rc, seed):
    """Antenna arrays in tril order (as SimCorrcalParams) and random gains."""
    n_ant = rc[0] * rc[1]
    a1, a2 = np.tril_indices(n_ant, k=-1)
    rng = np.random.default_rng(seed + 1)
    gains = rng.random(2 * n_ant).astype(np.float32)
    return gains, a1.astype(np.int32), a2.astype(np.int32)


def _run(label, mat, rc, seed, edges):
    gains, a1, a2 = _gains_inputs(rc, seed)
    assert mat.shape[0] == 2 * a1.size, "baseline count does not match edges"

    ref = ref_apply_gains(gains, mat, a1, a2)

    out = cp.full(mat.shape, cp.nan, dtype=cp.float32)
    gpu_apply_gains(cp.asarray(gains), cp.ascontiguousarray(cp.asarray(mat)),
                    cp.asarray(a1), cp.asarray(a2), out=out)
    cp.cuda.Device().synchronize()

    errs = block_rel_err(out, ref, edges)
    passed, line = summarize(label, errs, TOL)
    print("\n  " + line)
    assert passed, line


def test_apply_gains_diffuse(rc, n_eig, seed):
    d = make_host_data(rc, n_eig, 1, seed)
    _run("G Delta", d["diffuse"], rc, seed, d["edges"])


def test_apply_gains_source(rc, n_src, seed):
    d = make_host_data(rc, 1, n_src, seed)
    _run("G Sigma", d["source"], rc, seed, d["edges"])