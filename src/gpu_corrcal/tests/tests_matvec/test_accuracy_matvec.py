"""
Accuracy: sparse covariance times vector (kernels/sparse_cov_times_vec.cu)
against CorrCal's CPU SparseCov.__matmul__ in float64.

    test_matvec_cov   C    @ v,  isinv=False, inputs (N, Delta, Sigma)
    test_matvec_inv   C^-1 @ v,  isinv=True,  inputs (N^-1, Delta_bar, Sigma_bar)
                      taken from CorrCal's SparseCov.inv(), so this isolates the
                      matvec kernel from the GPU inverse.

Metric: per-block ||got - ref|| / ||term_scale||  (see cpu_reference.term_scale).
The output buffer is pre-filled with NaN, so rows the kernel never writes, or
an unsupported (n_eig, n_src) that makes the launcher return early, fail.
"""

import cupy as cp
import pytest

from accuracy_helpers import block_rel_err, cached, make_host_data, summarize, to_gpu
from cpu_reference import ref_matvec_cov, ref_matvec_inv
from gpu_corrcal.linalg import MatvecWorkspace, sparse_cov_vec_mul

# float32 reductions over up to ~3000 rows per block and ~3000 blocks for
# Sigma^T v. A NumPy float32 mirror of the kernel measures <= 2e-6 on 18x32.
TOL_COV = 5e-5
TOL_INV = 5e-5


def _gpu_matvec(noise, diffuse, source, vec, edges, isinv):
    nb = edges.shape[0] - 1
    ws = MatvecWorkspace(nb, diffuse.shape[1], source.shape[1])
    out = cp.full(vec.shape[0], cp.nan, dtype=cp.float32)
    sparse_cov_vec_mul(noise, diffuse, source, vec, edges, ws, isinv=isinv, out=out)
    cp.cuda.Device().synchronize()
    return out


def _check(label, got, ref, edges, tol):
    errs = block_rel_err(got, ref["out"], edges, scale=ref["scale"])
    passed, line = summarize(label, errs, tol)
    print("\n  " + line)
    assert passed, line


def test_matvec_cov(rc, n_eig, n_src, seed, refresh):
    d = make_host_data(rc, n_eig, n_src, seed)
    ref = cached(("matvec_cov", rc, n_eig, n_src, seed), lambda: ref_matvec_cov(d), refresh)

    g = to_gpu(d)
    got = _gpu_matvec(g["noise"], g["diffuse"], g["source"], g["vec"], g["edges"], isinv=False)

    _check("C @ v", got, ref, d["edges"], TOL_COV)


def test_matvec_inv(rc, n_eig, n_src, seed, refresh):
    d = make_host_data(rc, n_eig, n_src, seed)
    ref = cached(("matvec_inv", rc, n_eig, n_src, seed), lambda: ref_matvec_inv(d), refresh)

    g = to_gpu(d)
    got = _gpu_matvec(
        cp.asarray(ref["noise"]),
        cp.ascontiguousarray(cp.asarray(ref["diffuse"])),
        cp.ascontiguousarray(cp.asarray(ref["source"])),
        g["vec"], g["edges"], isinv=True,
    )

    _check("C^-1 @ v", got, ref, d["edges"], TOL_INV)