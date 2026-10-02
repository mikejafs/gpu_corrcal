"""
Accuracy: log determinant from inv_cov(compute_det=True)
(kernels/fused_chol_inv_kernel.cu for the diffuse blocks,
 kernels/sig_prime_kernels_pta.cu for the source level)
against CorrCal's CPU SparseCov.inv(return_det=True) in float64.

Both sides compute log det C minus log|N|:

    total = 2 sum_b sum_i log L_Delta,b[i,i]  +  2 sum_i log L_Sigma[i,i]

Three checks, so a failure points at the responsible kernel:

    diffuse blocks  per-block |err| / max(|ref_b|, 1), from ws.out_diffuse_det
    source part     |err| / max(|ref|, 1),           from ws.out_source_det
    total           |err| / |ref|,                   the value inv_cov returns

An absolute error in a log is a relative error in the determinant, so the
block metric is relative-error-in-det for blocks with |logdet_b| < 1.
"""

import cupy as cp
import numpy as np

from accuracy_helpers import cached, f64, make_host_data, summarize, to_gpu
from cpu_reference import ref_logdet
from gpu_corrcal.inverse_covariance import InvCovWorkspace, inv_cov

# NumPy float32 mirror on 18x32 measures: blocks <= 4e-5 (the n_eig=20 blocks
# are ill-conditioned because noise reaches 1e-6), source <= 2e-5, total <= 2e-7.
TOL_BLOCKS = 2e-4
TOL_SOURCE = 1e-4
TOL_TOTAL = 1e-6


def test_logdet(rc, n_eig, n_src, seed, refresh):
    d = make_host_data(rc, n_eig, n_src, seed)
    ref = cached(("logdet", rc, n_eig, n_src, seed), lambda: ref_logdet(d), refresh)

    g = to_gpu(d)
    ws = InvCovWorkspace(g["diffuse"], g["source"], g["edges"])
    ws.out_diffuse_det.fill(cp.nan)
    ws.out_source_det.fill(cp.nan)
    _, _, logdet = inv_cov(g["noise"], g["diffuse"], g["source"], g["edges"], ws, compute_det=True)
    cp.cuda.Device().synchronize()

    got_blocks = 2.0 * f64(ws.out_diffuse_det)
    got_source = 2.0 * float(ws.out_source_det[0])
    got_total = float(logdet)

    ref_blocks = ref["diffuse_blocks"]
    ref_source = float(ref["source"])
    ref_total = float(ref["total"])

    err_blocks = np.abs(got_blocks - ref_blocks) / np.maximum(np.abs(ref_blocks), 1.0)
    err_source = abs(got_source - ref_source) / max(abs(ref_source), 1.0)
    err_total = abs(got_total - ref_total) / abs(ref_total)

    results = [
        summarize("diffuse blocks", err_blocks, TOL_BLOCKS),
        summarize("source part", err_source, TOL_SOURCE),
        summarize("total", err_total, TOL_TOTAL),
    ]
    report = "\n  ".join(line for _, line in results)
    report += (f"\n  total: gpu={got_total:.8e}  cpu={ref_total:.8e}  "
               f"|diff|={abs(got_total - ref_total):.3e}")
    print("\n  " + report)

    assert all(ok for ok, _ in results), report