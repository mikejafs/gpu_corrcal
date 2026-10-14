"""
Accuracy: GpuSparseCov end to end against CorrCal's SparseCov in float64.

Runs the exact call sequence the likelihood uses,

    cov = cov.copy(); cov.apply_gains(g, a1, a2); cinv, logdet = cov.inv(True); cinv @ v

on both sides. The per-kernel tests isolate each launcher; this one checks the
composition and the class wiring (workspace aliasing, which arrays feed which
kernel, the N^-1 the inverse object carries).

Metrics:
    logdet     |err| / |ref|
    C^-1 @ v   per-block ||got - ref|| / ||term_scale||, with term_scale built
               from CorrCal's float64 inverse factors (see cpu_reference).

TOL_INV is provisional: unlike test_matvec_inv, the GPU inverse factors carry
their own float32 error here. Read the printed max over a first run and
tighten.
"""

import cupy as cp
import numpy as np

from accuracy_helpers import block_rel_err, cached, make_host_data, summarize, to_gpu
from cpu_reference import _cov, term_scale
from gpu_corrcal.sparse import GpuSparseCov

TOL_LOGDET = 1e-5
TOL_INV = 1e-3


def _gains_inputs(rc, seed):
    n_ant = rc[0] * rc[1]
    a1, a2 = np.tril_indices(n_ant, k=-1)
    rng = np.random.default_rng(seed + 1)
    gains = rng.random(2 * n_ant).astype(np.float32)
    return gains, a1.astype(np.int32), a2.astype(np.int32)


def _ref(d, gains, a1, a2):
    n_eig = d["diffuse"].shape[1]
    cov = _cov(d["noise"], d["diffuse"], d["source"], d["edges"], n_eig).copy()
    cov.apply_gains(gains.astype(np.float64), a1.astype(np.int64), a2.astype(np.int64))
    cinv, logdet = cov.inv(return_det=True)
    vec = np.asarray(d["vec"], dtype=np.float64)
    return {
        "logdet": np.array(logdet),
        "out": cinv @ vec,
        "scale": term_scale(cinv.noise, cinv.diff_mat, cinv.src_mat, vec, d["edges"]),
    }


def test_sparsecov_gains_inv_matvec(rc, n_eig, n_src, seed, refresh):
    d = make_host_data(rc, n_eig, n_src, seed)
    gains, a1, a2 = _gains_inputs(rc, seed)
    ref = cached(("sparsecov_e2e", rc, n_eig, n_src, seed),
                 lambda: _ref(d, gains, a1, a2), refresh)

    g = to_gpu(d)
    cov = GpuSparseCov(g["noise"], g["source"], g["diffuse"], g["edges"], n_eig)
    c = cov.copy()
    c.apply_gains(cp.asarray(gains), cp.asarray(a1), cp.asarray(a2))
    cinv, logdet = c.inv(return_det=True)
    got = cinv @ g["vec"]
    cp.cuda.Device().synchronize()

    err_ld = abs(float(logdet) - float(ref["logdet"])) / abs(float(ref["logdet"]))
    err_mv = block_rel_err(got, ref["out"], d["edges"], scale=ref["scale"])

    results = [
        summarize("logdet (gained)", err_ld, TOL_LOGDET),
        summarize("C^-1 @ v (gained)", err_mv, TOL_INV),
    ]
    report = "\n  ".join(line for _, line in results)
    print("\n  " + report)
    assert all(ok for ok, _ in results), report