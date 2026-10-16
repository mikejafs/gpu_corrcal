"""
Accuracy: gpu_corrcal.optimize.grad_nll against corrcal.optimize.grad_nll in
float64.

scale=1 in every case: CorrCal's grad_nll divides the gains by scale twice,
the GPU version once, and the two only agree at scale=1.

    plain   phs_norm_fac=inf   (chi^2 + trace terms only)
    prior   phs_norm_fac=0.5   (adds the phase-prior term)

Metric: ||g_gpu - g_cpu|| / ||g_cpu||. Starting tolerance is loose
(float32 inverse feeding a sum of ~n_ant terms per antenna with cancellation
between the chi^2 and trace parts); tighten after the first run.
"""

import cupy as cp
import numpy as np
import pytest

from accuracy_helpers import cached, make_host_data, summarize, to_gpu
from cpu_reference import _cov
from corrcal.optimize import grad_nll as cpu_grad_nll
from gpu_corrcal.linalg import GradWorkspace
from gpu_corrcal.optimize import grad_nll as gpu_grad_nll
from gpu_corrcal.sparse import GpuSparseCov

TOL = 1e-4

CASES = {"plain": np.inf, "prior": 0.5}


def _gains_inputs(rc, seed):
    n_ant = rc[0] * rc[1]
    a1, a2 = np.tril_indices(n_ant, k=-1)
    rng = np.random.default_rng(seed + 1)
    gains = rng.random(2 * n_ant).astype(np.float32)
    return gains, a1.astype(np.int32), a2.astype(np.int32)


@pytest.mark.parametrize("case", list(CASES))
def test_grad_nll(rc, n_eig, n_src, seed, refresh, case):
    phs = CASES[case]
    d = make_host_data(rc, n_eig, n_src, seed)
    gains, a1, a2 = _gains_inputs(rc, seed)

    def _ref():
        cov = _cov(d["noise"], d["diffuse"], d["source"], d["edges"], n_eig)
        g = cpu_grad_nll(
            gains.astype(np.float64), cov, np.asarray(d["vec"], dtype=np.float64),
            a1.astype(np.int64), a2.astype(np.int64), scale=1, phs_norm_fac=phs,
        )
        return {"grad": np.asarray(g)}

    ref = cached(("grad_nll", case, rc, n_eig, n_src, seed), _ref, refresh)

    g = to_gpu(d)
    cov = GpuSparseCov(g["noise"], g["source"], g["diffuse"], g["edges"], n_eig)
    a1_d, a2_d = cp.asarray(a1), cp.asarray(a2)
    ws = GradWorkspace(a1_d, a2_d, gains.size // 2, cov.n_grp, n_eig, n_src)
    got = gpu_grad_nll(cp.asarray(gains), cov, g["vec"], a1_d, a2_d,
                       scale=1, phs_norm_fac=phs, grad_ws=ws)

    want = ref["grad"]
    err = np.linalg.norm(got - want) / np.linalg.norm(want)
    passed, line = summarize(f"grad_nll ({case})", err, TOL)
    worst = int(np.argmax(np.abs(got - want)))
    print(f"\n  {line}\n  worst entry {worst}: gpu={got[worst]:.6e}  cpu={want[worst]:.6e}")
    assert passed, line