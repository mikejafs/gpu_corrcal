"""
Accuracy: gpu_corrcal.optimize.nll against corrcal.optimize.nll in float64.

Both sides receive the same float32 model, data and gains. Two cases:

    plain   scale=1, phs_norm_fac=inf   (chi^2 + logdet only)
    scaled  scale=2, phs_norm_fac=0.5   (exercises the gain scaling and the
                                         phase prior)

Metric: |nll_gpu - nll_cpu| / |nll_cpu|.
"""

import cupy as cp
import numpy as np
import pytest

from accuracy_helpers import cached, make_host_data, summarize, to_gpu
from cpu_reference import _cov
from corrcal.optimize import nll as cpu_nll
from gpu_corrcal.optimize import nll as gpu_nll
from gpu_corrcal.sparse import GpuSparseCov

TOL = 1e-5

CASES = {"plain": (1.0, np.inf), "scaled": (2.0, 0.5)}


def _gains_inputs(rc, seed):
    n_ant = rc[0] * rc[1]
    a1, a2 = np.tril_indices(n_ant, k=-1)
    rng = np.random.default_rng(seed + 1)
    gains = rng.random(2 * n_ant).astype(np.float32)
    return gains, a1.astype(np.int32), a2.astype(np.int32)


@pytest.mark.parametrize("case", list(CASES))
def test_nll(rc, n_eig, n_src, seed, refresh, case):
    scale, phs = CASES[case]
    d = make_host_data(rc, n_eig, n_src, seed)
    gains, a1, a2 = _gains_inputs(rc, seed)
    gains_in = (gains * np.float32(scale)).astype(np.float32)

    def _ref():
        cov = _cov(d["noise"], d["diffuse"], d["source"], d["edges"], n_eig)
        val = cpu_nll(
            gains_in.astype(np.float64), cov, np.asarray(d["vec"], dtype=np.float64),
            a1.astype(np.int64), a2.astype(np.int64), scale=scale, phs_norm_fac=phs,
        )
        return {"nll": np.array(val)}

    ref = cached(("nll", case, rc, n_eig, n_src, seed), _ref, refresh)

    g = to_gpu(d)
    cov = GpuSparseCov(g["noise"], g["source"], g["diffuse"], g["edges"], n_eig)
    got = gpu_nll(cp.asarray(gains_in), cov, g["vec"], cp.asarray(a1), cp.asarray(a2),
                  scale=scale, phs_norm_fac=phs)

    want = float(ref["nll"])
    err = abs(got - want) / abs(want)
    passed, line = summarize(f"nll ({case})", err, TOL)
    print(f"\n  {line}\n  gpu={got:.10e}  cpu={want:.10e}")
    assert passed, line