"""
Accuracy: the fused objectives nll_and_grad_cupy_tail and nll_and_grad.

Three tests, run from src/gpu_corrcal/tests:

    pytest -s tests_nll_and_grad                       # everything, default grid
    pytest -s tests_nll_and_grad -k separate           # 1. only fused vs separate
    pytest -s tests_nll_and_grad --sizes 6x8 -k fd     # 3. only finite differences

1. test_fused_vs_separate
   Fused vs the separate GPU nll + grad_nll on the same inputs. Both paths run
   the same kernels, so the only allowed difference is run-to-run float32
   noise from the float atomics in the inverse (cap_reduce) and the matvec.
   That noise floor is measured in the test by calling nll and grad_nll twice;
   the fused results must sit within max(TOL, 10 x floor) of the separate ones.
   A buffer-ordering or workspace-reuse bug shows up orders of magnitude above.

2. test_fused_vs_cpu
   Fused vs CorrCal (float64), same metrics and tolerances as the nll and
   grad_nll accuracy tests. CorrCal's grad_nll divides by scale twice, so the
   reference is always computed at scale=1 and rescaled exactly:
       nll(x, scale)  = nll(x/scale, 1)
       grad(x, scale) = grad_nll(x/scale, 1) / scale

3. test_fused_fd
   The fused GPU gradient projected on random unit directions v, against a
   central difference of CorrCal's float64 nll along v, at scale=1 and
   scale=2 with the phase prior on. Checks the gradient against the exact
   derivative of the objective (sign, 1/scale, prior coefficient), small grid
   only. The difference is taken on the CPU objective because the float32 GPU
   objective's noise divided by 2h caps a GPU-side difference near 1e-2.

   Noise is set equal on the Re and Im rows of each baseline. CorrCal's
   gradient (and so grad_nll, which ports it) uses noise[2k] alone in the
   trace term, which is exact only when noise[2k] == noise[2k+1]. With the
   independent per-row noise of make_host_data, CorrCal's own float64 grad_nll
   misses finite differences of its own nll by ~100%; with equal Re/Im noise
   it agrees to ~1e-8. Real visibility noise is circular (equal Re/Im
   variance), so this is the case that matters.
"""

import cupy as cp
import numpy as np
import pytest

from accuracy_helpers import cached, make_host_data, summarize, to_gpu
from cpu_reference import _cov
from corrcal.optimize import grad_nll as cpu_grad_nll
from corrcal.optimize import nll as cpu_nll
from gpu_corrcal.linalg import GradWorkspace, NllWorkspace
from gpu_corrcal.optimize import grad_nll, nll, nll_and_grad, nll_and_grad_cupy_tail
from gpu_corrcal.sparse import GpuSparseCov

TOL_VS_SEP_F = 1e-6    # relative, nll
TOL_VS_SEP_G = 1e-6    # relative 2-norm, gradient
TOL_CPU_F = 1e-5       # as tests_accuracy_nll
TOL_CPU_G = 1e-4       # as tests_accuracy_grad_nll
TOL_FD = 1e-3

# (scale, phs_norm_fac)
CASES = {"plain": (1.0, np.inf), "prior": (1.0, 0.5), "scaled": (2.0, 0.5)}

FUSED = {"cupy_tail": nll_and_grad_cupy_tail, "kernel_tail": nll_and_grad}


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------
def _gains_inputs(rc, seed):
    """Same gains and baselines as the nll / grad_nll accuracy tests."""
    n_ant = rc[0] * rc[1]
    a1, a2 = np.tril_indices(n_ant, k=-1)
    rng = np.random.default_rng(seed + 1)
    gains = rng.random(2 * n_ant).astype(np.float32)
    return gains, a1.astype(np.int32), a2.astype(np.int32)


def _gpu_problem(rc, n_eig, n_src, seed, equal_reim_noise=False):
    d = make_host_data(rc, n_eig, n_src, seed)
    if equal_reim_noise:
        d["noise"] = d["noise"].copy()
        d["noise"][1::2] = d["noise"][0::2]
    g = to_gpu(d)
    cov = GpuSparseCov(g["noise"], g["source"], g["diffuse"], g["edges"], n_eig)
    gains, a1, a2 = _gains_inputs(rc, seed)
    a1_d, a2_d = cp.asarray(a1), cp.asarray(a2)
    gws = GradWorkspace(a1_d, a2_d, gains.size // 2, cov.n_grp, n_eig, n_src)
    return d, g["vec"], cov, gains, a1, a2, a1_d, a2_d, gws


def _call_fused(name, x, cov, data, a1_d, a2_d, scale, phs, gws, nws):
    if name == "kernel_tail":
        return nll_and_grad(x, cov, data, a1_d, a2_d, scale, phs, gws, nws)
    return nll_and_grad_cupy_tail(x, cov, data, a1_d, a2_d, scale, phs, gws)


def _rel(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# ---------------------------------------------------------------------------
# 1. fused vs separate GPU
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("which", list(FUSED))
@pytest.mark.parametrize("case", list(CASES))
def test_fused_vs_separate(rc, n_eig, n_src, seed, case, which):
    scale, phs = CASES[case]
    _, data, cov, gains, _, _, a1_d, a2_d, gws = _gpu_problem(rc, n_eig, n_src, seed)
    nws = NllWorkspace()

    # scipy hands the objective host float64; do the same here.
    x = gains.astype(np.float64) * scale

    f_sep = nll(x, cov, data, a1_d, a2_d, scale, phs)
    g_sep = grad_nll(x, cov, data, a1_d, a2_d, scale, phs, gws)
    f_sep2 = nll(x, cov, data, a1_d, a2_d, scale, phs)
    g_sep2 = grad_nll(x, cov, data, a1_d, a2_d, scale, phs, gws)
    floor_f = abs(f_sep2 - f_sep) / abs(f_sep)
    floor_g = _rel(g_sep2, g_sep)

    f, g = _call_fused(which, x, cov, data, a1_d, a2_d, scale, phs, gws, nws)

    assert isinstance(f, float)
    assert g.dtype == np.float64 and g.shape == x.shape

    err_f = abs(f - f_sep) / abs(f_sep)
    err_g = _rel(g, g_sep)
    tol_f = max(TOL_VS_SEP_F, 10 * floor_f)
    tol_g = max(TOL_VS_SEP_G, 10 * floor_g)
    ok_f, line_f = summarize(f"{which} nll ({case})", err_f, tol_f)
    ok_g, line_g = summarize(f"{which} grad ({case})", err_g, tol_g)
    print(f"\n  {line_f}\n  {line_g}"
          f"\n  run-to-run floor: nll {floor_f:.1e}, grad {floor_g:.1e}"
          f"\n  fused nll={f:.10e}  separate nll={f_sep:.10e}")
    assert ok_f and ok_g, f"{line_f}\n{line_g}"


def test_kernel_tail_parts(rc, n_eig, n_src, seed):
    """The three terms nll_tail sums, each against its CuPy counterpart."""
    _, data, cov, gains, _, _, a1_d, a2_d, gws = _gpu_problem(rc, n_eig, n_src, seed)
    x = gains.astype(np.float64)
    phs = 0.5

    _, _, parts = nll_and_grad(x, cov, data, a1_d, a2_d, 1.0, phs, gws, NllWorkspace(),
                               return_parts=True)

    gained = cov.copy()
    gained.apply_gains(cp.asarray(x, dtype=cp.float32), a1_d, a2_d)
    cinv, logdet = gained.inv(return_det=True)
    p = cinv @ data
    chisq = float(cp.sum(data * p, dtype=cp.float64))
    ph = np.arctan2(x[1::2], x[0::2])
    want = np.array([chisq, float(logdet), np.mean(ph) ** 2 / phs ** 2])

    errs = np.abs(parts - want) / np.abs(want)
    print(f"\n  chisq  kernel={parts[0]:.10e}  cupy={want[0]:.10e}  rel={errs[0]:.1e}"
          f"\n  logdet kernel={parts[1]:.10e}  cupy={want[1]:.10e}  rel={errs[1]:.1e}"
          f"\n  phs    kernel={parts[2]:.10e}  numpy={want[2]:.10e}  rel={errs[2]:.1e}")
    # chisq / logdet: float32 inverse, run-to-run noise; phs: double vs double.
    assert errs[0] < 1e-5 and errs[1] < 1e-5 and errs[2] < 1e-12


# ---------------------------------------------------------------------------
# 2. fused vs CorrCal
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", list(CASES))
def test_fused_vs_cpu(rc, n_eig, n_src, seed, refresh, case):
    scale, phs = CASES[case]
    d, data, cov, gains, a1, a2, a1_d, a2_d, gws = _gpu_problem(rc, n_eig, n_src, seed)

    def _ref():
        c = _cov(d["noise"], d["diffuse"], d["source"], d["edges"], n_eig)
        args = (gains.astype(np.float64), c, np.asarray(d["vec"], dtype=np.float64),
                a1.astype(np.int64), a2.astype(np.int64))
        f = cpu_nll(*args, scale=1, phs_norm_fac=phs)
        g = cpu_grad_nll(*args, scale=1, phs_norm_fac=phs)
        return {"nll": np.array(f), "grad": np.asarray(g)}

    ref = cached(("nll_and_grad", phs, rc, n_eig, n_src, seed), _ref, refresh)
    want_f = float(ref["nll"])
    want_g = ref["grad"] / scale

    x = gains.astype(np.float64) * scale
    f, g = nll_and_grad(x, cov, data, a1_d, a2_d, scale, phs, gws, NllWorkspace())

    err_f = abs(f - want_f) / abs(want_f)
    err_g = _rel(g, want_g)
    ok_f, line_f = summarize(f"nll_and_grad nll ({case})", err_f, TOL_CPU_F)
    ok_g, line_g = summarize(f"nll_and_grad grad ({case})", err_g, TOL_CPU_G)
    print(f"\n  {line_f}\n  {line_g}\n  gpu nll={f:.10e}  cpu nll={want_f:.10e}")
    assert ok_f and ok_g, f"{line_f}\n{line_g}"


# ---------------------------------------------------------------------------
# 3. finite differences of the fused objective
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_fused_fd(rc, n_eig, n_src, seed, scale):
    """
    grad . v from the fused GPU call against a central difference of CorrCal's
    float64 nll along v.

    The difference is taken on the CPU objective, not the GPU one: the GPU nll
    is a float32 computation with run-to-run noise from float atomics, and
    dividing that noise by 2h limits a GPU-side difference to ~1e-2 relative
    for any h, which hides real errors at that level. CorrCal's float64 nll
    differences cleanly to ~1e-8, so the residual here is the GPU gradient's
    own error (float32, ~1e-4) plus any formula mistake. The fused nll itself
    is checked against CorrCal's nll in test_fused_vs_cpu.
    """
    if rc != (6, 8):
        pytest.skip("finite differences on the 6x8 grid only")
    d, data, cov, _, a1, a2, a1_d, a2_d, gws = _gpu_problem(rc, n_eig, n_src, seed,
                                                            equal_reim_noise=True)
    nws = NllWorkspace()
    phs = 0.5
    n_ant = rc[0] * rc[1]

    c = _cov(d["noise"], d["diffuse"], d["source"], d["edges"], n_eig)
    data_h = np.asarray(d["vec"], dtype=np.float64)
    a1_h, a2_h = a1.astype(np.int64), a2.astype(np.int64)
    f_cpu = lambda y: cpu_nll(y, c, data_h, a1_h, a2_h, scale=scale, phs_norm_fac=phs)

    # Gains near 1 (a realistic operating point). Random gains in [0, 1) put
    # some |g_k1 g_k2| near zero, where the objective's curvature wrecks
    # finite differences.
    rng = np.random.default_rng(seed + 2)
    g = 1 + 0.1 * rng.standard_normal(n_ant) + 0.1j * rng.standard_normal(n_ant)
    x = np.empty(2 * n_ant)
    x[0::2], x[1::2] = g.real, g.imag
    x *= scale

    _, grad = nll_and_grad(x, cov, data, a1_d, a2_d, scale, phs, gws, nws)

    # Error normalised by ||grad||, the same scale as the 2-norm metric of
    # test_fused_vs_cpu. Normalising by |grad . v| instead blows up for
    # directions nearly orthogonal to grad, where grad . v is small.
    gnorm = float(np.linalg.norm(grad))
    h = 1e-4
    worst = 0.0
    for trial in range(3):
        v = rng.standard_normal(x.size)
        v /= np.linalg.norm(v)
        an = float(grad @ v)
        fd = (f_cpu(x + h * v) - f_cpu(x - h * v)) / (2 * h)
        err = abs(fd - an) / gnorm
        worst = max(worst, err)
        print(f"\n  scale={scale} dir {trial}: gpu grad.v={an:.8e}  cpu fd={fd:.8e}  |fd - grad.v|/|grad|={err:.1e}")
    ok, line = summarize(f"fd (scale={scale})", worst, TOL_FD)
    print(f"  {line}")
    assert ok, line
