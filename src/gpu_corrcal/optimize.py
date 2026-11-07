"""
GPU counterpart of corrcal.optimize.

nll and grad_nll follow corrcal.optimize.nll / grad_nll, with GpuSparseCov in
place of SparseCov. nll_and_grad returns both from one inverse, for
scipy.optimize.minimize(..., jac=True).
"""

import cupy as cp
import numpy as np

from gpu_corrcal.sparse import GpuSparseCov
from gpu_corrcal.linalg import GradWorkspace, NllWorkspace, gpu_grad_nll, gpu_nll_tail

# def chisq_f64(cinv, data):
#     """d^T C^-1 d with C^-1 = N^-1 - Dbar Dbar^T - Sbar Sbar^T, accumulated in float64."""
#     d = data.astype(cp.float64)
#     t_noise = cp.sum(d * d * cinv.noise.astype(cp.float64))
#     prod = cinv.diff_mat.astype(cp.float64) * d[:, None]              # (n_row, n_eig)
#     cs = cp.concatenate([cp.zeros((1, prod.shape[1])), cp.cumsum(prod, axis=0)])
#     y_del = cs[cinv.edges[1:]] - cs[cinv.edges[:-1]]                  # (n_grp, n_eig): Dbar_b^T d_b
#     y_src = cinv.src_mat.astype(cp.float64).T @ d                     # (n_src,)
#     return t_noise - cp.sum(y_del ** 2) - cp.sum(y_src ** 2)

def nll(
    gains,
    cov: GpuSparseCov,
    data,
    ant_1_inds,
    ant_2_inds,
    scale: float = 1,
    phs_norm_fac: float = np.inf,
) -> float:
    """
    CorrCal negative log-likelihood (Pascua+ Eq. 61), up to a constant:

        log det C - log det N  +  d^T C^-1 d  +  mean(phi)^2 / phs_norm_fac^2

    Parameters
    ----------
    gains : (2*n_ant,) alternating Re/Im, host or device. Optimizers such as
        scipy hand in host float64; it is uploaded as float32 here.
    cov : GpuSparseCov of the ungained model. Not modified: the gained copy
        writes into cov.ws (see sparse.py).
    data : (n_row,) device float32, Re/Im-split and sorted into redundant
        groups like the model. Convert once before the optimizer loop.
    ant_1_inds, ant_2_inds : (n_bls,) device int32.
    scale : factor the gains were multiplied by before optimization.
    phs_norm_fac : width of the Gaussian prior on the mean gain phase.
        np.inf disables the prior.

    Returns
    -------
    Host float. Converting it forces one device sync per evaluation, which
    a host-side optimizer needs anyway.
    """
    gains = cp.asarray(gains)

    cov = cov.copy()
    cov.apply_gains(gains / scale, ant_1_inds, ant_2_inds)
    cinv, logdet = cov.inv(return_det=True)

    # p = C^-1 d in float32 (kernel). The length-n_row reduction d . p
    # accumulates in float64 so chi^2 keeps resolution as n_row grows.
    p = cinv @ data
    chisq = cp.sum(data * p, dtype=cp.float64)

    # chisq = chisq_f64(cinv, data)

    phases = cp.arctan2(gains[1::2], gains[::2])
    phs_norm = cp.mean(phases) ** 2 / phs_norm_fac ** 2

    return float(chisq + logdet + phs_norm)


# def grad_nll(
#     gains,
#     cov: GpuSparseCov,
#     data,
#     ant_1_inds,
#     ant_2_inds,
#     scale: float = 1,
#     phs_norm_fac: float = np.inf,
#     grad_ws: GradWorkspace = None,
# ):
#     """
#     Gradient of the CorrCal negative log-likelihood (Pascua+ Eq. 71) with
#     respect to the Re/Im gains, as corrcal.optimize.grad_nll.
 
#     Same arguments as nll, plus
#     grad_ws : GradWorkspace built once from (ant_1_inds, ant_2_inds). Built
#         here if None, which costs a sort over 2 n_bl entries per call: pass
#         one in for benchmarking and inside the solver.
 
#     Divides the gains by scale ONCE (CorrCal divides twice; the two agree at
#     scale=1). Everything after p = C^-1 d is one launcher (grad_nll_kernels.cu).
 
#     Returns
#     -------
#     (2*n_ant,) host float64, the layout scipy.optimize.minimize expects. The
#     copy forces one device sync, as float() does in nll.
#     """
#     gains = cp.asarray(gains, dtype=cp.float32) / cp.float32(scale)
#     a1 = cp.asarray(ant_1_inds, dtype=cp.int32)
#     a2 = cp.asarray(ant_2_inds, dtype=cp.int32)
#     n_ant = gains.shape[0] // 2
 
#     if grad_ws is None:
#         grad_ws = GradWorkspace(a1, a2, n_ant, cov.n_grp, cov.n_eig, cov.n_src)
 
#     gained = cov.copy()
#     gained.apply_gains(gains, a1, a2)
#     cinv = gained.inv(return_det=False)
#     p = cinv @ data
 
#     phs_coef = 0.0 if np.isinf(phs_norm_fac) else 2.0 / (n_ant**2 * phs_norm_fac**2)
#     out = gpu_grad_nll(gains, p, gained, cinv, cov, a1, a2, grad_ws,
#                        phs_coef=phs_coef, inv_scale=1.0 / scale, check=cov.check)
#     return cp.asnumpy(out).astype(np.float64)



def grad_nll(
    gains,
    cov: GpuSparseCov,
    data,
    ant_1_inds,
    ant_2_inds,
    scale: float = 1,
    phs_norm_fac: float = np.inf,
    grad_ws: GradWorkspace = None,
):
    """
    Gradient of the CorrCal negative log-likelihood (Pascua+ Eq. 71) with
    respect to the Re/Im gains, as corrcal.optimize.grad_nll.
 
    Same arguments as nll, plus
    grad_ws : GradWorkspace built once from (ant_1_inds, ant_2_inds). Built
        here if None, which costs a sort over 2 n_bl entries per call: pass
        one in for benchmarking and inside the solver.
 
    Divides the gains by scale ONCE (CorrCal divides twice; the two agree at
    scale=1). Everything after p = C^-1 d is one launcher (grad_nll_kernels.cu).
 
    Returns
    -------
    (2*n_ant,) host float64, the layout scipy.optimize.minimize expects. The
    copy forces one device sync, as float() does in nll.
    """
    gains = cp.asarray(gains, dtype=cp.float32) / cp.float32(scale)
    a1 = cp.asarray(ant_1_inds, dtype=cp.int32)
    a2 = cp.asarray(ant_2_inds, dtype=cp.int32)
    data = cp.ascontiguousarray(cp.asarray(data, dtype=cp.float32))
    n_ant = gains.shape[0] // 2
 
    if grad_ws is None:
        grad_ws = GradWorkspace(a1, a2, n_ant, cov.n_grp, cov.n_eig, cov.n_src)
 
    gained = cov.copy()
    gained.apply_gains(gains, a1, a2)
    cinv = gained.inv(return_det=False)
    p = cinv @ data
 
    phs_coef = 0.0 if np.isinf(phs_norm_fac) else 2.0 / (n_ant**2 * phs_norm_fac**2)
    out = gpu_grad_nll(gains, p, data, gained, cinv, cov, a1, a2, grad_ws,
                       phs_coef=phs_coef, inv_scale=1.0 / scale, check=cov.check)
    return out.get().astype(np.float64)
 



# ---------------------------------------------------------------------------
# Fused objective: (nll, gradient) from ONE apply_gains + ONE inverse
# ---------------------------------------------------------------------------
# nll and grad_nll above each run apply_gains, the Woodbury inverse and
# p = C^-1 d. scipy evaluates both at every trial point, so separately they
# pay for all three twice. Both functions below run them once and return the
# pair, in the form scipy.optimize.minimize(fun, x0, jac=True) expects.
#
#   nll_and_grad_cupy_tail : nll's CuPy tail (chi^2, logdet sum, phase prior)
#                            kept as is. Pure reuse of the code in nll and
#                            grad_nll; the reference for nll_and_grad.
#   nll_and_grad           : same pipeline, the CuPy tail replaced by one
#                            nll_tail launch. Production version.
#
# Gains: uploaded once as float64 (g64). nll's phase prior is computed from
# g64, exactly as nll does. apply_gains and the gradient kernels get
# g32 = float32(g64) / scale, exactly as grad_nll builds its gains. At scale=1
# this is also what nll feeds apply_gains; at scale != 1 nll rounds g64/scale
# instead, a float32-ulp difference in the gained model.

def _fused_setup(gains, cov, data, ant_1_inds, ant_2_inds, scale, grad_ws):
    """Shared preamble: one upload, int32 indices, workspace, gained inverse inputs."""
    g64 = cp.ascontiguousarray(cp.asarray(gains, dtype=cp.float64))
    g32 = g64.astype(cp.float32)
    if scale != 1:
        g32 /= cp.float32(scale)
    a1 = cp.asarray(ant_1_inds, dtype=cp.int32)
    a2 = cp.asarray(ant_2_inds, dtype=cp.int32)
    data = cp.ascontiguousarray(cp.asarray(data, dtype=cp.float32))
    n_ant = g32.shape[0] // 2
    if grad_ws is None:
        grad_ws = GradWorkspace(a1, a2, n_ant, cov.n_grp, cov.n_eig, cov.n_src)
    gained = cov.copy()
    gained.apply_gains(g32, a1, a2)
    return g64, g32, a1, a2, data, n_ant, grad_ws, gained


def _grad_phs_coef(n_ant, phs_norm_fac):
    """Phase-prior coefficient the gradient launcher takes (as in grad_nll)."""
    return 0.0 if np.isinf(phs_norm_fac) else 2.0 / (n_ant**2 * phs_norm_fac**2)


def nll_and_grad_cupy_tail(
    gains,
    cov: GpuSparseCov,
    data,
    ant_1_inds,
    ant_2_inds,
    scale: float = 1,
    phs_norm_fac: float = np.inf,
    grad_ws: GradWorkspace = None,
):
    """
    (nll, grad_nll) from one apply_gains + one inverse + one p = C^-1 d.
    The scalar tail is nll's CuPy code unchanged.

    Arguments as grad_nll.

    Returns
    -------
    (float, (2*n_ant,) host float64) : the pair scipy's jac=True expects.
    """
    g64, g32, a1, a2, data, n_ant, grad_ws, gained = _fused_setup(
        gains, cov, data, ant_1_inds, ant_2_inds, scale, grad_ws)

    # One inverse, with the log-determinant, shared by both outputs.
    cinv, logdet = gained.inv(return_det=True)
    p = cinv @ data

    # nll tail (queued on the stream; nothing below writes p or the det buffers)
    chisq = cp.sum(data * p, dtype=cp.float64)
    phases = cp.arctan2(g64[1::2], g64[::2])
    phs_norm = cp.mean(phases) ** 2 / phs_norm_fac ** 2
    nll_dev = chisq + logdet + phs_norm

    # gradient from the same p and the same Woodbury factors in gained.ws.inv
    grad = gpu_grad_nll(g32, p, data, gained, cinv, cov, a1, a2, grad_ws,
                        phs_coef=_grad_phs_coef(n_ant, phs_norm_fac),
                        inv_scale=1.0 / scale, check=cov.check)

    return float(nll_dev), grad.get().astype(np.float64)


def nll_and_grad(
    gains,
    cov: GpuSparseCov,
    data,
    ant_1_inds,
    ant_2_inds,
    scale: float = 1,
    phs_norm_fac: float = np.inf,
    grad_ws: GradWorkspace = None,
    nll_ws: NllWorkspace = None,
    return_parts: bool = False,
):
    """
    (nll, grad_nll) from one apply_gains + one inverse + one p = C^-1 d, with
    chi^2, the log-determinant sum and the phase prior in one nll_tail launch.

    Arguments as grad_nll, plus
    nll_ws : NllWorkspace. Built here if None (three small allocations from
        the CuPy pool); pass one in inside the solver and for benchmarking.
    return_parts : also return [chisq, logdet, phs] as a host float64 array,
        for diagnostics.

    Returns
    -------
    (float, (2*n_ant,) host float64), or (float, grad, parts) with return_parts.
    """
    if nll_ws is None:
        nll_ws = NllWorkspace()

    g64, g32, a1, a2, data, n_ant, grad_ws, gained = _fused_setup(
        gains, cov, data, ant_1_inds, ant_2_inds, scale, grad_ws)

    # One inverse. Block log-determinants stay in gained.ws.inv for nll_tail.
    cinv, _ = gained.inv(return_det=True, assemble_det=False)
    p = cinv @ data

    iw = gained.ws.inv
    phs_coef = 0.0 if np.isinf(phs_norm_fac) else 1.0 / phs_norm_fac**2
    tail = gpu_nll_tail(data, p, iw.out_diffuse_det, iw.out_source_det,
                        g64, phs_coef, nll_ws, check=cov.check)

    grad = gpu_grad_nll(g32, p, data, gained, cinv, cov, a1, a2, grad_ws,
                        phs_coef=_grad_phs_coef(n_ant, phs_norm_fac),
                        inv_scale=1.0 / scale, check=cov.check)

    tail_h = tail.get()                       # waits for the whole stream
    grad_h = grad.get().astype(np.float64)    # already complete; copy only
    if return_parts:
        return float(tail_h[0]), grad_h, tail_h[1:].copy()
    return float(tail_h[0]), grad_h
