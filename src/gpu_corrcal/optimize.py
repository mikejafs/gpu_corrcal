"""
GPU counterpart of corrcal.optimize.

nll follows corrcal.optimize.nll line for line, with GpuSparseCov in place of
SparseCov. grad_nll comes next, once the accumulate_gradient kernel exists.
"""

import cupy as cp
import numpy as np

from gpu_corrcal.sparse import GpuSparseCov
from gpu_corrcal.linalg import GradWorkspace, gpu_grad_nll



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
    return out
 