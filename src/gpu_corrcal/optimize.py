"""
GPU counterpart of corrcal.optimize.

nll follows corrcal.optimize.nll line for line, with GpuSparseCov in place of
SparseCov. grad_nll comes next, once the accumulate_gradient kernel exists.
"""

import cupy as cp
import numpy as np

from gpu_corrcal.sparse import GpuSparseCov


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

    return float(chisq + logdet.astype(cp.float64) + phs_norm)