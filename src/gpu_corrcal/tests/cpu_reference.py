"""
CPU references: thin adapters over r-pascua/corrcal (float64).

Each function takes host data from accuracy_helpers.make_host_data and
returns a dict of NumPy arrays in gpu_corrcal's flat layout, so the cache
can store it directly. Every computation goes through CorrCal's own code
path (SparseCov, corrcal.linalg); nothing here reimplements the math.
"""

import numpy as np
from corrcal import linalg as cc_linalg
from corrcal.sparse import SparseCov


def _cov(noise, diffuse, source, edges, n_eig, isinv=False):
    return SparseCov(
        noise=np.asarray(noise, dtype=np.float64),
        src_mat=np.asarray(source, dtype=np.float64),
        diff_mat=np.asarray(diffuse, dtype=np.float64),
        edges=np.asarray(edges, dtype=np.int64),
        n_eig=n_eig,
        isinv=isinv,
    )


def ref_logdet(d):
    """
    log det of C excluding log|N|, matching both SparseCov.inv(return_det=True)
    and gpu_corrcal.inv_cov(compute_det=True):

        total   = 2 sum_b sum_i log L_Delta,b[i,i]  +  2 sum_i log L_Sigma[i,i]

    Also returns the per-block diffuse contributions (2 sum_i log L_Delta,b[i,i]),
    computed with CorrCal's make_small_blocks + Cholesky, and the source part
    as total minus the diffuse sum.
    """
    n_eig = d["diffuse"].shape[1]
    cov = _cov(d["noise"], d["diffuse"], d["source"], d["edges"], n_eig)
    _, total = cov.inv(return_det=True)

    small = cc_linalg.make_small_blocks(
        noise_diag=cov.noise, diff_mat=cov.diff_mat, edges=cov.edges
    )
    L = np.linalg.cholesky(np.eye(n_eig)[None] + small)
    diffuse_blocks = 2.0 * np.log(np.diagonal(L, axis1=1, axis2=2)).sum(axis=1)

    return {
        "total": np.array(total),
        "diffuse_blocks": diffuse_blocks,
        "source": np.array(total - diffuse_blocks.sum()),
    }


def term_scale(noise, diffuse, source, vec, edges):
    """
    Per-row magnitude of the matvec terms,

        |N v| + |Delta (Delta^T v)| + |Sigma (Sigma^T v)|,

    in float64. Used as the error normalization: for C^-1 v the three terms
    cancel heavily (N^-1 reaches 1e6 in the sim data), so ||err|| / ||out||
    reports ~1e-3 for a correct float32 kernel. ||err|| / ||scale|| is the
    error relative to what float32 rounding can actually resolve (~1e-7 x
    reduction length), and an indexing bug still shows up at O(1).
    Only used for normalization; the reference product itself comes from CorrCal.
    """
    N, D, S, v = (np.asarray(x, dtype=np.float64) for x in (noise, diffuse, source, vec))
    starts = np.asarray(edges[:-1], dtype=np.int64)
    counts = np.diff(np.asarray(edges, dtype=np.int64))

    dt = np.add.reduceat(D * v[:, None], starts, axis=0)        # (nb, n_eig)
    del_term = np.einsum("ij,ij->i", D, np.repeat(dt, counts, axis=0))
    sig_term = S @ (S.T @ v)
    return np.abs(N * v) + np.abs(del_term) + np.abs(sig_term)


def ref_matvec_cov(d):
    """C @ v with C = N + Delta Delta^T + Sigma Sigma^T."""
    n_eig = d["diffuse"].shape[1]
    cov = _cov(d["noise"], d["diffuse"], d["source"], d["edges"], n_eig)
    return {
        "out": cov @ np.asarray(d["vec"], dtype=np.float64),
        "scale": term_scale(d["noise"], d["diffuse"], d["source"], d["vec"], d["edges"]),
    }


def ref_matvec_inv(d):
    """
    C^-1 @ v using CorrCal's sparse inverse.

    The inverse factors (N^-1, Delta_bar, Sigma_bar) are rounded to float32
    and returned, because those rounded arrays are what the GPU kernel gets
    as input. The reference product is then computed in float64 from the
    same rounded factors, so the comparison isolates the matvec kernel.
    """
    n_eig = d["diffuse"].shape[1]
    cinv = _cov(d["noise"], d["diffuse"], d["source"], d["edges"], n_eig).inv()

    ninv32 = cinv.noise.astype(np.float32)
    dbar32 = cinv.diff_mat.astype(np.float32)
    sbar32 = cinv.src_mat.astype(np.float32)

    cinv_rounded = _cov(ninv32, dbar32, sbar32, d["edges"], n_eig, isinv=True)
    out = cinv_rounded @ np.asarray(d["vec"], dtype=np.float64)

    return {
        "noise": ninv32,
        "diffuse": dbar32,
        "source": sbar32,
        "out": out,
        "scale": term_scale(ninv32, dbar32, sbar32, d["vec"], d["edges"]),
    }