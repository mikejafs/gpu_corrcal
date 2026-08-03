from gpu_corrcal.linalg import *
from gpu_corrcal.utils.tools import *
import time

class InvCovWorkspace:
    def __init__(self, diffuse, src, edges):

        nb = len(edges) - 1
        n_eig = diffuse.shape[1]
        n_src = src.shape[1]
        n_total = diffuse.shape[0]

        # ---- Diffuse level (Del_prime chain) ----
        self.temp2 = cp.empty((len(edges)-1, diffuse.shape[1], diffuse.shape[1]), dtype=cp.float32)
        self.L_del = cp.empty((len(edges)-1, diffuse.shape[1], diffuse.shape[1]), dtype=cp.float32)
        # self.out_diffuse_det = cp.empty((len(edges)-1), dtype=cp.float32)
        self.diffuse_bar = cp.empty((diffuse.shape[0], diffuse.shape[1]), dtype=cp.float32)
        # ... etc, one buffer per intermediate the chain needs


        
        # ---- source level (Sig_prime chain) ----
        # B: per-group factor, written by kernel 1, read by kernel 3.
        #    Fully overwritten each call -> cp.empty is safe.
        self.B = cp.empty((nb, n_src, n_eig), dtype=cp.float32)
        # M_sig: the single (n_src x n_src) capacitance Lambda. ACCUMULATOR
        #    (atomicAdd across groups) -> must be zeroed before every cap_reduce.
        self.M_sig = cp.zeros((n_src, n_src), dtype=cp.float32)
        # L_sig_inv: lower-tri inverse of chol(I + Lambda). Fully overwritten.
        self.L_sig_inv = cp.empty((n_src, n_src), dtype=cp.float32)
        # sig_prime (Sig_bar): the source-level output. Fully overwritten.
        self.sig_prime = cp.empty((n_total, n_src), dtype=cp.float32)

        #log determinant components
        self.out_diffuse_det = cp.empty((nb), dtype=cp.float32)
        self.out_source_det = cp.empty((1), dtype = cp.float32)


def inv_cov(noise, diffuse, source, edges, init_workspace, compute_det=False):
    """
    Custom kernel inverse covariance.
    
    Function description and params
    fill later ...
    """

    temp2 = make_small_blocks(
        noise, diffuse, edges, out = init_workspace.temp2
    )
    # print(f"temp2: max={float(cp.max(cp.abs(temp2))):.2e}, nan={int(cp.sum(cp.isnan(temp2)))}, inf={int(cp.sum(cp.isinf(temp2)))}")

    L_del_inv_T = fused_cholesky_inverse(
        temp2, edges, out=init_workspace.L_del, out_diffuse_det=init_workspace.out_diffuse_det if compute_det else None,
    )
    # print(f"L_inv: max={float(cp.max(cp.abs(L_del_inv_T))):.2e}, nan={int(cp.sum(cp.isnan(L_del_inv_T)))}, inf={int(cp.sum(cp.isinf(L_del_inv_T)))}")

    diffuse_bar = mul_temp_by_diffuse_chol(
        noise, diffuse, L_del_inv_T, init_workspace.diffuse_bar, edges
    )

    # ===================== source level (Sig_prime) ======================
    # M_sig is an accumulator: reset to zero before cap_reduce atomicAdds.
    init_workspace.M_sig.fill(0.0)
 
    # Kernel 1: build per-group B, accumulate Lambda into M_sig.
    # diffuse_bar plays the role of Del_prime (the whitened diffuse factor).
    # sync()
    # _t0 = time.perf_counter()
    B, M_sig = make_cap_reduce(
        source, diffuse_bar, noise, edges,
        init_workspace.B, init_workspace.M_sig
    )
    # sync()
    # print(f"cap_reduce real: {(time.perf_counter()-_t0)*1e6:.1f} us")

    # print(f"M_sig max={float(cp.abs(M_sig).max()):.3e}")
 
    # print("M_sig ptr in :", init_workspace.M_sig.data.ptr)
    # print("M_sig ptr out:", M_sig.data.ptr)
    # print(f"B max={float(cp.abs(B).max()):.3e}")
    # Kernel 2: L_sig_inv = (chol(I + Lambda))^{-1}.
    L_sig_inv = fused_cholesky_inverse_sig(
        M_sig, out=init_workspace.L_sig_inv, out_source_det=init_workspace.out_source_det if compute_det else None
    )
 
    # Kernel 3: assemble Gamma and apply (L_sig^T)^{-1} -> sig_prime.
    sig_prime = apply_sig_prime_launch(
        source, diffuse_bar, noise, edges,
        B, L_sig_inv, init_workspace.sig_prime
    )
    # ret = apply_sig_prime_launch(source, diffuse_bar, noise, edges, B, L_sig_inv, init_workspace.sig_prime)
    # if ret != 0:
    #     raise RuntimeError(f"apply_sig_prime dispatch failed: n_src={source.shape[1]}, n_eig={len(edges)-1}, ret={ret}")

    if compute_det:
        logdet = 2.0* (cp.sum(init_workspace.out_diffuse_det) + init_workspace.out_source_det[0]) #FOR NOW ONLY INCLUDE THE DIFFUSE PART
        old_logdet = 2.0* (cp.sum(init_workspace.out_diffuse_det)) #FOR NOW ONLY INCLUDE THE DIFFUSE PART

        # print(f"CUSTOM: old={old_logdet}, new={logdet}")

        return diffuse_bar, sig_prime, logdet
    else:
        return diffuse_bar, sig_prime

    return diffuse_bar