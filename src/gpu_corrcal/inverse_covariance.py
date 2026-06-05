from gpu_corrcal.linalg import *
from gpu_corrcal.utils.tools import *

class InvCovWorkspace:
    def __init__(self, diffuse, src, edges):
        self.temp2 = cp.empty((len(edges)-1, diffuse.shape[1], diffuse.shape[1]), dtype=cp.float32)
        self.L_del = cp.empty((len(edges)-1, diffuse.shape[1], diffuse.shape[1]), dtype=cp.float32)
        self.diffuse_bar = cp.empty((diffuse.shape[0], diffuse.shape[1]), dtype=cp.float32)
        # ... etc, one buffer per intermediate the chain needs

def inv_cov(noise, diffuse, source, edges, init_workspace):
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
        temp2, edges, out=init_workspace.L_del
    )
    # print(f"L_inv: max={float(cp.max(cp.abs(L_del_inv_T))):.2e}, nan={int(cp.sum(cp.isnan(L_del_inv_T)))}, inf={int(cp.sum(cp.isinf(L_del_inv_T)))}")

    diffuse_bar = mul_temp_by_diffuse_chol(
        noise, diffuse, L_del_inv_T, init_workspace.diffuse_bar, edges
    )

    return diffuse_bar