from gpu_corrcal.linalg import *
from gpu_corrcal.utils.tools import *

class InvCovWorkspace:
    def __init__(self, nb, n_eig, n_src):
        self.temp2 = cp.empty((nb, n_eig, n_eig), dtype=cp.float32)
        self.L_del = cp.empty((nb, n_eig, n_eig), dtype=cp.float32)
        self.diffuse_bar = cp.empty((nb, n_eig, n_eig), dtype=cp.float32)
        # ... etc, one buffer per intermediate the chain needs

def inv_cov(noise, diffuse, source, edges, init_workspace):

    temp2 = make_small_blocks(
        noise, diffuse, edges, out = init_workspace.temp2
    )

    # return temp2

    sync()
    L_del_inv = fused_cholesky_inverse(
        temp2, edges, out=init_workspace.L_del
    )
    
    return L_del_inv