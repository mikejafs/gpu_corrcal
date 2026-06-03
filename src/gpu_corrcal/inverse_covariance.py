from gpu_corrcal.linalg import *

class InvCovWorkspace:
    def __init__(self, nb, n_eig, n_src):
        self.temp2 = cp.empty((nb, n_eig, n_eig), dtype=cp.float32)
        self.L_del = cp.empty((nb, n_eig, n_eig), dtype=cp.float32)
        self.Del_prime = cp.empty((nb, n_eig, n_eig), dtype=cp.float32)
        # ... etc, one buffer per intermediate the chain needs

def inv_cov(noise, diffuse, source, edges, init_workspace):

    temp2 = make_small_blocks(
        noise, diffuse, edges, out = init_workspace.temp2
        )
    
    return temp2