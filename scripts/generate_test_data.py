import cupy as cp
from gpu_corrcal.utils.gridding import *
from gpu_corrcal.utils.simulate_params import *

# ============================================================
# Test data generation
# ============================================================

def make_test_data(n_eig, rc_list, seed):
    """Generate random test data using the simulate params library"""
    cp.random.seed(seed)

    rows = rc_list[0]
    cols = rc_list[1]

    n_ant = rows*cols
    # print(f" n_eig={n_eig}", end="", flush=True)

    spms = SimCorrcalParams(n_ant, n_eig, n_src=1, precision='float32', xp=cp)
    edges = spms.edges(rows, cols, use_random=False)
    edges_gpu = cp.asarray(edges)

    sim_data = spms.sim_data()
    noise = sim_data[0]
    diffuse = sim_data[1]

    return {
            "diffuse": diffuse,
            "noise": noise,
            "edges": edges_gpu
            }
