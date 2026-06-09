# ncu_target.py  -- single 35x35 inv_cov call for profiling
import cupy as cp
from gpu_corrcal.utils.simulate_params import *
from gpu_corrcal.utils.gridding import *
from gpu_corrcal.inverse_covariance import InvCovWorkspace, inv_cov
from gpu_corrcal.utils.tools import sync

rows = cols = 35
n_eig, n_src = 5, 10
cp.random.seed(12)
spms = SimCorrcalParams(rows*cols, n_eig, n_src=n_src, precision="float32", xp=cp)
edges = cp.asarray(spms.edges(rows, cols, use_random=False))
noise, diffuse, source, _, _ = spms.sim_data()

ws = InvCovWorkspace(diffuse, source, edges)

# warmup once (JIT / pool), NOT profiled because we limit launches below
inv_cov(noise, diffuse, source, edges, ws)
sync()

# the call we actually profile
inv_cov(noise, diffuse, source, edges, ws)
sync()