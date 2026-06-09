"""
Per-kernel timing for the inv_cov pipeline.
 
Times each of the six kernels in inv_cov SEPARATELY using cupyx.profiler.benchmark
(which handles warmup + device sync and reports GPU time with a std dev), so you
can see which kernel owns the runtime as antenna count / n_src / n_eig grow.
 
The six stages:
  Diffuse level (Del_prime):
    1. make_small_blocks        (warp reduction -> temp2)
    2. fused_cholesky_inverse   (batched chol+inv -> L_del_inv_T)
    3. mul_temp_by_diffuse_chol (-> diffuse_bar)
  Source level (Sig_prime):
    4. make_cap_reduce          (-> B, M_sig)            <-- atomic-contention suspect
    5. fused_cholesky_inverse_sig (-> L_sig_inv)
    6. apply_sig_prime_launch   (-> sig_prime)
 
Each stage is timed in isolation but fed the REAL outputs of the prior stages,
so the inputs are physically correct (not random junk that could change timing
via different branch / NaN behavior).
"""
 
import cupy as cp
import numpy as np
from cupyx.profiler import benchmark
 
from gpu_corrcal.utils.simulate_params import *
from gpu_corrcal.utils.gridding import *
from gpu_corrcal.utils.zp_puregpu_funcs_py import *
from gpu_corrcal.inverse_covariance import InvCovWorkspace
from gpu_corrcal.linalg import (
    make_small_blocks,
    fused_cholesky_inverse,
    mul_temp_by_diffuse_chol,
    make_cap_reduce,
    fused_cholesky_inverse_sig,
    apply_sig_prime_launch,
)
from gpu_corrcal.utils.tools import sync
 
 
# ----------------------------------------------------------------------
# data
# ----------------------------------------------------------------------
def make_data(rows, cols, n_eig, n_src, seed=12):
    cp.random.seed(seed)
    n_ant = rows * cols
    spms = SimCorrcalParams(n_ant, n_eig, n_src=n_src, precision="float32", xp=cp)
    edges = cp.asarray(spms.edges(rows, cols, use_random=False))
    noise, diffuse, source, _, _ = spms.sim_data()
    return noise, diffuse, source, edges
 
 
# ----------------------------------------------------------------------
# per-kernel timing
# ----------------------------------------------------------------------
def time_kernels(rows, cols, n_eig, n_src, n_repeat=20, n_warmup=5):
    noise, diffuse, source, edges = make_data(rows, cols, n_eig, n_src)
    nb = len(edges) - 1
    group_sizes = cp.diff(edges)
    max_group = int(group_sizes.max())
    mean_group = float(group_sizes.mean())
 
    ws = InvCovWorkspace(diffuse, source, edges)
 
    # --- pre-run the full chain ONCE so every stage has correct inputs ---
    temp2 = make_small_blocks(noise, diffuse, edges, out=ws.temp2)
    L_del_inv_T = fused_cholesky_inverse(temp2, edges, out=ws.L_del)
    diffuse_bar = mul_temp_by_diffuse_chol(noise, diffuse, L_del_inv_T, ws.diffuse_bar, edges)
    ws.M_sig.fill(0.0)
    B, M_sig = make_cap_reduce(source, diffuse_bar, noise, edges, ws.B, ws.M_sig)
    L_sig_inv = fused_cholesky_inverse_sig(M_sig, out=ws.L_sig_inv)
    _ = apply_sig_prime_launch(source, diffuse_bar, noise, edges, B, L_sig_inv, ws.sig_prime)
    sync()
 
    # --- wrap each stage as a zero-arg callable for benchmark() ---
    # NOTE: cap_reduce accumulates into M_sig, so its callable MUST re-zero
    # M_sig each repeat, or you'd be timing an accumulate-on-top (wrong result,
    # and atomic traffic into ever-growing values). The .fill is cheap and is
    # part of the real per-call cost anyway.
    def stage_small_blocks():
        make_small_blocks(noise, diffuse, edges, out=ws.temp2)
 
    def stage_chol_del():
        fused_cholesky_inverse(ws.temp2, edges, out=ws.L_del)
 
    def stage_diffuse_bar():
        mul_temp_by_diffuse_chol(noise, diffuse, ws.L_del, ws.diffuse_bar, edges)
 
    def stage_cap_reduce():
        ws.M_sig.fill(0.0)
        make_cap_reduce(source, ws.diffuse_bar, noise, edges, ws.B, ws.M_sig)
 
    def stage_chol_sig():
        fused_cholesky_inverse_sig(ws.M_sig, out=ws.L_sig_inv)
 
    def stage_apply():
        apply_sig_prime_launch(source, ws.diffuse_bar, noise, edges,
                               ws.B, ws.L_sig_inv, ws.sig_prime)
 
    stages = [
        ("1 small_blocks   ", stage_small_blocks),
        ("2 chol_del       ", stage_chol_del),
        ("3 diffuse_bar    ", stage_diffuse_bar),
        ("4 cap_reduce     ", stage_cap_reduce),
        ("5 chol_sig       ", stage_chol_sig),
        ("6 apply_sig_prime", stage_apply),
    ]
 
    print(f"\n=== rows={rows} cols={cols}  n_ant={rows*cols}  "
          f"n_eig={n_eig}  n_src={n_src}  nb={nb}  "
          f"group(max={max_group}, mean={mean_group:.1f}) ===")
    print(f"{'stage':<18}  {'gpu (us)':>10}  {'std (us)':>9}  {'% of total':>10}")
 
    results = []
    for name, fn in stages:
        r = benchmark(fn, n_repeat=n_repeat, n_warmup=n_warmup)
        gpu_us = r.gpu_times.mean() * 1e6
        gpu_std = r.gpu_times.std() * 1e6
        results.append((name, gpu_us, gpu_std))
 
    total = sum(g for _, g, _ in results)
    for name, gpu_us, gpu_std in results:
        print(f"{name:<18}  {gpu_us:>10.1f}  {gpu_std:>9.1f}  {100*gpu_us/total:>9.1f}%")
    print(f"{'TOTAL (sum)':<18}  {total:>10.1f}")
    return results
 
 
if __name__ == "__main__":
    # sweep the antenna grid at fixed (n_eig, n_src); same shape as your bench
    rc_list = [[5, 5], [9, 9], [14, 14], [20, 20], [27, 27], [35, 35]]
    n_eig = 3
    n_src = 15
 
    for rows, cols in rc_list:
        time_kernels(rows, cols, n_eig, n_src, n_repeat=20, n_warmup=5)