# ============================================================
# run_benchmarks.py
# Runs timing benchmarks and saves results to .npz files.
# NO plotting in this module — see plot_benchmarks.py.
#
# What to benchmark is defined ONCE, in _build_targets().
# Adding a new function to test = add one entry to that dict.
# Everything else (timing_test, the sweeps, the save schema)
# is generic and operates only on "the three times that came back".
# ============================================================

# This kills any multithreaded activity:
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import cupy as cp
import ctypes
import math
from datetime import datetime

from gpu_corrcal.utils.gridding import *
from gpu_corrcal.utils.simulate_params import *
from gpu_corrcal.utils.zp_puregpu_funcs_py import *
from gpu_corrcal.inverse_covariance import *
from gpu_corrcal.utils.tools import *
from gpu_corrcal.cupy_reference.cupy_inverse_covariance import *
from cupyx.profiler import benchmark

cudart = ctypes.CDLL("libcudart.so")

from corrcal.sparse import *

DATA_DIR = "/home/mikej/main_phd_work/thesis_projects/gpu_corrcal/src/gpu_corrcal/tests2/benchmark_data"


# ============================================================
# Target registry — THE ONLY PLACE THAT KNOWS ABOUT SPECIFIC
# FUNCTIONS. Each entry maps a target name to the three calls a
# benchmark needs, packaged as (callable, args_tuple):
#
#     {
#       "custom": (fn, args),   # custom GPU kernel
#       "cupy":   (fn, args),   # CuPy reference
#       "cpu":    (fn, args),   # single-core CPU corrcal reference
#     }
#
# timing_test() just does benchmark(fn, args, ...) on each — it
# never needs to know what the function is.
#
# For functions that need per-call setup (e.g. an accumulating
# kernel that must be .fill(0.0)'d each call, like cap_reduce /
# M_sig), wrap the raw kernel in a tiny closure here and register
# the closure. That keeps the setup discipline inside the registry
# instead of leaking into timing_test.
# ============================================================
def _build_targets(test_data, n_eig):
    """Build the registry of benchmarkable targets for one test dataset."""
    diffuse = test_data["diffuse"]
    noise = test_data["noise"]
    edges = test_data["edges"]
    source = test_data["source"]

    # CPU-side copies (float64) for the corrcal reference
    cpu_diffuse = cp.asnumpy(diffuse).astype(np.float64)
    cpu_noise = cp.asnumpy(noise).astype(np.float64)
    cpu_edges = cp.asnumpy(edges).astype(np.float64)
    cpu_source = cp.asnumpy(source).astype(np.float64)

    # Shared setup ------------------------------------------------------
    ws = InvCovWorkspace(diffuse, source, edges)

    zp_inv_noise, lb, nb = zeroPad(noise, edges, return_inv=True, dtype=cp.float32)
    zp_diffuse, lb, nb = zeroPad(diffuse, edges, return_inv=False, dtype=cp.float32)
    zp_source, lb, nb = zeroPad(source, edges, return_inv=False, dtype=cp.float32)

    cpu_sparse_cov = SparseCov(cpu_noise, cpu_source, cpu_diffuse, cpu_edges, n_eig, False)

    # Registry ----------------------------------------------------------
    targets = {
        # inverse covariance only (compute_det defaults to False)
        "inv": {
            "custom": (inv_cov, (noise, diffuse, source, edges, ws)),
            "cupy":   (inverse_covariance,
                       (zp_inv_noise, zp_diffuse, zp_source, cp, False, True)),
            "cpu":    (cpu_sparse_cov.inv, ()),
        },

        # ---- stubs for functions still to come; fill in and uncomment ----
        # inverse covariance + log-determinant (compute_det=True)
        "logdet": {
            "custom": (inv_cov, (noise, diffuse, source, edges, ws, True)),
            "cupy":   (inverse_covariance,
                       (zp_inv_noise, zp_diffuse, zp_source, cp, True, True)),
            "cpu":    (cpu_sparse_cov.inv, (True,)),
        },

        "logdet_vs_onlyinv": {
            "inv_logdet": (inv_cov, (noise, diffuse, source, edges, ws, True)),
            "inv": (inv_cov, (noise, diffuse, source, edges, ws, False)),
        },


        # matrix-times-vector
        # "matvec": {
        #     "custom": (<gpu_matvec>, (...)),
        #     "cupy":   (<cupy_matvec>, (...)),
        #     "cpu":    (<cpu_matvec>, ()),
        # },

        # apply gains to the covariance
        # "apply_gains": {
        #     "custom": (<gpu_apply_gains>, (...)),
        #     "cupy":   (<cupy_apply_gains>, (...)),
        #     "cpu":    (<cpu_apply_gains>, ()),
        # },

        # full likelihood
        # "likelihood": {
        #     "custom": (<gpu_likelihood>, (...)),
        #     "cupy":   (<cupy_likelihood>, (...)),
        #     "cpu":    (<cpu_likelihood>, ()),
        # },
    }
    return targets


# ============================================================
# Single timing test — generic, driven by the registry
# ============================================================
def timing_test(n_eig, n_src, rc_tuple, threads_per_block, seed,
                target="inv", n_rep=400):
    """Time custom GPU kernel vs CuPy reference vs single-core CPU corrcal
    for the chosen `target` (see _build_targets)."""
    test_data = make_test_data(n_eig, n_src, rc_tuple, seed)

    targets = _build_targets(test_data, n_eig)
    if target not in targets:
        raise KeyError(f"unknown target '{target}'; "
                       f"available: {list(targets)}")
    spec = targets[target]

    # Custom GPU kernel ------------------------------------------------
    custom_fn, custom_args = spec["custom"]
    times = benchmark(custom_fn, custom_args, n_repeat=n_rep)
    avg_gpu = float(cp.mean(times.gpu_times))
    avg_cpu = float(cp.mean(times.cpu_times))

    custom_fn, custom_args = spec["inv"]
    times = benchmark(custom_fn, custom_args, n_repeat=n_rep)
    avg_gpu = float(cp.mean(times.gpu_times))
    avg_cpu = float(cp.mean(times.cpu_times))

    custom_fn_ld, custom_args_ld = spec["inv_logdet"]
    times_ld = benchmark(custom_fn_ld, custom_args_ld, n_repeat=n_rep)
    avg_gpu_ld = float(cp.mean(times_ld.gpu_times))
    avg_cpu_ld = float(cp.mean(times_ld.cpu_times))

    # # CuPy reference ---------------------------------------------------
    # cupy_fn, cupy_args = spec["cupy"]
    # ref_times = benchmark(cupy_fn, cupy_args, n_repeat=n_rep)
    # ref_avg_gpu = float(cp.mean(ref_times.gpu_times))

    # # CPU reference (pinned to a single core) --------------------------
    # cpu_fn, cpu_args = spec["cpu"]
    # original_affinity = os.sched_getaffinity(0)
    # os.sched_setaffinity(0, {0})
    # ref_cpu_times = benchmark(cpu_fn, cpu_args, n_repeat=n_rep)
    # os.sched_setaffinity(0, original_affinity)
    # ref_cpu_avg_cpu = float(cp.mean(ref_cpu_times.cpu_times))

    # print(f"  [{target}]  [n_eig={n_eig:>2}  n_src={n_src:>2}]   "
    #       f"Custom: gpu={avg_gpu*1e6:>7.1f} us  cpu={avg_cpu*1e6:>7.1f} us | "
    #       f"CuPy: gpu={ref_avg_gpu*1e6:>7.1f} us | "
    #       f"CPU: cpu={ref_cpu_avg_cpu*1e6:>7.1f} us  "
    #       f"cpu_std={cp.std(ref_cpu_times.cpu_times)}")

    return avg_gpu, avg_gpu_ld

    return avg_gpu, ref_avg_gpu, ref_cpu_avg_cpu


# ============================================================
# Helpers for antenna-count sweeps
# ============================================================
def pop_row_col_input(end_iter_num):
    row_col_list = []
    j = 0
    for i in range(2, end_iter_num):
        j += i
        row_col_list.append([j, j])
    return row_col_list


def pop_row_col_input_tens(iter_num):
    """Generates the same nant inputs as Bobby's paper plot."""
    row_col_list = []
    coeff = [1, 2, 3, 5]
    for n in range(iter_num + 1):
        for c in coeff:
            ant_val = c * 10 ** (n + 1)
            for i in range(int(math.isqrt(ant_val)), 0, -1):
                if ant_val % i == 0:
                    row_col_list.append([i, ant_val // i])
                    break
    return row_col_list


# ============================================================
# Shared save helper — one schema for every sweep type
# ============================================================
def _save_results(sweep, x_values, test_times, reference_times,
                  cpu_reference_times, params):
    """Save a benchmark sweep to .npz with a uniform schema.

    Every file contains:
      sweep                : str, one of 'nant', 'neig', 'nsrc', 'neig_nsrc'
      target               : str, which function was benchmarked (in params)
      x_values             : the independent variable
      test_times           : custom GPU kernel times (s)
      reference_times      : CuPy reference times (s)
      cpu_reference_times  : single-core CPU corrcal times (s)
      title_label / file_label : machine labels (captured at RUN time,
                                 so plotting never needs the GPU)
      ...plus all fixed params passed in `params`
    """
    title_label, file_label = get_machine_label()
    os.makedirs(DATA_DIR, exist_ok=True)
    tstamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    param_str = "_".join(f"{k}{v}" for k, v in params.items())
    data_file = f"{DATA_DIR}/{sweep}_{param_str}_{file_label}_{tstamp}.npz"

    np.savez(data_file,
             sweep=sweep,
             x_values=x_values,
             test_times=test_times,
             reference_times=reference_times,
             cpu_reference_times=cpu_reference_times,
             title_label=title_label,
             file_label=file_label,
             **params)
    print(f"Saved benchmark data to {data_file}")
    return data_file


# ============================================================
# Sweep runners — data generation ONLY.
# Generic: they loop an x-axis, call timing_test, collect the three
# numbers, and save. `target` is just a passenger threaded through.
# ============================================================
def run_nant_sweep(n_eig, n_src, n_trials, tpb, seed, target="inv"):
    row_col_inputs = pop_row_col_input_tens(n_trials)
    n_ants = np.array([rc[0] * rc[1] for rc in row_col_inputs], dtype=float)
    print(f"row-col combos used are: {row_col_inputs}")

    test_times = np.zeros(len(row_col_inputs))
    reference_times = np.zeros(len(row_col_inputs))
    cpu_reference_times = np.zeros(len(row_col_inputs))

    for i, rc in enumerate(row_col_inputs):
        print(f"on iteration {i}")
        # gpu_t, ref_gpu_t, ref_cpu_t = timing_test(n_eig, n_src, rc, tpb, seed, target=target)
        gpu_t, ref_gpu_t = timing_test(n_eig, n_src, rc, tpb, seed, target=target)
        test_times[i] = gpu_t
        reference_times[i] = ref_gpu_t
        # cpu_reference_times[i] = ref_cpu_t

    return _save_results("nant", n_ants, test_times, reference_times,
                         cpu_reference_times,
                         dict(target=target, n_eig=n_eig, n_src=n_src, tpb=tpb, seed=seed))


def run_neig_sweep(rc_tuple, eig_range, n_src, tpb, seed, target="inv"):
    neigs = np.arange(eig_range[0], eig_range[1])
    n_ant = rc_tuple[0] * rc_tuple[1]
    print(f"n_ant fixed at {n_ant}")

    test_times = np.zeros(len(neigs))
    reference_times = np.zeros(len(neigs))
    cpu_reference_times = np.zeros(len(neigs))

    for i, n_eig in enumerate(neigs):
        gpu_t, ref_gpu_t, ref_cpu_t = timing_test(n_eig, n_src, rc_tuple, tpb, seed, target=target)
        test_times[i] = gpu_t
        reference_times[i] = ref_gpu_t
        cpu_reference_times[i] = ref_cpu_t

    return _save_results("neig", neigs, test_times, reference_times,
                         cpu_reference_times,
                         dict(target=target, n_ant=n_ant, n_src=n_src, tpb=tpb, seed=seed))


def run_nsrc_sweep(rc_tuple, src_range, n_eig, tpb, seed, target="inv"):
    nsrcs = np.arange(src_range[0], src_range[1])
    n_ant = rc_tuple[0] * rc_tuple[1]
    print(f"n_ant fixed at {n_ant}")

    test_times = np.zeros(len(nsrcs))
    reference_times = np.zeros(len(nsrcs))
    cpu_reference_times = np.zeros(len(nsrcs))

    for i, n_src in enumerate(nsrcs):
        gpu_t, ref_gpu_t, ref_cpu_t = timing_test(n_eig, n_src, rc_tuple, tpb, seed, target=target)
        test_times[i] = gpu_t
        reference_times[i] = ref_gpu_t
        cpu_reference_times[i] = ref_cpu_t

    return _save_results("nsrc", nsrcs, test_times, reference_times,
                         cpu_reference_times,
                         dict(target=target, n_ant=n_ant, n_eig=n_eig, tpb=tpb, seed=seed))


def run_neig_nsrc_sweep(rc_tuple, eig_src_range, tpb, seed, target="inv"):
    vals = np.arange(eig_src_range[0], eig_src_range[1], 2)
    n_ant = rc_tuple[0] * rc_tuple[1]
    print(f"n_ant fixed at {n_ant}")

    test_times = np.zeros(len(vals))
    reference_times = np.zeros(len(vals))
    cpu_reference_times = np.zeros(len(vals))

    for i, v in enumerate(vals):
        gpu_t, ref_gpu_t, ref_cpu_t = timing_test(v, v, rc_tuple, tpb, seed, target=target)
        test_times[i] = gpu_t
        reference_times[i] = ref_gpu_t
        cpu_reference_times[i] = ref_cpu_t

    return _save_results("neig_nsrc", vals, test_times, reference_times,
                         cpu_reference_times,
                         dict(target=target, n_ant=n_ant, tpb=tpb, seed=seed))


# ============================================================
def time_multiple(rc_tuple, tpb, eig_range, src_range, seed,
                  time_eigs=False, time_srcs=False, target="inv"):
    """Console-only sweep, no data saved (quick sanity checks)."""
    print()
    print("=" * 65)
    print(f"TIMING TESTS -> n_ant = {rc_tuple[0]} * {rc_tuple[1]} = {rc_tuple[0]*rc_tuple[1]}")
    print("=" * 65)

    if time_eigs:
        for n_eig in range(eig_range[0], eig_range[1]):
            timing_test(n_eig, 3, rc_tuple, tpb, seed, target=target)
    elif time_srcs:
        for n_src in range(src_range[0], src_range[1]):
            timing_test(3, n_src, rc_tuple, tpb, seed, target=target)

    print("=" * 65)


# =========================================================================
if __name__ == "__main__":
    # Test params
    # -------------------------------
    rows = 16
    cols = 32
    n_eig = 3
    n_src = 5
    rc = (rows, cols)
    random_seed = 42
    tpb = 128
    T = True
    F = False
    # -------------------------------

    # Which function to benchmark this run
    # -------------------------------
    # target = "inv"   # "inv" | "logdet" | "matvec" | "apply_gains" | "likelihood"
    target = "logdet"

    # Switch board
    # -------------------------------
    one_timing_test = F
    many_timing_tests = F
    bench_nant = T
    bench_neig = F
    bench_nsrc = F
    bench_neig_nsrc = F

    if one_timing_test:
        timing_test(n_eig, n_src, rc, tpb, random_seed, target=target)

    if many_timing_tests:
        time_multiple(rc, tpb, (1, 10), (1, 10), random_seed, time_srcs=True, target=target)

    if bench_nant:
        run_nant_sweep(n_eig, n_src, n_trials=1, tpb=tpb, seed=random_seed, target=target)

    if bench_neig:
        run_neig_sweep(rc, (1, 20), n_src, tpb, random_seed, target=target)

    if bench_nsrc:
        run_nsrc_sweep(rc, (1, 20), n_eig, tpb, random_seed, target=target)

    if bench_neig_nsrc:
        run_neig_nsrc_sweep(rc, (2, 22), tpb, random_seed, target=target)