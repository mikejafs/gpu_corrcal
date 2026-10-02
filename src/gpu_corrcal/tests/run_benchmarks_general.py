# ============================================================
# run_benchmarks.py
# Runs timing benchmarks and saves results to .npz files.
# NO plotting in this module — see plot_benchmarks.py.
#
# What to benchmark is defined ONCE, in _build_targets().
# Adding a new function to test = add one entry to that dict.
#
# A target is a dict of NAMED (fn, args) series. timing_test runs
# each series and returns (names, times) of matching length — so a
# target can have 2, 3, 4, ... series and everything downstream
# (sweeps, save schema, plotting) adapts automatically. This is
# what lets inv-vs-logdet (2 series) and the eventual
# inv/logdet/likelihood/gradient plot (4 series) share one code path.
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
from gpu_corrcal.cupy_reference import cupy_utils
from gpu_corrcal.linalg import *

from cupyx.profiler import benchmark

cudart = ctypes.CDLL("libcudart.so")

from corrcal.sparse import *
from corrcal import linalg


# DATA_DIR = "/home/mikejafs/gpu_corrcal/gpu_corrcal/src/gpu_corrcal/tests2/benchmark_data"

def _resolve_data_dir():
    """Pick the benchmark_data dir based on which machine we're on,
    using the same get_machine_label() the filenames/plots key on."""
    _, file_label = get_machine_label()
    if "5070" in file_label:
        return "/home/mikej/main_phd_work/thesis_projects/gpu_corrcal/src/gpu_corrcal/tests2/benchmark_data"
    elif "A40" in file_label:
        return "/home/mikejafs/gpu_corrcal/gpu_corrcal/src/gpu_corrcal/tests2/benchmark_data"
    else:
        raise RuntimeError(f"unrecognized machine label {file_label!r}; "
                           f"add its data dir to _resolve_data_dir()")

DATA_DIR = _resolve_data_dir()

# ============================================================
# Target registry — THE ONLY PLACE THAT KNOWS ABOUT SPECIFIC
# FUNCTIONS. Each target is a dict of NAMED series:
#
#     "target_name": {
#         "series_label_1": (fn, args),
#         "series_label_2": (fn, args),
#         ...                              # 2, 3, 4, ... allowed
#     }
#
# The series labels become the plot legend labels and are saved
# with the data, so the plot module never needs to know what a
# target contains. Insertion order is preserved.
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
    vec = test_data["data_vec"]

    # CPU-side copies (float64) for the corrcal reference
    cpu_diffuse = cp.asnumpy(diffuse).astype(np.float64)
    cpu_noise = cp.asnumpy(noise).astype(np.float64)
    cpu_edges = cp.asnumpy(edges).astype(np.float64)
    cpu_source = cp.asnumpy(source).astype(np.float64)
    cpu_vec = cp.asnumpy(vec).astype(np.float64)

    # Shared setup ------------------------------------------------------
    ws = InvCovWorkspace(diffuse, source, edges)
    matvec_ws = MatvecWorkspace(diffuse.shape[0], diffuse.shape[1], source.shape[1])

    zp_inv_noise, lb, nb = zeroPad(noise, edges, return_inv=True, dtype=cp.float32)
    zp_diffuse, lb, nb = zeroPad(diffuse, edges, return_inv=False, dtype=cp.float32)
    zp_source, lb, nb = zeroPad(source, edges, return_inv=False, dtype=cp.float32)
    zp_vec, _, _ = zeroPad(vec, edges, return_inv=False, dtype=cp.float32)

    cpu_sparse_cov = SparseCov(cpu_noise, cpu_source, cpu_diffuse, cpu_edges, n_eig, False)

    # Registry ----------------------------------------------------------
    targets = {
        # custom vs CuPy vs CPU, inverse covariance only
        "inv": {
            "Custom GPU": (inv_cov, (noise, diffuse, source, edges, ws)),
            "CuPy":       (inverse_covariance,
                           (zp_inv_noise, zp_diffuse, zp_source, cp, False, True)),
            "CPU":        (cpu_sparse_cov.inv, ()),
        },

        # custom vs CuPy vs CPU, inverse covariance + log-determinant
        "logdet": {
            "Custom GPU": (inv_cov, (noise, diffuse, source, edges, ws, True)),
            "CuPy":       (inverse_covariance,
                           (zp_inv_noise, zp_diffuse, zp_source, cp, True, True)),
            "CPU":        (cpu_sparse_cov.inv, (True,)),
        },

        # head-to-head: custom inv vs custom inv+logdet (the logdet overhead)
        "logdet_vs_onlyinv": {
            "Inv only":   (inv_cov, (noise, diffuse, source, edges, ws, False)),
            "Inv+logdet": (inv_cov, (noise, diffuse, source, edges, ws, True)),
        },

        "matvec": {
            "Custom GPU": (sparse_cov_vec_mul, (noise, diffuse, source, vec, edges, matvec_ws, True)),
            "CuPy":       (cupy_utils.sparse_cov_times_vec, (zp_inv_noise, zp_diffuse, zp_source, zp_vec, True)),
            "CPU":        (linalg.sparse_cov_times_vec, (cpu_sparse_cov, cpu_vec)),
        },

        # ---- stubs for functions still to come; fill in and uncomment ----
        
        
        # "apply_gains": {
        #     "Custom GPU": (<gpu_apply_gains>, (...)),
        #     "CuPy":       (<cupy_apply_gains>, (...)),
        #     "CPU":        (<cpu_apply_gains>, ()),
        # },
        # the eventual 4-line plot — all custom GPU, one axis:
        # "likelihood_stack": {
        #     "Inv":         (inv_cov, (...)),
        #     "Inv+logdet":  (inv_cov, (..., True)),
        #     "Likelihood":  (<gpu_likelihood>, (...)),
        #     "Gradient":    (<gpu_gradient>, (...)),
        # },
    }
    return targets


# ============================================================
# Single timing test — generic, driven by the registry.
# Returns (names, times): parallel lists of length len(spec).
# ============================================================
def timing_test(n_eig, n_src, rc_tuple, threads_per_block, seed,
                target="inv", n_rep=1000):
    """Benchmark every series in `target` (see _build_targets).
    Returns (names, times) where times are mean GPU times in seconds."""
    test_data = make_test_data(n_eig, n_src, rc_tuple, seed)

    targets = _build_targets(test_data, n_eig)
    if target not in targets:
        raise KeyError(f"unknown target '{target}'; "
                       f"available: {list(targets)}")
    spec = targets[target]

    names, times = [], []
    for name, (fn, args) in spec.items():
        t = benchmark(fn, args, n_repeat=n_rep)
        names.append(name)
        times.append(float(cp.mean(t.gpu_times)))

    print(f"  [{target}]  [n_eig={n_eig:>2} n_src={n_src:>2}]  " +
          "  ".join(f"{nm}={v*1e6:>7.1f}us" for nm, v in zip(names, times)))

    return names, times


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
# Shared save helper — one schema for every sweep type.
# Series are stored as a 2-D array (n_series, n_points) plus the
# parallel list of series names, so any number of series round-trips.
# ============================================================
def _save_results(sweep, x_values, series_names, series_times, params):
    """Save a benchmark sweep to .npz with a uniform, N-series schema.

    Every file contains:
      sweep         : str, one of 'nant', 'neig', 'nsrc', 'neig_nsrc'
      target        : str, which target was benchmarked (in params)
      x_values      : the independent variable, shape (n_points,)
      series_names  : str array, shape (n_series,) — legend labels
      series_times  : float array, shape (n_series, n_points) — GPU times (s)
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
             series_names=np.array(series_names),
             series_times=np.array(series_times),
             title_label=title_label,
             file_label=file_label,
             **params)
    print(f"Saved benchmark data to {data_file}")
    return data_file


def _run_sweep(sweep, x_values, param_iter, timing_call, params):
    """Generic sweep driver.

    param_iter : iterable of the loop variable for each point
    timing_call: fn(loop_var) -> (names, times) for that point
    Collects into a (n_series, n_points) array and saves once.
    """
    names = None
    series_times = None
    for i, var in enumerate(param_iter):
        print(f"on iteration {i}")
        pt_names, pt_times = timing_call(var)
        if series_times is None:
            names = pt_names
            series_times = np.zeros((len(pt_names), len(x_values)))
        series_times[:, i] = pt_times
    return _save_results(sweep, x_values, names, series_times, params)


# ============================================================
# Sweep runners — data generation ONLY.
# Each just defines its x-axis and a per-point timing call; the
# generic _run_sweep handles collection + save. `target` rides
# through as a passenger.
# ============================================================
def run_nant_sweep(n_eig, n_src, n_trials, tpb, seed, target="inv"):
    row_col_inputs = pop_row_col_input_tens(n_trials)
    n_ants = np.array([rc[0] * rc[1] for rc in row_col_inputs], dtype=float)
    print(f"row-col combos used are: {row_col_inputs}")

    return _run_sweep(
        "nant", n_ants, row_col_inputs,
        lambda rc: timing_test(n_eig, n_src, rc, tpb, seed, target=target),
        dict(target=target, n_eig=n_eig, n_src=n_src, tpb=tpb, seed=seed))


def run_neig_sweep(rc_tuple, eig_range, n_src, tpb, seed, target="inv"):
    neigs = np.arange(eig_range[0], eig_range[1])
    n_ant = rc_tuple[0] * rc_tuple[1]
    print(f"n_ant fixed at {n_ant}")

    return _run_sweep(
        "neig", neigs, neigs,
        lambda n_eig: timing_test(n_eig, n_src, rc_tuple, tpb, seed, target=target),
        dict(target=target, n_ant=n_ant, n_src=n_src, tpb=tpb, seed=seed))


def run_nsrc_sweep(rc_tuple, src_range, n_eig, tpb, seed, target="inv"):
    nsrcs = np.arange(src_range[0], src_range[1])
    n_ant = rc_tuple[0] * rc_tuple[1]
    print(f"n_ant fixed at {n_ant}")

    return _run_sweep(
        "nsrc", nsrcs, nsrcs,
        lambda n_src: timing_test(n_eig, n_src, rc_tuple, tpb, seed, target=target),
        dict(target=target, n_ant=n_ant, n_eig=n_eig, tpb=tpb, seed=seed))


def run_neig_nsrc_sweep(rc_tuple, eig_src_range, tpb, seed, target="inv"):
    vals = np.arange(eig_src_range[0], eig_src_range[1], 2)
    n_ant = rc_tuple[0] * rc_tuple[1]
    print(f"n_ant fixed at {n_ant}")

    return _run_sweep(
        "neig_nsrc", vals, vals,
        lambda v: timing_test(v, v, rc_tuple, tpb, seed, target=target),
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
    rows = 32
    cols = 16
    n_eig = 3
    n_src = 5
    rc = (rows, cols)
    random_seed = 42
    tpb = 128
    T = True
    F = False
    # -------------------------------

    # Which target to benchmark this run
    # -------------------------------
    # "inv" | "logdet" | "logdet_vs_onlyinv" | (later) "matvec" | ...
    # target = "logdet_vs_onlyinv"
    target = "matvec"

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
        time_multiple(rc, tpb, (1, 10), (1, 10), random_seed, time_srcs=True, time_eigs=True, target=target)

    if bench_nant:
        run_nant_sweep(n_eig, n_src, n_trials=2, tpb=tpb, seed=random_seed, target=target)

    if bench_neig:
        run_neig_sweep(rc, (1, 20), n_src, tpb, random_seed, target=target)

    if bench_nsrc:
        run_nsrc_sweep(rc, (1, 20), n_eig, tpb, random_seed, target=target)

    if bench_neig_nsrc:
        run_neig_nsrc_sweep(rc, (2, 22), tpb, random_seed, target=target)
