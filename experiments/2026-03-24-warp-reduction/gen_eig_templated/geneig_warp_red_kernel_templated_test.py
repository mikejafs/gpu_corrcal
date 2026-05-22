"""
ctypes interface and correctness/timing test for the templated
general eigenmode warp reduction kernel.

Compile:
    nvcc -Xcompiler -fPIC -shared -o geneig_warp_red_kernel.so geneig_warp_red_kernel.cu

Usage:
    python geneig_warp_red_kernel_test.py

TODO:
- [DONE] Add benchmark tests comparing gen eig to hardcoded red kernel for neig = 3 (NOT TO BE ADDED TO BOILERPLATE STUFF) -> do by inspection per nant in the terminal

ADD TO BOILERPLATE:
- Add option to produce benchmark plot comparing to reference script (ie. cupy)
- [DONE] Modify current benchmark to terminal output so that it compares times directly to reference script for same inputs
- Add full cupy reference to scripts to then 'pull parts from' when it comes time to building new kernels
"""

import numpy as np
import cupy as cp
import ctypes
import os
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FixedLocator, FixedFormatter
from warp_red_kern_r3 import *
from gpu_corrcal.utils.gridding import *
from gpu_corrcal.utils.simulate_params import *
from gpu_corrcal.utils.zp_puregpu_funcs_py import *
from cupyx.profiler import benchmark

# cudart = ctypes.CDLL("libcudart.so")

# ============================================================
# ctypes wrapper
# ============================================================

def load_kernel(so_path=None):

    # Wether or not to use pragma unroll
    unroll = True
    """Load the shared library and set up function signatures."""
    if so_path is None:
        #if NO PRAGMA UNROLL
        if not unroll:
            so_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "geneig_warp_red_kernel_templated_NOpragma.so"
            )
            print(f"NO unroll")

        #if YES PRAGMA UNROLL
        elif unroll:
            so_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "geneig_warp_red_kernel_templated.so"
            )
            print(f"WITH unroll")

    lib = ctypes.CDLL(so_path)

    # void launch_two_level_warp_reduction(
    #     const float* diffuse, const float* noise, const int* edges,
    #     float* out, int nb, int n_eig, int threads_per_block
    # )
    lib.launch_two_level_warp_reduction.restype = None
    lib.launch_two_level_warp_reduction.argtypes = [
        ctypes.c_void_p,  # diffuse
        ctypes.c_void_p,  # noise
        ctypes.c_void_p,  # edges
        ctypes.c_void_p,  # out
        ctypes.c_int,     # nb
        ctypes.c_int,     # n_eig
        ctypes.c_int,     # threads_per_block
    ]

    lib.sync_device.restype = None
    lib.sync_device.argtypes = []

    return lib


def call_kernel(lib, diffuse_gpu, noise_gpu, edges_gpu, out_gpu,
                nb, n_eig, threads_per_block):
    """Launch the kernel through ctypes."""
    lib.launch_two_level_warp_reduction(
        diffuse_gpu.data.ptr,
        noise_gpu.data.ptr,
        edges_gpu.data.ptr,
        out_gpu.data.ptr,
        ctypes.c_int(nb),
        ctypes.c_int(n_eig),
        ctypes.c_int(threads_per_block),
    )


def sync(lib):
    """Synchronize the device."""
    lib.sync_device()


# ============================================================
# CPU & CUPY References
# ============================================================

def cpu_ref(diffuse, noise, edges, n_eig):
    """
    CPU reference: for each block b, compute
        out[b] = sum_i (1/noise[i]) * diffuse[i,:] @ diffuse[i,:].T
    where i ranges over edges[b] .. edges[b+1]-1.

    Returns (nb, n_eig, n_eig) array.
    """
    nb = len(edges) - 1
    out = np.zeros((nb, n_eig, n_eig), dtype=np.float32)
    for b in range(nb):
        for i in range(edges[b], edges[b + 1]):
            d = diffuse[i, :]
            out[b] += np.outer(d, d) / noise[i]
    return out


def cupy_ref(noise, diffuse, edges):
    zp_inv_noise, lb, nb = zeroPad(noise, edges, return_inv=True, dtype=cp.float32)
    zp_diffuse, lb, nb = zeroPad(diffuse, edges, return_inv=False, dtype=cp.float32)
    temp = zp_inv_noise[..., None] * zp_diffuse
    out = cp.transpose(zp_diffuse, [0, 2, 1]) @ temp
    cp.cuda.Stream.null.synchronize()
    return out, nb


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


# ============================================================
# Correctness test
# ============================================================

def correctness_test(n_eig, rc_tuple, threads_per_block):
    """Compare GPU kernel output to CPU reference."""
    diffuse, noise, edges = make_test_data(n_eig, rc_tuple, 12)

    # CPU reference
    ref_cpu = cpu_ref(cp.asnumpy(diffuse), cp.asnumpy(noise), cp.asnumpy(edges), n_eig)
    ref_cpu = cp.asarray(ref_cpu)

    # CuPy reference
    ref_cupy, nb = cupy_ref(noise, diffuse, edges)

    # ---------------------------------------------------------------
    # REMOVE FROM BOILERPLATE
    # Hardcoded reference (REMEMBER ONLY WORKS FOR NEIG = 3)
    out_3neig = cov_reduce_sym_r3(noise, diffuse, edges, 128)
    # ---------------------------------------------------------------

    # GPU
    out_kernel = cp.zeros((nb, n_eig, n_eig), dtype=cp.float32)

    call_kernel(lib, diffuse, noise, edges, out_kernel,
                nb, n_eig, threads_per_block)
    sync(lib)

    # Check against cupy
    match_cupy = cp.allclose(ref_cupy, out_kernel, atol=1e-4, rtol=1e-4)
    max_diff_cupy = np.max(np.abs(ref_cupy - out_kernel))

    # Check against cpu ref
    match_cpu = cp.allclose(ref_cpu, out_kernel, atol=1e-4, rtol=1e-4)
    max_diff_cpu = np.max(np.abs(ref_cpu - out_kernel))

    # ---------------------------------------------------------------
    # REMOVE FROM BOILERPLATE
    # Check against hardcoded version
    match_hardcode = cp.allclose(out_3neig, out_kernel, atol=1e-4, rtol=1e-4)
    max_diff_hardcode = np.max(np.abs(out_3neig - out_kernel))
    # ---------------------------------------------------------------

    # print(f"AGAINST HARDCODED VERSION:  n_eig={n_eig:2d}  |  allclose: {match_hardcode}  |  max |diff|: {max_diff_hardcode:.2e}")

    print(f"AGAINST CUPY:  n_eig={n_eig:2d}  |  allclose: {match_cupy}  |  max |diff|: {max_diff_cupy:.2e}")    
    print(f"AGAINST CPU:  n_eig={n_eig:2d}  |  allclose: {match_cpu}  |  max |diff|: {max_diff_cpu:.2e}")
    print(80*"-")

    debug_match = False
    if debug_match:
        if not match_cupy:
            with open(f"debug_n_eig.txt", "a") as f:
                f.write(f"n_eig={n_eig}\n")
                f.write(f"ref_cupy sum: {cp.sum(ref_cupy):.6e}\n")
                f.write(f"kernel  sum: {cp.sum(out_kernel):.6e}\n")
                f.write(f"ref[0]:\n{cp.asnumpy(ref_cupy[0])}\n\n")
                f.write(f"gpu[0]:\n{cp.asnumpy(out_kernel[0])}\n\n")
                
                # check if kernel output is all zeros
                f.write(f"kernel all zeros: {cp.all(out_kernel == 0)}\n")
                f.write(f"kernel max: {cp.max(cp.abs(out_kernel)):.6e}\n")
                f.write(f"ref max: {cp.max(cp.abs(ref_cupy)):.6e}\n")

        print(f"  n_eig={n_eig} CUDA error: {cudart.cudaGetLastError()}")

    return match_cupy, match_cpu


def test_multiple_correct():
    pass


# ============================================================
# Timing test
# ============================================================


def timing_test(lib, n_eig, rc_tuple, threads_per_block, seed):
    """Time the GPU kernel."""
    # Generate the test data

    test_data = make_test_data(n_eig, rc_tuple, seed)
    ref_cupy, nb = cupy_ref(
        test_data["noise"], 
        test_data["diffuse"], 
        test_data["edges"]
        ) # we need nb for the kernel

    # Initialize the output matrix
    out_kernel = cp.zeros((nb, n_eig, n_eig), dtype=cp.float32)


    # Benchmark gen_eig kernel
    times = benchmark(call_kernel, 
                      (lib, test_data["diffuse"], test_data["noise"], test_data["edges"], out_kernel,
                    nb, n_eig, threads_per_block), 
                    n_repeat= 1000)
    
    avg_gpu = float(cp.mean(times.gpu_times)) * 1e6  # microseconds
    avg_cpu = float(cp.mean(times.cpu_times)) * 1e6
    

    # Benchmark [INSERT REF FUNCTION] -> CuPy ref right now
    ref_times = benchmark(cupy_ref, 
                          (test_data["noise"], test_data["diffuse"], test_data["edges"]), 
                          n_repeat= 1000)
    
    ref_avg_gpu = float(cp.mean(ref_times.gpu_times)) * 1e6  # microseconds
    ref_avg_cpu = float(cp.mean(ref_times.cpu_times)) * 1e6


    n_sym = n_eig * (n_eig + 1) // 2
    # Print out the results
    print(f"  [n_eig={n_eig}  n_sym={n_sym:>3}]   "
            f"KERNEL: gpu={avg_gpu:>5.1f} us  cpu={avg_cpu:>5.1f} us | "
            f"CUPY: gpu={ref_avg_gpu:>7.1f} us  cpu={ref_avg_cpu:>5.1f} us")
    
    return avg_gpu, ref_avg_gpu
    

def time_multiple(lib, rc_tuple, tpb, eig_range, random_seed):
    eig_start, eig_stop = eig_range[0], eig_range[1]
    print()
    print("=" * 65)
    print(f"TIMING TESTS -> n_ant = {rows} * {cols} = {n_ant}")
    print("=" * 65)

    for n_eig in range(eig_start, eig_stop):
        timing_test(lib, n_eig, rc_tuple, 128, random_seed)

    print("=" * 65)


#for generating combos of rows and columns for full benchmarking        
def pop_row_col_input(end_iter_num):
    row_col_list = []
    j = 0
    for i in range(2, end_iter_num):
        j += i
        row_col = [j, j]
        row_col_list.append(row_col)
    return row_col_list


def timing_plot_nant_varies(
        lib, n_eig, rc_tuple, n_trials, tpb, random_seed, save_plot=True
        ):

    n_iter = n_trials + 2
    row_col_inputs = pop_row_col_input(n_iter)
    n_ants = np.zeros(len(row_col_inputs))
    for i, rc in enumerate(row_col_inputs):
        n_ant = rc[0]*rc[1]
        n_ants[i] = n_ant
    print(f"row-col combos used are: {row_col_inputs}")

    test_times = np.zeros(len(row_col_inputs))
    reference_times = np.zeros(len(row_col_inputs))

    for i, rc in enumerate(row_col_inputs):
        print(f"on iteration {i}")
        # test_data = make_test_data(n_eig, rc, random_seed)
        gpu_t, ref_gpu_t = timing_test(lib, n_eig, rc_tuple, tpb, random_seed)
        test_times[i] = gpu_t
        reference_times[i] = ref_gpu_t

    # ------------------------------------------------------
    # CHANGE TO DESIRED OUTPUT FOLDER AND TITLE NAME
    dir_name = 'varying_nant_plots'
    title = r"$\mathbf{On\ Cluster\ \rightarrow\ A40\ GPU}$"
    # ------------------------------------------------------

    file_name = f'n_trials={n_iter-2}'

    #plotting
    plt.rcParams["text.usetex"] = False
    plt.rcParams['axes.labelsize'] = 13
    plt.rcParams['figure.figsize'] = (8, 5)
    plt.rcParams.update({
        "mathtext.fontset": "cm",
        "font.family": "serif",
        })

    fig, ax = plt.subplots()
    ax.loglog(n_ants, test_times, '-x', ms = 7,  label = 'cupy')
    ax.loglog(n_ants, reference_times, '-p', ms = 7, label = 'corrcal')
    
    ax.xaxis.set_major_locator(FixedLocator(n_ants))
    ax.xaxis.set_major_formatter(
        FixedFormatter([rf"${int(np.sqrt(n))}^2$" for n in n_ants])
    )
    ax.tick_params(axis='both', which='major',
        labelsize=13, length=6, width=1.5)
    
    # ax.tick_params(axis='x', labelrotation=-20)
    
    ax.set_xlabel(r"$\mathbf{Number\ of\ Antennas}$")
    ax.set_ylabel(r"$\mathbf{Time\ (s)}$")
    ax.set_title(title, fontsize='14')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    
    if save_plot:
        plt.savefig(f'{dir_name}/{file_name}.png', format = 'png', dpi = 300, bbox_inches = 'tight')
 
    plt.show()


def timing_plot_neig_varies():
    pass

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    lib = load_kernel()

    # Test params
    rows = 32
    cols = 24
    n_eig = 3
    rc = (rows, cols)
    n_ant = rows*cols
    random_seed=42
    n_trials = 5

    # Test correctness
    correctness = False
    if correctness:
        print("=" * 65)
        print("CORRECTNESS TESTS")
        print("=" * 65)

        all_pass = True
        for n_eig in range(3, 4):
            ok = correctness_test(n_eig, rc, threads_per_block=128)
            all_pass = all_pass and ok

        print("-" * 65)
        if all_pass:
            print("ALL PASSED")
        else:
            print("SOME TESTS FAILED")

    # Run benchmark tests

    one_timing_test = False
    many_timing_tests = False

    if one_timing_test:
        timing_test(lib, n_eig, rc, 128, random_seed)
    
    if many_timing_tests:
        eig_range = (1, 6)
        time_multiple(lib, rc, 128, eig_range, random_seed)
    
    plot_benchmark = True

    if plot_benchmark:
        timing_plot_nant_varies(lib, n_eig, rc, n_trials, 128, random_seed, save_plot=True)
