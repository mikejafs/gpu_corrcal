"""
ctypes interface and correctness/timing test for the templated
general eigenmode warp reduction kernel.

Compile:
    nvcc -Xcompiler -fPIC -shared -o geneig_warp_red_kernel.so geneig_warp_red_kernel.cu

Usage:
    python geneig_warp_red_kernel_test.py


CAUTION:
    NEED TO REWORK THE TESTS TO REPRESENT TRUE REDUNDANT ARRAY SHAPES
    CURRENTLY NOT ASSUMING CORRECT REDUNDANT BLOCK SHAPES

    THEN DO THE SAME FOR THE SHAREDMEM FILE
"""

import numpy as np
import cupy as cp
import ctypes
import os
import time
from utils.gridding import *
from utils.simulate_params import *
from utils.zp_puregpu_funcs_py import *
from cupyx.profiler import benchmark

cudart = ctypes.CDLL("libcudart.so")


# ============================================================
# ctypes wrapper
# ============================================================

def load_kernel(so_path=None):
    """Load the shared library and set up function signatures."""
    if so_path is None:
        so_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "geneig_warp_red_kernel_templated.so"
        )
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
# CPU reference
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

def make_test_data(n_eig, rows, cols, seed):
    """Generate random test data using the simulate params library"""
    cp.random.seed(seed)

    n_ant = rows*cols
    # print(f" n_eig={n_eig}", end="", flush=True)

    spms = SimCorrcalParams(n_ant, n_eig, n_src=1, precision='float32', xp=cp)
    edges = spms.edges(rows, cols, use_random=False)
    edges_gpu = cp.asarray(edges)

    sim_data = spms.sim_data()
    noise = sim_data[0]
    diffuse = sim_data[1]

    return diffuse, noise, edges_gpu


# ============================================================
# Correctness test
# ============================================================

def correctness_test(n_eig, rows, cols, threads_per_block):
    """Compare GPU kernel output to CPU reference."""
    diffuse, noise, edges = make_test_data(n_eig, rows, cols, 12)

    # CPU reference
    ref_cpu = cpu_ref(cp.asnumpy(diffuse), cp.asnumpy(noise), cp.asnumpy(edges), n_eig)
    ref_cpu = cp.asarray(ref_cpu)

    # CuPy reference
    ref_cupy, nb = cupy_ref(noise, diffuse, edges)
    # print(ref_cupy.shape)

    # GPU
    # diffuse_gpu = cp.asarray(diffuse)
    # noise_gpu = cp.asarray(noise)
    # edges_gpu = cp.asarray(edges)
    out_kernel = cp.zeros((nb, n_eig, n_eig), dtype=cp.float32)
    # print(out_kernel.shape)

    call_kernel(lib, diffuse, noise, edges, out_kernel,
                nb, n_eig, threads_per_block)
    sync(lib)

    # Check against cupy
    match_cupy = cp.allclose(ref_cupy, out_kernel, atol=1e-4, rtol=1e-4)
    max_diff_cupy = np.max(np.abs(ref_cupy - out_kernel))

    # Check against cpu ref
    match_cpu = cp.allclose(ref_cpu, out_kernel, atol=1e-4, rtol=1e-4)
    max_diff_cpu = np.max(np.abs(ref_cpu - out_kernel))

    # print()
    print(f"AGAINST CUPY:  n_eig={n_eig:2d}  |  allclose: {match_cupy}  |  max |diff|: {max_diff_cupy:.2e}")
    # print(80*"-")

    # print(f"AGAINST CPU:  n_eig={n_eig:2d}  |  allclose: {match_cpu}  |  max |diff|: {max_diff_cpu:.2e}")
    # print(80*"-")
    # print()



    # if not match_cupy:
    #     block_diffs = cp.array([cp.max(cp.abs(ref_cupy[b] - out_kernel[b])) for b in range(nb)])
    #     worst = cp.argmax(block_diffs)
    #     print(f"    worst block: {worst}, max diff: {block_diffs[worst]:.2e}")
    #     print(f"    ref[{worst}]:\n{ref_cupy[worst]}")
    #     print(f"    gpu[{worst}]:\n{out_kernel[worst]}")

    check_zeros = True
    if check_zeros:
        pass


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


# ============================================================
# Timing test
# ============================================================

def timing_test(lib, n_eig, rows, cols, threads_per_block, seed):
    """Time the GPU kernel."""
    diffuse, noise, edges = make_test_data(n_eig, rows, cols, seed)
    ref_cupy, nb = cupy_ref(noise, diffuse, edges)

    # diffuse_gpu = cp.asarray(diffuse)
    # noise_gpu = cp.asarray(noise)
    # edges_gpu = cp.asarray(edges)
    out_kernel = cp.zeros((nb, n_eig, n_eig), dtype=cp.float32)

    # Warmup
    # for _ in range(5):
    #     call_kernel(lib, diffuse, noise, edges, out_kernel,
    #                 nb, n_eig, threads_per_block)
    # sync(lib)

    # # Timed
    # sync(lib)
    # t0 = time.perf_counter()
    # for _ in range(n_iter):
    #     call_kernel(ç)
    # sync(lib)
    # elapsed = time.perf_counter() - t0

    # per_call_ms = (elapsed / n_iter) * 1e6
    # n_total = int(edges[-1])
    # print(f"  n_eig={n_eig:2d}  |  nb={nb}  |  n_total={n_total:>8d}  |  "
    #       f"{per_call_ms:.8f} us/call  ({n_iter} iters)")

    times = benchmark(call_kernel, (lib, diffuse, noise, edges, out_kernel,
                    nb, n_eig, threads_per_block), n_repeat= 100)
    avg_gpu = float(cp.mean(times.gpu_times)) * 1e6  # microseconds
    avg_cpu = float(cp.mean(times.cpu_times)) * 1e6
    n_sym = n_eig * (n_eig + 1) // 2
    print(f"  n_eig={n_eig}  n_sym={n_sym:>3}  "
            f"gpu={avg_gpu:>8.1f} us  cpu={avg_cpu:>8.1f} us")

    # return per_call_ms


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    lib = load_kernel()

    print("=" * 65)
    print("CORRECTNESS TESTS")
    print("=" * 65)

    rows = 16
    cols = 21
    n_ant = rows*cols
    random_seed=42

    all_pass = True
    for n_eig in range(1, 21):
        ok_cupy, ok_cpu = correctness_test(n_eig, rows, cols, threads_per_block=128)
        # ok_cupy = correctness_test(n_eig, rows, cols, threads_per_block=512)
        all_pass = all_pass and ok_cupy

    print("-" * 65)
    if all_pass:
        print("ALL PASSED")
    else:
        print("SOME TESTS FAILED")

    print()
    print("=" * 65)
    print("TIMING TESTS")
    print("=" * 65)

    for n_eig in range(1, 21):
        timing_test(lib, n_eig, rows, cols, 128, random_seed)

    print("=" * 65)
