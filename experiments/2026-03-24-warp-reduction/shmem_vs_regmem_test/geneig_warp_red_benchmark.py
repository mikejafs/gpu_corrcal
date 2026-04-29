"""
Benchmark: templated (register) vs shared memory warp reduction kernel.

Compile both first:
    nvcc -Xcompiler -fPIC -shared -o geneig_warp_red_kernel.so geneig_warp_red_kernel.cu
    nvcc -Xcompiler -fPIC -shared -o geneig_warp_red_kernel_shmem.so geneig_warp_red_kernel_shmem.cu

Usage:
    python geneig_warp_red_benchmark.py
"""

import numpy as np
import cupy as cp
import ctypes
import os
import time


# ============================================================
# Load both libraries
# ============================================================

def load_lib(name):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    lib = ctypes.CDLL(path)
    lib.sync_device.restype = None
    lib.sync_device.argtypes = []
    return lib


def setup_register_lib(lib):
    lib.launch_two_level_warp_reduction.restype = None
    lib.launch_two_level_warp_reduction.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    return lib


def setup_shmem_lib(lib):
    lib.launch_two_level_warp_reduction_shmem.restype = None
    lib.launch_two_level_warp_reduction_shmem.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    return lib


# ============================================================
# Kernel launchers
# ============================================================

def call_register(lib, diffuse, noise, edges, out, nb, n_eig, tpb):
    lib.launch_two_level_warp_reduction(
        diffuse.data.ptr, noise.data.ptr, edges.data.ptr, out.data.ptr,
        ctypes.c_int(nb), ctypes.c_int(n_eig), ctypes.c_int(tpb),
    )

def call_shmem(lib, diffuse, noise, edges, out, nb, n_eig, tpb):
    lib.launch_two_level_warp_reduction_shmem(
        diffuse.data.ptr, noise.data.ptr, edges.data.ptr, out.data.ptr,
        ctypes.c_int(nb), ctypes.c_int(n_eig), ctypes.c_int(tpb),
    )


# ============================================================
# CPU reference
# ============================================================

def ref_cpu(diffuse, noise, edges, n_eig):
    nb = len(edges) - 1
    out = np.zeros((nb, n_eig, n_eig), dtype=np.float32)
    for b in range(nb):
        for i in range(edges[b], edges[b + 1]):
            d = diffuse[i, :]
            out[b] += np.outer(d, d) / noise[i]
    return out


# ============================================================
# Test data
# ============================================================

def make_test_data(nb, n_per_block, n_eig, seed=42):
    rng = np.random.default_rng(seed)
    edges = np.zeros(nb + 1, dtype=np.int32)
    for b in range(nb):
        edges[b + 1] = edges[b] + n_per_block
    n_total = int(edges[-1])
    diffuse = rng.standard_normal((n_total, n_eig)).astype(np.float32)
    noise = rng.uniform(0.1, 10.0, size=n_total).astype(np.float32)
    return diffuse, noise, edges


# ============================================================
# Correctness check
# ============================================================

def correctness_test(lib_reg, lib_shm, n_eig, nb=32, n_per_block=512, tpb=128):
    diffuse, noise, edges = make_test_data(nb, n_per_block, n_eig)
    ref = ref_cpu(diffuse, noise, edges, n_eig)

    diffuse_gpu = cp.asarray(diffuse)
    noise_gpu = cp.asarray(noise)
    edges_gpu = cp.asarray(edges)

    # Register version
    out_reg = cp.zeros((nb, n_eig, n_eig), dtype=cp.float32)
    call_register(lib_reg, diffuse_gpu, noise_gpu, edges_gpu, out_reg, nb, n_eig, tpb)
    lib_reg.sync_device()
    out_reg_h = cp.asnumpy(out_reg)

    # Shared memory version
    out_shm = cp.zeros((nb, n_eig, n_eig), dtype=cp.float32)
    call_shmem(lib_shm, diffuse_gpu, noise_gpu, edges_gpu, out_shm, nb, n_eig, tpb)
    lib_shm.sync_device()
    out_shm_h = cp.asnumpy(out_shm)

    match_reg = np.allclose(ref, out_reg_h, atol=1e-4, rtol=1e-4)
    match_shm = np.allclose(ref, out_shm_h, atol=1e-4, rtol=1e-4)
    max_diff_reg = np.max(np.abs(ref - out_reg_h))
    max_diff_shm = np.max(np.abs(ref - out_shm_h))

    print(f"  n_eig={n_eig:2d}  |  register: {match_reg} (max diff {max_diff_reg:.2e})"
          f"  |  shmem: {match_shm} (max diff {max_diff_shm:.2e})")

    return match_reg and match_shm


# ============================================================
# Timing
# ============================================================

def time_kernel(call_fn, lib, diffuse, noise, edges, out, nb, n_eig, tpb, n_iter=100):
    """Warmup then time."""
    for _ in range(10):
        call_fn(lib, diffuse, noise, edges, out, nb, n_eig, tpb)
    lib.sync_device()

    lib.sync_device()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        call_fn(lib, diffuse, noise, edges, out, nb, n_eig, tpb)
    lib.sync_device()
    elapsed = time.perf_counter() - t0
    return (elapsed / n_iter) * 1e3  # ms


def timing_test(lib_reg, lib_shm, n_eig, nb=512, n_per_block=1024, tpb=128, n_iter=100):
    diffuse, noise, edges = make_test_data(nb, n_per_block, n_eig)

    diffuse_gpu = cp.asarray(diffuse)
    noise_gpu = cp.asarray(noise)
    edges_gpu = cp.asarray(edges)
    out_gpu = cp.zeros((nb, n_eig, n_eig), dtype=cp.float32)

    ms_reg = time_kernel(call_register, lib_reg,
                         diffuse_gpu, noise_gpu, edges_gpu, out_gpu,
                         nb, n_eig, tpb, n_iter)

    ms_shm = time_kernel(call_shmem, lib_shm,
                         diffuse_gpu, noise_gpu, edges_gpu, out_gpu,
                         nb, n_eig, tpb, n_iter)

    n_total = int(edges[-1])
    slowdown = ms_shm / ms_reg if ms_reg > 0 else float('inf')

    print(f"  n_eig={n_eig:2d}  |  n_total={n_total:>8d}  |  "
          f"register: {ms_reg:7.3f} ms  |  shmem: {ms_shm:7.3f} ms  |  "
          f"shmem/reg: {slowdown:.2f}x")

    return ms_reg, ms_shm


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    lib_reg = setup_register_lib(load_lib("geneig_warp_red_kernel.so"))
    lib_shm = setup_shmem_lib(load_lib("geneig_warp_red_kernel_shmem.so"))

    print("=" * 80)
    print("CORRECTNESS TESTS")
    print("=" * 80)

    all_pass = True
    for n_eig in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20]:
        ok = correctness_test(lib_reg, lib_shm, n_eig)
        all_pass = all_pass and ok

    print("-" * 80)
    print("ALL PASSED" if all_pass else "SOME TESTS FAILED")

    print()
    print("=" * 80)
    print("TIMING TESTS  (nb=512, n_per_block=1024, tpb=128, 100 iters)")
    print("=" * 80)

    for n_eig in [2, 4, 6, 8, 10, 12, 16, 20]:
        timing_test(lib_reg, lib_shm, n_eig)

    print("=" * 80)
