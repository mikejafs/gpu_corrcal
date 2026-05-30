import numpy as np
import cupy as cp
import ctypes

from gpu_corrcal.utils.gridding import *
from gpu_corrcal.utils.simulate_params import *
from gpu_corrcal.utils.zp_puregpu_funcs_py import *
cudart = ctypes.CDLL("libcudart.so")

# ============================================================
# Correctness test
# ============================================================

def correctness_test(lib, n_eig, rc_tuple, threads_per_block):
    """Compare GPU kernel output to CPU reference."""

    test_data = make_test_data(n_eig, rc_tuple, 12)
    diffuse = test_data["diffuse"]
    noise = test_data["noise"]
    edges = test_data["edges"]

    # CPU reference
    ref_cpu = cpu_ref(cp.asnumpy(diffuse), cp.asnumpy(noise), cp.asnumpy(edges), n_eig)
    ref_cpu = cp.asarray(ref_cpu)

    # CuPy reference
    ref_cupy, nb = cupy_ref(noise, diffuse, edges)

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


def test_multiple_correct(lib, rc_tuple, eig_range, threads_per_block=128):
    eig_start, eig_stop = eig_range
    print("=" * 65)
    print("CORRECTNESS TESTS")
    print("=" * 65)

    all_pass = True
    for n_eig in range(eig_start, eig_stop):
        match_cupy, match_cpu = correctness_test(lib, n_eig, rc_tuple, threads_per_block)
        all_pass = all_pass and match_cupy and match_cpu

    print("-" * 65)
    print("ALL PASSED" if all_pass else "SOME TESTS FAILED")
    return all_pass
