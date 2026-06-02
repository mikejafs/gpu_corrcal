import cupy as cp
import ctypes
from gpu_corrcal.utils.gridding import *
from gpu_corrcal.utils.simulate_params import *
from gpu_corrcal.utils.zp_puregpu_funcs_py import *
from gpu_corrcal.inverse_covariance import *
from gpu_corrcal.utils.tools import *
from gpu_corrcal.cupy_reference.cupy_inverse_covariance import *
from cupyx.profiler import benchmark
cudart = ctypes.CDLL("libcudart.so")

# ============================================================
# Correctness test
# ============================================================

def correctness_test(n_eig, rc_tuple, threads_per_block):
    """Compare GPU kernel output to CPU reference."""

    test_data = make_test_data(n_eig, rc_tuple, 12)
    diffuse = test_data["diffuse"]
    noise = test_data["noise"]
    edges = test_data["edges"]

    # CuPy reference
    ref_cupy, nb = cupy_ref(noise, diffuse, edges)

    # GPU
    inv_noise, out_warp_red = inv_cov(noise, diffuse, 1, edges)

    sync()

    print(f" red kernel out {out_warp_red.shape}")
    print(f"cupy out {ref_cupy.shape}")

    # Check against cupy
    match_cupy = cp.allclose(ref_cupy, out_warp_red, atol=1e-4, rtol=1e-4)
    max_diff_cupy = np.max(np.abs(ref_cupy - out_warp_red))

    print(f"AGAINST CUPY:  n_eig={n_eig:2d}  |  allclose: {match_cupy}  |  max |diff|: {max_diff_cupy:.2e}")    
    print(80*"-")

    return match_cupy


def test_multiple_correct(rc_tuple, eig_range, threads_per_block=128):
    eig_start, eig_stop = eig_range
    print("=" * 65)
    print("CORRECTNESS TESTS")
    print("=" * 65)

    all_pass = True
    for n_eig in range(eig_start, eig_stop):
        match_cupy = correctness_test(n_eig, rc_tuple, threads_per_block)
        all_pass = all_pass and match_cupy

    print("-" * 65)
    print("ALL PASSED" if all_pass else "SOME TESTS FAILED")
    return all_pass

if __name__ == "__main__":
    # Test params
    # -------------------------------
    rows = 32
    cols = 18
    n_eig = 3
    rc = (rows, cols)
    n_ant = rows*cols
    random_seed=42
    T = True
    F = False
    # -------------------------------


    # Switch board for running tests
    # -------------------------------
    correctness = T

    if correctness:
        eig_range = (1, 20)   
        test_multiple_correct(rc, eig_range, threads_per_block=128)
