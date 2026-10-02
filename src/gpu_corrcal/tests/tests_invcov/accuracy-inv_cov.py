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
    source = test_data["source"]
    print(len(edges))

    # CuPy reference --------------------------------
    #!!! Note that setup_c.. has everything up to the point
    # we actually need it -- not cupy_ref
    zp_diffuse_ref_cupy, zp_source_ref_cupy = setup_cupy_ref(noise, diffuse, source, edges)
    # print(f"cupy shape {zp_ref_cupy.shape}")
    
    # scale probes — do this BEFORE any undo_zeroPad
    print("REF  source_bar:  max", float(cp.abs(zp_source_ref_cupy).max()),
        "mean", float(cp.abs(zp_source_ref_cupy).mean()))

    # kernel, also in padded space if you can get it pre-undo:
    # (whatever inv_cov returns before undo)

    diffuse_ref_cupy = undo_zeroPad(zp_diffuse_ref_cupy, edges, dtype=cp.float32, ReImsplit=True)
    source_ref_cupy = undo_zeroPad(zp_source_ref_cupy, edges, dtype=cp.float32, ReImsplit=True)
    # print(source_ref_cupy)

    # print(f"cupy shape {ref_cupy.shape}")
    

    # GPU --------------------------------------------
    # Initialize the workspace:
    ws = InvCovWorkspace(diffuse, source, edges)
    diffuse_out_kernel, source_out_kernel = inv_cov(noise, diffuse, source, edges, ws)

    print("KERN source_out:  max", float(cp.abs(source_out_kernel).max()),
        "mean", float(cp.abs(source_out_kernel).mean()))

    # print(source_out_kernel)
    # print(f"out kern shape {out_kernel.shape}")
    # print(type(zp_ref_cupy))
    # print(type(zp_out_kernel))
    # print(zp_out_kernel.shape)

    sync()

    # Check against cupy

    # match_cupy = cp.allclose(diffuse_ref_cupy, diffuse_out_kernel, atol=1e-1, rtol=1e-5)
    # max_diff_cupy = np.max(np.abs(diffuse_ref_cupy - diffuse_out_kernel))

    match_cupy = cp.allclose(source_ref_cupy, source_out_kernel, atol=1e-1, rtol=1e-5)
    max_diff_cupy = np.max(np.abs(source_ref_cupy - source_out_kernel))

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
        eig_range = (1, 21)   
        test_multiple_correct(rc, eig_range, threads_per_block=128)
