import cupy as cp
import ctypes
from pathlib import Path
from gpu_corrcal.inverse_covariance import *
from gpu_corrcal.utils.gridding import *
from gpu_corrcal.utils.simulate_params import *
from gpu_corrcal.utils.zp_puregpu_funcs_py import *
from gpu_corrcal.utils.tools import *
from gpu_corrcal.cupy_reference.cupy_inverse_covariance import *

lib_dir = Path(__file__).resolve().parent

#change this if more than one .so file in this parent directory
file_path = list(lib_dir.glob("*.so"))[0]

lib = ctypes.cdll.LoadLibrary(file_path)

fused_chol_inv = lib.launch_batched_cholesky_inv
fused_chol_inv.argtypes = [
    ctypes.c_void_p,   #temp2 input
    ctypes.c_void_p,   #output mat
    ctypes.c_int,      #num blocks
    ctypes.c_int       #n_eig
]


def fused_cholesky_inverse(temp2, edges, out):
    n_eig = temp2.shape[1]
    num_blocks = len(edges) - 1
    fused_chol_inv(
        temp2.data.ptr,
        out.data.ptr,
        num_blocks,
        n_eig
    )
    return out


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



    # Set up both refs
    temp2_cupy, nb = setup_cupy_ref(noise, diffuse, edges)

    ws = InvCovWorkspace(nb, diffuse.shape[1], source.shape[1])
    temp2_kernel = inv_cov(noise, diffuse, source, edges, ws)
    sync()

    # CuPy reference --------------------------------
    ref_cupy = cupy_ref(temp2_cupy)
    # ref_cupy = cupy_ref(temp2_kernel)
    cp.cuda.Stream.null.synchronize()

    # GPU --------------------------------------------
    # Initialize the workspace:
    sync()

    #
    # print(f"Inf in temp2_cupy: {cp.sum(cp.isinf(temp2_cupy))}")
    # print(f"Inf in temp2_kernel: {cp.sum(cp.isinf(temp2_kernel))}")

    # print(f"NaN in noise: {cp.sum(cp.isnan(noise))}")
    # print(f"NaN in diffuse: {cp.sum(cp.isnan(diffuse))}")
    # print(f"Zeros in noise: {cp.sum(noise == 0)}")
    # print(f"NaN in temp2_cupy: {cp.sum(cp.isnan(temp2_cupy))}")
    # print(f"NaN in temp2_kernel: {cp.sum(cp.isnan(temp2_kernel))}")

    # print(f"temp2 match: {cp.allclose(temp2_cupy, temp2_kernel, atol=1e-3)}")
    # print(f"temp2 max diff: {cp.max(cp.abs(temp2_cupy - temp2_kernel)):.2e}")

    #new kernel
    out_kernel = fused_cholesky_inverse(temp2_kernel, edges, out=ws.L_del)
    sync()

    # print(ref_cupy)
    # print(out_kernel)
    

    # Check against cupy
    match_cupy = cp.allclose(ref_cupy, out_kernel, atol=1e-3, rtol=1e-3)
    max_diff_cupy = np.max(np.abs(ref_cupy - out_kernel))

    print(f"AGAINST CUPY:  n_eig={n_eig:2d}  |  allclose: {match_cupy}  |  max |diff|: {max_diff_cupy:.2e}")    
    print(80*"-")
    return match_cupy


def test_multiple_correct(rc_tuple, eig_range, threads_per_block):
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
    rows = 38
    cols = 21
    print(f"{rows*cols} antennas")
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
        eig_range = (3, 9)   
        test_multiple_correct(rc, eig_range, threads_per_block=128)
