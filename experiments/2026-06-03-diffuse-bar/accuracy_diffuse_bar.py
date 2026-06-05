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

mul_temp_by_diffuse_chol_inv = lib.launch_mul_temp_by_chol
mul_temp_by_diffuse_chol_inv.argtypes = [
    ctypes.c_void_p,   #noise input
    ctypes.c_void_p,   #diffuse input
    ctypes.c_void_p,   #diffuse chol inv T mat
    ctypes.c_void_p,   #output mat    
    ctypes.c_void_p,      #edges
    ctypes.c_int,      #num blocks
    ctypes.c_int       #n_eig
]


def mul_temp_by_diffuse_chol(noise, diffuse, L_inv_T, out, edges):
    n_eig = diffuse.shape[1]
    nb = len(edges) - 1
    mul_temp_by_diffuse_chol_inv(
        noise.data.ptr,
        diffuse.data.ptr,
        L_inv_T.data.ptr,
        out.data.ptr,
        edges.data.ptr,
        nb,
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
    temp, t3 = setup_cupy_ref(noise, diffuse, edges)

    ws = InvCovWorkspace(diffuse, source, edges)
    L_del_inv_T = inv_cov(noise, diffuse, source, edges, ws)
    sync()



    # --------------------------------
    # CuPy reference 
    out_cupy = cupy_ref(temp, edges, L_del_inv_T)
    cp.cuda.Stream.null.synchronize()
    ref_cupy = undo_zeroPad(out_cupy, edges, dtype=cp.float32, ReImsplit=True)

    # ref_cupy = ref_cupy.reshape(diffuse.shape[0], diffuse.shape[1])
    cp.cuda.Stream.null.synchronize()

    #--------------------------------------------
    # New GPU kernel
    
    out_kernel = mul_temp_by_diffuse_chol(
        noise, diffuse, L_del_inv_T, ws.diffuse_bar, edges
    )
    sync()
    # print(f"out:   max={float(cp.max(cp.abs(out_kernel))):.2e}, nan={int(cp.sum(cp.isnan(out_kernel)))}, inf={int(cp.sum(cp.isinf(out_kernel)))}")

    # print(ref_cupy[0])
    # print(out_kernel[0])
    

    # Check against cupy
    match_cupy = cp.allclose(ref_cupy, out_kernel, atol=1e-4, rtol=1e-4)
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
        eig_range = (1, 21)   
        test_multiple_correct(rc, eig_range, threads_per_block=128)
