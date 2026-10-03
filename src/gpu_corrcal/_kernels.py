import numpy as np
import cupy as cp
import ctypes
import os
from pathlib import Path

lib_dir = Path(__file__).resolve().parent / "kernels"

# nvcc -Xcompiler -fPIC -shared -o geneig_warp_red_kernel.so geneig_warp_red_kernel.cu

def _load(name):
    matches = list(lib_dir.glob(f"{name}*.so"))
    if not matches:
        raise FileNotFoundError(f"No .so matching {name!r} in {lib_dir}")
    return ctypes.cdll.LoadLibrary(str(matches[0]))


_warp_reduction_kernel = _load("geneig_warp_red_kernel_templated")
_fused_chol_inv = _load("fused_chol_inv_kernel")
_diffuse_bar_kernel = _load("diffuse_bar_kernel")
_sig_prime_kernels = _load("sig_prime_kernels_pta")
_matvec_mul_kernels = _load("sparse_cov_times_vec")
_apply_gains_kernel = _load("apply_gains")


sync_device = _warp_reduction_kernel.sync_device
sync_device.restype = None
sync_device.argtypes = []


# INVERSE COVARIANCE

warp_reduction = _warp_reduction_kernel.launch_two_level_warp_reduction
warp_reduction.restype = None
warp_reduction.argtypes = [
    ctypes.c_void_p,  # diffuse
    ctypes.c_void_p,  # noise
    ctypes.c_void_p,  # edges
    ctypes.c_void_p,  # out
    ctypes.c_int,     # nb
    ctypes.c_int,     # n_eig
    ctypes.c_int,     # threads_per_block
]
    
fused_chol_inv = _fused_chol_inv.launch_batched_cholesky_inv
fused_chol_inv.restype = None
fused_chol_inv.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]

mul_temp_by_diffuse_chol_inv = _diffuse_bar_kernel.launch_mul_temp_by_chol
mul_temp_by_diffuse_chol_inv.restype = None
mul_temp_by_diffuse_chol_inv.argtypes = [
    ctypes.c_void_p,   #noise input
    ctypes.c_void_p,   #diffuse input
    ctypes.c_void_p,   #diffuse chol inv T mat
    ctypes.c_void_p,   #output mat    
    ctypes.c_void_p,      #edges
    ctypes.c_int,      #num blocks
    ctypes.c_int,      #n_eig
]

# ============================================================================
# Additions to gpu_corrcal/_kernels.py  --  Sig_prime (second Woodbury level)
# Append these to the existing _kernels.py, after the diffuse_bar bindings.
# Compile:  nvcc -Xcompiler -fPIC -shared -o sig_prime_kernels.so sig_prime_kernels.cu
# ============================================================================
  
# ---- KERNEL 1: cap_reduce -> writes B and M_sig ----------------------------
# cap_reduce = _sig_prime_kernels.launch_cap_reduce
# cap_reduce.restype = None
# cap_reduce.argtypes = [
#     ctypes.c_void_p,   # Sig
#     ctypes.c_void_p,   # Del_prime  (== diffuse_bar from the diffuse level)
#     ctypes.c_void_p,   # noise
#     ctypes.c_void_p,   # edges
#     ctypes.c_void_p,   # B          (output, (nb, n_src, n_eig))
#     ctypes.c_void_p,   # M_sig      (accumulated, (n_src, n_src) -- zero before launch)
#     ctypes.c_int,      # nb
#     ctypes.c_int,      # n_src
#     ctypes.c_int,      # n_eig
# ]

cap_reduce = _sig_prime_kernels.launch_cap_reduce
cap_reduce.restype = ctypes.c_int          # it returns int, not None
cap_reduce.argtypes = [
    ctypes.c_int,      # n_src
    ctypes.c_int,      # n_eig
    ctypes.c_int,      # nb
    ctypes.c_void_p,   # Sig
    ctypes.c_void_p,   # Del_prime
    ctypes.c_void_p,   # noise
    ctypes.c_void_p,   # edges
    ctypes.c_void_p,   # B
    ctypes.c_void_p,   # M_sig
    ctypes.c_void_p,   # stream
]
 
# # ---- KERNEL 2: chol_inv_fused (single (n_src x n_src)) ----------------------
# chol_inv_fused = _sig_prime_kernels.launch_chol_inv_fused
# chol_inv_fused.restype = None
# chol_inv_fused.argtypes = [
#     ctypes.c_void_p,   # M_sig  (lower tri = Lambda)
#     ctypes.c_void_p,   # L_inv  (output)
#     ctypes.c_int,      # n_src
# ]
 
# ---- KERNEL 3: apply_sig_prime -> writes Sig_prime -------------------------
# apply_sig_prime = _sig_prime_kernels.launch_apply_sig_prime
# apply_sig_prime.restype = None
# apply_sig_prime.argtypes = [
#     ctypes.c_void_p,   # Sig
#     ctypes.c_void_p,   # Del_prime
#     ctypes.c_void_p,   # noise
#     ctypes.c_void_p,   # edges
#     ctypes.c_void_p,   # B          (from kernel 1)
#     ctypes.c_void_p,   # L_inv
#     ctypes.c_void_p,   # Sig_prime  (output, (n_total, n_src))
#     ctypes.c_int,      # nb
#     ctypes.c_int,      # n_src
#     ctypes.c_int,      # n_eig
# ]


chol_inv_fused = _sig_prime_kernels.launch_chol_inv_fused
chol_inv_fused.restype = ctypes.c_int
chol_inv_fused.argtypes = [
    ctypes.c_int,      # n_src
    ctypes.c_void_p,   # M_sig
    ctypes.c_void_p,   # L_inv
    ctypes.c_void_p,   # logdet
    ctypes.c_int,      # return_det (bool)
    ctypes.c_void_p,   # stream
]


apply_sig_prime = _sig_prime_kernels.launch_apply_sig_prime
apply_sig_prime.restype = ctypes.c_int
apply_sig_prime.argtypes = [
    ctypes.c_int,      # n_src
    ctypes.c_int,      # n_eig
    ctypes.c_int,      # nb
    ctypes.c_void_p,   # Sig
    ctypes.c_void_p,   # Del_prime
    ctypes.c_void_p,   # noise
    ctypes.c_void_p,   # edges
    ctypes.c_void_p,   # B
    ctypes.c_void_p,   # L_inv
    ctypes.c_void_p,   # Sig_prime
    ctypes.c_void_p,   # stream
]

# --------------------------
# MULTIPLY SPARSE COV BY VEC
# --------------------------
sparse_cov_times_vec = _matvec_mul_kernels.launch_sparse_cov_times_vec
sparse_cov_times_vec.restype = None
sparse_cov_times_vec.argtypes = [
    ctypes.c_void_p,   # noise
    ctypes.c_void_p,   # diffuse
    ctypes.c_void_p,   # source
    ctypes.c_void_p,   # vec
    ctypes.c_void_p,   # out
    ctypes.c_void_p,   # del_tmp workspace
    ctypes.c_void_p,   # sig_tmp workspace
    ctypes.c_void_p,   # edges (int32)
    ctypes.c_int,      # nb
    ctypes.c_int,      # n_src
    ctypes.c_int,      # n_eig
    ctypes.c_int,      # is_inv
    ctypes.c_void_p,   # cudaStream_t
]

apply_gains_launch = _apply_gains_kernel.launch_apply_gains
apply_gains_launch.restype = ctypes.c_int
apply_gains_launch.argtypes = [
    ctypes.c_void_p,   # gains  (2*n_ant,) float32, Re/Im alternating
    ctypes.c_void_p,   # ant_1  (n_bl,) int32
    ctypes.c_void_p,   # ant_2  (n_bl,) int32
    ctypes.c_void_p,   # mat    (2*n_bl, n_col) float32
    ctypes.c_void_p,   # out    (2*n_bl, n_col) float32
    ctypes.c_int,      # n_bl
    ctypes.c_int,      # n_col
    ctypes.c_void_p,   # cudaStream_t
]