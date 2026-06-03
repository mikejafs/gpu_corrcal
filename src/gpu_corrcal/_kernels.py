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

sync_device = _warp_reduction_kernel.sync_device
sync_device.restype = None
sync_device.argtypes = []


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
    ctypes.c_int,
    ctypes.c_int
]
