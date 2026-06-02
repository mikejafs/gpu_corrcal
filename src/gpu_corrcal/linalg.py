# Folder for storing various linear algebra functions written over top of cuda kernels

import ctypes
from gpu_corrcal._kernels import *

# warp_reduction_kernel_lib = load_warp_reduction_kernel()

def make_small_blocks(
        noise_gpu, diffuse_gpu, edges_gpu
        ):
    
    """Launch the kernel through ctypes."""
    n_eig = diffuse_gpu.shape[1]
    nb = len(edges_gpu) - 1
    threads_per_block = 128
    inv_noise = 1.0 / noise_gpu
    out = cp.zeros((nb, n_eig, n_eig), dtype=cp.float32)

    warp_reduction(
        diffuse_gpu.data.ptr,
        inv_noise.data.ptr,
        edges_gpu.data.ptr,
        out.data.ptr,
        ctypes.c_int(nb),
        ctypes.c_int(n_eig),
        ctypes.c_int(threads_per_block),
    )

    return inv_noise, out
