# Folder for storing various linear algebra functions written over top of cuda kernels

import ctypes
from gpu_corrcal._kernels import *

# warp_reduction_kernel_lib = load_warp_reduction_kernel()

def make_small_blocks(
        noise_gpu, diffuse_gpu, edges_gpu, out
        ):
    
    """Launch the kernel through ctypes."""
    n_eig = diffuse_gpu.shape[1]
    nb = len(edges_gpu) - 1
    threads_per_block = 128

    warp_reduction(
        diffuse_gpu.data.ptr,
        noise_gpu.data.ptr,
        edges_gpu.data.ptr,
        out.data.ptr,
        ctypes.c_int(nb),
        ctypes.c_int(n_eig),
        ctypes.c_int(threads_per_block),
    )

    return out


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
