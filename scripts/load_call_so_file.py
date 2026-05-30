import numpy as np
import cupy as cp
import ctypes
import os
import time
import re
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FixedLocator, FixedFormatter
from gpu_corrcal.utils.gridding import *
from gpu_corrcal.utils.simulate_params import *
from gpu_corrcal.utils.zp_puregpu_funcs_py import *
from cupyx.profiler import benchmark

# cudart = ctypes.CDLL("libcudart.so")

# ============================================================
# ctypes wrapper
# ============================================================

def load_kernel(so_path=None):

    # Wether or not to use pragma unroll
    unroll = True
    """Load the shared library and set up function signatures."""
    if so_path is None:
        #if NO PRAGMA UNROLL
        if not unroll:
            so_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "geneig_warp_red_kernel_templated_NOpragma.so"
            )
            print(f"NO unroll")

        #if YES PRAGMA UNROLL
        elif unroll:
            so_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "geneig_warp_red_kernel_templated.so"
            )
            print(f"WITH unroll")

    lib = ctypes.CDLL(so_path)

    # void launch_two_level_warp_reduction(
    #     const float* diffuse, const float* noise, const int* edges,
    #     float* out, int nb, int n_eig, int threads_per_block
    # )
    lib.launch_two_level_warp_reduction.restype = None
    lib.launch_two_level_warp_reduction.argtypes = [
        ctypes.c_void_p,  # diffuse
        ctypes.c_void_p,  # noise
        ctypes.c_void_p,  # edges
        ctypes.c_void_p,  # out
        ctypes.c_int,     # nb
        ctypes.c_int,     # n_eig
        ctypes.c_int,     # threads_per_block
    ]

    lib.sync_device.restype = None
    lib.sync_device.argtypes = []

    return lib


def call_kernel(lib, diffuse_gpu, noise_gpu, edges_gpu, out_gpu,
                nb, n_eig, threads_per_block):
    """Launch the kernel through ctypes."""
    lib.launch_two_level_warp_reduction(
        diffuse_gpu.data.ptr,
        noise_gpu.data.ptr,
        edges_gpu.data.ptr,
        out_gpu.data.ptr,
        ctypes.c_int(nb),
        ctypes.c_int(n_eig),
        ctypes.c_int(threads_per_block),
    )


def sync(lib):
    """Synchronize the device."""
    lib.sync_device()