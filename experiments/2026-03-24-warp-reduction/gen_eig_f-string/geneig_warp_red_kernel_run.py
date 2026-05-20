#rewrite the gen_eig warp reduction kernel here from scratch

"""
Kernel plus the Python Interface for the General Eigenmode Version of
the Warp Reduction Implimention

nvcc -Xcompiler -fPIC -shared -o geneig_warp_red_kernel-notemplate.so geneig_warp_red_kernel.cu


WHAT WE'RE TRYING TO DO HERE...
1. Run the "dynamically" programmed gen warp red kernel written in .cu file to see if there is an error and what the error message is
2. If some how there is no error:
    a. need to benchmark agains both cupy and hardcoded red kernel to try and understand what is going on
        - My hunch is that if somehow the compiler let's us do this, then the arrays will be dynamically initialized at the main memory level resulting in substantial loss of performance

UPDATE:
    THIS FILE WAS NOT EVEN NECESSARY -- The geneig CUDA file won't even compile as a result of trying to declar dynamic kernel variables. This simply isn't possible in CUDA C. 
        
"""

from utils.gridding import *
from utils.simulate_params import *
from cupyx.profiler import benchmark

import numpy as np
import cupy as cp
import ctypes
import os

def load_lib(name):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    lib = ctypes.CDLL(path)
    lib.sync_device.restype = None
    lib.sync_device.argtypes = []
    return lib

def setup_warp_red_lib(lib):
    lib.launch_two_level_warp_reduction.restype = None
    lib.launch_two_level_warp_reduction.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int
    ]
    return lib

def call_warp_red_lib(lib, diffuse, noise, edges, out, nb, n_eig, tpb):
    lib.launch_two_level_waarp_reduction(
        diffuse.data.ptr,
        noise.data.ptr,
        edges.data.ptr,
        out.data.ptr,
        ctypes.c_int(nb),
        ctypes.c_int(n_eig),
        ctypes.c_int(tpb),
    )

def test_gen_warp_kernel():
    n_eig = 3
    rows = 10
    cols = 10
    n_ant = rows*cols

    #generate the simulate params object
    spms = SimCorrcalParams(n_ant, n_eig, n_src=1, precision='float32', xp=cp)

    #create the noise and diffuse matrices
    sim_data = spms.sim_data()
    noise = sim_data[0]
    diffuse = sim_data[1]
    edges = spms.edges(rows, cols, use_random = False)

    


if __name__ == "__main__":
    test_gen_warp_kernel()
    # print("Everything is working")













