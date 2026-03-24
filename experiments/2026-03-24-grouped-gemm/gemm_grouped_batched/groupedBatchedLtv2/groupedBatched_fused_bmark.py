import numpy as np
import cupy as cp
import ctypes
import sys
import os
from cupyx.profiler import benchmark

sys.path.insert(0, os.path.abspath('..'))

from simulate_params import *
from zp_puregpu_funcs_py import *


# ======================================================================
# Load and compile the fused CUDA kernel via CuPy RawModule
# ======================================================================

def _load_kernel():
    file_dir = os.path.dirname(os.path.abspath(__file__))
    kernel_path = os.path.join(file_dir, "corrcal_fused_blocks.cu")

    if not os.path.exists(kernel_path):
        raise FileNotFoundError(f"Kernel source not found: {kernel_path}")

    # Read file as raw bytes → decode as UTF-8 → ignore weird characters
    with open(kernel_path, "rb") as f:
        kernel_src = f.read().decode("utf-8", errors="ignore")

    module = cp.RawModule(code=kernel_src, backend="nvrtc")
    return module.get_function("corrcal_fused_blocks")

# Global so we only compile once
_fused_kernel = None


# ======================================================================
# Public user-facing function
# ======================================================================

def corrcal_fused_matmul(diff, temp, edges, threads_per_block=256):
    """
    Compute blockwise C_b = diff_b^T @ temp_b using a single fused CUDA kernel.

    Parameters
    ----------
    diff : CuPy array, shape (n_bl, n_eig)
    temp : CuPy array, shape (n_bl, n_eig)
            temp = noise_inv * diff (already computed by caller)
    edges : CuPy or numpy int array, shape (n_blocks+1,)
            Defines block boundaries: block b = rows edges[b] ... edges[b+1]-1
    threads_per_block : int, optional
            CUDA threadblock size (default: 256)

    Returns
    -------
    C_out : CuPy array, shape (n_blocks, n_eig, n_eig)
            Blockwise results.
    """

    global _fused_kernel
    if _fused_kernel is None:
        _fused_kernel = _load_kernel()

    # Ensure correct types
    diff = diff.astype(cp.float32)
    temp = temp.astype(cp.float32)
    edges = cp.asarray(edges, dtype=cp.int32)

    n_bl, r = diff.shape
    n_blocks = len(edges) - 1

    # Output
    C_out = cp.zeros((n_blocks, r, r), dtype=cp.float32)

    # Shared memory requirement:
    # each warp holds an r×r tile
    warps = threads_per_block // 32
    shmem_bytes = warps * (r*r) * 4   # float32 → 4 bytes

    # Launch the CUDA kernel
    _fused_kernel(
        (n_blocks,),                # grid
        (threads_per_block,),       # block
        (
            diff.ravel(),
            temp.ravel(),
            edges,
            r,
            C_out.ravel()
        ),
        shared_mem = shmem_bytes
    )

    return C_out


if __name__ == "__main__":

    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SET UP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    #Parameter set up
    ant_dim_x = 32
    ant_dim_y = 64
    n_ant = ant_dim_x * ant_dim_y    
    n_eig = 3
    n_src = 1

    print('Parameters:')
    print('Number of antennas (Re Im split):', n_ant,
        '\nNumber of eigenmodes:', n_eig,
        '\nNumbe of sources:', n_src)
    print()

    cp.random.seed(10)
    spms = SimCorrcalParams(n_ant, n_eig, n_src, precision='float32', xp=cp)
    edges = spms.edges(ant_dim_x, ant_dim_y, use_random=False)
    # print(edges)

    #simulated matrices with correct shapes
    sim_data = spms.sim_data()
    noise = sim_data[0]
    diff = sim_data[1]
    # print(len(diff))

    #zeropad diff and noise 
    zp_noise, nb, lb = zeroPad(noise, edges, return_inv=True, dtype=cp.float32)
    zp_diff, nb, lb = zeroPad(diff, edges, return_inv=False, dtype=cp.float32)


    #need this if wanting to compare to zped stuff since the true matmul is diff.T@N^-1@diff
    noise = 1/noise

    """~~~~~~~~~~~~~~~~~~~~~ COMPUTATION OF diff.T @ N^-1 @ diff ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

    #set up the temp mat just before the mat mul we are interested in

    #cublas temp (no zeropadding)
    temp = noise[..., None] * diff

    #zeropadded temp
    zp_temp = zp_noise[..., None] * zp_diff


    #cupy computation of diff.T @ N^-1 @ diff
    def cupy_block_mul(diff, tmp):
        res = cp.transpose(diff, [0, 2, 1]) @ tmp
        cp.cuda.Stream.null.synchronize()
        return res

    #return temp2 = diff.T @ N^-1 @ diff
    temp2 = cupy_block_mul(zp_diff, zp_temp)
    # print(temp2)

    #cublas compuation of diff.T @ N^-1 @ diff
    # params = prepare_grouped_batched_params_lt(diff, temp, edges)
    out = corrcal_fused_matmul(diff, temp, edges, threads_per_block=256)
    # out = reshape_out(C_blocks, edges)
    # print(out)
    

    print()
    print('cuBLAS output has dtype:', cp.dtype(out))
    print()


    #CHECK IF CUPY AND CUBLAS ARE COMPUTING THE SAME THING
    if np.allclose(temp2, out):
        print('Checking correctness with np.allclose (default params):' \
        '\nCuPy and cuBLAS match')
    else:
        print('Checking correctness with np.allclose (default params):' \
        '\nCuPy and cuBLAS DO NOT match')




"""~~~~~~~~~~~~~~~~~~~~~ BENCHMARKING CUPY AND CUBLAS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

#CUPY TIMES
times = (benchmark(cupy_block_mul, (zp_diff, zp_temp), n_repeat = 100))
# gpu_times = gpu_times.split()
# gpu_cpu_t = float(gpu_times[3])/1e6
# gpu_gpu_t = float(gpu_times[14])/1e6

gpu_t_s = times.gpu_times
cpu_t_s = times.cpu_times

avg_gpu_t = cp.mean(gpu_t_s)
avg_cpu_t = cp.mean(cpu_t_s)

# print(gpu_cpu_t, gpu_gpu_t)
print()
print('CuPy times:')
print('cpu:', avg_cpu_t, ' gpu:', avg_gpu_t)

#CUBLAS TIMES
times = (benchmark(corrcal_fused_matmul, (diff, temp, edges, 256,), n_repeat = 100))
# gpu_times = gpu_times.split()
# gpu_cpu_t = float(gpu_times[3])/1e6
# gpu_gpu_t = float(gpu_times[14])/1e6

gpu_t_s = times.gpu_times
cpu_t_s = times.cpu_times

avg_gpu_t = cp.mean(gpu_t_s)
avg_cpu_t = cp.mean(cpu_t_s)

# print(gpu_cpu_t, gpu_gpu_t)
print()
print('CuBLAS times:')
print('cpu:', avg_cpu_t, ' gpu:', avg_gpu_t)




