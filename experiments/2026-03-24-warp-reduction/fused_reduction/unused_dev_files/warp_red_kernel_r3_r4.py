"""
Working implimentation of the warp reduction kernel that computes the matrix product
in the first application of the WB identity. Benchmarks against CuPy implimentation are
included at the very bottom. Results show a roughly 10x improvement in speed over CuPy,
which was already still faster than virtually any implimentation using cuBLAS given the 
problem shape. 

r3 and r4 kernels are hardcoded for either 3 or 4 eignemodes so this may have to change 
in the future. Accompanying files in this directory were used to build up to these tests,
and this file includes potentially very useful code in terms of the GPU implimentation 
of CorrCal.
"""

import cupy as cp
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath("../cuBLAS/gemm_grouped_batched"))

from gridding import *
from simulate_params import *
from zp_puregpu_funcs_py import *

# ----------------------------
# r=3: 6 unique symmetric entries
# order: (0,0),
#        (0,1)(1,1),
#        (0,2)(1,2)(2,2)
# ----------------------------


# ----------------------------
# r=4: 10 unique symmetric entries
# order: (0,0),
#        (0,1)(1,1),
#        (0,2)(1,2)(2,2),
#        (0,3)(1,3)(2,3)(3,3)
# ----------------------------


kernel_r3 = r'''
__device__ __forceinline__ float warp_sum(float v) {
    unsigned mask = 0xffffffffu;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}

__device__ __forceinline__
void accumulate_sym6_r3(const float* __restrict__ D,
                        const float* __restrict__ w,
                        int start, int stop,
                        float* __restrict__ acc)
{
    #pragma unroll
    for (int t=0;t<6;++t) acc[t] = 0.0f;

    int tid = (int)threadIdx.x;
    int T   = (int)blockDim.x;

    for (int i = start + tid; i < stop; i += T) {
        float wi = w[i];

        float d0 = D[i*3 + 0];
        float d1 = D[i*3 + 1];
        float d2 = D[i*3 + 2];

        float wd0 = wi * d0;
        float wd1 = wi * d1;
        float wd2 = wi * d2;

        // upper triangle
        acc[0] += wd0*d0;   // 00
        acc[1] += wd0*d1;   // 01
        acc[2] += wd1*d1;   // 11
        acc[3] += wd0*d2;   // 02
        acc[4] += wd1*d2;   // 12
        acc[5] += wd2*d2;   // 22
    }
}

extern "C" __global__
void cov_reduce_r3_sym6_warp(const float* __restrict__ D,
                             const float* __restrict__ w,
                             const int* __restrict__ edges,
                             float* __restrict__ C6,
                             int nb)
{
    int b = (int)blockIdx.x;
    if (b >= nb) return;

    int start = edges[b];
    int stop  = edges[b+1];

    float acc[6];
    accumulate_sym6_r3(D, w, start, stop, acc);

    #pragma unroll
    for (int t=0;t<6;++t) acc[t] = warp_sum(acc[t]);

    int lane = (int)(threadIdx.x & 31);
    int warp = (int)(threadIdx.x >> 5);

    extern __shared__ float sh[];
    int num_warps = (int)(blockDim.x >> 5);

    if (lane == 0) {
        #pragma unroll
        for (int t=0;t<6;++t) {
            sh[t * num_warps + warp] = acc[t];
        }
    }
    __syncthreads();

    if (warp == 0) {
        float v[6];
        if (lane < num_warps) {
            #pragma unroll
            for (int t=0;t<6;++t) v[t] = sh[t * num_warps + lane];
        } else {
            #pragma unroll
            for (int t=0;t<6;++t) v[t] = 0.0f;
        }

        #pragma unroll
        for (int t=0;t<6;++t) v[t] = warp_sum(v[t]);

        if (lane == 0) {
            //float* out = C6 + (size_t)b * 6;
            float* out = C6 + (size_t)b * 9;

            //#pragma unroll
            //for (int t=0;t<6;++t) out[t] = v[t];


            // v ordering: [00,01,11,02,12,22]
            out[0] = v[0];  out[1] = v[1];  out[2] = v[3];
            out[3] = v[1];  out[4] = v[2];  out[5] = v[4];
            out[6] = v[3];  out[7] = v[4];  out[8] = v[5];
        }
    }
}
'''

kernel_r4 = r'''
__device__ __forceinline__ float warp_sum(float v) {
    unsigned mask = 0xffffffffu;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}

__device__ __forceinline__
void accumulate_sym10_r4(const float* __restrict__ D,
                         const float* __restrict__ w,
                         int start, int stop,
                         float* __restrict__ acc)
{
    #pragma unroll
    for (int t=0;t<10;++t) acc[t] = 0.0f;

    int tid = (int)threadIdx.x;
    int T   = (int)blockDim.x;

    for (int i = start + tid; i < stop; i += T) {
        float wi = w[i];

        float d0 = D[i*4 + 0];
        float d1 = D[i*4 + 1];
        float d2 = D[i*4 + 2];
        float d3 = D[i*4 + 3];

        float wd0 = wi * d0;
        float wd1 = wi * d1;
        float wd2 = wi * d2;
        float wd3 = wi * d3;

        // upper triangle
        acc[0] += wd0*d0;   // 00
        acc[1] += wd0*d1;   // 01
        acc[2] += wd1*d1;   // 11
        acc[3] += wd0*d2;   // 02
        acc[4] += wd1*d2;   // 12
        acc[5] += wd2*d2;   // 22
        acc[6] += wd0*d3;   // 03
        acc[7] += wd1*d3;   // 13
        acc[8] += wd2*d3;   // 23
        acc[9] += wd3*d3;   // 33
    }
}

extern "C" __global__
void cov_reduce_r4_sym10_warp(const float* __restrict__ D,
                              const float* __restrict__ w,
                              const int* __restrict__ edges,
                              float* __restrict__ C10,
                              int nb)
{
    int b = (int)blockIdx.x;
    if (b >= nb) return;

    int start = edges[b];
    int stop  = edges[b+1];

    float acc[10];
    accumulate_sym10_r4(D, w, start, stop, acc);

    #pragma unroll
    for (int t=0;t<10;++t) acc[t] = warp_sum(acc[t]);

    int lane = (int)(threadIdx.x & 31);
    int warp = (int)(threadIdx.x >> 5);

    extern __shared__ float sh[];
    int num_warps = (int)(blockDim.x >> 5);

    if (lane == 0) {
        #pragma unroll
        for (int t=0;t<10;++t) {
            sh[t * num_warps + warp] = acc[t];
        }
    }
    __syncthreads();

    if (warp == 0) {
        float v[10];
        if (lane < num_warps) {
            #pragma unroll
            for (int t=0;t<10;++t) v[t] = sh[t * num_warps + lane];
        } else {
            #pragma unroll
            for (int t=0;t<10;++t) v[t] = 0.0f;
        }

        #pragma unroll
        for (int t=0;t<10;++t) v[t] = warp_sum(v[t]);

        if (lane == 0) {
            float* out = C10 + (size_t)b * 10;
            #pragma unroll
            for (int t=0;t<10;++t) out[t] = v[t];
        }
    }
}
'''


k_r3 = cp.RawKernel(kernel_r3, "cov_reduce_r3_sym6_warp")
k_r4 = cp.RawKernel(kernel_r4, "cov_reduce_r4_sym10_warp")

def cov_reduce_sym_r3(D, w, edges, threads=128):
    """
    D: (N,3) float32 C-contig
    w: (N,) float32 C-contig
    edges: (nb+1,) int32
    returns: (nb,6) float32
    """
    assert D.dtype == cp.float32 and D.ndim == 2 and D.shape[1] == 3 and D.flags.c_contiguous
    assert w.dtype == cp.float32 and w.ndim == 1 and w.flags.c_contiguous and w.shape[0] == D.shape[0]
    assert edges.dtype == cp.int32 and edges.ndim == 1

    nb = edges.size - 1
    # out = cp.empty((nb, 6), dtype=cp.float32)
    out = cp.empty((nb, 3, 3), dtype=cp.float32)


    num_warps = threads // 32
    shared = 6 * num_warps * 4

    k_r3((nb,), (threads,), (D, w, edges, out, np.int32(nb)), shared_mem=shared)
    return out


def cov_reduce_sym_r4(D, w, edges, threads=128):
    """
    D: (N,4) float32 C-contig
    w: (N,) float32 C-contig
    edges: (nb+1,) int32
    returns: (nb,10) float32
    """
    assert D.dtype == cp.float32 and D.ndim == 2 and D.shape[1] == 4 and D.flags.c_contiguous
    assert w.dtype == cp.float32 and w.ndim == 1 and w.flags.c_contiguous and w.shape[0] == D.shape[0]
    assert edges.dtype == cp.int32 and edges.ndim == 1

    nb = edges.size - 1
    out = cp.empty((nb, 10), dtype=cp.float32)

    num_warps = threads // 32
    shared = 10 * num_warps * 4

    k_r4((nb,), (threads,), (D, w, edges, out, np.int32(nb)), shared_mem=shared)
    return out


from cupyx.profiler import benchmark

def ref_sym(D, w, edges):
    nb = edges.size - 1
    r = D.shape[1]
    out = []
    for b in range(nb):
        s = int(edges[b].item()); e = int(edges[b+1].item())
        Db = D[s:e]
        C = Db.T @ (Db * w[s:e, None])
        # pack upper triangle
        vals = []
        for j in range(r):
            for i in range(j+1):
                vals.append(C[i,j])
        out.append(cp.stack(vals))
    return cp.stack(out)


def unpack_symmetric(triu_vals):
    nb, n = triu_vals.shape

    r = int((cp.sqrt(8*n + 1) - 1) / 2)

    C = cp.zeros((nb, r, r), dtype=triu_vals.dtype)
    
    idx = 0
    for j in range(r):
        for i in range(j+1):
            C[:, i, j] = triu_vals[:, idx]
            C[:, j, i] = triu_vals[:, idx]
            idx += 1

    return C

def unpack_sym6_kernel(C6):
    nb = C6.shape[0]
    C = cp.zeros((nb, 3, 3), dtype=C6.dtype)

    C[:,0,0] = C6[:,0]

    C[:,0,1] = C[:,1,0] = C6[:,1]
    C[:,1,1] = C6[:,2]

    C[:,0,2] = C[:,2,0] = C6[:,3]
    C[:,1,2] = C[:,2,1] = C6[:,4]
    C[:,2,2] = C6[:,5]

    return C

# Example
# edges = cp.asarray(grid_redundancy_edges(3, 2)[0], dtype=cp.int32)
# N = int(edges[-1].get())
cp.random.seed(0)

# r=3
# D3 = cp.random.standard_normal((N,3), dtype=cp.float32)
# w  = cp.random.random((N,), dtype=cp.float32)

# D3 = cp.random.rand(N,3, dtype=cp.float32)
# w  = cp.random.rand(N, dtype=cp.float32)

#----------------------------------------------------
#Simulating the parameters
n_eig = 3
rows = 48
cols = 24
n_ant = rows*cols
print(f"Number of antennas: {n_ant}")

spms = SimCorrcalParams(n_ant, n_eig, n_src=1, precision='float32', xp = cp)
edges = spms.edges(rows, cols, use_random=False)
edges = cp.asarray(edges)

sim_data = spms.sim_data()
w = sim_data[0]
D3 = sim_data[1]

zp_w, nb, lb = zeroPad(w, edges, return_inv=True, dtype=cp.float32)
zp_D3, nb, lb = zeroPad(D3, edges, return_inv=False, dtype=cp.float32)

w = 1/w


"""~~~~~~~~~~~~~~~~~~~~~ COMPUTATION OF diff.T @ N^-1 @ diff ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

"""
diff.T @ temp

temp == N^-1 @ diff
"""

zp_temp = zp_w[..., None] * zp_D3

#cupy computation of diff.T @ N^-1 @ diff
def cupy_block_mul(diff, tmp):
    res = cp.transpose(diff, [0, 2, 1]) @ tmp
    cp.cuda.Stream.null.synchronize()
    return res

# return temp2 = diff.T @ N^-1 @ diff
temp2 = cupy_block_mul(zp_D3, zp_temp)

#use warp reduction kernel to compute matrix product
C3 = cov_reduce_sym_r3(D3, w, edges)
# full_block_C3 = unpack_sym6_kernel(C3)

print_results_one_block = False
if print_results_one_block:
    print(temp2[0])
    print()
    # print(full_block_C3[0])
    print(C3[0])

check_match = True
if check_match:
    print("------------------------------------------------------------")
    #CHECK IF CUPY AND CUBLAS ARE COMPUTING THE SAME THING
    # if np.allclose(temp2, full_block_C3):
    if np.allclose(temp2, C3):
        print('Checking correctness with np.allclose (default params):' \
        '\nCuPy and Warp Reduction Implimentation MATCH')
    else:
        print('Checking correctness with np.allclose (default params):' \
        '\nCuPy and Warp Reduction Implimentation DO NOT MATCH')
    print("------------------------------------------------------------")


"""~~~~~~~~~~~~~~~~~~~~~ BENCHMARKING CUPY AND reduction kernel ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
time = True
if time:
    #CUPY TIMES
    times = (benchmark(cupy_block_mul, (zp_D3, zp_temp), n_repeat = 100))
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

    #WARP REDUCTION TIMES
    times = (benchmark(cov_reduce_sym_r3, (D3, w, edges), n_repeat = 100))
    # gpu_times = gpu_times.split()
    # gpu_cpu_t = float(gpu_times[3])/1e6
    # gpu_gpu_t = float(gpu_times[14])/1e6

    gpu_t_s = times.gpu_times
    cpu_t_s = times.cpu_times

    avg_gpu_t = cp.mean(gpu_t_s)
    avg_cpu_t = cp.mean(cpu_t_s)

    # print(gpu_cpu_t, gpu_gpu_t)
    print()
    print('Warp Reduction times:')
    print('cpu:', avg_cpu_t, ' gpu:', avg_gpu_t)

    unpack = False
    if unpack:
        #REASSEMBLE FROM LOWER TRIANGLUAR TIMES
        times = (benchmark(unpack_sym6_kernel, (C3,), n_repeat = 100))
        # gpu_times = gpu_times.split()
        # gpu_cpu_t = float(gpu_times[3])/1e6
        # gpu_gpu_t = float(gpu_times[14])/1e6

        gpu_t_s = times.gpu_times
        cpu_t_s = times.cpu_times

        avg_gpu_t = cp.mean(gpu_t_s)
        avg_cpu_t = cp.mean(cpu_t_s)

        # print(gpu_cpu_t, gpu_gpu_t)
        print()
        print('Times to reassemble block form:')
        print('cpu:', avg_cpu_t, ' gpu:', avg_gpu_t)

    print()