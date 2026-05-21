"""
Solely the kernel + tests for the warp reduction kernel for the case that N_eig = 3.

Providing only the implimentatin for the warp reduction kernel to be able to better
understand what is going on with this implementation.
"""


import cupy as cp
import numpy as np
import sys
import os

# current_path = os.path.dirname(os.path.abspath(__file__))
# base_dir = os.path.abspath(os.path.join(current_path, "..", "gemm_grouped_batched"))
# sys.path.insert(0, base_dir)

from gpu_corrcal.utils.gridding import *
from gpu_corrcal.utils.simulate_params import *
from cupyx.profiler import benchmark


# ----------------------------
# r=3: 6 unique symmetric entries
# order: (0,0),
#        (0,1)(1,1),
#        (0,2)(1,2)(2,2)
# ----------------------------



kernel_r3 = r'''

/*----------------------------------------------------------------------
Note that warp_sum and accumulate_sym6_r3 are helper functions
designed to execute inline in the location they are written in the
following main function. 

__device__ indicates they are only meant to be ran from the GPU code
__forceinline__ indicates they are meant to act like helper functions
                and the compiler is to paste the function body directly
                into the call site
----------------------------------------------------------------------*/

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

    // This loop means the threads are distributed for each i so long as i_total < T
    // if i_total > T, then another iteration of the loop unfolds and each i member
    // is sent forward by a factor equal to T

    for (int i = start + tid; i < stop; i += T) {
        float wi = 1/w[i];

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

/*----------The beginning of the "main" kernel----------*/
// Two helper functions are called within...

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
            sh[t * num_warps + warp] = acc[t];  //write stuff from accumulate function to shared mem
        }
    }
    __syncthreads();  //make sure all warps finish writing before trying to do anything else

    if (warp == 0) {
        float v[6];
        if (lane < num_warps) {     //This is now a new warp for the warp in sh mem, 
                                    //since we only have warps 0, 1, 2, 3 (128 threads)
                                    //this is a clever way of noting the answer will live
                                    //in either index 0, 1, 2, or 3 in the sh mem warp ~~ I think...

            #pragma unroll
            for (int t=0;t<6;++t) v[t] = sh[t * num_warps + lane];
        } else {
            #pragma unroll
            for (int t=0;t<6;++t) v[t] = 0.0f;      //This is only here so that all lanes in the warp can participate
                                                    //in the warm sum primitive to follow
        }

        #pragma unroll
        for (int t=0;t<6;++t) v[t] = warp_sum(v[t]);

        // This last bit is just creating the 3x3 matrix from the upper triangular data
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

k_r3 = cp.RawKernel(kernel_r3, "cov_reduce_r3_sym6_warp")

def cov_reduce_sym_r3(w, D, edges, threads=128):
    """
    D: (N,3) float32 C-contig
    w: (N,) float32 C-contig
    edges: (nb+1,) int32
    returns: (nb,6) float32
    """
    assert D.dtype == cp.float32 and D.ndim == 2 and D.shape[1] == 3 and D.flags.c_contiguous
    assert w.dtype == cp.float32 and w.ndim == 1 and w.flags.c_contiguous and w.shape[0] == D.shape[0]
    assert edges.dtype == cp.int32 and edges.ndim == 1

    nb = edges.size - 1 # Note this is number of blocks not number of baselines
    # out = cp.empty((nb, 6), dtype=cp.float32)
    out = cp.empty((nb, 3, 3), dtype=cp.float32)


    num_warps = threads // 32
    shared = 6 * num_warps * 4

    k_r3((nb,), (threads,), (D, w, edges, out, np.int32(nb)), shared_mem=shared)
    return out


if __name__ == "__main__":
    """----------------------------------Simulation Land-----------------------------------------"""
    #random seed
    cp.random.seed(0)

    #Simulating the parameters
    n_eig = 3
    rows = 20
    cols = 31
    n_ant = rows*cols
    print(f"Number of antennas: {n_ant}")

    spms = SimCorrcalParams(n_ant, n_eig, n_src=1, precision='float32', xp = cp)
    edges = spms.edges(rows, cols, use_random=False)
    edges = cp.asarray(edges)

    sim_data = spms.sim_data()
    w = sim_data[0]
    D3 = sim_data[1]

    # w = 1/w


    """
    ~~~~~~~~~~~~~~~~~~~~~ COMPUTATION OF diff.T @ N^-1 @ diff ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Note the computation we are doing is:

    temp == N^-1 @ diff
    diff.T @ temp

    """

    #use warp reduction kernel to compute matrix product
    C3 = cov_reduce_sym_r3(w, D3, edges) #note that result has desired 3D block shape 

    print_results_one_block = False
    if print_results_one_block:
        print(C3[0])


    """~~~~~~~~~~~~~~~~~~~~~ Reduction Kernel benchmark tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

    time = False
    if time:
        #WARP REDUCTION TIMES
        times = (benchmark(cov_reduce_sym_r3, (w, D3, edges), n_repeat = 1000))
        # gpu_times = gpu_times.split()
        # gpu_cpu_t = float(gpu_times[3])/1e6
        # gpu_gpu_t = float(gpu_times[14])/1e6

        gpu_t_s = times.gpu_times
        cpu_t_s = times.cpu_times

        avg_gpu_t = cp.mean(gpu_t_s) * 1e6
        avg_cpu_t = cp.mean(cpu_t_s) * 1e6

        # print(gpu_cpu_t, gpu_gpu_t)
        print()
        print('Warp Reduction times:')
        print('cpu:', avg_cpu_t, ' gpu:', avg_gpu_t)
        print()
