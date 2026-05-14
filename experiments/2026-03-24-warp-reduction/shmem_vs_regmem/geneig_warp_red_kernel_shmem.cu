// nvcc -Xcompiler -fPIC -shared -o geneig_warp_red_kernel_shmem.so geneig_warp_red_kernel_shmem.cu

/*
Shared-memory variant of the general eigenmode warp reduction kernel.

All per-thread arrays (d, acc, warp0_red) are placed in shared memory
instead of registers, so n_eig can be a runtime parameter.

This exists purely for benchmarking against the templated register version
to quantify the performance cost of shared memory vs registers.

Shared memory layout (per block):
    d:          blockDim.x * n_eig   floats   (per-thread diffuse vector)
    acc:        blockDim.x * n_sym   floats   (per-thread outer product accumulator)
    warp_red:   n_sym * num_warps    floats   (warp reduction workspace, same as original)

Total shared mem = blockDim.x * (n_eig + n_sym) + n_sym * num_warps   floats
*/

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>


__device__ __forceinline__ float warp_sum(float val) {
    unsigned mask = 0xffffffffu;
    val += __shfl_down_sync(mask, val, 16);
    val += __shfl_down_sync(mask, val, 8);
    val += __shfl_down_sync(mask, val, 4);
    val += __shfl_down_sync(mask, val, 2);
    val += __shfl_down_sync(mask, val, 1);
    return val;
}


extern "C" __global__
void two_level_warp_reduction_shmem(
    const float* __restrict__ diffuse,
    const float* __restrict__ noise,
    const int* __restrict__ edges,
    float* __restrict__ out,
    int nb, int n_eig, int n_sym
){
    int b = (int)blockIdx.x;
    if (b >= nb) return;

    int tid = (int)threadIdx.x;
    int bdim = (int)blockDim.x;
    int num_warps = bdim >> 5;
    int lane = tid & 31;
    int warp = tid >> 5;

    // Partition shared memory:
    //   d_sh:    bdim * n_eig    floats
    //   acc_sh:  bdim * n_sym    floats
    //   warp_sh: n_sym * num_warps  floats
    extern __shared__ float shmem[];
    float* d_sh    = shmem;                                   // [bdim * n_eig]
    float* acc_sh  = d_sh   + bdim * n_eig;                   // [bdim * n_sym]
    float* warp_sh = acc_sh + bdim * n_sym;                   // [n_sym * num_warps]

    // Per-thread pointers into shared memory
    float* d   = d_sh   + tid * n_eig;
    float* acc = acc_sh + tid * n_sym;

    int start = edges[b];
    int stop  = edges[b + 1];

    // ---- accumulate outer product ----
    for (int t = 0; t < n_sym; ++t) acc[t] = 0.0f;

    for (int i = start + tid; i < stop; i += bdim) {
        float ni_inv = 1.0f / noise[i];

        for (int j = 0; j < n_eig; ++j) {
            d[j] = diffuse[i * n_eig + j];
        }

        int idx = 0;
        for (int j = 0; j < n_eig; ++j) {
            float d_ni_inv = ni_inv * d[j];
            for (int row = 0; row <= j; ++row) {
                acc[idx] += d[row] * d_ni_inv;
                idx++;
            }
        }
    }

    // ---- warp-level reduction ----
    for (int i = 0; i < n_sym; ++i) {
        acc[i] = warp_sum(acc[i]);
    }

    if (lane == 0) {
        for (int i = 0; i < n_sym; ++i) {
            warp_sh[i * num_warps + warp] = acc[i];
        }
    }
    __syncthreads();

    // ---- cross-warp reduction in warp 0 ----
    if (warp == 0) {
        for (int i = 0; i < n_sym; ++i) {
            float val = (lane < num_warps) ? warp_sh[i * num_warps + lane] : 0.0f;
            val = warp_sum(val);

            if (lane == 0) {
                // Temporarily stash in warp_sh since we need all n_sym values
                // before writing the symmetric output
                warp_sh[i] = val;
            }
        }

        if (lane == 0) {
            float* res = out + (size_t)b * n_eig * n_eig;

            int idx = 0;
            for (int col = 0; col < n_eig; ++col) {
                for (int row = 0; row <= col; ++row) {
                    res[row * n_eig + col] = warp_sh[idx];
                    res[col * n_eig + row] = warp_sh[idx];
                    idx++;
                }
            }
        }
    }
}


// ============================================================
// Host-side launcher (callable from ctypes)
// ============================================================

extern "C"
void launch_two_level_warp_reduction_shmem(
    const float* diffuse,
    const float* noise,
    const int* edges,
    float* out,
    int nb,
    int n_eig,
    int threads_per_block
){
    int n_sym = n_eig * (n_eig + 1) / 2;
    int num_warps = threads_per_block / 32;

    // d + acc per thread, plus warp reduction workspace
    int shared_mem = (threads_per_block * (n_eig + n_sym)
                      + n_sym * num_warps) * sizeof(float);

    dim3 grid(nb);
    dim3 block(threads_per_block);

    two_level_warp_reduction_shmem<<<grid, block, shared_mem>>>(
        diffuse, noise, edges, out, nb, n_eig, n_sym
    );
}


extern "C"
void sync_device() {
    cudaDeviceSynchronize();
}
