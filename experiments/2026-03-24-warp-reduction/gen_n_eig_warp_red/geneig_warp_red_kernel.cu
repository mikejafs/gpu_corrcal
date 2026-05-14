// nvcc -Xcompiler -fPIC -shared -o geneig_warp_red_kernel.so geneig_warp_red_kernel.cu -lcublas


/*
General number of eigenmodes warp reduction kernel.

Some comments:
    The python-based string literal verion of this before used many #pragma_unrolls.
    This may be unnecessary -- especially if the compile flags account for loop optimization
    using the e.g., -o flags... This is something worth profiling for. I'll leave them out 
    EVERYWHERE for now, though they may be worth adding back in for the future.
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

__device__ __forceinline__ void accumulate_outer_prod(
    const float* __restrict__ diffuse,
    const float* __restrict__ noise,
    int start, 
    int stop,
    float* __restrict__ acc,
    int n_eig,
    int n_sym
){
    for (int t = 0; t < n_sym; ++t) acc[t] = 0.0f;

    int tid = (int)threadIdx.x;
    int bdim = (int)blockDim.x;

    for (int i = start + tid; i < stop; i += bdim) {
        float ni_inv = 1.0f / noise[i];

        // Load the n_eig-element vector for row i.
        // Using a local array lets the compiler keep these in registers.
        float d[n_eig];
        for (int j = 0; j < n_eig; ++j) {
            d[j] = diffuse[i * n_eig + j];   //this only works beceuse each thread sees a different d array
        }

        // Accumulate upper triangle of the outer product.
        // idx walks through the packed symmetric storage in the same
        // order as the original kernel: column-major upper triangle.
        int idx = 0;
        for (int j = 0; j < n_eig; ++j) {
            float d_ni_inv = ni_inv * d[j];
            for (int row = 0; row <= j; ++row) {
                acc[idx] += d[row] * d_ni_inv;
                idx++;
            }
        }
    }
}


extern "C" __global__
void two_level_warp_reduction(
    const float* __restrict__ diffuse,
    const float* __restrict__ noise,
    const int* __restrict__ edges,
    float* __restrict__ out,
    int nb, int n_eig, int n_sym
){
    int b = (int)blockIdx.x;
    if (b >= nb) return;

    int start = edges[b];
    int stop = edges[b + 1];

    float acc[n_sym];
    accumulate_outer_prod(diffuse, noise, start, stop, acc, n_eig, n_sym);

    for (int i = 0; i < n_sym; ++i){
        acc[i] = warp_sum(acc[i]);
    }

    int lane = (int)threadIdx.x & 31;
    int warp = (int)threadIdx.x >> 5;  //right shift -- equiv to division by 2^n = 32 in this case

    extern __shared__ float sh[];
    int num_warps = (int)blockDim.x >> 5;

    if (lane == 0){
        for (int i = 0; i < n_sym; ++i){
            sh[i*num_warps + warp] = acc[i];
        }
    }
    __syncthreads();

    //Take all the shared mem writes and re-write them to the 0th warp
    if (warp == 0){
        float warp0_red[n_sym];
        if (lane < num_warps){
            for (int i = 0; i < n_sym; ++i){
                warp0_red[i] = sh[i*num_warps + lane];
            } 
        } else {
            for (int i = 0; i < n_sym; ++i){
                warp0_red[i] = 0.0f;
            }
        }

        for (int j = 0; j < n_sym; ++j){
            warp_sum(warp0_red[j]);
        }

        if (lane == 0){
            float* res = out + (size_t)b * n_eig * n_eig;

            int idx = 0;
            for (int col = 0; col < n_eig; ++col){
                for (int row = 0; row <= col; ++row){
                    res[row * n_eig + col] = warp0_red[idx];
                    res[col * n_eig + row] = warp0_red[idx];
                    idx++;
                }
            }
        }
    }
}


// ================================================
// Host-side launcher
// ================================================

extern "C"
void launch_two_level_warp_reduction(
    const float* diffuse,
    const float* noise,
    const int* edges,
    float* out,
    int nb,
    int n_eig,
    int threads_per_block
){
    int n_sym = n_eig * (n_eig + 1) / 2;
    int ThreadsPerBlock = 128;

    dim3 grid(nb);
    dim3 block(threads_per_block);

    two_level_warp_reduction<<<grid, block>>>(
        diffuse, noise, edges, out, nb, n_eig, n_sym
    );
}

extern "C"
void sync_device(){
    cudaDeviceSynchronize();
}

