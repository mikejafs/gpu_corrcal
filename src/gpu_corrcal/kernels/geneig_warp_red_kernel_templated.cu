// nvcc -Xcompiler -fPIC -shared -o geneig_warp_red_kernel_templated.so geneig_warp_red_kernel_templated.cu

/*
General number of eigenmodes warp reduction kernel.
Templated on N_EIG so the compiler can allocate registers optimally.
Host-side dispatch via extern "C" functions callable from ctypes.
*/


/*
TODO: 

- Once done fixing python side tests, profile with #pragma unroll added back in


*/

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>


// ============================================================
// Device helpers
// ============================================================

__device__ __forceinline__ float warp_sum(float val) {
    unsigned mask = 0xffffffffu;
    val += __shfl_down_sync(mask, val, 16);
    val += __shfl_down_sync(mask, val, 8);
    val += __shfl_down_sync(mask, val, 4);
    val += __shfl_down_sync(mask, val, 2);
    val += __shfl_down_sync(mask, val, 1);
    return val;
}


template <int N_EIG>
__device__ __forceinline__ void accumulate_outer_prod(
    const float* __restrict__ diffuse,
    const float* __restrict__ noise,
    int start, 
    int stop,
    float* __restrict__ acc
){
    constexpr int N_SYM = N_EIG * (N_EIG + 1) / 2;

    #pragma unroll
    for (int t = 0; t < N_SYM; ++t) acc[t] = 0.0f;

    int tid = (int)threadIdx.x;
    int bdim = (int)blockDim.x;
    
    for (int i = start + tid; i < stop; i += bdim) {
        float ni_inv = noise[i];
        // float noise[i] = 1.0f / noise[i];

        float d[N_EIG];
        #pragma unroll
        for (int j = 0; j < N_EIG; ++j) {
            d[j] = diffuse[i * N_EIG + j];
        }

        int idx = 0;
        #pragma unroll
        for (int j = 0; j < N_EIG; ++j) {
            float d_ni_inv = ni_inv * d[j];
            // float d_ni_inv = noise[i] * d[j];
            #pragma unroll
            for (int row = 0; row <= j; ++row) {
                acc[idx] += d[row] * d_ni_inv;
                // acc[idx] += d[row] * noise[i];
                idx++;
            }
        }
    }
}


// ============================================================
// Templated kernel
// ============================================================

template <int N_EIG>
__global__
void two_level_warp_reduction(
    const float* __restrict__ diffuse,
    const float* __restrict__ noise,
    const int* __restrict__ edges,
    float* __restrict__ out,
    int nb
){
    constexpr int N_SYM = N_EIG * (N_EIG + 1) / 2;

    int b = (int)blockIdx.x;
    if (b >= nb) return;

    int start = edges[b];
    int stop = edges[b + 1];

    /*
    int b = (int)blockIdx.x

    int start = edges[b]
    int stop = edges[b + 1]
    */

    float acc[N_SYM];
    accumulate_outer_prod<N_EIG>(diffuse, noise, start, stop, acc);

    #pragma unroll
    for (int i = 0; i < N_SYM; ++i){
        acc[i] = warp_sum(acc[i]);
    }

    int lane = (int)threadIdx.x & 31;
    int warp = (int)threadIdx.x >> 5;

    extern __shared__ float sh[];
    int num_warps = (int)blockDim.x >> 5;

    if (lane == 0){
        #pragma unroll
        for (int i = 0; i < N_SYM; ++i){
            sh[i * num_warps + warp] = acc[i];
        }
    }
    __syncthreads();

    if (warp == 0){
        float warp0_red[N_SYM];
        if (lane < num_warps){
            #pragma unroll
            for (int i = 0; i < N_SYM; ++i){
                warp0_red[i] = sh[i * num_warps + lane];
            } 
        } else {
            #pragma unroll
            for (int i = 0; i < N_SYM; ++i){
                warp0_red[i] = 0.0f;
            }
        }

        #pragma unroll
        for (int j = 0; j < N_SYM; ++j){
            warp0_red[j] = warp_sum(warp0_red[j]);
        }

        if (lane == 0){
            float* res = out + (size_t)b * N_EIG * N_EIG;

            int idx = 0;
            #pragma unroll
            for (int col = 0; col < N_EIG; ++col){
                #pragma unroll
                for (int row = 0; row <= col; ++row){
                    res[row * N_EIG + col] = warp0_red[idx];
                    res[col * N_EIG + row] = warp0_red[idx];
                    idx++;
                }
            }
        }
    }
}


// ============================================================
// Explicit instantiations
// ============================================================

template __global__ void two_level_warp_reduction<1>(const float*, const float*, const int*, float*, int);
template __global__ void two_level_warp_reduction<2>(const float*, const float*, const int*, float*, int);
template __global__ void two_level_warp_reduction<3>(const float*, const float*, const int*, float*, int);
template __global__ void two_level_warp_reduction<4>(const float*, const float*, const int*, float*, int);
template __global__ void two_level_warp_reduction<5>(const float*, const float*, const int*, float*, int);
template __global__ void two_level_warp_reduction<6>(const float*, const float*, const int*, float*, int);
template __global__ void two_level_warp_reduction<7>(const float*, const float*, const int*, float*, int);
template __global__ void two_level_warp_reduction<8>(const float*, const float*, const int*, float*, int);
template __global__ void two_level_warp_reduction<9>(const float*, const float*, const int*, float*, int);
template __global__ void two_level_warp_reduction<10>(const float*, const float*, const int*, float*, int);
template __global__ void two_level_warp_reduction<11>(const float*, const float*, const int*, float*, int);
template __global__ void two_level_warp_reduction<12>(const float*, const float*, const int*, float*, int);
template __global__ void two_level_warp_reduction<13>(const float*, const float*, const int*, float*, int);
template __global__ void two_level_warp_reduction<14>(const float*, const float*, const int*, float*, int);
template __global__ void two_level_warp_reduction<15>(const float*, const float*, const int*, float*, int);
template __global__ void two_level_warp_reduction<16>(const float*, const float*, const int*, float*, int);
template __global__ void two_level_warp_reduction<17>(const float*, const float*, const int*, float*, int);
template __global__ void two_level_warp_reduction<18>(const float*, const float*, const int*, float*, int);
template __global__ void two_level_warp_reduction<19>(const float*, const float*, const int*, float*, int);
template __global__ void two_level_warp_reduction<20>(const float*, const float*, const int*, float*, int);


// ============================================================
// Host-side dispatch (callable from ctypes)
// ============================================================

// Helper macro to reduce boilerplate in the switch
#define LAUNCH_CASE(N) \
    case N: \
        two_level_warp_reduction<N><<<grid, block, shared_mem>>>(diffuse, noise, edges, out, nb); \
        break;

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
    int num_warps = threads_per_block / 32;
    int shared_mem = n_sym * num_warps * sizeof(float);

    dim3 grid(nb);
    dim3 block(threads_per_block);

    switch (n_eig) {
        LAUNCH_CASE(1)
        LAUNCH_CASE(2)
        LAUNCH_CASE(3)
        LAUNCH_CASE(4)
        LAUNCH_CASE(5)
        LAUNCH_CASE(6)
        LAUNCH_CASE(7)
        LAUNCH_CASE(8)
        LAUNCH_CASE(9)
        LAUNCH_CASE(10)
        LAUNCH_CASE(11)
        LAUNCH_CASE(12)
        LAUNCH_CASE(13)
        LAUNCH_CASE(14)
        LAUNCH_CASE(15)
        LAUNCH_CASE(16)
        LAUNCH_CASE(17)
        LAUNCH_CASE(18)
        LAUNCH_CASE(19)
        LAUNCH_CASE(20)
        default:
            fprintf(stderr, "ERROR: n_eig=%d not supported (max 20). "
                    "Recompile with additional template instantiations.\n", n_eig);
            break;
    }
}


extern "C"
void sync_device() {
    cudaDeviceSynchronize();
}
