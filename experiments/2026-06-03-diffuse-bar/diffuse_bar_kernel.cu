// nvcc -Xcompiler -fPIC -shared -o diffuse_bar_kernel.so diffuse_bar_kernel.cu

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

template <int N>
__global__ void multiply_temp_by_chol(
    const float* __restrict__ noise,
    const float* __restrict__ diffuse,
    const float* __restrict__ L_inv_T,
    float* __restrict__ out,
    const int* edges,
    int nb 
){
    //first cuda thread specification
    int b = blockIdx.x;
    if (b >= nb) return;

    //starts and stops from edges
    int start = edges[b];
    int stop = edges[b+1];

    // load in L_inv_T for this block and do to shmem since shared for the
    // whole block -- be sure to sync the threads once done
    __shared__ float L[N*N];
    for (int i = threadIdx.x; i < N*N; i += blockDim.x){
        L[i] = L_inv_T[b * N * N + i];
    }
        __syncthreads();

    // Full arithmetic:
    // 3 loops:
    /*
    1. load in temp and assign each row to a variable t
    2. do the actual matrix multiplication:
        out[row] += t[k] * L_block[row * N + k]
    */

    for (int row = start + threadIdx.x; row < stop; row += blockDim.x){
        float t[N];
        #pragma unroll
        for (int k = 0; k < N; ++k){
            // t[k] = temp[row * N + k];
            t[k] = (1.0f/noise[row])*diffuse[row * N + k];
        }

        #pragma unroll
        for (int col = 0; col < N; ++col){
            float sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < N; ++j){
                sum += t[j] * L[j * N + col];
            }
            out[row * N + col] = sum;
        }
    }
}


template __global__ void multiply_temp_by_chol<1>(const float*, const float*, const float*, float*, const int*, int);
template __global__ void multiply_temp_by_chol<2>(const float*, const float*, const float*, float*, const int*, int);
template __global__ void multiply_temp_by_chol<3>(const float*, const float*, const float*, float*, const int*, int);
template __global__ void multiply_temp_by_chol<4>(const float*, const float*, const float*, float*, const int*, int);
template __global__ void multiply_temp_by_chol<5>(const float*, const float*, const float*, float*, const int*, int);
template __global__ void multiply_temp_by_chol<6>(const float*, const float*, const float*, float*, const int*, int);
template __global__ void multiply_temp_by_chol<7>(const float*, const float*, const float*, float*, const int*, int);
template __global__ void multiply_temp_by_chol<8>(const float*, const float*, const float*, float*, const int*, int);
template __global__ void multiply_temp_by_chol<9>(const float*, const float*, const float*, float*, const int*, int);
template __global__ void multiply_temp_by_chol<10>(const float*, const float*, const float*, float*, const int*, int);
template __global__ void multiply_temp_by_chol<11>(const float*, const float*, const float*, float*, const int*, int);
template __global__ void multiply_temp_by_chol<12>(const float*, const float*, const float*, float*, const int*, int);
template __global__ void multiply_temp_by_chol<13>(const float*, const float*, const float*, float*, const int*, int);
template __global__ void multiply_temp_by_chol<14>(const float*, const float*, const float*, float*, const int*, int);
template __global__ void multiply_temp_by_chol<15>(const float*, const float*, const float*, float*, const int*, int);
template __global__ void multiply_temp_by_chol<16>(const float*, const float*, const float*, float*, const int*, int);
template __global__ void multiply_temp_by_chol<17>(const float*, const float*, const float*, float*, const int*, int);
template __global__ void multiply_temp_by_chol<18>(const float*, const float*, const float*, float*, const int*, int);
template __global__ void multiply_temp_by_chol<19>(const float*, const float*, const float*, float*, const int*, int);
template __global__ void multiply_temp_by_chol<20>(const float*, const float*, const float*, float*, const int*, int);


#define LAUNCH_MUL_TEMP_BY_CHOL(N) \
    case N: \
        multiply_temp_by_chol<N><<<nb, 128>>>(noise, diffuse, L_inv_T, out, edges, nb); \
        break;


extern "C"
void launch_mul_temp_by_chol(
    const float* noise,
    const float* diffuse,
    const float* L_inv_T,
    float* out,
    const int* edges,
    int nb,
    int n_eig

){
    switch(n_eig){
        LAUNCH_MUL_TEMP_BY_CHOL(1)
        LAUNCH_MUL_TEMP_BY_CHOL(2)
        LAUNCH_MUL_TEMP_BY_CHOL(3)
        LAUNCH_MUL_TEMP_BY_CHOL(4)
        LAUNCH_MUL_TEMP_BY_CHOL(5)
        LAUNCH_MUL_TEMP_BY_CHOL(6)
        LAUNCH_MUL_TEMP_BY_CHOL(7)
        LAUNCH_MUL_TEMP_BY_CHOL(8)
        LAUNCH_MUL_TEMP_BY_CHOL(9)
        LAUNCH_MUL_TEMP_BY_CHOL(10)
        LAUNCH_MUL_TEMP_BY_CHOL(11)
        LAUNCH_MUL_TEMP_BY_CHOL(12)
        LAUNCH_MUL_TEMP_BY_CHOL(13)
        LAUNCH_MUL_TEMP_BY_CHOL(14)
        LAUNCH_MUL_TEMP_BY_CHOL(15)
        LAUNCH_MUL_TEMP_BY_CHOL(16)
        LAUNCH_MUL_TEMP_BY_CHOL(17)
        LAUNCH_MUL_TEMP_BY_CHOL(18)
        LAUNCH_MUL_TEMP_BY_CHOL(19)
        LAUNCH_MUL_TEMP_BY_CHOL(20)
        default:
            fprintf(stderr, "ERROR: n_eig=%d not supported (max 20). "
                    "Recompile with additional template instantiations.\n", n_eig);
            break;
    }
}