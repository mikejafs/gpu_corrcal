// nvcc -Xcompiler -fPIC -shared -o fused_chol_inv_kernel.so fused_chol_inv_kernel.cu

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>


template <int N>
__global__ void batched_cholesky_inv(
    const float* __restrict__ in,   // (n_mat, N, N) - input matrices
    float* __restrict__ out,         // (n_mat, N, N) - L_inv output
    int n_mat
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_mat) return;

    const float* A = in + idx * N * N;
    float* R = out + idx * N * N;

    // Local storage for L
    float L[N * N];

    // Zero out L
    #pragma unroll
    for (int i = 0; i < N * N; ++i) L[i] = 0.0f;

    // ---- Step 1: Cholesky decomposition ----
    // A = L @ L^T, store L in lower triangle
    #pragma unroll
    for (int j = 0; j < N; ++j) {
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < j; ++k)
            sum += L[j * N + k] * L[j * N + k];
        L[j * N + j] = sqrtf(A[j * N + j] + 1.0f - sum);

        float inv_ljj = 1.0f / L[j * N + j];
        #pragma unroll
        for (int i = j + 1; i < N; ++i) {
            float sum2 = 0.0f;
            #pragma unroll
            for (int k = 0; k < j; ++k)
                sum2 += L[i * N + k] * L[j * N + k];
            L[i * N + j] = (A[i * N + j] - sum2) * inv_ljj;
        }
    }

    // ---- Step 2: Triangular inverse via forward substitution ----
    // Solve L @ L_inv = I column by column
    float Linv[N * N];

    #pragma unroll
    for (int i = 0; i < N * N; ++i) Linv[i] = 0.0f;

    #pragma unroll
    for (int col = 0; col < N; ++col) {
        // Diagonal: L_inv[col][col] = 1 / L[col][col]
        Linv[col * N + col] = 1.0f / L[col * N + col];

        // Below diagonal: forward substitution
        #pragma unroll
        for (int row = col + 1; row < N; ++row) {
            float sum = 0.0f;
            #pragma unroll
            for (int k = col; k < row; ++k)
                sum += L[row * N + k] * Linv[k * N + col];
            Linv[row * N + col] = -sum / L[row * N + row];
        }
    }

    // ---- Write L_inv to global memory ----
    #pragma unroll
    for (int i = 0; i < N * N; ++i) R[i] = Linv[i];
}


template __global__ void batched_cholesky_inv<1>(const float*, float*, int);
template __global__ void batched_cholesky_inv<2>(const float*, float*, int);
template __global__ void batched_cholesky_inv<3>(const float*, float*, int);
template __global__ void batched_cholesky_inv<4>(const float*, float*, int);
template __global__ void batched_cholesky_inv<5>(const float*, float*, int);
template __global__ void batched_cholesky_inv<6>(const float*, float*, int);
template __global__ void batched_cholesky_inv<7>(const float*, float*, int);
template __global__ void batched_cholesky_inv<8>(const float*, float*, int);
template __global__ void batched_cholesky_inv<9>(const float*, float*, int);
template __global__ void batched_cholesky_inv<10>(const float*, float*, int);
template __global__ void batched_cholesky_inv<11>(const float*, float*, int);
template __global__ void batched_cholesky_inv<12>(const float*, float*, int);
template __global__ void batched_cholesky_inv<13>(const float*, float*, int);
template __global__ void batched_cholesky_inv<14>(const float*, float*, int);
template __global__ void batched_cholesky_inv<15>(const float*, float*, int);
template __global__ void batched_cholesky_inv<16>(const float*, float*, int);
template __global__ void batched_cholesky_inv<17>(const float*, float*, int);
template __global__ void batched_cholesky_inv<18>(const float*, float*, int);
template __global__ void batched_cholesky_inv<19>(const float*, float*, int);
template __global__ void batched_cholesky_inv<20>(const float*, float*, int);


#define LAUNCH_CHOL_INV(N) \
    case N: \
        batched_cholesky_inv<N><<<(n_mat+255)/256, 256>>>(in, out, n_mat); \
        break;

extern "C"
void launch_batched_cholesky_inv(
    const float* in, float* out, int n_mat, int n_eig
){
    switch(n_eig) {
        LAUNCH_CHOL_INV(1)
        LAUNCH_CHOL_INV(2)
        LAUNCH_CHOL_INV(3)
        LAUNCH_CHOL_INV(4)
        LAUNCH_CHOL_INV(5)
        LAUNCH_CHOL_INV(6)
        LAUNCH_CHOL_INV(7)
        LAUNCH_CHOL_INV(8)
        LAUNCH_CHOL_INV(9)
        LAUNCH_CHOL_INV(10)
        LAUNCH_CHOL_INV(11)
        LAUNCH_CHOL_INV(12)
        LAUNCH_CHOL_INV(13)
        LAUNCH_CHOL_INV(14)
        LAUNCH_CHOL_INV(15)
        LAUNCH_CHOL_INV(16)
        LAUNCH_CHOL_INV(17)
        LAUNCH_CHOL_INV(18)
        LAUNCH_CHOL_INV(19)
        LAUNCH_CHOL_INV(20)
        default:
            fprintf(stderr, "ERROR: n_eig=%d not supported (max 20). "
                    "Recompile with additional template instantiations.\n", n_eig);
            break;
        // ... up to 20
    }
}