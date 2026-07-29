// nvcc -Xcompiler -fPIC -shared -o fused_chol_inv_kernel.so fused_chol_inv_kernel.cu

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>


template <int N, bool COMPUTE_DET>
__global__ void batched_cholesky_inv(
    const float* __restrict__ in,   // (n_mat, N, N) - input matrices
    float* __restrict__ out,         // (n_mat, N, N) - L_inv output
    float* __restrict__ out_diffuse_det, // storage of 1st half of logdet expression
    int n_mat

){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_mat) return;

    const float* A = in + idx * N * N;
    float* R = out + idx * N * N;

    float logdiag = 0.0f;

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
        // L[j * N + j] = sqrtf(A[j * N + j] + 1.0f - sum);
        float ljj = sqrtf(A[j * N + j] + 1.0f - sum);
        L[j * N + j] = ljj;
        if (COMPUTE_DET) logdiag += logf(ljj);

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

    if (COMPUTE_DET) out_diffuse_det[idx] = logdiag;

    // for (int i = 0; i < N; ++i){
    //     out_diffuse_det = 
    // }

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
    // for (int i = 0; i < N * N; ++i) R[i] = Linv[i];
    for (int row = 0; row < N; ++row){
        for (int col = 0; col < N; ++col){
            R[row * N + col] = Linv[col * N + row];
        }
    }
}


# define INST_DIFFUSE(N) \
    template __global__ void batched_cholesky_inv<N, true>(const float*, float*, float*, int); \
    template __global__ void batched_cholesky_inv<N, false>(const float*, float*, float*, int);

INST_DIFFUSE(1)  INST_DIFFUSE(2)  INST_DIFFUSE(3)  INST_DIFFUSE(4)
INST_DIFFUSE(5)  INST_DIFFUSE(6)  INST_DIFFUSE(7)  INST_DIFFUSE(8)
INST_DIFFUSE(9)  INST_DIFFUSE(10) INST_DIFFUSE(11) INST_DIFFUSE(12)
INST_DIFFUSE(13) INST_DIFFUSE(14) INST_DIFFUSE(15) INST_DIFFUSE(16)
INST_DIFFUSE(17) INST_DIFFUSE(18) INST_DIFFUSE(19) INST_DIFFUSE(20)
#undef INST_DIFFUSE

// template __global__ void batched_cholesky_inv<1>(const float*, float*, int);
// template __global__ void batched_cholesky_inv<2>(const float*, float*, int);
// template __global__ void batched_cholesky_inv<3>(const float*, float*, int);
// template __global__ void batched_cholesky_inv<4>(const float*, float*, int);
// template __global__ void batched_cholesky_inv<5>(const float*, float*, int);
// template __global__ void batched_cholesky_inv<6>(const float*, float*, int);
// template __global__ void batched_cholesky_inv<7>(const float*, float*, int);
// template __global__ void batched_cholesky_inv<8>(const float*, float*, int);
// template __global__ void batched_cholesky_inv<9>(const float*, float*, int);
// template __global__ void batched_cholesky_inv<10>(const float*, float*, int);
// template __global__ void batched_cholesky_inv<11>(const float*, float*, int);
// template __global__ void batched_cholesky_inv<12>(const float*, float*, int);
// template __global__ void batched_cholesky_inv<13>(const float*, float*, int);
// template __global__ void batched_cholesky_inv<14>(const float*, float*, int);
// template __global__ void batched_cholesky_inv<15>(const float*, float*, int);
// template __global__ void batched_cholesky_inv<16>(const float*, float*, int);
// template __global__ void batched_cholesky_inv<17>(const float*, float*, int);
// template __global__ void batched_cholesky_inv<18>(const float*, float*, int);
// template __global__ void batched_cholesky_inv<19>(const float*, float*, int);
// template __global__ void batched_cholesky_inv<20>(const float*, float*, int);


#define LAUNCH_CHOL_INV(N) \
    case N: \
        if (compute_det) \
            batched_cholesky_inv<N, true><<<(n_mat+255)/256, 256>>>(in, out, out_diffuse_det, n_mat); \
        else \
            batched_cholesky_inv<N, false><<<(n_mat+255)/256, 256>>>(in, out, nullptr, n_mat); \
        break;

extern "C"
void launch_batched_cholesky_inv(
    const float* in, float* out, float* out_diffuse_det, int n_mat, int n_eig, int compute_det
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