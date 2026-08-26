// nvcc -Xcompiler -fPIC -shared -o sparse_cov_times_vec.so sparse_cov_times_vec.cu
//
// out = N*vec + s*( Del @ (Del^T vec) + Sig @ (Sig^T vec) ),  s = -1 if IS_INV else +1
//
// Layout (flat / ragged, matching diffuse_bar_kernel.cu):
//   noise    : (n_row,)            row-diagonal N, caller passes N^-1 when IS_INV
//   diffuse  : (n_row, N_EIG)      row-major
//   source   : (n_row, N_SRC)      row-major
//   vec      : (n_row,)
//   out      : (n_row,)
//   edges    : (nb+1,)             block b spans rows [edges[b], edges[b+1])
//
// Two passes are required because Sig^T vec is a reduction over ALL rows,
// i.e. a global barrier. Del^T vec is block-local but is staged to global
// anyway so pass 2 can be a single fused sweep.

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define TPB 128  // must be a multiple of 32

// ---------------------------------------------------------------------------
// Pass 1: del_tmp[b] = Del_b^T vec_b   (per block, nb x N_EIG)
//         sig_tmp    = Sig^T vec       (global,   N_SRC)
// ---------------------------------------------------------------------------
template <int N_SRC, int N_EIG>
__global__ void sparse_cov_reduce(
    const float* __restrict__ diffuse,
    const float* __restrict__ source,
    const float* __restrict__ vec,
    float* __restrict__ del_tmp,
    float* __restrict__ sig_tmp,   // MUST be zeroed before launch
    const int* __restrict__ edges,
    int nb
){
    const int b = blockIdx.x;
    if (b >= nb) return;

    const int start = edges[b];
    const int stop  = edges[b + 1];

    // per-thread partials
    float d_acc[N_EIG];
    float s_acc[N_SRC];
    #pragma unroll
    for (int k = 0; k < N_EIG; ++k) d_acc[k] = 0.0f;
    #pragma unroll
    for (int i = 0; i < N_SRC; ++i) s_acc[i] = 0.0f;

    for (int row = start + threadIdx.x; row < stop; row += blockDim.x){
        const float v = vec[row];
        #pragma unroll
        for (int k = 0; k < N_EIG; ++k)
            d_acc[k] = fmaf(diffuse[row * N_EIG + k], v, d_acc[k]);
        #pragma unroll
        for (int i = 0; i < N_SRC; ++i)
            s_acc[i] = fmaf(source[row * N_SRC + i], v, s_acc[i]);
    }

    // intra-warp reduction
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1){
        #pragma unroll
        for (int k = 0; k < N_EIG; ++k)
            d_acc[k] += __shfl_down_sync(0xffffffff, d_acc[k], off);
        #pragma unroll
        for (int i = 0; i < N_SRC; ++i)
            s_acc[i] += __shfl_down_sync(0xffffffff, s_acc[i], off);
    }

    // cross-warp reduction through shared memory
    __shared__ float d_red[N_EIG];
    __shared__ float s_red[N_SRC];
    for (int k = threadIdx.x; k < N_EIG; k += blockDim.x) d_red[k] = 0.0f;
    for (int i = threadIdx.x; i < N_SRC; i += blockDim.x) s_red[i] = 0.0f;
    __syncthreads();

    if ((threadIdx.x & 31) == 0){
        #pragma unroll
        for (int k = 0; k < N_EIG; ++k) atomicAdd(&d_red[k], d_acc[k]);
        #pragma unroll
        for (int i = 0; i < N_SRC; ++i) atomicAdd(&s_red[i], s_acc[i]);
    }
    __syncthreads();

    // del_tmp is block-private -> plain store
    for (int k = threadIdx.x; k < N_EIG; k += blockDim.x)
        del_tmp[b * N_EIG + k] = d_red[k];

    // sig_tmp is shared across the whole grid -> atomic
    for (int i = threadIdx.x; i < N_SRC; i += blockDim.x)
        atomicAdd(&sig_tmp[i], s_red[i]);
}

// ---------------------------------------------------------------------------
// Pass 2: out = N*vec + s*( Del @ del_tmp + Sig @ sig_tmp )
// ---------------------------------------------------------------------------
template <int N_SRC, int N_EIG, bool IS_INV>
__global__ void sparse_cov_apply(
    const float* __restrict__ noise,
    const float* __restrict__ diffuse,
    const float* __restrict__ source,
    const float* __restrict__ vec,
    const float* __restrict__ del_tmp,
    const float* __restrict__ sig_tmp,
    float* __restrict__ out,
    const int* __restrict__ edges,
    int nb
){
    // IS_INV touches exactly this one constant -- see note in the summary,
    // it is a cheap runtime argument if you want to halve the instantiations.
    constexpr float sgn = IS_INV ? -1.0f : 1.0f;

    const int b = blockIdx.x;
    if (b >= nb) return;

    const int start = edges[b];
    const int stop  = edges[b + 1];

    __shared__ float d[N_EIG];
    __shared__ float s[N_SRC];
    for (int k = threadIdx.x; k < N_EIG; k += blockDim.x) d[k] = del_tmp[b * N_EIG + k];
    for (int i = threadIdx.x; i < N_SRC; i += blockDim.x) s[i] = sig_tmp[i];
    __syncthreads();

    for (int row = start + threadIdx.x; row < stop; row += blockDim.x){
        float acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < N_EIG; ++k)
            acc = fmaf(diffuse[row * N_EIG + k], d[k], acc);
        #pragma unroll
        for (int i = 0; i < N_SRC; ++i)
            acc = fmaf(source[row * N_SRC + i], s[i], acc);

        out[row] = fmaf(noise[row], vec[row], sgn * acc);
    }
}

// ---------------------------------------------------------------------------
// Explicit instantiations (X-macro over the N_SRC x N_EIG grid)
// ---------------------------------------------------------------------------
#define INSTANTIATE_PAIR(NS, NE)                                               \
    template __global__ void sparse_cov_reduce<NS, NE>(                        \
        const float*, const float*, const float*, float*, float*,              \
        const int*, int);                                                      \
    template __global__ void sparse_cov_apply<NS, NE, true>(                   \
        const float*, const float*, const float*, const float*, const float*,  \
        const float*, float*, const int*, int);                                \
    template __global__ void sparse_cov_apply<NS, NE, false>(                  \
        const float*, const float*, const float*, const float*, const float*,  \
        const float*, float*, const int*, int);

#define EIG_LIST(NS)                                                           \
    INSTANTIATE_PAIR(NS,  1) INSTANTIATE_PAIR(NS,  2) INSTANTIATE_PAIR(NS,  3) \
    INSTANTIATE_PAIR(NS,  4) INSTANTIATE_PAIR(NS,  5) INSTANTIATE_PAIR(NS,  6) \
    INSTANTIATE_PAIR(NS,  7) INSTANTIATE_PAIR(NS,  8) INSTANTIATE_PAIR(NS,  9) \
    INSTANTIATE_PAIR(NS, 10) INSTANTIATE_PAIR(NS, 11) INSTANTIATE_PAIR(NS, 12) \
    INSTANTIATE_PAIR(NS, 13) INSTANTIATE_PAIR(NS, 14) INSTANTIATE_PAIR(NS, 15) \
    INSTANTIATE_PAIR(NS, 16) INSTANTIATE_PAIR(NS, 17) INSTANTIATE_PAIR(NS, 18) \
    INSTANTIATE_PAIR(NS, 19) INSTANTIATE_PAIR(NS, 20)

EIG_LIST( 1) EIG_LIST( 2) EIG_LIST( 3) EIG_LIST( 4) EIG_LIST( 5)
EIG_LIST( 6) EIG_LIST( 7) EIG_LIST( 8) EIG_LIST( 9) EIG_LIST(10)
EIG_LIST(11) EIG_LIST(12) EIG_LIST(13) EIG_LIST(14) EIG_LIST(15)
EIG_LIST(16) EIG_LIST(17) EIG_LIST(18) EIG_LIST(19) EIG_LIST(20)

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------
#define LAUNCH_EIG_CASE(NS, NE)                                                \
    case NE:                                                                   \
        sparse_cov_reduce<NS, NE><<<nb, TPB, 0, stream>>>(                     \
            diffuse, source, vec, del_tmp, sig_tmp, edges, nb);                \
        if (is_inv)                                                            \
            sparse_cov_apply<NS, NE, true><<<nb, TPB, 0, stream>>>(            \
                noise, diffuse, source, vec, del_tmp, sig_tmp, out, edges, nb);\
        else                                                                   \
            sparse_cov_apply<NS, NE, false><<<nb, TPB, 0, stream>>>(           \
                noise, diffuse, source, vec, del_tmp, sig_tmp, out, edges, nb);\
        break;

#define LAUNCH_SRC_CASE(NS)                                                    \
    case NS:                                                                   \
        switch (n_eig){                                                        \
            LAUNCH_EIG_CASE(NS,  1) LAUNCH_EIG_CASE(NS,  2)                    \
            LAUNCH_EIG_CASE(NS,  3) LAUNCH_EIG_CASE(NS,  4)                    \
            LAUNCH_EIG_CASE(NS,  5) LAUNCH_EIG_CASE(NS,  6)                    \
            LAUNCH_EIG_CASE(NS,  7) LAUNCH_EIG_CASE(NS,  8)                    \
            LAUNCH_EIG_CASE(NS,  9) LAUNCH_EIG_CASE(NS, 10)                    \
            LAUNCH_EIG_CASE(NS, 11) LAUNCH_EIG_CASE(NS, 12)                    \
            LAUNCH_EIG_CASE(NS, 13) LAUNCH_EIG_CASE(NS, 14)                    \
            LAUNCH_EIG_CASE(NS, 15) LAUNCH_EIG_CASE(NS, 16)                    \
            LAUNCH_EIG_CASE(NS, 17) LAUNCH_EIG_CASE(NS, 18)                    \
            LAUNCH_EIG_CASE(NS, 19) LAUNCH_EIG_CASE(NS, 20)                    \
            default: bad_eig = 1; break;                                       \
        }                                                                      \
        break;

extern "C"
void launch_sparse_cov_times_vec(
    const float* noise,
    const float* diffuse,
    const float* source,
    const float* vec,
    float* out,
    float* del_tmp,      // workspace, nb * n_eig
    float* sig_tmp,      // workspace, n_src  (zeroed here, not by the caller)
    const int* edges,
    int nb,
    int n_src,
    int n_eig,
    int is_inv,
    cudaStream_t stream
){
    int bad_eig = 0;

    // sig_tmp is an atomic accumulator, not a fully-overwritten buffer.
    // Zeroed here so a CG iteration cannot forget it.
    cudaMemsetAsync(sig_tmp, 0, (size_t)n_src * sizeof(float), stream);

    switch (n_src){
        LAUNCH_SRC_CASE( 1) LAUNCH_SRC_CASE( 2) LAUNCH_SRC_CASE( 3)
        LAUNCH_SRC_CASE( 4) LAUNCH_SRC_CASE( 5) LAUNCH_SRC_CASE( 6)
        LAUNCH_SRC_CASE( 7) LAUNCH_SRC_CASE( 8) LAUNCH_SRC_CASE( 9)
        LAUNCH_SRC_CASE(10) LAUNCH_SRC_CASE(11) LAUNCH_SRC_CASE(12)
        LAUNCH_SRC_CASE(13) LAUNCH_SRC_CASE(14) LAUNCH_SRC_CASE(15)
        LAUNCH_SRC_CASE(16) LAUNCH_SRC_CASE(17) LAUNCH_SRC_CASE(18)
        LAUNCH_SRC_CASE(19) LAUNCH_SRC_CASE(20)
        default:
            fprintf(stderr, "ERROR: n_src=%d not supported (max 20).\n", n_src);
            return;
    }

    if (bad_eig)
        fprintf(stderr, "ERROR: n_eig=%d not supported (max 20).\n", n_eig);
}
