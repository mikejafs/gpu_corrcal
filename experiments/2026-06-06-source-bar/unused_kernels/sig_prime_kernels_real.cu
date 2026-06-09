// sig_prime_kernels.cu
// Second-level Woodbury: fold Sigma Sigma^T onto (N + Delta Delta^T)^{-1}.
//
// REAL-VALUED throughout. The complex calibration problem is handled upstream
// via an alternating Re/Im layout (Re1 Im1 Re2 Im2 ...); from the kernel's
// view every entry is a plain float and every ^H is just ^T (no conjugation).
//
// Assumes Del_prime is the WHITENED diffuse factor:
//     (N + Delta Delta^T)^{-1} = N^{-1} - Del_prime Del_prime^T.
//
// Ragged CSR layout via `edges` (NO zero-padding; padding exists only in the
// CuPy reference). Block b owns flat rows edges[b]:edges[b+1] = the true
// redundant-group size. One redundant group per CUDA block.
//
// Shapes (row-major):
//   Sig        : (n_total, N_SRC)   float   -- source factor, baseline rows
//   Del_prime  : (n_total, N_EIG)   float   -- whitened diffuse factor
//   noise      : (n_total,)         float   -- N (reciprocal in-register)
//   edges      : (nb+1,)            int     -- CSR block boundaries
//   M_sig      : (N_SRC, N_SRC)     float   -- single capacitance, reduced over ALL blocks
//   L_inv      : (N_SRC, N_SRC)     float   -- lower-tri inverse of chol(I+M_sig)
//   Sig_prime  : (n_total, N_SRC)   float   -- OUTPUT (only large write)
//
// Per block b (contraction over rows i in edges[b]:edges[b+1]):
//   A[i,s]    = Sig[i,s] / N[i]                          (in-register, never stored)
//   B[s,e]    = sum_{i in b} Sig[i,s] Del'[i,e]          (N_SRC x N_EIG shared tile)
//   M_sig    += sum_i Sig[i,s] Sig[i,t]/N[i]  -  sum_e B[s,e] B[t,e]
//   W[i,s]    = A[i,s] - sum_e Del'[i,e] B[s,e]
//   Sig_prime[i,s] = sum_{t<=s} W[i,t] L_inv[s,t]
//
// B B^T carries TWO independent row indices within a block (i and i'), so it
// does NOT factor to a per-row scalar -- B is a real per-block tile, rebuilt
// in kernel 3 rather than written to global.

// ============================================================================
// KERNEL 1: capacitance reduction  (one redundant group per CUDA block)
//   H_b[s,t] = sum_i Sig[i,s] Sig[i,t]/N[i]  -  sum_e B[s,e] B[t,e]   (lower tri)
//   atomicAdd into single (N_SRC x N_SRC) M_sig => M_sig = sum_b H_b.
//   Identity added in kernel 2.
// ============================================================================
template <int N_SRC, int N_EIG>
__global__ void cap_reduce(
    const float* __restrict__ Sig,        // (n_total, N_SRC)
    const float* __restrict__ Del_prime,  // (n_total, N_EIG)
    const float* __restrict__ noise,      // (n_total,)
    const int*   __restrict__ edges,      // (nb+1,)
    float*       __restrict__ M_sig,      // (N_SRC, N_SRC) lower tri accumulated
    int nb
){
    const int b = blockIdx.x;
    if (b >= nb) return;
    const int start = edges[b];
    const int stop  = edges[b + 1];

    constexpr int NB_TILE = N_SRC * N_EIG;
    constexpr int NT      = N_SRC * (N_SRC + 1) / 2;

    __shared__ float Bsh[NB_TILE];
    __shared__ float SHA[NT];

    for (int t = threadIdx.x; t < NB_TILE; t += blockDim.x) Bsh[t] = 0.f;
    for (int t = threadIdx.x; t < NT;      t += blockDim.x) SHA[t] = 0.f;
    __syncthreads();

    // ---- Phase 1: accumulate B and SHA over the block's rows -------------
    for (int i = start + threadIdx.x; i < stop; i += blockDim.x) {
        const float invN = 1.0f / noise[i];           // reciprocal fusion

        float s[N_SRC];
        #pragma unroll
        for (int c = 0; c < N_SRC; ++c) s[c] = Sig[(size_t)i * N_SRC + c];

        float d[N_EIG];
        #pragma unroll
        for (int e = 0; e < N_EIG; ++e) d[e] = Del_prime[(size_t)i * N_EIG + e];

        // B[s,e] += Sig[i,s] Del'[i,e]
        #pragma unroll
        for (int sidx = 0; sidx < N_SRC; ++sidx) {
            #pragma unroll
            for (int e = 0; e < N_EIG; ++e)
                atomicAdd(&Bsh[sidx * N_EIG + e], s[sidx] * d[e]);
        }

        // SHA[s,t] += Sig[i,s] Sig[i,t] * invN   (lower tri s>=t)
        int k = 0;
        #pragma unroll
        for (int col = 0; col < N_SRC; ++col) {        // t
            #pragma unroll
            for (int row = col; row < N_SRC; ++row) {  // s
                atomicAdd(&SHA[k], s[row] * s[col] * invN);
                ++k;
            }
        }
    }
    __syncthreads();

    // ---- Phase 2: H_b = SHA - B B^T, atomicAdd into global M_sig ---------
    for (int k = threadIdx.x; k < NT; k += blockDim.x) {
        // recover (row=s, col=t) from packed col-major lower-tri index k
        int col = 0, row = 0, kk = k;
        #pragma unroll
        for (int c = 0; c < N_SRC; ++c) {
            int len = N_SRC - c;
            if (kk < len) { col = c; row = c + kk; break; }
            kk -= len;
        }

        float bbt = 0.f;                               // (B B^T)[s,t] = sum_e B[s,e] B[t,e]
        #pragma unroll
        for (int e = 0; e < N_EIG; ++e)
            bbt += Bsh[row * N_EIG + e] * Bsh[col * N_EIG + e];

        atomicAdd(&M_sig[(size_t)row * N_SRC + col], SHA[k] - bbt);
    }
}

// ============================================================================
// KERNEL 2: fused Cholesky + triangular inverse on single (N_SRC x N_SRC).
//   M = I + H, H = lower tri stored in M_sig (symmetric). One thread.
//   Output L_inv = (chol(M))^{-1}, lower triangular.
// ============================================================================
template <int N_SRC>
__global__ void chol_inv_fused(
    const float* __restrict__ M_sig,   // (N_SRC,N_SRC) lower tri = H
    float*       __restrict__ L_inv    // (N_SRC,N_SRC) output, lower tri
){
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float L[N_SRC * N_SRC];
    #pragma unroll
    for (int i = 0; i < N_SRC * N_SRC; ++i) L[i] = 0.f;

    // Cholesky: M = L L^T, M = I + H (symmetric, lower tri stored)
    for (int j = 0; j < N_SRC; ++j) {
        float diag = M_sig[(size_t)j * N_SRC + j] + 1.0f;     // + I
        for (int k = 0; k < j; ++k) {
            float ljk = L[j * N_SRC + k];
            diag -= ljk * ljk;
        }
        float ljj = sqrtf(diag);
        L[j * N_SRC + j] = ljj;
        float inv_ljj = 1.0f / ljj;

        for (int i = j + 1; i < N_SRC; ++i) {
            float sum = M_sig[(size_t)i * N_SRC + j];          // H[i,j], i>j
            for (int k = 0; k < j; ++k)
                sum -= L[i * N_SRC + k] * L[j * N_SRC + k];
            L[i * N_SRC + j] = sum * inv_ljj;
        }
    }

    // Triangular inverse (forward substitution)
    for (int j = 0; j < N_SRC; ++j) {
        L_inv[(size_t)j * N_SRC + j] = 1.0f / L[j * N_SRC + j];
        for (int i = j + 1; i < N_SRC; ++i) {
            float sum = 0.f;
            for (int k = j; k < i; ++k)
                sum += L[i * N_SRC + k] * L_inv[(size_t)k * N_SRC + j];
            L_inv[(size_t)i * N_SRC + j] = -sum / L[i * N_SRC + i];
        }
    }
    // zero strictly-upper (output buffer may be cp.empty)
    for (int i = 0; i < N_SRC; ++i)
        for (int j = i + 1; j < N_SRC; ++j)
            L_inv[(size_t)i * N_SRC + j] = 0.f;
}

// ============================================================================
// KERNEL 3: apply  (one redundant group per CUDA block)
//   Rebuild B[s,e] = sum_i Sig[i,s] Del'[i,e] (shared, recomputed), then per row:
//     W[i,s]         = Sig[i,s]/N[i] - sum_e Del'[i,e] B[s,e]
//     Sig_prime[i,s] = sum_{t<=s} W[i,t] L_inv[s,t]
// ============================================================================
template <int N_SRC, int N_EIG>
__global__ void apply_sig_prime(
    const float* __restrict__ Sig,        // (n_total, N_SRC)
    const float* __restrict__ Del_prime,  // (n_total, N_EIG)
    const float* __restrict__ noise,      // (n_total,)
    const int*   __restrict__ edges,      // (nb+1,)
    const float* __restrict__ L_inv,      // (N_SRC, N_SRC) lower-tri
    float*       __restrict__ Sig_prime,  // (n_total, N_SRC) OUTPUT
    int nb
){
    const int b = blockIdx.x;
    if (b >= nb) return;
    const int start = edges[b];
    const int stop  = edges[b + 1];

    constexpr int NB_TILE = N_SRC * N_EIG;
    __shared__ float Bsh[NB_TILE];
    __shared__ float Ls[N_SRC * N_SRC];

    for (int t = threadIdx.x; t < N_SRC * N_SRC; t += blockDim.x) Ls[t] = L_inv[t];
    for (int t = threadIdx.x; t < NB_TILE;       t += blockDim.x) Bsh[t] = 0.f;
    __syncthreads();

    // ---- Phase 1: rebuild B for this block -------------------------------
    for (int i = start + threadIdx.x; i < stop; i += blockDim.x) {
        float s[N_SRC];
        #pragma unroll
        for (int c = 0; c < N_SRC; ++c) s[c] = Sig[(size_t)i * N_SRC + c];
        float d[N_EIG];
        #pragma unroll
        for (int e = 0; e < N_EIG; ++e) d[e] = Del_prime[(size_t)i * N_EIG + e];

        #pragma unroll
        for (int sidx = 0; sidx < N_SRC; ++sidx) {
            #pragma unroll
            for (int e = 0; e < N_EIG; ++e)ygt
                atomicAdd(&Bsh[sidx * N_EIG + e], s[sidx] * d[e]);
        }
    }
    __syncthreads();

    // ---- Phase 2: per-row W and triangular solve -------------------------
    for (int i = start + threadIdx.x; i < stop; i += blockDim.x) {
        const float invN = 1.0f / noise[i];

        float s[N_SRC];
        #pragma unroll
        for (int c = 0; c < N_SRC; ++c) s[c] = Sig[(size_t)i * N_SRC + c];
        float d[N_EIG];
        #pragma unroll
        for (int e = 0; e < N_EIG; ++e) d[e] = Del_prime[(size_t)i * N_EIG + e];

        // W[s] = A[i,s] - sum_e Del'[i,e] B[s,e]
        float W[N_SRC];
        #pragma unroll
        for (int sidx = 0; sidx < N_SRC; ++sidx) {
            float w = s[sidx] * invN;                  // A[i,s]
            #pragma unroll
            for (int e = 0; e < N_EIG; ++e)
                w -= d[e] * Bsh[sidx * N_EIG + e];
            W[sidx] = w;
        }

        // Sig_prime[i,s] = sum_{t<=s} W[t] L_inv[s,t]
        #pragma unroll
        for (int row = 0; row < N_SRC; ++row) {
            float out = 0.f;
            #pragma unroll
            for (int col = 0; col <= row; ++col)
                out += W[col] * Ls[row * N_SRC + col];
            Sig_prime[(size_t)i * N_SRC + row] = out;
        }
    }
}
