// sig_prime_kernels.cu
// Second-level Woodbury: produce the whitened source factor Sig_prime (Sig_bar)
// such that  C^{-1} = N^{-1} - Del_bar Del_bar^T - Sig_bar Sig_bar^T
// for  C = N + Delta Delta^T + Sigma Sigma^T.
//
// REAL-VALUED throughout. The complex calibration problem is handled upstream
// via an alternating Re/Im layout (Re1 Im1 Re2 Im2 ...); from the kernel's
// view every entry is a plain float and every ^H is just ^T (no conjugation).
//
// Del_prime (== Del_bar) is the WHITENED diffuse factor:
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
//   B          : (nb, N_SRC, N_EIG) float   -- per-group factor, WRITTEN by k1, READ by k3
//   M_sig      : (N_SRC, N_SRC)     float   -- single capacitance, reduced over ALL blocks
//   L_inv      : (N_SRC, N_SRC)     float   -- lower-tri inverse of chol(I+Lambda)
//   Sig_prime  : (n_total, N_SRC)   float   -- OUTPUT (only large write)
//
// Pipeline:
//   Step 1 (cap_reduce):  B[b][s,e]  = sum_{i in b} Sig[i,s] Del'[i,e]      -> stored
//                         Lambda     = sum_b ( Sig^T A - B B^T )            -> M_sig (lower tri)
//                         where A[i,s] = Sig[i,s]/N[i]
//   Step 2 (chol_inv_fused): L_sig L_sig^T = chol(I + Lambda); L_inv = L_sig^{-1}
//   Step 3 (apply_sig_prime):
//                         Gamma[i,s] = A[i,s] - sum_e Del'[i,e] B[b][s,e]
//                         Sig_prime  = Gamma (L_sig^T)^{-1}
//                                    => Sig_prime[i,s] = sum_{t<=s} Gamma[i,t] L_inv[s,t]
//
// B carries TWO independent row indices within a group (i and i') via B B^T, so
// it cannot factor to a per-row scalar. B is (nb, N_SRC, N_EIG): its size is fixed
// by sources/modes and does NOT grow with n_total. At CHORD redundancy (large
// group sizes, n_total ~ 1.3e5+) rebuilding B in step 3 would re-run a reduction
// over all baselines; instead step 1 writes B once and step 3 reads it back --
// one small write + one small read vs. recomputing the full reduction.

// ============================================================================
// KERNEL 1: capacitance reduction  (one redundant group per CUDA block)
//   B[b][s,e] = sum_i Sig[i,s] Del'[i,e]                      -> global B
//   H_b[s,t]  = sum_i Sig[i,s] Sig[i,t]/N[i]  -  sum_e B[s,e] B[t,e]  (lower tri)
//   atomicAdd H_b into single (N_SRC x N_SRC) M_sig => M_sig = sum_b H_b = Lambda.
//   Identity added in kernel 2.
// ============================================================================
template <int N_SRC, int N_EIG>
__global__ void cap_reduce(
    const float* __restrict__ Sig,        // (n_total, N_SRC)
    const float* __restrict__ Del_prime,  // (n_total, N_EIG)
    const float* __restrict__ noise,      // (n_total,)
    const int*   __restrict__ edges,      // (nb+1,)
    float*       __restrict__ B,          // (nb, N_SRC, N_EIG) OUTPUT
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

    // ---- Phase 1: accumulate B and SHA over the group's rows -------------
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

    // ---- Write the completed B tile out to global (fully overwritten) -----
    {
        const size_t boff = (size_t)b * NB_TILE;
        for (int t = threadIdx.x; t < NB_TILE; t += blockDim.x)
            B[boff + t] = Bsh[t];
    }

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
//   M = I + Lambda, Lambda = lower tri stored in M_sig (symmetric). One thread.
//   Output L_inv = (chol(M))^{-1}, lower triangular.
// ============================================================================
template <int N_SRC>
__global__ void chol_inv_fused(
    const float* __restrict__ M_sig,   // (N_SRC,N_SRC) lower tri = Lambda
    float*       __restrict__ L_inv    // (N_SRC,N_SRC) output, lower tri
){
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float L[N_SRC * N_SRC];
    #pragma unroll
    for (int i = 0; i < N_SRC * N_SRC; ++i) L[i] = 0.f;

    // Cholesky: M = L L^T, M = I + Lambda (symmetric, lower tri stored)
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
            float sum = M_sig[(size_t)i * N_SRC + j];          // Lambda[i,j], i>j
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
//   Reads the precomputed B[b] tile (from kernel 1), then per row:
//     Gamma[i,s]     = Sig[i,s]/N[i] - sum_e Del'[i,e] B[b][s,e]
//     Sig_prime[i,s] = sum_{t<=s} Gamma[i,t] L_inv[s,t]
//   No B rebuild, no second reduction over baselines.
// ============================================================================
template <int N_SRC, int N_EIG>
__global__ void apply_sig_prime(
    const float* __restrict__ Sig,        // (n_total, N_SRC)
    const float* __restrict__ Del_prime,  // (n_total, N_EIG)
    const float* __restrict__ noise,      // (n_total,)
    const int*   __restrict__ edges,      // (nb+1,)
    const float* __restrict__ B,          // (nb, N_SRC, N_EIG) from kernel 1
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

    // load this group's precomputed B tile + L_inv into shared
    const size_t boff = (size_t)b * NB_TILE;
    for (int t = threadIdx.x; t < NB_TILE;       t += blockDim.x) Bsh[t] = B[boff + t];
    for (int t = threadIdx.x; t < N_SRC * N_SRC; t += blockDim.x) Ls[t]  = L_inv[t];
    __syncthreads();

    // ---- per-row Gamma and triangular solve ------------------------------
    for (int i = start + threadIdx.x; i < stop; i += blockDim.x) {
        const float invN = 1.0f / noise[i];

        float s[N_SRC];
        #pragma unroll
        for (int c = 0; c < N_SRC; ++c) s[c] = Sig[(size_t)i * N_SRC + c];
        float d[N_EIG];
        #pragma unroll
        for (int e = 0; e < N_EIG; ++e) d[e] = Del_prime[(size_t)i * N_EIG + e];

        // Gamma[s] = A[i,s] - sum_e Del'[i,e] B[s,e]
        float G[N_SRC];
        #pragma unroll
        for (int sidx = 0; sidx < N_SRC; ++sidx) {
            float g = s[sidx] * invN;                  // A[i,s]
            #pragma unroll
            for (int e = 0; e < N_EIG; ++e)
                g -= d[e] * Bsh[sidx * N_EIG + e];
            G[sidx] = g;
        }

        // Sig_prime[i,s] = sum_{t<=s} Gamma[t] L_inv[s,t]
        #pragma unroll
        for (int row = 0; row < N_SRC; ++row) {
            float out = 0.f;
            #pragma unroll
            for (int col = 0; col <= row; ++col)
                out += G[col] * Ls[row * N_SRC + col];
            Sig_prime[(size_t)i * N_SRC + row] = out;
        }
    }
}

// ============================================================================
// EXPLICIT TEMPLATE INSTANTIATIONS
// ============================================================================

// ---- explicit instantiations: cap_reduce<N_SRC, N_EIG> ----
template __global__ void cap_reduce<1, 1>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<1, 2>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<1, 3>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<1, 4>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<1, 5>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<1, 6>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<1, 7>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<1, 8>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<1, 9>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<1, 10>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<1, 11>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<1, 12>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<1, 13>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<1, 14>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<1, 15>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<1, 16>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<1, 17>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<1, 18>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<1, 19>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<1, 20>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<2, 1>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<2, 2>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<2, 3>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<2, 4>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<2, 5>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<2, 6>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<2, 7>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<2, 8>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<2, 9>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<2, 10>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<2, 11>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<2, 12>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<2, 13>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<2, 14>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<2, 15>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<2, 16>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<2, 17>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<2, 18>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<2, 19>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<2, 20>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<3, 1>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<3, 2>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<3, 3>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<3, 4>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<3, 5>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<3, 6>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<3, 7>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<3, 8>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<3, 9>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<3, 10>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<3, 11>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<3, 12>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<3, 13>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<3, 14>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<3, 15>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<3, 16>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<3, 17>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<3, 18>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<3, 19>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<3, 20>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<4, 1>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<4, 2>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<4, 3>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<4, 4>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<4, 5>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<4, 6>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<4, 7>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<4, 8>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<4, 9>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<4, 10>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<4, 11>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<4, 12>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<4, 13>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<4, 14>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<4, 15>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<4, 16>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<4, 17>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<4, 18>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<4, 19>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<4, 20>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<5, 1>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<5, 2>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<5, 3>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<5, 4>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<5, 5>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<5, 6>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<5, 7>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<5, 8>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<5, 9>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<5, 10>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<5, 11>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<5, 12>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<5, 13>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<5, 14>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<5, 15>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<5, 16>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<5, 17>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<5, 18>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<5, 19>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<5, 20>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<6, 1>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<6, 2>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<6, 3>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<6, 4>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<6, 5>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<6, 6>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<6, 7>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<6, 8>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<6, 9>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<6, 10>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<6, 11>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<6, 12>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<6, 13>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<6, 14>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<6, 15>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<6, 16>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<6, 17>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<6, 18>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<6, 19>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<6, 20>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<7, 1>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<7, 2>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<7, 3>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<7, 4>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<7, 5>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<7, 6>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<7, 7>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<7, 8>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<7, 9>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<7, 10>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<7, 11>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<7, 12>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<7, 13>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<7, 14>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<7, 15>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<7, 16>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<7, 17>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<7, 18>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<7, 19>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<7, 20>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<8, 1>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<8, 2>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<8, 3>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<8, 4>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<8, 5>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<8, 6>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<8, 7>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<8, 8>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<8, 9>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<8, 10>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<8, 11>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<8, 12>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<8, 13>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<8, 14>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<8, 15>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<8, 16>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<8, 17>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<8, 18>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<8, 19>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<8, 20>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<9, 1>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<9, 2>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<9, 3>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<9, 4>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<9, 5>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<9, 6>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<9, 7>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<9, 8>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<9, 9>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<9, 10>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<9, 11>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<9, 12>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<9, 13>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<9, 14>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<9, 15>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<9, 16>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<9, 17>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<9, 18>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<9, 19>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<9, 20>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<10, 1>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<10, 2>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<10, 3>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<10, 4>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<10, 5>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<10, 6>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<10, 7>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<10, 8>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<10, 9>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<10, 10>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<10, 11>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<10, 12>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<10, 13>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<10, 14>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<10, 15>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<10, 16>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<10, 17>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<10, 18>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<10, 19>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<10, 20>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<11, 1>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<11, 2>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<11, 3>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<11, 4>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<11, 5>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<11, 6>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<11, 7>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<11, 8>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<11, 9>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<11, 10>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<11, 11>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<11, 12>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<11, 13>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<11, 14>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<11, 15>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<11, 16>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<11, 17>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<11, 18>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<11, 19>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<11, 20>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<12, 1>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<12, 2>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<12, 3>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<12, 4>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<12, 5>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<12, 6>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<12, 7>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<12, 8>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<12, 9>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<12, 10>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<12, 11>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<12, 12>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<12, 13>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<12, 14>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<12, 15>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<12, 16>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<12, 17>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<12, 18>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<12, 19>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<12, 20>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<13, 1>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<13, 2>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<13, 3>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<13, 4>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<13, 5>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<13, 6>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<13, 7>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<13, 8>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<13, 9>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<13, 10>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<13, 11>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<13, 12>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<13, 13>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<13, 14>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<13, 15>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<13, 16>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<13, 17>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<13, 18>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<13, 19>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<13, 20>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<14, 1>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<14, 2>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<14, 3>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<14, 4>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<14, 5>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<14, 6>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<14, 7>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<14, 8>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<14, 9>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<14, 10>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<14, 11>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<14, 12>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<14, 13>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<14, 14>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<14, 15>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<14, 16>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<14, 17>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<14, 18>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<14, 19>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<14, 20>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<15, 1>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<15, 2>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<15, 3>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<15, 4>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<15, 5>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<15, 6>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<15, 7>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<15, 8>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<15, 9>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<15, 10>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<15, 11>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<15, 12>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<15, 13>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<15, 14>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<15, 15>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<15, 16>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<15, 17>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<15, 18>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<15, 19>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<15, 20>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<16, 1>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<16, 2>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<16, 3>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<16, 4>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<16, 5>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<16, 6>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<16, 7>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<16, 8>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<16, 9>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<16, 10>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<16, 11>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<16, 12>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<16, 13>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<16, 14>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<16, 15>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<16, 16>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<16, 17>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<16, 18>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<16, 19>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<16, 20>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<17, 1>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<17, 2>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<17, 3>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<17, 4>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<17, 5>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<17, 6>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<17, 7>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<17, 8>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<17, 9>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<17, 10>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<17, 11>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<17, 12>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<17, 13>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<17, 14>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<17, 15>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<17, 16>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<17, 17>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<17, 18>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<17, 19>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<17, 20>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<18, 1>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<18, 2>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<18, 3>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<18, 4>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<18, 5>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<18, 6>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<18, 7>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<18, 8>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<18, 9>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<18, 10>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<18, 11>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<18, 12>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<18, 13>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<18, 14>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<18, 15>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<18, 16>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<18, 17>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<18, 18>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<18, 19>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<18, 20>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<19, 1>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<19, 2>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<19, 3>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<19, 4>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<19, 5>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<19, 6>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<19, 7>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<19, 8>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<19, 9>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<19, 10>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<19, 11>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<19, 12>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<19, 13>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<19, 14>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<19, 15>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<19, 16>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<19, 17>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<19, 18>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<19, 19>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<19, 20>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<20, 1>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<20, 2>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<20, 3>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<20, 4>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<20, 5>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<20, 6>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<20, 7>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<20, 8>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<20, 9>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<20, 10>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<20, 11>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<20, 12>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<20, 13>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<20, 14>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<20, 15>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<20, 16>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<20, 17>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<20, 18>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<20, 19>(
    const float*, const float*, const float*, const int*, float*, float*, int);
template __global__ void cap_reduce<20, 20>(
    const float*, const float*, const float*, const int*, float*, float*, int);

// ---- explicit instantiations: chol_inv_fused<N_SRC> ----
template __global__ void chol_inv_fused<1>(
    const float*, float*);
template __global__ void chol_inv_fused<2>(
    const float*, float*);
template __global__ void chol_inv_fused<3>(
    const float*, float*);
template __global__ void chol_inv_fused<4>(
    const float*, float*);
template __global__ void chol_inv_fused<5>(
    const float*, float*);
template __global__ void chol_inv_fused<6>(
    const float*, float*);
template __global__ void chol_inv_fused<7>(
    const float*, float*);
template __global__ void chol_inv_fused<8>(
    const float*, float*);
template __global__ void chol_inv_fused<9>(
    const float*, float*);
template __global__ void chol_inv_fused<10>(
    const float*, float*);
template __global__ void chol_inv_fused<11>(
    const float*, float*);
template __global__ void chol_inv_fused<12>(
    const float*, float*);
template __global__ void chol_inv_fused<13>(
    const float*, float*);
template __global__ void chol_inv_fused<14>(
    const float*, float*);
template __global__ void chol_inv_fused<15>(
    const float*, float*);
template __global__ void chol_inv_fused<16>(
    const float*, float*);
template __global__ void chol_inv_fused<17>(
    const float*, float*);
template __global__ void chol_inv_fused<18>(
    const float*, float*);
template __global__ void chol_inv_fused<19>(
    const float*, float*);
template __global__ void chol_inv_fused<20>(
    const float*, float*);

// ---- explicit instantiations: apply_sig_prime<N_SRC, N_EIG> ----
template __global__ void apply_sig_prime<1, 1>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<1, 2>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<1, 3>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<1, 4>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<1, 5>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<1, 6>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<1, 7>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<1, 8>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<1, 9>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<1, 10>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<1, 11>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<1, 12>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<1, 13>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<1, 14>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<1, 15>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<1, 16>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<1, 17>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<1, 18>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<1, 19>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<1, 20>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<2, 1>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<2, 2>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<2, 3>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<2, 4>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<2, 5>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<2, 6>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<2, 7>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<2, 8>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<2, 9>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<2, 10>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<2, 11>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<2, 12>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<2, 13>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<2, 14>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<2, 15>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<2, 16>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<2, 17>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<2, 18>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<2, 19>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<2, 20>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<3, 1>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<3, 2>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<3, 3>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<3, 4>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<3, 5>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<3, 6>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<3, 7>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<3, 8>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<3, 9>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<3, 10>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<3, 11>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<3, 12>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<3, 13>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<3, 14>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<3, 15>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<3, 16>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<3, 17>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<3, 18>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<3, 19>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<3, 20>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<4, 1>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<4, 2>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<4, 3>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<4, 4>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<4, 5>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<4, 6>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<4, 7>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<4, 8>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<4, 9>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<4, 10>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<4, 11>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<4, 12>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<4, 13>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<4, 14>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<4, 15>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<4, 16>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<4, 17>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<4, 18>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<4, 19>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<4, 20>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<5, 1>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<5, 2>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<5, 3>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<5, 4>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<5, 5>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<5, 6>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<5, 7>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<5, 8>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<5, 9>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<5, 10>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<5, 11>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<5, 12>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<5, 13>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<5, 14>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<5, 15>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<5, 16>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<5, 17>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<5, 18>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<5, 19>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<5, 20>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<6, 1>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<6, 2>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<6, 3>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<6, 4>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<6, 5>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<6, 6>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<6, 7>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<6, 8>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<6, 9>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<6, 10>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<6, 11>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<6, 12>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<6, 13>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<6, 14>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<6, 15>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<6, 16>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<6, 17>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<6, 18>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<6, 19>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<6, 20>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<7, 1>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<7, 2>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<7, 3>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<7, 4>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<7, 5>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<7, 6>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<7, 7>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<7, 8>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<7, 9>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<7, 10>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<7, 11>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<7, 12>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<7, 13>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<7, 14>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<7, 15>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<7, 16>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<7, 17>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<7, 18>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<7, 19>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<7, 20>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<8, 1>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<8, 2>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<8, 3>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<8, 4>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<8, 5>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<8, 6>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<8, 7>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<8, 8>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<8, 9>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<8, 10>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<8, 11>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<8, 12>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<8, 13>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<8, 14>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<8, 15>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<8, 16>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<8, 17>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<8, 18>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<8, 19>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<8, 20>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<9, 1>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<9, 2>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<9, 3>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<9, 4>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<9, 5>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<9, 6>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<9, 7>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<9, 8>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<9, 9>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<9, 10>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<9, 11>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<9, 12>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<9, 13>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<9, 14>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<9, 15>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<9, 16>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<9, 17>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<9, 18>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<9, 19>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<9, 20>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<10, 1>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<10, 2>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<10, 3>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<10, 4>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<10, 5>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<10, 6>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<10, 7>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<10, 8>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<10, 9>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<10, 10>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<10, 11>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<10, 12>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<10, 13>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<10, 14>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<10, 15>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<10, 16>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<10, 17>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<10, 18>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<10, 19>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<10, 20>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<11, 1>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<11, 2>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<11, 3>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<11, 4>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<11, 5>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<11, 6>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<11, 7>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<11, 8>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<11, 9>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<11, 10>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<11, 11>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<11, 12>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<11, 13>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<11, 14>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<11, 15>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<11, 16>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<11, 17>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<11, 18>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<11, 19>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<11, 20>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<12, 1>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<12, 2>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<12, 3>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<12, 4>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<12, 5>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<12, 6>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<12, 7>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<12, 8>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<12, 9>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<12, 10>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<12, 11>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<12, 12>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<12, 13>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<12, 14>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<12, 15>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<12, 16>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<12, 17>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<12, 18>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<12, 19>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<12, 20>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<13, 1>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<13, 2>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<13, 3>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<13, 4>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<13, 5>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<13, 6>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<13, 7>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<13, 8>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<13, 9>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<13, 10>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<13, 11>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<13, 12>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<13, 13>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<13, 14>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<13, 15>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<13, 16>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<13, 17>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<13, 18>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<13, 19>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<13, 20>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<14, 1>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<14, 2>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<14, 3>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<14, 4>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<14, 5>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<14, 6>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<14, 7>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<14, 8>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<14, 9>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<14, 10>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<14, 11>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<14, 12>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<14, 13>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<14, 14>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<14, 15>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<14, 16>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<14, 17>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<14, 18>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<14, 19>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<14, 20>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<15, 1>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<15, 2>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<15, 3>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<15, 4>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<15, 5>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<15, 6>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<15, 7>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<15, 8>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<15, 9>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<15, 10>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<15, 11>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<15, 12>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<15, 13>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<15, 14>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<15, 15>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<15, 16>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<15, 17>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<15, 18>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<15, 19>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<15, 20>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<16, 1>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<16, 2>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<16, 3>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<16, 4>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<16, 5>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<16, 6>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<16, 7>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<16, 8>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<16, 9>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<16, 10>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<16, 11>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<16, 12>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<16, 13>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<16, 14>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<16, 15>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<16, 16>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<16, 17>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<16, 18>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<16, 19>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<16, 20>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<17, 1>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<17, 2>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<17, 3>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<17, 4>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<17, 5>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<17, 6>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<17, 7>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<17, 8>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<17, 9>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<17, 10>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<17, 11>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<17, 12>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<17, 13>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<17, 14>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<17, 15>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<17, 16>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<17, 17>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<17, 18>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<17, 19>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<17, 20>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<18, 1>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<18, 2>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<18, 3>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<18, 4>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<18, 5>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<18, 6>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<18, 7>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<18, 8>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<18, 9>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<18, 10>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<18, 11>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<18, 12>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<18, 13>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<18, 14>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<18, 15>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<18, 16>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<18, 17>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<18, 18>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<18, 19>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<18, 20>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<19, 1>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<19, 2>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<19, 3>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<19, 4>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<19, 5>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<19, 6>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<19, 7>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<19, 8>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<19, 9>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<19, 10>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<19, 11>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<19, 12>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<19, 13>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<19, 14>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<19, 15>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<19, 16>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<19, 17>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<19, 18>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<19, 19>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<19, 20>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<20, 1>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<20, 2>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<20, 3>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<20, 4>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<20, 5>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<20, 6>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<20, 7>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<20, 8>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<20, 9>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<20, 10>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<20, 11>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<20, 12>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<20, 13>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<20, 14>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<20, 15>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<20, 16>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<20, 17>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<20, 18>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<20, 19>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
template __global__ void apply_sig_prime<20, 20>(
    const float*, const float*, const float*, const int*, const float*, const float*, float*, int);
