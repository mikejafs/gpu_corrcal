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
// Template instantiations + host launchers
//
// cap_reduce and apply_sig_prime are templated on TWO params (N_SRC, N_EIG),
// so the instantiation set is the product 1..20 x 1..20 = 400 each. These are
// stamped out at COMPILE time -- they cannot be generated by a runtime loop
// (template args must be compile-time constants). We use nested preprocessor
// macros so the 1..20 ranges are written once and expanded by the compiler.
//
// Dispatch uses a single combined key:  KEY(n_src, n_eig) = n_src*32 + n_eig
// (32 > 20 so the two indices never overlap; *32 is a shift, not a multiply).
// ============================================================================

#include <stdio.h>
#include <cuda_runtime.h>

#define KEY(NS, NE) ((NS) * 32 + (NE))

// ---- X-macro over 1..20 ----------------------------------------------------
#define FOR_EACH_DIM(DO) \
    DO(1)  DO(2)  DO(3)  DO(4)  DO(5)  DO(6)  DO(7)  DO(8)  DO(9)  DO(10) \
    DO(11) DO(12) DO(13) DO(14) DO(15) DO(16) DO(17) DO(18) DO(19) DO(20)

// ===================== explicit template instantiations =====================
// chol_inv_fused: single param N_SRC (20 total)
#define INST_CHOL(NS) \
    template __global__ void chol_inv_fused<NS>(const float*, float*);
FOR_EACH_DIM(INST_CHOL)
#undef INST_CHOL

// cap_reduce / apply_sig_prime: product N_SRC x N_EIG (400 each)
// Nested expansion: outer DO(NS) re-runs FOR_EACH_DIM with an inner that fixes NS.
#define INST_CAP_PAIR(NS, NE) \
    template __global__ void cap_reduce<NS, NE>( \
        const float*, const float*, const float*, const int*, float*, float*, int);
#define INST_APPLY_PAIR(NS, NE) \
    template __global__ void apply_sig_prime<NS, NE>( \
        const float*, const float*, const float*, const int*, const float*, float*, int);

// for a fixed NS, sweep NE over 1..20
#define INST_CAP_ROW(NS)   \
    INST_CAP_PAIR(NS,1)  INST_CAP_PAIR(NS,2)  INST_CAP_PAIR(NS,3)  INST_CAP_PAIR(NS,4)  \
    INST_CAP_PAIR(NS,5)  INST_CAP_PAIR(NS,6)  INST_CAP_PAIR(NS,7)  INST_CAP_PAIR(NS,8)  \
    INST_CAP_PAIR(NS,9)  INST_CAP_PAIR(NS,10) INST_CAP_PAIR(NS,11) INST_CAP_PAIR(NS,12) \
    INST_CAP_PAIR(NS,13) INST_CAP_PAIR(NS,14) INST_CAP_PAIR(NS,15) INST_CAP_PAIR(NS,16) \
    INST_CAP_PAIR(NS,17) INST_CAP_PAIR(NS,18) INST_CAP_PAIR(NS,19) INST_CAP_PAIR(NS,20)
#define INST_APPLY_ROW(NS) \
    INST_APPLY_PAIR(NS,1)  INST_APPLY_PAIR(NS,2)  INST_APPLY_PAIR(NS,3)  INST_APPLY_PAIR(NS,4)  \
    INST_APPLY_PAIR(NS,5)  INST_APPLY_PAIR(NS,6)  INST_APPLY_PAIR(NS,7)  INST_APPLY_PAIR(NS,8)  \
    INST_APPLY_PAIR(NS,9)  INST_APPLY_PAIR(NS,10) INST_APPLY_PAIR(NS,11) INST_APPLY_PAIR(NS,12) \
    INST_APPLY_PAIR(NS,13) INST_APPLY_PAIR(NS,14) INST_APPLY_PAIR(NS,15) INST_APPLY_PAIR(NS,16) \
    INST_APPLY_PAIR(NS,17) INST_APPLY_PAIR(NS,18) INST_APPLY_PAIR(NS,19) INST_APPLY_PAIR(NS,20)

FOR_EACH_DIM(INST_CAP_ROW)
FOR_EACH_DIM(INST_APPLY_ROW)

#undef INST_CAP_PAIR
#undef INST_APPLY_PAIR
#undef INST_CAP_ROW
#undef INST_APPLY_ROW

// ============================== host launchers ==============================
// Block size 128 matches the diffuse_bar convention. cap_reduce and
// apply_sig_prime launch one CUDA block per redundant group (<<<nb, 128>>>);
// chol_inv_fused is a single-thread kernel (<<<1, 1>>>).

// ---- KERNEL 1: cap_reduce ----
#define CASE_CAP(NS, NE) \
    case KEY(NS, NE): \
        cap_reduce<NS, NE><<<nb, 128>>>(Sig, Del_prime, noise, edges, B, M_sig, nb); \
        break;
#define CASE_CAP_ROW(NS) \
    CASE_CAP(NS,1)  CASE_CAP(NS,2)  CASE_CAP(NS,3)  CASE_CAP(NS,4)  CASE_CAP(NS,5)  \
    CASE_CAP(NS,6)  CASE_CAP(NS,7)  CASE_CAP(NS,8)  CASE_CAP(NS,9)  CASE_CAP(NS,10) \
    CASE_CAP(NS,11) CASE_CAP(NS,12) CASE_CAP(NS,13) CASE_CAP(NS,14) CASE_CAP(NS,15) \
    CASE_CAP(NS,16) CASE_CAP(NS,17) CASE_CAP(NS,18) CASE_CAP(NS,19) CASE_CAP(NS,20)

extern "C"
void launch_cap_reduce(
    const float* Sig,
    const float* Del_prime,
    const float* noise,
    const int*   edges,
    float*       B,        // (nb, n_src, n_eig)
    float*       M_sig,    // (n_src, n_src), must be zeroed before launch
    int nb,
    int n_src,
    int n_eig
){
    switch (KEY(n_src, n_eig)) {
        FOR_EACH_DIM(CASE_CAP_ROW)
        default:
            fprintf(stderr, "ERROR: cap_reduce (n_src=%d, n_eig=%d) not supported "
                    "(each max 20). Recompile with additional instantiations.\n",
                    n_src, n_eig);
            break;
    }
}
#undef CASE_CAP
#undef CASE_CAP_ROW

// ---- KERNEL 2: chol_inv_fused ----
#define CASE_CHOL(NS) \
    case NS: \
        chol_inv_fused<NS><<<1, 1>>>(M_sig, L_inv); \
        break;

extern "C"
void launch_chol_inv_fused(
    const float* M_sig,    // (n_src, n_src) lower tri = Lambda
    float*       L_inv,    // (n_src, n_src) output
    int n_src
){
    switch (n_src) {
        FOR_EACH_DIM(CASE_CHOL)
        default:
            fprintf(stderr, "ERROR: chol_inv_fused (n_src=%d) not supported "
                    "(max 20). Recompile with additional instantiations.\n", n_src);
            break;
    }
}
#undef CASE_CHOL

// ---- KERNEL 3: apply_sig_prime ----
#define CASE_APPLY(NS, NE) \
    case KEY(NS, NE): \
        apply_sig_prime<NS, NE><<<nb, 128>>>(Sig, Del_prime, noise, edges, B, L_inv, Sig_prime, nb); \
        break;
#define CASE_APPLY_ROW(NS) \
    CASE_APPLY(NS,1)  CASE_APPLY(NS,2)  CASE_APPLY(NS,3)  CASE_APPLY(NS,4)  CASE_APPLY(NS,5)  \
    CASE_APPLY(NS,6)  CASE_APPLY(NS,7)  CASE_APPLY(NS,8)  CASE_APPLY(NS,9)  CASE_APPLY(NS,10) \
    CASE_APPLY(NS,11) CASE_APPLY(NS,12) CASE_APPLY(NS,13) CASE_APPLY(NS,14) CASE_APPLY(NS,15) \
    CASE_APPLY(NS,16) CASE_APPLY(NS,17) CASE_APPLY(NS,18) CASE_APPLY(NS,19) CASE_APPLY(NS,20)

extern "C"
void launch_apply_sig_prime(
    const float* Sig,
    const float* Del_prime,
    const float* noise,
    const int*   edges,
    const float* B,         // (nb, n_src, n_eig) from kernel 1
    const float* L_inv,     // (n_src, n_src)
    float*       Sig_prime, // (n_total, n_src) OUTPUT
    int nb,
    int n_src,
    int n_eig
){
    switch (KEY(n_src, n_eig)) {
        FOR_EACH_DIM(CASE_APPLY_ROW)
        default:
            fprintf(stderr, "ERROR: apply_sig_prime (n_src=%d, n_eig=%d) not supported "
                    "(each max 20). Recompile with additional instantiations.\n",
                    n_src, n_eig);
            break;
    }
}
#undef CASE_APPLY
#undef CASE_APPLY_ROW

#undef FOR_EACH_DIM
#undef KEY
