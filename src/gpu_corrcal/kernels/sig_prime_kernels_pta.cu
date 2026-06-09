// nvcc -Xcompiler -fPIC -shared -o geneig_warp_red_kernel_templated.so geneig_warp_red_kernel_templated.cu

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
//
//   REGISTER-ACCUMULATE: each thread keeps private B/SHA partials in registers
//   (spilling to L1-cached local memory for large N_SRC/N_EIG). The per-row
//   hot loop uses plain +=, eliminating all shared-memory atomics from the
//   inner loop. After the row loop, an intra-warp shuffle reduction combines
//   32 threads -> 1, then lane 0 of each warp does a single atomicAdd into
//   shared. Total atomics per element: (blockDim/32) instead of group_size.
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

    // ---- Per-thread private accumulators ----
    float rB[NB_TILE];
    float rS[NT];
    #pragma unroll
    for (int t = 0; t < NB_TILE; ++t) rB[t] = 0.f;
    #pragma unroll
    for (int t = 0; t < NT; ++t) rS[t] = 0.f;

    // ---- Phase 1: accumulate into registers (plain +=, NO atomics) -------
    for (int i = start + threadIdx.x; i < stop; i += blockDim.x) {
        const float invN = 1.0f / noise[i];

        float s[N_SRC];
        #pragma unroll
        for (int c = 0; c < N_SRC; ++c) s[c] = Sig[(size_t)i * N_SRC + c];

        float d[N_EIG];
        #pragma unroll
        for (int e = 0; e < N_EIG; ++e) d[e] = Del_prime[(size_t)i * N_EIG + e];

        #pragma unroll
        for (int sidx = 0; sidx < N_SRC; ++sidx) {
            #pragma unroll
            for (int e = 0; e < N_EIG; ++e)
                rB[sidx * N_EIG + e] += s[sidx] * d[e];
        }

        int k = 0;
        #pragma unroll
        for (int col = 0; col < N_SRC; ++col) {
            #pragma unroll
            for (int row = col; row < N_SRC; ++row) {
                rS[k] += s[row] * s[col] * invN;
                ++k;
            }
        }
    }

    // ---- Intra-warp reduction via shuffle (32 threads -> lane 0) ----------
    for (int t = 0; t < NB_TILE; ++t) {
        float val = rB[t];
        val += __shfl_down_sync(0xFFFFFFFF, val, 16);
        val += __shfl_down_sync(0xFFFFFFFF, val, 8);
        val += __shfl_down_sync(0xFFFFFFFF, val, 4);
        val += __shfl_down_sync(0xFFFFFFFF, val, 2);
        val += __shfl_down_sync(0xFFFFFFFF, val, 1);
        rB[t] = val;
    }
    for (int t = 0; t < NT; ++t) {
        float val = rS[t];
        val += __shfl_down_sync(0xFFFFFFFF, val, 16);
        val += __shfl_down_sync(0xFFFFFFFF, val, 8);
        val += __shfl_down_sync(0xFFFFFFFF, val, 4);
        val += __shfl_down_sync(0xFFFFFFFF, val, 2);
        val += __shfl_down_sync(0xFFFFFFFF, val, 1);
        rS[t] = val;
    }

    // ---- Cross-warp reduction: lane 0 of each warp -> shared -------------
    __shared__ float Bsh[NB_TILE];
    __shared__ float SHA[NT];

    for (int t = threadIdx.x; t < NB_TILE; t += blockDim.x) Bsh[t] = 0.f;
    for (int t = threadIdx.x; t < NT;      t += blockDim.x) SHA[t] = 0.f;
    __syncthreads();

    if ((threadIdx.x & 31) == 0) {
        for (int t = 0; t < NB_TILE; ++t) atomicAdd(&Bsh[t], rB[t]);
        for (int t = 0; t < NT; ++t)      atomicAdd(&SHA[t], rS[t]);
    }
    __syncthreads();

    // ---- Write completed B tile to global --------------------------------
    {
        const size_t boff = (size_t)b * NB_TILE;
        for (int t = threadIdx.x; t < NB_TILE; t += blockDim.x)
            B[boff + t] = Bsh[t];
    }

    // ---- Phase 2: H_b = SHA - B B^T, atomicAdd into global M_sig --------
    for (int k = threadIdx.x; k < NT; k += blockDim.x) {
        int col = 0, row = 0, kk = k;
        #pragma unroll
        for (int c = 0; c < N_SRC; ++c) {
            int len = N_SRC - c;
            if (kk < len) { col = c; row = c + kk; break; }
            kk -= len;
        }

        float bbt = 0.f;
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
// EXPLICIT INSTANTIATIONS via X-macro.
// Each kernel's signature is written ONCE (in INST_CAP / INST_APPLY / INST_CHOL).
// The preprocessor sweeps the 1..MAX grid -- edit the bounds in NSRC_LIST and
// GRID_2D below, in ONE place, to change the supported range.
// ============================================================================

// ---- the N_SRC axis (1..20); reused by every kernel ----
#define NSRC_LIST(F, ...) \
    F(1,__VA_ARGS__)  F(2,__VA_ARGS__)  F(3,__VA_ARGS__)  F(4,__VA_ARGS__)  \
    F(5,__VA_ARGS__)  F(6,__VA_ARGS__)  F(7,__VA_ARGS__)  F(8,__VA_ARGS__)  \
    F(9,__VA_ARGS__)  F(10,__VA_ARGS__) F(11,__VA_ARGS__) F(12,__VA_ARGS__) \
    F(13,__VA_ARGS__) F(14,__VA_ARGS__) F(15,__VA_ARGS__) F(16,__VA_ARGS__) \
    F(17,__VA_ARGS__) F(18,__VA_ARGS__) F(19,__VA_ARGS__) F(20,__VA_ARGS__)

// ---- 2-D grid: for each N_EIG (1..20), sweep the whole N_SRC axis ----
#define EIG_ROW(NE, F) NSRC_LIST(F, NE)
#define GRID_2D(F) \
    EIG_ROW(1,F)  EIG_ROW(2,F)  EIG_ROW(3,F)  EIG_ROW(4,F)  \
    EIG_ROW(5,F)  EIG_ROW(6,F)  EIG_ROW(7,F)  EIG_ROW(8,F)  \
    EIG_ROW(9,F)  EIG_ROW(10,F) EIG_ROW(11,F) EIG_ROW(12,F) \
    EIG_ROW(13,F) EIG_ROW(14,F) EIG_ROW(15,F) EIG_ROW(16,F) \
    EIG_ROW(17,F) EIG_ROW(18,F) EIG_ROW(19,F) EIG_ROW(20,F)

// ---- per-kernel instantiation bodies (the ONLY place each signature lives) ----
#define INST_CAP(NS, NE) \
    template __global__ void cap_reduce<NS,NE>( \
        const float*,const float*,const float*,const int*,float*,float*,int);
#define INST_APPLY(NS, NE) \
    template __global__ void apply_sig_prime<NS,NE>( \
        const float*,const float*,const float*,const int*, \
        const float*,const float*,float*,int);
#define INST_CHOL(NS, _) \
    template __global__ void chol_inv_fused<NS>(const float*,float*);

GRID_2D(INST_CAP)        // N_SRC x N_EIG  cap_reduce specializations
GRID_2D(INST_APPLY)      // N_SRC x N_EIG  apply_sig_prime specializations
NSRC_LIST(INST_CHOL, 0)  // N_SRC          chol_inv_fused (N_EIG arg ignored)

#undef INST_CAP
#undef INST_APPLY
#undef INST_CHOL

// ============================================================================
// HOST DISPATCH (extern "C", ctypes-friendly).
// Function-pointer tables built with the same X-macros: index [n_src-1][n_eig-1].
// blockDim for the group kernels is GROUP_BLOCK (tune as needed).
// `stream` is a cudaStream_t passed as void* (CuPy stream.ptr); pass 0 for default.
// ============================================================================
#include <cuda_runtime.h>

#ifndef NSRC_MAX
#define NSRC_MAX 20
#endif
#ifndef NEIG_MAX
#define NEIG_MAX 20
#endif
#ifndef GROUP_BLOCK
#define GROUP_BLOCK 256
#endif

// kernel function-pointer typedefs (match the __global__ signatures exactly)
typedef void (*cap_fn)(const float*,const float*,const float*,const int*,float*,float*,int);
typedef void (*apply_fn)(const float*,const float*,const float*,const int*,
                         const float*,const float*,float*,int);
typedef void (*chol_fn)(const float*,float*);

// ---- table fillers (reuse the grid macros) ----
#define FILL_CAP(NS,NE)   cap_tab  [(NS)-1][(NE)-1] = &cap_reduce<NS,NE>;
#define FILL_APPLY(NS,NE) apply_tab[(NS)-1][(NE)-1] = &apply_sig_prime<NS,NE>;
#define FILL_CHOL(NS,_)   chol_tab [(NS)-1]         = &chol_inv_fused<NS>;

static cap_fn   cap_tab  [NSRC_MAX][NEIG_MAX];
static apply_fn apply_tab[NSRC_MAX][NEIG_MAX];
static chol_fn  chol_tab [NSRC_MAX];
static bool     tables_ready = false;

static void build_tables() {
    if (tables_ready) return;
    GRID_2D(FILL_CAP)
    GRID_2D(FILL_APPLY)
    NSRC_LIST(FILL_CHOL, 0)
    tables_ready = true;
}
#undef FILL_CAP
#undef FILL_APPLY
#undef FILL_CHOL

static inline bool in_range(int n_src, int n_eig) {
    return n_src >= 1 && n_src <= NSRC_MAX && n_eig >= 1 && n_eig <= NEIG_MAX;
}

extern "C" {

// returns 0 on success, nonzero on bad (n_src,n_eig)
int launch_cap_reduce(
    int n_src, int n_eig, int nb,
    const float* Sig, const float* Del_prime, const float* noise,
    const int* edges, float* B, float* M_sig, void* stream)
{
    if (!in_range(n_src, n_eig)) return 1;
    build_tables();
    cudaStream_t s = (cudaStream_t)stream;
    cap_tab[n_src-1][n_eig-1]<<<nb, GROUP_BLOCK, 0, s>>>(
        Sig, Del_prime, noise, edges, B, M_sig, nb);
    return 0;
}

int launch_chol_inv_fused(
    int n_src, const float* M_sig, float* L_inv, void* stream)
{
    if (n_src < 1 || n_src > NSRC_MAX) return 1;
    build_tables();
    cudaStream_t s = (cudaStream_t)stream;
    chol_tab[n_src-1]<<<1, 1, 0, s>>>(M_sig, L_inv);
    return 0;
}

int launch_apply_sig_prime(
    int n_src, int n_eig, int nb,
    const float* Sig, const float* Del_prime, const float* noise,
    const int* edges, const float* B, const float* L_inv,
    float* Sig_prime, void* stream)
{
    if (!in_range(n_src, n_eig)) return 1;
    build_tables();
    cudaStream_t s = (cudaStream_t)stream;
    apply_tab[n_src-1][n_eig-1]<<<nb, GROUP_BLOCK, 0, s>>>(
        Sig, Del_prime, noise, edges, B, L_inv, Sig_prime, nb);
    return 0;
}

} // extern "C"
