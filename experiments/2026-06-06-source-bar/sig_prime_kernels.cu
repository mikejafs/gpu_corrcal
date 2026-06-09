// sig_prime_kernels.cu
// Second-level Woodbury: fold Sigma Sigma^H onto (N + Delta Delta^H)^{-1}.
//
// Assumes Del_prime is the WHITENED diffuse factor, i.e.
//     (N + Delta Delta^H)^{-1} = N^{-1} - Del_prime Del_prime^H.
//
// Ragged CSR layout via `edges` (NO zero-padding; padding exists only in the
// CuPy reference). Block b owns flat rows edges[b]:edges[b+1] = the true
// redundant-group size. One redundant group per CUDA block.
//
// Shapes (row-major):
//   Sig        : (n_total, N_SRC)   complex   -- source factor, baseline rows
//   Del_prime  : (n_total, N_EIG)   complex   -- whitened diffuse factor
//   noise      : (n_total,)         real      -- N (reciprocal in-register)
//   edges      : (nb+1,)            int       -- CSR block boundaries
//   M_sig      : (N_SRC, N_SRC)     complex   -- single capacitance, reduced over ALL blocks
//   L_inv      : (N_SRC, N_SRC)     complex   -- lower-tri inverse of chol(I+M_sig)
//   Sig_prime  : (n_total, N_SRC)   complex   -- OUTPUT (only large write)
//
// Per block b (contraction over rows i in edges[b]:edges[b+1]):
//   A[i,s] = Sig[i,s] / N[i]                         (in-register, never stored)
//   B[b][s,e] = sum_{i in b} conj(Sig[i,s]) Del'[i,e]   (N_SRC x N_EIG shared tile)
//   M_sig += sum_{i in b} conj(Sig[i,s]) Sig[i,t]/N[i]  -  sum_e B[s,e] conj(B[t,e])
//   W[i,s] = A[i,s] - sum_e Del'[i,e] conj(B[b][s,e])
//   Sig_prime[i,s] = sum_{t<=s} W[i,t] conj(L_inv[s,t])
//
// NOTE: B B^H carries TWO independent row indices within a block (i and i'),
// so it does NOT factor to a per-row scalar. B must be formed as a per-block
// tile. Recomputed in kernel 3 rather than written to global.

#include <cuComplex.h>

// ---- complex helpers (CUDA C has no operator overloads) --------------------
__device__ __forceinline__ cuFloatComplex cax(cuFloatComplex a, cuFloatComplex b) { return cuCaddf(a, b); }
__device__ __forceinline__ cuFloatComplex csu(cuFloatComplex a, cuFloatComplex b) { return cuCsubf(a, b); }
__device__ __forceinline__ cuFloatComplex cmu(cuFloatComplex a, cuFloatComplex b) { return cuCmulf(a, b); }
__device__ __forceinline__ cuFloatComplex ccj(cuFloatComplex a)                   { return cuConjf(a); }
__device__ __forceinline__ cuFloatComplex czero()                                 { return make_cuFloatComplex(0.f, 0.f); }
__device__ __forceinline__ cuFloatComplex crscale(cuFloatComplex a, float s)      { return make_cuFloatComplex(cuCrealf(a)*s, cuCimagf(a)*s); }

// ============================================================================
// KERNEL 1: capacitance reduction  (one redundant group per CUDA block)
//
//   For block b, over rows i in edges[b]:edges[b+1]:
//     B[s,e]   = sum_i conj(Sig[i,s]) Del'[i,e]          (N_SRC x N_EIG)
//     SHA[s,t] = sum_i conj(Sig[i,s]) Sig[i,t] / N[i]    (Sig^H A, lower tri)
//   then contribute  H_b[s,t] = SHA[s,t] - sum_e B[s,e] conj(B[t,e])
//   via atomicAdd into the single (N_SRC x N_SRC) M_sig (lower tri).
//   Across all blocks this realizes  M_sig = sum_b H_b  (== xp.sum(axis=0)).
//   Identity is added in kernel 2.
//
//   B B^H carries two independent row indices i,i' within the block, so B
//   must be formed fully (per-block shared tile) before the B B^H contraction.
//   Threads stride over the block's rows; partials reduce through shared mem.
//
//   Layout note: this matches the CuPy reference's zero-padded (nb,lb,.) @ /
//   sum(axis=0) exactly, because the only rows that differ are CuPy's zero
//   pad rows, which contribute nothing -- identical to summing real rows only.
// ============================================================================
template <int N_SRC, int N_EIG>
__global__ void cap_reduce(
    const cuFloatComplex* __restrict__ Sig,        // (n_total, N_SRC)
    const cuFloatComplex* __restrict__ Del_prime,  // (n_total, N_EIG)
    const float*          __restrict__ noise,      // (n_total,)
    const int*            __restrict__ edges,      // (nb+1,)
    cuFloatComplex*       __restrict__ M_sig,      // (N_SRC, N_SRC) lower tri accumulated
    int nb
){
    const int b = blockIdx.x;
    if (b >= nb) return;
    const int start = edges[b];
    const int stop  = edges[b + 1];

    constexpr int NB_TILE = N_SRC * N_EIG;             // B tile size
    constexpr int NT      = N_SRC * (N_SRC + 1) / 2;   // lower-tri SHA size

    // shared accumulators for this block (single warp's worth reduced here)
    __shared__ cuFloatComplex Bsh[NB_TILE];
    __shared__ cuFloatComplex SHA[NT];

    for (int t = threadIdx.x; t < NB_TILE; t += blockDim.x) Bsh[t] = czero();
    for (int t = threadIdx.x; t < NT;      t += blockDim.x) SHA[t] = czero();
    __syncthreads();

    // ---- Phase 1: accumulate B and SHA over the block's rows -------------
    // Each thread holds register partials, strides over rows, then we push
    // into shared via per-lane warp reduction + atomic-in-shared.
    // With one group per block and small tiles, accumulate directly into
    // shared using float atomics (deterministic result under allclose).
    for (int i = start + threadIdx.x; i < stop; i += blockDim.x) {
        const float invN = 1.0f / noise[i];           // reciprocal fusion

        cuFloatComplex s[N_SRC];
        #pragma unroll
        for (int c = 0; c < N_SRC; ++c) s[c] = Sig[(size_t)i * N_SRC + c];

        cuFloatComplex d[N_EIG];
        #pragma unroll
        for (int e = 0; e < N_EIG; ++e) d[e] = Del_prime[(size_t)i * N_EIG + e];

        // B[s,e] += conj(Sig[i,s]) Del'[i,e]
        #pragma unroll
        for (int sidx = 0; sidx < N_SRC; ++sidx) {
            cuFloatComplex cs = ccj(s[sidx]);
            #pragma unroll
            for (int e = 0; e < N_EIG; ++e) {
                cuFloatComplex v = cmu(cs, d[e]);
                cuFloatComplex* dst = &Bsh[sidx * N_EIG + e];
                atomicAdd(&(((float*)dst)[0]), cuCrealf(v));
                atomicAdd(&(((float*)dst)[1]), cuCimagf(v));
            }
        }

        // SHA[s,t] += conj(Sig[i,s]) Sig[i,t] * invN   (lower tri s>=t)
        int k = 0;
        #pragma unroll
        for (int col = 0; col < N_SRC; ++col) {        // t
            #pragma unroll
            for (int row = col; row < N_SRC; ++row) {  // s
                cuFloatComplex v = crscale(cmu(ccj(s[row]), s[col]), invN);
                cuFloatComplex* dst = &SHA[k];
                atomicAdd(&(((float*)dst)[0]), cuCrealf(v));
                atomicAdd(&(((float*)dst)[1]), cuCimagf(v));
                ++k;
            }
        }
    }
    __syncthreads();

    // ---- Phase 2: H_b = SHA - B B^H, atomicAdd into global M_sig ---------
    // Parallelize the lower-tri (s,t) entries across threads.
    for (int k = threadIdx.x; k < NT; k += blockDim.x) {
        // recover (row=s, col=t) from packed lower-tri index k
        // k enumerates col-major lower tri: for col in 0..N, row in col..N
        int col = 0, row = 0, kk = k;
        #pragma unroll
        for (int c = 0; c < N_SRC; ++c) {
            int len = N_SRC - c;
            if (kk < len) { col = c; row = c + kk; break; }
            kk -= len;
        }

        // (B B^H)[s,t] = sum_e B[s,e] conj(B[t,e])
        cuFloatComplex bbh = czero();
        #pragma unroll
        for (int e = 0; e < N_EIG; ++e)
            bbh = cax(bbh, cmu(Bsh[row * N_EIG + e], ccj(Bsh[col * N_EIG + e])));

        cuFloatComplex h = csu(SHA[k], bbh);

        cuFloatComplex* dst = &M_sig[(size_t)row * N_SRC + col];
        atomicAdd(&(((float*)dst)[0]), cuCrealf(h));
        atomicAdd(&(((float*)dst)[1]), cuCimagf(h));
    }
}

// ============================================================================
// KERNEL 2: fused Cholesky + triangular inverse on the single (N_SRC x N_SRC)
//   M_sig. One thread does it all (matrix is tiny). Adds Identity first.
//   Input M_sig holds lower-tri of H (strictly: H accumulated, upper untouched).
//   We treat M = I + H using only lower triangle (Hermitian).
//   Output L_sig_inv = (chol(M))^{-1}, lower triangular.
// ============================================================================
template <int N_SRC>
__global__ void chol_inv_fused(
    const cuFloatComplex* __restrict__ M_sig,   // (N_SRC,N_SRC) lower tri = H
    cuFloatComplex*       __restrict__ L_inv    // (N_SRC,N_SRC) output, lower tri
){
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    cuFloatComplex L[N_SRC * N_SRC];
    #pragma unroll
    for (int i = 0; i < N_SRC * N_SRC; ++i) L[i] = czero();

    // Cholesky: M = L L^H, M = I + H (Hermitian, lower tri stored)
    for (int j = 0; j < N_SRC; ++j) {
        // diagonal
        cuFloatComplex mjj = M_sig[(size_t)j * N_SRC + j];
        float diag = cuCrealf(mjj) + 1.0f;          // + I
        #pragma unroll
        for (int k = 0; k < N_SRC; ++k) {
            if (k < j) {
                cuFloatComplex ljk = L[j * N_SRC + k];
                diag -= cuCrealf(ljk)*cuCrealf(ljk) + cuCimagf(ljk)*cuCimagf(ljk);
            }
        }
        float ljj = sqrtf(diag);
        L[j * N_SRC + j] = make_cuFloatComplex(ljj, 0.f);
        float inv_ljj = 1.0f / ljj;

        // below-diagonal column j
        for (int i = j + 1; i < N_SRC; ++i) {
            cuFloatComplex mij = M_sig[(size_t)i * N_SRC + j];   // H[i,j], i>j: stored
            cuFloatComplex sum = mij;                            // no +I off-diagonal
            for (int k = 0; k < j; ++k) {
                cuFloatComplex lik = L[i * N_SRC + k];
                cuFloatComplex ljk = L[j * N_SRC + k];
                sum = csu(sum, cmu(lik, ccj(ljk)));
            }
            L[i * N_SRC + j] = crscale(sum, inv_ljj);
        }
    }

    // Triangular inverse (forward substitution), overwrite into L_inv.
    // Linv lower-tri: Linv[j,j] = 1/L[j,j]; Linv[i,j] = -L[i,i]^{-1} sum_{k=j}^{i-1} L[i,k] Linv[k,j]
    for (int j = 0; j < N_SRC; ++j) {
        float ljj = cuCrealf(L[j * N_SRC + j]);
        L_inv[(size_t)j * N_SRC + j] = make_cuFloatComplex(1.0f / ljj, 0.f);
        for (int i = j + 1; i < N_SRC; ++i) {
            cuFloatComplex sum = czero();
            for (int k = j; k < i; ++k)
                sum = cax(sum, cmu(L[i * N_SRC + k], L_inv[(size_t)k * N_SRC + j]));
            float inv_lii = 1.0f / cuCrealf(L[i * N_SRC + i]);
            L_inv[(size_t)i * N_SRC + j] = crscale(sum, -inv_lii);
        }
    }
    // zero strictly-upper of L_inv (output buffer may be cp.empty)
    for (int i = 0; i < N_SRC; ++i)
        for (int j = i + 1; j < N_SRC; ++j)
            L_inv[(size_t)i * N_SRC + j] = czero();
}

// ============================================================================
// KERNEL 3: apply  (one redundant group per CUDA block)
//
//   For block b: first rebuild B[s,e] = sum_{i in b} conj(Sig[i,s]) Del'[i,e]
//   (per-block shared tile, recomputed -- not read from global), then for each
//   real row i in edges[b]:edges[b+1]:
//     A[i,s] = Sig[i,s] / N[i]
//     W[i,s] = A[i,s] - sum_e Del'[i,e] conj(B[s,e])      (Del' B^H, per row)
//     Sig_prime[i,s] = sum_{t<=s} W[i,t] conj(L_inv[s,t]) (W L_sig^{-H})
//
//   B genuinely depends on the whole block (two row indices), so it is built
//   once per block in shared, then reused across all rows of the block. This
//   is the recompute (vs. returning B from kernel 1) we settled on: one extra
//   per-block contraction, no baseline-sized global round-trip.
// ============================================================================
template <int N_SRC, int N_EIG>
__global__ void apply_sig_prime(
    const cuFloatComplex* __restrict__ Sig,        // (n_total, N_SRC)
    const cuFloatComplex* __restrict__ Del_prime,  // (n_total, N_EIG)
    const float*          __restrict__ noise,      // (n_total,)
    const int*            __restrict__ edges,      // (nb+1,)
    const cuFloatComplex* __restrict__ L_inv,      // (N_SRC, N_SRC) lower-tri
    cuFloatComplex*       __restrict__ Sig_prime,  // (n_total, N_SRC) OUTPUT
    int nb
){
    const int b = blockIdx.x;
    if (b >= nb) return;
    const int start = edges[b];
    const int stop  = edges[b + 1];

    constexpr int NB_TILE = N_SRC * N_EIG;
    __shared__ cuFloatComplex Bsh[NB_TILE];
    __shared__ cuFloatComplex Ls[N_SRC * N_SRC];

    // cache L_inv (tiny) and zero B tile
    for (int t = threadIdx.x; t < N_SRC * N_SRC; t += blockDim.x) Ls[t] = L_inv[t];
    for (int t = threadIdx.x; t < NB_TILE;       t += blockDim.x) Bsh[t] = czero();
    __syncthreads();

    // ---- Phase 1: rebuild B for this block -------------------------------
    for (int i = start + threadIdx.x; i < stop; i += blockDim.x) {
        cuFloatComplex s[N_SRC];
        #pragma unroll
        for (int c = 0; c < N_SRC; ++c) s[c] = Sig[(size_t)i * N_SRC + c];
        cuFloatComplex d[N_EIG];
        #pragma unroll
        for (int e = 0; e < N_EIG; ++e) d[e] = Del_prime[(size_t)i * N_EIG + e];

        #pragma unroll
        for (int sidx = 0; sidx < N_SRC; ++sidx) {
            cuFloatComplex cs = ccj(s[sidx]);
            #pragma unroll
            for (int e = 0; e < N_EIG; ++e) {
                cuFloatComplex v = cmu(cs, d[e]);
                cuFloatComplex* dst = &Bsh[sidx * N_EIG + e];
                atomicAdd(&(((float*)dst)[0]), cuCrealf(v));
                atomicAdd(&(((float*)dst)[1]), cuCimagf(v));
            }
        }
    }
    __syncthreads();

    // ---- Phase 2: per-row W and triangular solve -------------------------
    for (int i = start + threadIdx.x; i < stop; i += blockDim.x) {
        const float invN = 1.0f / noise[i];

        cuFloatComplex s[N_SRC];
        #pragma unroll
        for (int c = 0; c < N_SRC; ++c) s[c] = Sig[(size_t)i * N_SRC + c];
        cuFloatComplex d[N_EIG];
        #pragma unroll
        for (int e = 0; e < N_EIG; ++e) d[e] = Del_prime[(size_t)i * N_EIG + e];

        // W[s] = A[i,s] - sum_e Del'[i,e] conj(B[s,e])
        cuFloatComplex W[N_SRC];
        #pragma unroll
        for (int sidx = 0; sidx < N_SRC; ++sidx) {
            cuFloatComplex w = crscale(s[sidx], invN);     // A[i,s]
            #pragma unroll
            for (int e = 0; e < N_EIG; ++e)
                w = csu(w, cmu(d[e], ccj(Bsh[sidx * N_EIG + e])));
            W[sidx] = w;
        }

        // Sig_prime[i,s] = sum_{t<=s} W[t] conj(L_inv[s,t])
        #pragma unroll
        for (int row = 0; row < N_SRC; ++row) {
            cuFloatComplex out = czero();
            #pragma unroll
            for (int col = 0; col <= row; ++col)
                out = cax(out, cmu(W[col], ccj(Ls[row * N_SRC + col])));
            Sig_prime[(size_t)i * N_SRC + row] = out;
        }
    }
}
