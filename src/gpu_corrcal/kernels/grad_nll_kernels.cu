// nvcc -Xcompiler -fPIC -shared -o grad_nll_kernels.so grad_nll_kernels.cu
//
// Gradient of the CorrCal negative log-likelihood (Pascua+ Sec. 4.2.2),
// everything after p = C^-1 d. GPU port of the body of
// corrcal.optimize.grad_nll from "q = ..." through the phase prior, with the
// C accumulate_gradient loop folded in.
//
// Inputs that already live in VRAM after apply_gains + inv on the gained copy:
//   diff_g, src_g : G Delta, G Sigma          (CovWorkspace.diff_g / src_g)
//   dbar, sbar    : Delta_bar, Sigma_bar      (whitened factors of C^-1)
//   R_del         : L_b^-T per group           (InvCovWorkspace.L_del)
//   B             : (G Sigma)_b^T Delta_bar_b  (InvCovWorkspace.B)
//   L_sig_inv     : L_s^-1                     (InvCovWorkspace.L_sig_inv)
//   p             : C^-1 d                     (sparse_cov_times_vec, IS_INV)
// and from the ungained model:
//   diff, src     : Delta, Sigma
//   noise         : N  (unchanged by gains; noise[2k] alone enters the trace
//                   prefactor, as in CorrCal)
// plus the data d.
//
// Math, per baseline k with rows r0 = 2k, r1 = 2k+1, antennas k1, k2:
//
//   q = (C - N) G^T p = Delta (G Delta)^T C^-1 d + Sigma (G Sigma)^T C^-1 d
//                     = Delta del_tmp[b]          + Sigma sig_tmp
//
//   s_k    = p_r q_r + p_i q_i,       t_k = p_i q_r - p_r q_i
//   Pbar_k = sum_e (dbar[r0,e]^2 + dbar[r1,e]^2) + sum_c (sbar[r0,c]^2 + sbar[r1,c]^2)
//   G_k    = g_k1 conj(g_k2)
//   pf_k   = 2 N[r0] Pbar_k / |G_k|^2
//   four contributions (k1 Re, k1 Im, k2 Re, k2 Im), identical to c_funcs.c
//
// Projections: why not (G Sigma)^T p directly
// -------------------------------------------
// C^-1 nearly annihilates the source subspace: with N^-1 up to ~6e5 and ~3e5
// rows, (G Sigma)^T C^-1 d is ~1e6 times smaller than its individual terms.
// float32 p carries ~1e-5 relative error, so a direct reduction against p
// returns no correct digits (300% error on the test grid, independent of the
// accumulation precision). The same cancellation hits (G Delta)_b^T p for
// strongly constrained groups.
//
// The Woodbury factors from inv() give both projections without it. With
// M = N + (G Delta)(G Delta)^T, Lam = (G Sigma)^T M^-1 (G Sigma),
// L_s L_s^T = I + Lam, T_b = (G Delta)_b^T N^-1 (G Delta)_b, L_b L_b^T = I + T_b:
//
//   u        = (G Sigma)^T M^-1 d = sum_b [ (G Sigma)_b^T N^-1 d_b - B_b (Delta_bar_b^T d_b) ]
//   sig_tmp  = (G Sigma)^T C^-1 d = (I + Lam)^-1 u = L_s^-T L_s^-1 u
//   r        = d - (G Sigma) sig_tmp
//   del_tmp_b = (G Delta)_b^T C^-1 d = (I + T_b)^-1 (G Delta)_b^T N^-1 r_b
//                                    = R_b R_b^T (G Delta)_b^T N^-1 r_b
//
// Every reduction here sums terms of one sign scale, and the shrinkage is
// applied afterwards by the small triangular factors. Float32 emulation on
// the 18x32 test grid: 4.9e-6 relative gradient error, vs 2.9e-2 direct.
//
// Pipeline (one launcher, four kernels, one stream):
//
//   K1a grad_project_u    grid = nb, one redundant group per block
//        per row: Delta_bar^T d (block-local), (G Sigma)^T N^-1 d
//        block: u_b = (G Sigma)_b^T N^-1 d_b - B_b xd_b  -> atomicAdd (double) into u
//        phs_sum = sum_a atan2(g_i, g_r) (if prior on)
//      u over ALL rows is a global barrier, hence the separate K1b.
//
//   K1b grad_project_del  grid = nb, one redundant group per block
//        thread 0: sig_tmp = L_s^-T L_s^-1 u (double, in shared); block 0 stores it
//        per row: y_b += (G Delta)^T (d - (G Sigma) sig_tmp) / N
//        thread 0: del_tmp[b] = R_b R_b^T y_b
//
//   K2 grad_baseline      grid = nb, one redundant group per block
//        del_tmp[b], sig_tmp staged in shared (as sparse_cov_apply)
//        per baseline: q, s, t, Pbar, G_k, pf -> contrib[k] = float4
//      Every auxiliary vector of the CPU version (G^T p, q, s, t, inv_power)
//      exists only in registers. Only contrib (16 B per baseline) is written.
//
//   K3 grad_gather        one warp per antenna
//        out[a] = sum of that antenna's contributions, via a CSR of
//        (baseline, role) entries built once per solve from ant_1 / ant_2
//        lane-strided loads, warp_sum by __shfl_down_sync, lane 0 adds the
//        phase-prior term, multiplies by 1/scale and stores.
//
// Why a gather instead of atomics in K2: within a redundant group each antenna
// appears at most once as ant_1 and at most once as ant_2, so the 32 lanes of
// a warp in K2 target (almost) 32 distinct antennas and warp aggregation
// (__match_any_sync + shuffles) has nothing to merge. Every contribution would
// reach global memory as its own atomic, and float atomics make the gradient
// order-dependent across calls, which a line search does not like. The gather
// has no atomics, is bitwise reproducible, and the shuffle reduction runs over
// lanes that genuinely share a target. Cost: one 16 B/baseline write + read.
//
// Accumulation in K3 is double: each antenna sums ~n_ant terms whose
// chi^2 and trace parts cancel near the optimum, and K3 is tiny (n_ant warps).
//
// Layout:
//   diff, diff_g, dbar : (2 n_bl, N_EIG)  row-major float32
//   src, src_g, sbar   : (2 n_bl, N_SRC)  row-major float32
//   noise, p, data     : (2 n_bl,)
//   R_del              : (nb, N_EIG, N_EIG) row-major, R_b = L_b^-T (upper tri)
//   B                  : (nb, N_SRC, N_EIG)
//   L_sig_inv          : (N_SRC, N_SRC) row-major, lower tri
//   gains              : (2 n_ant,)  Re/Im alternating, ALREADY divided by scale
//   ant_1, ant_2       : (n_bl,) int32
//   edges              : (nb+1,) int32, Re/Im-split (all even)
//   u_acc              : (N_SRC,) double  workspace, zeroed by the launcher
//   phs_sum            : (1,)             workspace, zeroed by the launcher
//   del_tmp            : (nb, N_EIG)      workspace, fully overwritten
//   sig_tmp            : (N_SRC,)         workspace, fully overwritten (K1b block 0)
//   contrib            : (n_bl, 4)        workspace, fully overwritten
//   csr_ptr            : (n_ant+1,) int32
//   csr_idx            : (2 n_bl,) int32  entry e = 2k + role, role 0 = ant_1, 1 = ant_2;
//                                         contrib viewed as float2[2 n_bl] is indexed by e
//   out                : (2 n_ant,)       gradient, fully overwritten

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#ifndef NSRC_MAX
#define NSRC_MAX 20
#endif
#ifndef NEIG_MAX
#define NEIG_MAX 20
#endif
#define TPB 128         // group kernels; multiple of 32
#define GATHER_TPB 128  // 4 antennas per block

// ============================================================================
// Device helpers
// ============================================================================
__device__ __forceinline__ float warp_sum(float val) {
    val += __shfl_down_sync(0xffffffffu, val, 16);
    val += __shfl_down_sync(0xffffffffu, val, 8);
    val += __shfl_down_sync(0xffffffffu, val, 4);
    val += __shfl_down_sync(0xffffffffu, val, 2);
    val += __shfl_down_sync(0xffffffffu, val, 1);
    return val;
}

__device__ __forceinline__ double warp_sum_d(double val) {
    val += __shfl_down_sync(0xffffffffu, val, 16);
    val += __shfl_down_sync(0xffffffffu, val, 8);
    val += __shfl_down_sync(0xffffffffu, val, 4);
    val += __shfl_down_sync(0xffffffffu, val, 2);
    val += __shfl_down_sync(0xffffffffu, val, 1);
    return val;
}

// ============================================================================
// K1a: u = (G Sigma)^T M^-1 d, accumulated in double; phase sum.
// ============================================================================
template <int N_SRC, int N_EIG>
__global__ void grad_project_u(
    const float* __restrict__ src_g,    // (n_row, N_SRC)  G Sigma
    const float* __restrict__ dbar,     // (n_row, N_EIG)  Delta_bar
    const float* __restrict__ noise,    // (n_row,)
    const float* __restrict__ data,     // (n_row,)
    const float* __restrict__ B,        // (nb, N_SRC, N_EIG)
    const int*   __restrict__ edges,    // (nb+1,)
    const float* __restrict__ gains,    // (2 n_ant,)
    double*      __restrict__ u_acc,    // (N_SRC,)  atomic accumulator
    float*       __restrict__ phs_sum,  // (1,)      atomic accumulator
    int nb,
    int n_ant,
    int do_phs
){
    const int b = blockIdx.x;
    if (b >= nb) return;
    const int start = edges[b];
    const int stop  = edges[b + 1];

    float xd[N_EIG];   // Delta_bar_b^T d_b
    float us[N_SRC];   // (G Sigma)_b^T N^-1 d_b
    #pragma unroll
    for (int e = 0; e < N_EIG; ++e) xd[e] = 0.0f;
    #pragma unroll
    for (int c = 0; c < N_SRC; ++c) us[c] = 0.0f;

    for (int row = start + threadIdx.x; row < stop; row += blockDim.x) {
        const float v  = data[row];
        const float vn = v / noise[row];
        #pragma unroll
        for (int e = 0; e < N_EIG; ++e)
            xd[e] = fmaf(dbar[(size_t)row * N_EIG + e], v, xd[e]);
        #pragma unroll
        for (int c = 0; c < N_SRC; ++c)
            us[c] = fmaf(src_g[(size_t)row * N_SRC + c], vn, us[c]);
    }

    #pragma unroll
    for (int e = 0; e < N_EIG; ++e) xd[e] = warp_sum(xd[e]);
    #pragma unroll
    for (int c = 0; c < N_SRC; ++c) us[c] = warp_sum(us[c]);

    __shared__ float xd_red[N_EIG];
    __shared__ float us_red[N_SRC];
    for (int e = threadIdx.x; e < N_EIG; e += blockDim.x) xd_red[e] = 0.0f;
    for (int c = threadIdx.x; c < N_SRC; c += blockDim.x) us_red[c] = 0.0f;
    __syncthreads();

    if ((threadIdx.x & 31) == 0) {
        #pragma unroll
        for (int e = 0; e < N_EIG; ++e) atomicAdd(&xd_red[e], xd[e]);
        #pragma unroll
        for (int c = 0; c < N_SRC; ++c) atomicAdd(&us_red[c], us[c]);
    }
    __syncthreads();

    // u_b[s] = us[s] - sum_e B_b[s,e] xd[e]
    const size_t boff = (size_t)b * N_SRC * N_EIG;
    for (int c = threadIdx.x; c < N_SRC; c += blockDim.x) {
        double val = (double)us_red[c];
        #pragma unroll
        for (int e = 0; e < N_EIG; ++e)
            val -= (double)B[boff + c * N_EIG + e] * (double)xd_red[e];
        atomicAdd(&u_acc[c], val);
    }

    // Phase sum. do_phs is uniform across the grid, so every lane of every
    // warp reaches the shuffles.
    if (do_phs) {
        float ph = 0.0f;
        for (int a = blockIdx.x * blockDim.x + threadIdx.x; a < n_ant;
             a += gridDim.x * blockDim.x)
            ph += atan2f(gains[2 * a + 1], gains[2 * a]);
        ph = warp_sum(ph);
        if ((threadIdx.x & 31) == 0 && ph != 0.0f) atomicAdd(phs_sum, ph);
    }
}

// ============================================================================
// K1b: sig_tmp = (I + Lam)^-1 u;  del_tmp[b] = (I + T_b)^-1 (G Delta)_b^T N^-1 r_b
// ============================================================================
template <int N_SRC, int N_EIG>
__global__ void grad_project_del(
    const float*  __restrict__ diff_g,     // (n_row, N_EIG)  G Delta
    const float*  __restrict__ src_g,      // (n_row, N_SRC)  G Sigma
    const float*  __restrict__ noise,      // (n_row,)
    const float*  __restrict__ data,       // (n_row,)
    const float*  __restrict__ R_del,      // (nb, N_EIG, N_EIG)  L_b^-T
    const float*  __restrict__ L_sig_inv,  // (N_SRC, N_SRC)      L_s^-1, lower tri
    const double* __restrict__ u_acc,      // (N_SRC,)            from K1a
    const int*    __restrict__ edges,      // (nb+1,)
    float*        __restrict__ sig_tmp,    // (N_SRC,)            OUTPUT (block 0)
    float*        __restrict__ del_tmp,    // (nb, N_EIG)         OUTPUT
    int nb
){
    const int b = blockIdx.x;
    if (b >= nb) return;
    const int start = edges[b];
    const int stop  = edges[b + 1];

    __shared__ float w[N_SRC];
    __shared__ float y_red[N_EIG];

    // sig_tmp = L_s^-T (L_s^-1 u). N_SRC^2 FMAs, recomputed per block in
    // double so every block sees the identical w without another launch.
    if (threadIdx.x == 0) {
        double v[N_SRC];
        #pragma unroll
        for (int i = 0; i < N_SRC; ++i) {
            double acc = 0.0;
            for (int j = 0; j <= i; ++j)
                acc += (double)L_sig_inv[i * N_SRC + j] * u_acc[j];
            v[i] = acc;
        }
        #pragma unroll
        for (int i = 0; i < N_SRC; ++i) {
            double acc = 0.0;
            for (int j = i; j < N_SRC; ++j)
                acc += (double)L_sig_inv[j * N_SRC + i] * v[j];
            w[i] = (float)acc;
        }
    }
    for (int e = threadIdx.x; e < N_EIG; e += blockDim.x) y_red[e] = 0.0f;
    __syncthreads();

    if (b == 0)
        for (int c = threadIdx.x; c < N_SRC; c += blockDim.x) sig_tmp[c] = w[c];

    // y_b = (G Delta)_b^T N^-1 (d - (G Sigma) w)_b
    float y[N_EIG];
    #pragma unroll
    for (int e = 0; e < N_EIG; ++e) y[e] = 0.0f;

    for (int row = start + threadIdx.x; row < stop; row += blockDim.x) {
        float r = data[row];
        #pragma unroll
        for (int c = 0; c < N_SRC; ++c)
            r = fmaf(-src_g[(size_t)row * N_SRC + c], w[c], r);
        r /= noise[row];
        #pragma unroll
        for (int e = 0; e < N_EIG; ++e)
            y[e] = fmaf(diff_g[(size_t)row * N_EIG + e], r, y[e]);
    }

    #pragma unroll
    for (int e = 0; e < N_EIG; ++e) y[e] = warp_sum(y[e]);
    if ((threadIdx.x & 31) == 0) {
        #pragma unroll
        for (int e = 0; e < N_EIG; ++e) atomicAdd(&y_red[e], y[e]);
    }
    __syncthreads();

    // del_tmp[b] = R (R^T y),  R = L_b^-T upper triangular
    if (threadIdx.x == 0) {
        const float* R = R_del + (size_t)b * N_EIG * N_EIG;
        float z[N_EIG];
        #pragma unroll
        for (int i = 0; i < N_EIG; ++i) {
            float acc = 0.0f;
            for (int j = 0; j <= i; ++j)                 // R[j,i] nonzero for j <= i
                acc = fmaf(R[j * N_EIG + i], y_red[j], acc);
            z[i] = acc;
        }
        #pragma unroll
        for (int i = 0; i < N_EIG; ++i) {
            float acc = 0.0f;
            for (int j = i; j < N_EIG; ++j)              // R[i,j] nonzero for j >= i
                acc = fmaf(R[i * N_EIG + j], z[j], acc);
            del_tmp[(size_t)b * N_EIG + i] = acc;
        }
    }
}

// ============================================================================
// K2: per-baseline contributions. All CPU auxiliary vectors stay in registers.
// ============================================================================
template <int N_SRC, int N_EIG>
__global__ void grad_baseline(
    const float* __restrict__ diff,     // (n_row, N_EIG)  Delta (ungained)
    const float* __restrict__ src,      // (n_row, N_SRC)  Sigma (ungained)
    const float* __restrict__ dbar,     // (n_row, N_EIG)  Delta_bar
    const float* __restrict__ sbar,     // (n_row, N_SRC)  Sigma_bar
    const float* __restrict__ noise,    // (n_row,)        N (ungained)
    const float* __restrict__ p,        // (n_row,)
    const float* __restrict__ gains,    // (2 n_ant,)
    const int*   __restrict__ ant_1,    // (n_bl,)
    const int*   __restrict__ ant_2,    // (n_bl,)
    const float* __restrict__ del_tmp,  // (nb, N_EIG)     from K1
    const float* __restrict__ sig_tmp,  // (N_SRC,)        from K1
    const int*   __restrict__ edges,    // (nb+1,)
    float4*      __restrict__ contrib,  // (n_bl,)         OUTPUT
    int nb
){
    const int b = blockIdx.x;
    if (b >= nb) return;
    const int k_start = edges[b] >> 1;
    const int k_stop  = edges[b + 1] >> 1;

    __shared__ float dt[N_EIG];
    __shared__ float st[N_SRC];
    for (int e = threadIdx.x; e < N_EIG; e += blockDim.x) dt[e] = del_tmp[(size_t)b * N_EIG + e];
    for (int c = threadIdx.x; c < N_SRC; c += blockDim.x) st[c] = sig_tmp[c];
    __syncthreads();

    for (int k = k_start + threadIdx.x; k < k_stop; k += blockDim.x) {
        const size_t r0 = 2 * (size_t)k;
        const size_t r1 = r0 + 1;

        // q = (C - N) G^T p on this Re/Im pair, and the inverse "power" Pbar.
        float q0 = 0.0f, q1 = 0.0f, pbar = 0.0f;
        #pragma unroll
        for (int e = 0; e < N_EIG; ++e) {
            q0 = fmaf(diff[r0 * N_EIG + e], dt[e], q0);
            q1 = fmaf(diff[r1 * N_EIG + e], dt[e], q1);
            const float y0 = dbar[r0 * N_EIG + e];
            const float y1 = dbar[r1 * N_EIG + e];
            pbar = fmaf(y0, y0, fmaf(y1, y1, pbar));
        }
        #pragma unroll
        for (int c = 0; c < N_SRC; ++c) {
            q0 = fmaf(src[r0 * N_SRC + c], st[c], q0);
            q1 = fmaf(src[r1 * N_SRC + c], st[c], q1);
            const float y0 = sbar[r0 * N_SRC + c];
            const float y1 = sbar[r1 * N_SRC + c];
            pbar = fmaf(y0, y0, fmaf(y1, y1, pbar));
        }

        const float p0 = p[r0];
        const float p1 = p[r1];
        const float S = fmaf(p0, q0,  p1 * q1);   // Re(conj(q) p)
        const float T = fmaf(p1, q0, -p0 * q1);   // Im(conj(q) p)

        const int i1 = __ldg(&ant_1[k]);
        const int i2 = __ldg(&ant_2[k]);
        const float g1r = __ldg(&gains[2 * i1]);
        const float g1i = __ldg(&gains[2 * i1 + 1]);
        const float g2r = __ldg(&gains[2 * i2]);
        const float g2i = __ldg(&gains[2 * i2 + 1]);

        // G_k = g1 conj(g2)
        const float Gr = fmaf(g1r, g2r,  g1i * g2i);
        const float Gi = fmaf(g1i, g2r, -g1r * g2i);
        const float pf = 2.0f * noise[r0] * pbar / fmaf(Gr, Gr, Gi * Gi);

        float4 c;
        c.x = -2.0f * fmaf(g2r, S, -g2i * T) + pf * fmaf(Gr, g2r, -Gi * g2i);  // k1 Re
        c.y = -2.0f * fmaf(g2i, S,  g2r * T) + pf * fmaf(Gr, g2i,  Gi * g2r);  // k1 Im
        c.z = -2.0f * fmaf(g1r, S,  g1i * T) + pf * fmaf(Gr, g1r,  Gi * g1i);  // k2 Re
        c.w = -2.0f * fmaf(g1i, S, -g1r * T) + pf * fmaf(Gr, g1i, -Gi * g1r);  // k2 Im
        contrib[k] = c;
    }
}

// ============================================================================
// K3: per-antenna gather. One warp per antenna, no atomics.
// ============================================================================
__global__ void grad_gather(
    const float2* __restrict__ contrib2,  // (2 n_bl,)  contrib viewed as (Re, Im) pairs
    const int*    __restrict__ csr_ptr,   // (n_ant+1,)
    const int*    __restrict__ csr_idx,   // (2 n_bl,)
    const float*  __restrict__ gains,     // (2 n_ant,)
    const float*  __restrict__ phs_sum,   // (1,)
    float*        __restrict__ out,       // (2 n_ant,)
    int n_ant,
    float phs_coef,                       // 2 / (n_ant^2 phs_norm_fac^2), 0 disables
    float inv_scale
){
    const int a    = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (a >= n_ant) return;               // uniform per warp

    const int lo = csr_ptr[a];
    const int hi = csr_ptr[a + 1];

    double acc_r = 0.0, acc_i = 0.0;
    for (int e = lo + lane; e < hi; e += 32) {
        const float2 c = contrib2[csr_idx[e]];
        acc_r += (double)c.x;
        acc_i += (double)c.y;
    }
    acc_r = warp_sum_d(acc_r);
    acc_i = warp_sum_d(acc_i);

    if (lane == 0) {
        if (phs_coef != 0.0f) {
            // CorrCal: prefac = 2 sum(phi) / (|g| n^2 sigma^2);
            //          Re -= prefac sin(phi), Im += prefac cos(phi)
            // with sin(phi) = g_i/|g|, cos(phi) = g_r/|g|.
            const double gr = gains[2 * a];
            const double gi = gains[2 * a + 1];
            const double pref = (double)phs_coef * (double)(*phs_sum) / (gr * gr + gi * gi);
            acc_r -= pref * gi;
            acc_i += pref * gr;
        }
        out[2 * a]     = (float)(acc_r * inv_scale);
        out[2 * a + 1] = (float)(acc_i * inv_scale);
    }
}

// ============================================================================
// Explicit instantiations (X-macro grid, as sig_prime_kernels_pta.cu)
// ============================================================================
#define NSRC_LIST(F, ...) \
    F(1,__VA_ARGS__)  F(2,__VA_ARGS__)  F(3,__VA_ARGS__)  F(4,__VA_ARGS__)  \
    F(5,__VA_ARGS__)  F(6,__VA_ARGS__)  F(7,__VA_ARGS__)  F(8,__VA_ARGS__)  \
    F(9,__VA_ARGS__)  F(10,__VA_ARGS__) F(11,__VA_ARGS__) F(12,__VA_ARGS__) \
    F(13,__VA_ARGS__) F(14,__VA_ARGS__) F(15,__VA_ARGS__) F(16,__VA_ARGS__) \
    F(17,__VA_ARGS__) F(18,__VA_ARGS__) F(19,__VA_ARGS__) F(20,__VA_ARGS__)

#define EIG_ROW(NE, F) NSRC_LIST(F, NE)
#define GRID_2D(F) \
    EIG_ROW(1,F)  EIG_ROW(2,F)  EIG_ROW(3,F)  EIG_ROW(4,F)  \
    EIG_ROW(5,F)  EIG_ROW(6,F)  EIG_ROW(7,F)  EIG_ROW(8,F)  \
    EIG_ROW(9,F)  EIG_ROW(10,F) EIG_ROW(11,F) EIG_ROW(12,F) \
    EIG_ROW(13,F) EIG_ROW(14,F) EIG_ROW(15,F) EIG_ROW(16,F) \
    EIG_ROW(17,F) EIG_ROW(18,F) EIG_ROW(19,F) EIG_ROW(20,F)

#define INST_U(NS, NE) \
    template __global__ void grad_project_u<NS,NE>( \
        const float*,const float*,const float*,const float*,const float*, \
        const int*,const float*,double*,float*,int,int,int);
#define INST_DEL(NS, NE) \
    template __global__ void grad_project_del<NS,NE>( \
        const float*,const float*,const float*,const float*,const float*, \
        const float*,const double*,const int*,float*,float*,int);
#define INST_BASELINE(NS, NE) \
    template __global__ void grad_baseline<NS,NE>( \
        const float*,const float*,const float*,const float*,const float*, \
        const float*,const float*,const int*,const int*,const float*, \
        const float*,const int*,float4*,int);

GRID_2D(INST_U)
GRID_2D(INST_DEL)
GRID_2D(INST_BASELINE)

#undef INST_U
#undef INST_DEL
#undef INST_BASELINE

// ============================================================================
// Host dispatch (extern "C", ctypes). Tables indexed [n_src-1][n_eig-1].
// ============================================================================
typedef void (*u_fn)(const float*,const float*,const float*,const float*,const float*,
                     const int*,const float*,double*,float*,int,int,int);
typedef void (*del_fn)(const float*,const float*,const float*,const float*,const float*,
                       const float*,const double*,const int*,float*,float*,int);
typedef void (*baseline_fn)(const float*,const float*,const float*,const float*,const float*,
                            const float*,const float*,const int*,const int*,const float*,
                            const float*,const int*,float4*,int);

#define FILL_U(NS,NE)        u_tab       [(NS)-1][(NE)-1] = &grad_project_u<NS,NE>;
#define FILL_DEL(NS,NE)      del_tab     [(NS)-1][(NE)-1] = &grad_project_del<NS,NE>;
#define FILL_BASELINE(NS,NE) baseline_tab[(NS)-1][(NE)-1] = &grad_baseline<NS,NE>;

static u_fn        u_tab       [NSRC_MAX][NEIG_MAX];
static del_fn      del_tab     [NSRC_MAX][NEIG_MAX];
static baseline_fn baseline_tab[NSRC_MAX][NEIG_MAX];
static bool        tables_ready = false;

static void build_tables() {
    if (tables_ready) return;
    GRID_2D(FILL_U)
    GRID_2D(FILL_DEL)
    GRID_2D(FILL_BASELINE)
    tables_ready = true;
}
#undef FILL_U
#undef FILL_DEL
#undef FILL_BASELINE

extern "C" {

// Returns 0 on success, 1 on (n_src, n_eig) outside the compiled grid.
// gains must already be divided by scale; inv_scale = 1/scale is applied to
// the output, matching "return gradient / scale".
int launch_grad_nll(
    int n_src, int n_eig, int nb, int n_bl, int n_ant,
    const float* diff_g, const float* src_g,
    const float* diff,   const float* src,
    const float* dbar,   const float* sbar,
    const float* R_del,  const float* B,   const float* L_sig_inv,
    const float* noise,  const float* p,   const float* data, const float* gains,
    const int* ant_1,    const int* ant_2, const int* edges,
    const int* csr_ptr,  const int* csr_idx,
    double* u_acc, float* phs_sum, float* del_tmp, float* sig_tmp, float* contrib,
    float* out,
    float phs_coef, float inv_scale,
    void* stream)
{
    if (n_src < 1 || n_src > NSRC_MAX || n_eig < 1 || n_eig > NEIG_MAX) return 1;
    build_tables();
    cudaStream_t s = (cudaStream_t)stream;

    // Atomic accumulators: zeroed here so a CG iteration cannot forget them.
    cudaMemsetAsync(u_acc, 0, (size_t)n_src * sizeof(double), s);
    cudaMemsetAsync(phs_sum, 0, sizeof(float), s);

    const int do_phs = (phs_coef != 0.0f);

    u_tab[n_src-1][n_eig-1]<<<nb, TPB, 0, s>>>(
        src_g, dbar, noise, data, B, edges, gains, u_acc, phs_sum, nb, n_ant, do_phs);

    del_tab[n_src-1][n_eig-1]<<<nb, TPB, 0, s>>>(
        diff_g, src_g, noise, data, R_del, L_sig_inv, u_acc, edges, sig_tmp, del_tmp, nb);

    baseline_tab[n_src-1][n_eig-1]<<<nb, TPB, 0, s>>>(
        diff, src, dbar, sbar, noise, p, gains, ant_1, ant_2,
        del_tmp, sig_tmp, edges, (float4*)contrib, nb);

    const int warps_per_block = GATHER_TPB / 32;
    const int grid = (n_ant + warps_per_block - 1) / warps_per_block;
    grad_gather<<<grid, GATHER_TPB, 0, s>>>(
        (const float2*)contrib, csr_ptr, csr_idx, gains, phs_sum, out,
        n_ant, phs_coef, inv_scale);

    return 0;
}

} // extern "C"
