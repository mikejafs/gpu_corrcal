// nvcc -Xcompiler -fPIC -shared -o apply_gains.so apply_gains.cu
//
// out = G mat,  per-baseline complex gain applied to a Re/Im-split matrix.
// GPU port of corrcal.utils.apply_gains_to_mat.
//
// For baseline j with antennas p = ant_1[j], q = ant_2[j]:
//
//     g_j = g_p * conj(g_q)
//         = (gr_p gr_q + gi_p gi_q) + i (gi_p gr_q - gr_p gi_q)
//
// and for every column c, the Re/Im row pair (2j, 2j+1) is rotated by g_j:
//
//     out[2j,   c] = Re(g_j) mat[2j, c] - Im(g_j) mat[2j+1, c]
//     out[2j+1, c] = Im(g_j) mat[2j, c] + Re(g_j) mat[2j+1, c]
//
// Layout (flat, matching sparse_cov_times_vec.cu):
//   gains : (2 * n_ant,)          alternating Re/Im per antenna
//   ant_1 : (n_bl,)  int32        first antenna of each baseline
//   ant_2 : (n_bl,)  int32        second antenna of each baseline
//   mat   : (2 * n_bl, N_COL)     row-major, Re/Im alternating along rows
//   out   : (2 * n_bl, N_COL)     row-major, must not alias mat
//
// N_COL is n_eig when applied to the diffuse matrix and n_src when applied
// to the source matrix; one launcher serves both.
//
// There is no reduction and no block structure: every (baseline, column)
// element is independent, so `edges` is not needed and the kernel is a flat
// elementwise map. One thread handles one (j, c) pair, reading and writing
// both rows of that pair. Consecutive threads take consecutive c, then the
// next j, so a warp's Re loads and Im loads together cover a contiguous span
// of mat and out (coalesced). N_COL is a template parameter so the / and %
// that split the thread index into (j, c) compile to constant arithmetic.
//
// The gain product is recomputed by each of the N_COL threads that share a
// baseline. gains is 2 * n_ant floats (a few KB), so these gathers are served
// from L1/L2 and are cheaper than a shared-memory staging pass.

#include <stdio.h>
#include <cuda_runtime.h>

#define TPB 256  // must be a multiple of 32

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
template <int N_COL>
__global__ void apply_gains_kernel(
    const float* __restrict__ gains,
    const int*   __restrict__ ant_1,
    const int*   __restrict__ ant_2,
    const float* __restrict__ mat,
    float*       __restrict__ out,
    int n_bl
){
    const long long idx   = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)n_bl * N_COL;
    if (idx >= total) return;

    const int j = (int)(idx / N_COL);   // baseline
    const int c = (int)(idx % N_COL);   // column

    const int p = __ldg(&ant_1[j]);
    const int q = __ldg(&ant_2[j]);

    const float gr_p = __ldg(&gains[2 * p]);
    const float gi_p = __ldg(&gains[2 * p + 1]);
    const float gr_q = __ldg(&gains[2 * q]);
    const float gi_q = __ldg(&gains[2 * q + 1]);

    // g = g_p * conj(g_q)
    const float g_re = fmaf(gr_p, gr_q,  gi_p * gi_q);
    const float g_im = fmaf(gi_p, gr_q, -gr_p * gi_q);

    const long long re = (2LL * j)     * N_COL + c;
    const long long im = (2LL * j + 1) * N_COL + c;

    const float m_re = mat[re];
    const float m_im = mat[im];

    out[re] = fmaf(g_re, m_re, -g_im * m_im);
    out[im] = fmaf(g_im, m_re,  g_re * m_im);
}

// ---------------------------------------------------------------------------
// Explicit instantiations (X-macro over N_COL = 1 .. 20)
// ---------------------------------------------------------------------------
#define INSTANTIATE_COL(NC)                                                    \
    template __global__ void apply_gains_kernel<NC>(                           \
        const float*, const int*, const int*, const float*, float*, int);

INSTANTIATE_COL( 1) INSTANTIATE_COL( 2) INSTANTIATE_COL( 3) INSTANTIATE_COL( 4)
INSTANTIATE_COL( 5) INSTANTIATE_COL( 6) INSTANTIATE_COL( 7) INSTANTIATE_COL( 8)
INSTANTIATE_COL( 9) INSTANTIATE_COL(10) INSTANTIATE_COL(11) INSTANTIATE_COL(12)
INSTANTIATE_COL(13) INSTANTIATE_COL(14) INSTANTIATE_COL(15) INSTANTIATE_COL(16)
INSTANTIATE_COL(17) INSTANTIATE_COL(18) INSTANTIATE_COL(19) INSTANTIATE_COL(20)

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------
#define LAUNCH_COL_CASE(NC)                                                    \
    case NC: {                                                                 \
        const long long total = (long long)n_bl * NC;                          \
        const int grid = (int)((total + TPB - 1) / TPB);                       \
        apply_gains_kernel<NC><<<grid, TPB, 0, stream>>>(                      \
            gains, ant_1, ant_2, mat, out, n_bl);                              \
        break;                                                                 \
    }

extern "C"
int launch_apply_gains(
    const float* gains,
    const int*   ant_1,
    const int*   ant_2,
    const float* mat,
    float*       out,
    int n_bl,
    int n_col,
    cudaStream_t stream
){
    if (n_bl <= 0) return 0;

    switch (n_col){
        LAUNCH_COL_CASE( 1) LAUNCH_COL_CASE( 2) LAUNCH_COL_CASE( 3)
        LAUNCH_COL_CASE( 4) LAUNCH_COL_CASE( 5) LAUNCH_COL_CASE( 6)
        LAUNCH_COL_CASE( 7) LAUNCH_COL_CASE( 8) LAUNCH_COL_CASE( 9)
        LAUNCH_COL_CASE(10) LAUNCH_COL_CASE(11) LAUNCH_COL_CASE(12)
        LAUNCH_COL_CASE(13) LAUNCH_COL_CASE(14) LAUNCH_COL_CASE(15)
        LAUNCH_COL_CASE(16) LAUNCH_COL_CASE(17) LAUNCH_COL_CASE(18)
        LAUNCH_COL_CASE(19) LAUNCH_COL_CASE(20)
        default:
            fprintf(stderr, "ERROR: n_col=%d not supported (max 20).\n", n_col);
            return 1;
    }
    return 0;
}
