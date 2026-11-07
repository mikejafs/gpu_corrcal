// nvcc -Xcompiler -fPIC -shared -o nll_tail.so nll_tail.cu
//
// Scalar tail of the CorrCal negative log-likelihood, one launch.
//
// Replaces the CuPy operations that gpu_corrcal.optimize.nll runs after
// p = C^-1 d:
//
//   chisq  = sum_r d_r p_r                                  (n_row terms)
//   logdet = 2 (sum_b diffuse_det_b + source_det)            (log det C - log det N)
//   phs    = phs_coef * (mean_a atan2(g_i, g_r))^2           (phs_coef = 1/phs_norm_fac^2)
//   out[0] = chisq + logdet + phs,  out[1] = chisq,  out[2] = logdet,  out[3] = phs
//
// diffuse_det / source_det are the per-block outputs that inv_cov writes when
// compute_det=True (InvCovWorkspace.out_diffuse_det / out_source_det). They
// hold sum_i log L_ii per block, so logdet = 2 * their sum, the same assembly
// inv_cov does in CuPy.
//
// Structure: single kernel, "last block finishes" pattern.
//   1. Every block reduces its grid-stride share of d.p in double and writes
//      one partial to partials[blockIdx.x].
//   2. __threadfence() makes that write visible device-wide, then thread 0
//      increments counter. The block that sees counter == gridDim.x - 1 is
//      the last one to finish; every other partial is already visible.
//   3. The last block sums the partials in a fixed order (deterministic
//      result, no float atomics), sums the nb block log-determinants and the
//      n_ant phases, and writes out[0..3].
// counter is zeroed by the launcher with cudaMemsetAsync before every launch,
// so an aborted launch cannot leave a stale count behind.
//
// Precision: every product and sum is in double. The CuPy path forms d*p in
// float32 and sums in double, and sums the block determinants in float32, so
// the two agree to float32 rounding (~1e-7 relative), not bitwise.

#include <cuda_runtime.h>

#define TAIL_TPB        256
#define TAIL_MAX_BLOCKS 256      // must match NLL_TAIL_MAX_BLOCKS in linalg.py
#define TAIL_PER_THREAD 4        // target elements per thread for the grid size

// Sum of v over the block. All threads must call it. Every thread gets the
// result. sh needs TAIL_TPB/32 doubles. Starts and ends with __syncthreads so
// consecutive calls can reuse sh.
__device__ __forceinline__ double block_sum(double v, double* sh)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);

    __syncthreads();                        // sh free from any previous call
    if (lane == 0) sh[warp] = v;
    __syncthreads();

    double tot = 0.0;
    if (warp == 0) {
        tot = (lane < (TAIL_TPB >> 5)) ? sh[lane] : 0.0;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            tot += __shfl_down_sync(0xffffffffu, tot, off);
        if (lane == 0) sh[0] = tot;
    }
    __syncthreads();
    tot = sh[0];
    return tot;
}

__global__ void __launch_bounds__(TAIL_TPB)
nll_tail_kernel(
    const float*  __restrict__ data,         // (n_row,)  d
    const float*  __restrict__ p,            // (n_row,)  C^-1 d
    int n_row,
    const float*  __restrict__ diffuse_det,  // (nb,)     sum log L_ii per group
    int nb,
    const float*  __restrict__ source_det,   // (1,)      sum log L_ii, source level
    const double* __restrict__ gains,        // (2 n_ant,) Re/Im, NOT divided by scale
    int n_ant,
    double phs_coef,                         // 1/phs_norm_fac^2, 0 disables
    double*       partials,                  // (gridDim.x,) scratch
    unsigned int* counter,                   // (1,) zeroed before launch
    double*       out)                       // (4,) nll, chisq, logdet, phs
{
    __shared__ double sh[TAIL_TPB / 32];
    __shared__ int is_last;

    // ---- 1. chi^2 partial for this block --------------------------------
    double acc = 0.0;
    const int stride = gridDim.x * blockDim.x;
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < n_row; r += stride)
        acc += (double)data[r] * (double)p[r];
    acc = block_sum(acc, sh);

    // ---- 2. publish, then find out whether this block is last -------------
    if (threadIdx.x == 0) {
        partials[blockIdx.x] = acc;
        __threadfence();
        const unsigned int done = atomicAdd(counter, 1u);
        is_last = (done == gridDim.x - 1);
    }
    __syncthreads();
    if (!is_last) return;

    // ---- 3. last block: final sums ---------------------------------------
    // volatile: read the other blocks' partials from L2, not a stale L1 line.
    const volatile double* vpart = partials;
    double c = 0.0;
    for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x) c += vpart[i];
    const double chisq = block_sum(c, sh);

    double l = 0.0;
    for (int b = threadIdx.x; b < nb; b += blockDim.x) l += (double)diffuse_det[b];
    const double logdet = 2.0 * (block_sum(l, sh) + (double)source_det[0]);

    double phs = 0.0;
    if (phs_coef != 0.0) {                  // uniform across the block: no divergence at block_sum
        double ph = 0.0;
        for (int a = threadIdx.x; a < n_ant; a += blockDim.x)
            ph += atan2(gains[2 * a + 1], gains[2 * a]);
        const double mean = block_sum(ph, sh) / (double)n_ant;
        phs = phs_coef * mean * mean;
    }

    if (threadIdx.x == 0) {
        out[0] = chisq + logdet + phs;
        out[1] = chisq;
        out[2] = logdet;
        out[3] = phs;
    }
}

extern "C" {

// Returns 0 on success, otherwise the cudaError_t of the launch.
int launch_nll_tail(
    const float* data, const float* p, int n_row,
    const float* diffuse_det, int nb, const float* source_det,
    const double* gains, int n_ant, double phs_coef,
    double* partials, unsigned int* counter, double* out,
    void* stream)
{
    cudaStream_t s = (cudaStream_t)stream;

    int grid = (n_row + TAIL_TPB * TAIL_PER_THREAD - 1) / (TAIL_TPB * TAIL_PER_THREAD);
    if (grid < 1) grid = 1;
    if (grid > TAIL_MAX_BLOCKS) grid = TAIL_MAX_BLOCKS;

    cudaMemsetAsync(counter, 0, sizeof(unsigned int), s);
    nll_tail_kernel<<<grid, TAIL_TPB, 0, s>>>(
        data, p, n_row, diffuse_det, nb, source_det,
        gains, n_ant, phs_coef, partials, counter, out);

    return (int)cudaGetLastError();
}

} // extern "C"
