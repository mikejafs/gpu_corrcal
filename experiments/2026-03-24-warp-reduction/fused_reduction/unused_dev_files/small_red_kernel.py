import cupy as cp
import numpy as np
from cupyx.profiler import benchmark

# ----------------------------
# Make "edges" like your grid redundancy example (CPU)
# ----------------------------
def grid_redundancy_edges(n_rows, n_cols):
    ants = [(i, j) for i in range(n_rows) for j in range(n_cols)]
    N = len(ants)
    blocks = {}
    for a in range(N):
        x1, y1 = ants[a]
        for b in range(a + 1, N):
            x2, y2 = ants[b]
            dx = x2 - x1
            dy = y2 - y1
            key = (dx, dy)
            blocks.setdefault(key, []).append((a, b))

    keys_sorted = sorted(blocks.keys(), key=lambda x: (x[0], x[1]))
    block_sizes = [len(blocks[k]) for k in keys_sorted]
    edges = np.zeros(len(block_sizes) + 1, dtype=np.int32)
    edges[1:] = np.cumsum(block_sizes)
    return edges

# ----------------------------
# RawKernel: fused reduction for r=5
# Computes: C_b = sum_{i in block b} w_i * d_i d_i^T
# D is (N,5) row-major contiguous, w is (N,)
# edges is (nb+1,), output C is (nb, 25) row-major (5x5 flattened)
# ----------------------------
kernel_src = r'''
extern "C" __global__
void cov_reduce_r5(const float* __restrict__ D,
                   const float* __restrict__ w,
                   const int* __restrict__ edges,
                   float* __restrict__ C,
                   int nb)
{
    int b = (int)blockIdx.x;
    if (b >= nb) return;

    int start = edges[b];
    int stop  = edges[b+1];

    // 25 accumulators in registers (r=5 => 5x5)
    float c00=0, c01=0, c02=0, c03=0, c04=0;
    float c10=0, c11=0, c12=0, c13=0, c14=0;
    float c20=0, c21=0, c22=0, c23=0, c24=0;
    float c30=0, c31=0, c32=0, c33=0, c34=0;
    float c40=0, c41=0, c42=0, c43=0, c44=0;

    // Stride over rows in this block
    for (int i = start + (int)threadIdx.x; i < stop; i += (int)blockDim.x) {
        float wi = w[i];

        // Load d (length 5)
        // D is row-major: D[i*5 + j]
        float d0 = D[i*5 + 0];
        float d1 = D[i*5 + 1];
        float d2 = D[i*5 + 2];
        float d3 = D[i*5 + 3];
        float d4 = D[i*5 + 4];

        // Multiply by weight once, then outer product
        float wd0 = wi * d0;
        float wd1 = wi * d1;
        float wd2 = wi * d2;
        float wd3 = wi * d3;
        float wd4 = wi * d4;

        c00 += wd0*d0; c01 += wd0*d1; c02 += wd0*d2; c03 += wd0*d3; c04 += wd0*d4;
        c10 += wd1*d0; c11 += wd1*d1; c12 += wd1*d2; c13 += wd1*d3; c14 += wd1*d4;
        c20 += wd2*d0; c21 += wd2*d1; c22 += wd2*d2; c23 += wd2*d3; c24 += wd2*d4;
        c30 += wd3*d0; c31 += wd3*d1; c32 += wd3*d2; c33 += wd3*d3; c34 += wd3*d4;
        c40 += wd4*d0; c41 += wd4*d1; c42 += wd4*d2; c43 += wd4*d3; c44 += wd4*d4;
    }

    // Block reduction in shared memory (25 floats per thread -> heavy but simple)
    extern __shared__ float sh[];
    int t = (int)threadIdx.x;
    int T = (int)blockDim.x;

    // layout: sh[0*T + t] is entry 0 for thread t, sh[1*T+t] entry 1, ...
    sh[ 0*T + t] = c00; sh[ 1*T + t] = c01; sh[ 2*T + t] = c02; sh[ 3*T + t] = c03; sh[ 4*T + t] = c04;
    sh[ 5*T + t] = c10; sh[ 6*T + t] = c11; sh[ 7*T + t] = c12; sh[ 8*T + t] = c13; sh[ 9*T + t] = c14;
    sh[10*T + t] = c20; sh[11*T + t] = c21; sh[12*T + t] = c22; sh[13*T + t] = c23; sh[14*T + t] = c24;
    sh[15*T + t] = c30; sh[16*T + t] = c31; sh[17*T + t] = c32; sh[18*T + t] = c33; sh[19*T + t] = c34;
    sh[20*T + t] = c40; sh[21*T + t] = c41; sh[22*T + t] = c42; sh[23*T + t] = c43; sh[24*T + t] = c44;

    __syncthreads();

    // parallel reduction over threads (sum) for each of 25 entries
    for (int stride = T/2; stride > 0; stride >>= 1) {
        if (t < stride) {
            #pragma unroll
            for (int e = 0; e < 25; ++e) {
                sh[e*T + t] += sh[e*T + (t + stride)];
            }
        }
        __syncthreads();
    }

    // thread 0 writes result
    if (t == 0) {
        float* out = C + (size_t)b * 25;
        #pragma unroll
        for (int e = 0; e < 25; ++e) out[e] = sh[e*T + 0];
    }
}
'''
cov_reduce_r5 = cp.RawKernel(kernel_src, "cov_reduce_r5")

# ----------------------------
# Convenience wrappers
# ----------------------------
def cov_reduce_rawkernel(D, w, edges, threads=128):
    # D: (N,5) float32 contiguous, w: (N,) float32 contiguous
    # edges: (nb+1,) int32
    assert D.dtype == cp.float32 and w.dtype == cp.float32 and edges.dtype == cp.int32
    assert D.ndim == 2 and D.shape[1] == 5 and w.ndim == 1 and D.shape[0] == w.shape[0]
    nb = edges.size - 1
    C = cp.empty((nb, 25), dtype=cp.float32)
    shared = 25 * threads * 4  # bytes
    cov_reduce_r5((nb,), (threads,), (D, w, edges, C, np.int32(nb)), shared_mem=shared)
    return C.reshape(nb, 5, 5)

def cov_gemm_reference(D, w, edges):
    # Reference using per-block GEMM (slow but correct)
    nb = edges.size - 1
    C = cp.empty((nb, 5, 5), dtype=cp.float32)
    for b in range(nb):
        s = int(edges[b].item())
        e = int(edges[b+1].item())
        Db = D[s:e, :]                  # (h,5)
        WDb = Db * w[s:e, None]         # (h,5)
        C[b] = Db.T @ WDb               # (5,5)
    return C

# ----------------------------
# Demo + correctness + benchmark
# ----------------------------
if __name__ == "__main__":
    # Example similar scale to your earlier tests
    rows, cols = 32, 18
    edges_np = grid_redundancy_edges(rows, cols)
    nb = len(edges_np) - 1
    N = int(edges_np[-1])

    print(f"rows,cols=({rows},{cols})  antennas={rows*cols}  baselines(N_rows)={N}  blocks={nb}")

    # Random D and weights
    cp.random.seed(0)
    D = cp.random.standard_normal((N, 5), dtype=cp.float32)
    w = cp.random.random((N,), dtype=cp.float32)

    edges = cp.asarray(edges_np, dtype=cp.int32)

    # Warmup
    C1 = cov_reduce_rawkernel(D, w, edges, threads=128)
    cp.cuda.Stream.null.synchronize()

    # Correctness check vs GEMM reference (do it once; ref loop is slow)
    Cref = cov_gemm_reference(D, w, edges)
    cp.cuda.Stream.null.synchronize()

    ok = bool(cp.allclose(C1, Cref, rtol=1e-4, atol=1e-4).item())
    print("Correctness vs per-block GEMM reference:", ok)

    # Benchmark kernel-only (fast)
    def run_kernel():
        C = cov_reduce_rawkernel(D, w, edges, threads=128)
        cp.cuda.Stream.null.synchronize()
        return C

    # Benchmark a comparable "vectorized" approach across blocks is nontrivial without padding,
    # so we just report the kernel timing and (optionally) per-block GEMM timing.
    t_kernel = benchmark(lambda: run_kernel(), (), n_repeat=100)
    print("\nRawKernel fused reduction:")
    print("  cpu:", float(cp.mean(t_kernel.cpu_times).item()), "  gpu:", float(cp.mean(t_kernel.gpu_times).item()))

    # Optional: benchmark reference (will be much slower; reduce repeats)
    t_ref = benchmark(lambda: cov_gemm_reference(D, w, edges), (), n_repeat=10)
    print("\nPer-block GEMM reference (looping blocks):")
    print("  cpu:", float(cp.mean(t_ref.cpu_times).item()), "  gpu:", float(cp.mean(t_ref.gpu_times).item()))
