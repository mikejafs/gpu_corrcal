import cupy as cp
import numpy as np
from cupyx.profiler import benchmark

# ----------------------------
# Edges generator (CPU)
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
# Helpers: pack/unpack symmetric 5x5 <-> 15
# order: (0,0),
#        (0,1)(1,1),
#        (0,2)(1,2)(2,2),
#        (0,3)(1,3)(2,3)(3,3),
#        (0,4)(1,4)(2,4)(3,4)(4,4)
# ----------------------------
def sym15_to_mat5(C15: cp.ndarray) -> cp.ndarray:
    nb = C15.shape[0]
    C = cp.zeros((nb, 5, 5), dtype=C15.dtype)
    idx = 0
    for j in range(5):
        for i in range(j + 1):
            C[:, i, j] = C15[:, idx]
            C[:, j, i] = C15[:, idx]
            idx += 1
    return C

def mat5_to_sym15(C: cp.ndarray) -> cp.ndarray:
    nb = C.shape[0]
    out = cp.empty((nb, 15), dtype=C.dtype)
    idx = 0
    for j in range(5):
        for i in range(j + 1):
            out[:, idx] = C[:, i, j]
            idx += 1
    return out

# ----------------------------
# Reference (slow): per-block GEMM
# ----------------------------
def cov_gemm_reference_sym15(D, w, edges):
    nb = edges.size - 1
    out = cp.empty((nb, 15), dtype=cp.float32)
    for b in range(nb):
        s = int(edges[b].item())
        e = int(edges[b + 1].item())
        Db = D[s:e, :]                  # (h,5)
        WDb = Db * w[s:e, None]         # (h,5)
        C = Db.T @ WDb                  # (5,5)
        out[b] = mat5_to_sym15(C[None, ...])[0]
    return out

# ----------------------------
# CUDA kernels:
# 1) one-block-per-class, warp-shuffle reduction, 15 elems only
# 2) persistent blocks w/ atomic counter, same inner reduction
# ----------------------------
kernel_src = r'''
// Warp shuffle reduce sum
__device__ __forceinline__ float warp_sum(float v) {
    // full mask
    unsigned mask = 0xffffffffu;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}

// Accumulate symmetric 5x5 (15 unique) for one block [start, stop)
__device__ __forceinline__
void accumulate_sym15_r5(const float* __restrict__ D,
                         const float* __restrict__ w,
                         int start, int stop,
                         float* __restrict__ acc15)
{
    // 15 accumulators in registers
    // mapping indices:
    // 0:00
    // 1:01 2:11
    // 3:02 4:12 5:22
    // 6:03 7:13 8:23 9:33
    // 10:04 11:14 12:24 13:34 14:44
    #pragma unroll
    for (int t=0;t<15;++t) acc15[t] = 0.0f;

    int tid = (int)threadIdx.x;
    int T = (int)blockDim.x;

    for (int i = start + tid; i < stop; i += T) {
        float wi = w[i];
        float d0 = D[i*5 + 0];
        float d1 = D[i*5 + 1];
        float d2 = D[i*5 + 2];
        float d3 = D[i*5 + 3];
        float d4 = D[i*5 + 4];

        // weight once
        float wd0 = wi * d0;
        float wd1 = wi * d1;
        float wd2 = wi * d2;
        float wd3 = wi * d3;
        float wd4 = wi * d4;

        // upper triangle accum
        acc15[0]  += wd0*d0;           // 00
        acc15[1]  += wd0*d1;           // 01
        acc15[2]  += wd1*d1;           // 11
        acc15[3]  += wd0*d2;           // 02
        acc15[4]  += wd1*d2;           // 12
        acc15[5]  += wd2*d2;           // 22
        acc15[6]  += wd0*d3;           // 03
        acc15[7]  += wd1*d3;           // 13
        acc15[8]  += wd2*d3;           // 23
        acc15[9]  += wd3*d3;           // 33
        acc15[10] += wd0*d4;           // 04
        acc15[11] += wd1*d4;           // 14
        acc15[12] += wd2*d4;           // 24
        acc15[13] += wd3*d4;           // 34
        acc15[14] += wd4*d4;           // 44
    }
}

extern "C" __global__
void cov_reduce_r5_sym15_warp(const float* __restrict__ D,
                              const float* __restrict__ w,
                              const int* __restrict__ edges,
                              float* __restrict__ C15,
                              int nb)
{
    int b = (int)blockIdx.x;
    if (b >= nb) return;

    int start = edges[b];
    int stop  = edges[b+1];

    float acc[15];
    accumulate_sym15_r5(D, w, start, stop, acc);

    // Warp reduce (each of 15 scalars)
    #pragma unroll
    for (int t=0;t<15;++t) acc[t] = warp_sum(acc[t]);

    // One value per warp -> shared
    int lane = (int)(threadIdx.x & 31);
    int warp = (int)(threadIdx.x >> 5);

    // shared: [15][num_warps]
    extern __shared__ float sh[];
    int num_warps = (int)(blockDim.x >> 5);

    if (lane == 0) {
        #pragma unroll
        for (int t=0;t<15;++t) {
            sh[t * num_warps + warp] = acc[t];
        }
    }
    __syncthreads();

    // First warp reduces warp-sums
    if (warp == 0) {
        float v[15];
        if (lane < num_warps) {
            #pragma unroll
            for (int t=0;t<15;++t) v[t] = sh[t * num_warps + lane];
        } else {
            #pragma unroll
            for (int t=0;t<15;++t) v[t] = 0.0f;
        }

        #pragma unroll
        for (int t=0;t<15;++t) v[t] = warp_sum(v[t]);

        if (lane == 0) {
            float* out = C15 + (size_t)b * 15;
            #pragma unroll
            for (int t=0;t<15;++t) out[t] = v[t];
        }
    }
}

extern "C" __global__
void cov_reduce_r5_sym15_persistent(const float* __restrict__ D,
                                    const float* __restrict__ w,
                                    const int* __restrict__ edges,
                                    float* __restrict__ C15,
                                    int nb,
                                    int* __restrict__ counter)
{
    __shared__ int sb;   // shared "block id" for this CUDA block

    while (true) {

        // One thread claims the next work item
        if (threadIdx.x == 0) {
            sb = atomicAdd(counter, 1);
        }
        __syncthreads();

        int b = sb;
        if (b >= nb) return;

        int start = edges[b];
        int stop  = edges[b+1];

        float acc[15];
        accumulate_sym15_r5(D, w, start, stop, acc);

        #pragma unroll
        for (int t=0;t<15;++t) acc[t] = warp_sum(acc[t]);

        int lane = (int)(threadIdx.x & 31);
        int warp = (int)(threadIdx.x >> 5);

        extern __shared__ float sh[];
        int num_warps = (int)(blockDim.x >> 5);

        if (lane == 0) {
            #pragma unroll
            for (int t=0;t<15;++t) {
                sh[t * num_warps + warp] = acc[t];
            }
        }
        __syncthreads();

        if (warp == 0) {
            float v[15];
            if (lane < num_warps) {
                #pragma unroll
                for (int t=0;t<15;++t) v[t] = sh[t * num_warps + lane];
            } else {
                #pragma unroll
                for (int t=0;t<15;++t) v[t] = 0.0f;
            }

            #pragma unroll
            for (int t=0;t<15;++t) v[t] = warp_sum(v[t]);

            if (lane == 0) {
                float* out = C15 + (size_t)b * 15;
                #pragma unroll
                for (int t=0;t<15;++t) out[t] = v[t];
            }
        }

        __syncthreads(); // ensures all threads are done before next iteration
    }
}

'''
k_warp = cp.RawKernel(kernel_src, "cov_reduce_r5_sym15_warp")
k_persist = cp.RawKernel(kernel_src, "cov_reduce_r5_sym15_persistent")

# ----------------------------
# Wrappers
# ----------------------------
def cov_reduce_sym15_warp(D, w, edges, threads=128):
    assert D.dtype == cp.float32 and w.dtype == cp.float32 and edges.dtype == cp.int32
    assert D.flags.c_contiguous and w.flags.c_contiguous
    assert D.ndim == 2 and D.shape[1] == 5 and w.ndim == 1 and D.shape[0] == w.shape[0]
    nb = edges.size - 1
    out = cp.empty((nb, 15), dtype=cp.float32)

    num_warps = threads // 32
    shared = 15 * num_warps * 4  # bytes

    k_warp((nb,), (threads,), (D, w, edges, out, np.int32(nb)), shared_mem=shared)
    return out

def cov_reduce_sym15_persistent(D, w, edges, threads=128, persistent_blocks=80):
    assert D.dtype == cp.float32 and w.dtype == cp.float32 and edges.dtype == cp.int32
    assert D.flags.c_contiguous and w.flags.c_contiguous
    nb = edges.size - 1
    out = cp.empty((nb, 15), dtype=cp.float32)

    counter = cp.zeros((), dtype=cp.int32)  # device scalar counter

    num_warps = threads // 32
    shared = 15 * num_warps * 4

    k_persist((persistent_blocks,), (threads,),
              (D, w, edges, out, np.int32(nb), counter),
              shared_mem=shared)
    return out

# ----------------------------
# Main: correctness + benchmarks
# ----------------------------
if __name__ == "__main__":
    rows, cols = 32, 32
    edges_np = grid_redundancy_edges(rows, cols)
    nb = len(edges_np) - 1
    N = int(edges_np[-1])

    print(f"rows,cols=({rows},{cols})  antennas={rows*cols}  baselines={N}  blocks={nb}")

    cp.random.seed(0)
    D = cp.random.standard_normal((N, 5), dtype=cp.float32)  # row-major
    w = cp.random.random((N,), dtype=cp.float32)
    edges = cp.asarray(edges_np, dtype=cp.int32)

    # Warmup
    Cw = cov_reduce_sym15_warp(D, w, edges)
    Cp = cov_reduce_sym15_persistent(D, w, edges, persistent_blocks=80)
    cp.cuda.Stream.null.synchronize()

    # Correctness vs GEMM reference (run once; slow)
    Cref = cov_gemm_reference_sym15(D, w, edges)
    cp.cuda.Stream.null.synchronize()

    ok_w = bool(cp.allclose(Cw, Cref, rtol=1e-4, atol=1e-4).item())
    ok_p = bool(cp.allclose(Cp, Cref, rtol=1e-4, atol=1e-4).item())
    print("Correctness warp:", ok_w, " persistent:", ok_p)

    def run_warp():
        C = cov_reduce_sym15_warp(D, w, edges, threads=128)
        cp.cuda.Stream.null.synchronize()
        return C

    def run_persist():
        C = cov_reduce_sym15_persistent(D, w, edges, threads=128, persistent_blocks=80)
        cp.cuda.Stream.null.synchronize()
        return C

    tw = benchmark(lambda: run_warp(), (), n_repeat=200)
    tp = benchmark(lambda: run_persist(), (), n_repeat=200)

    print("\nWarp-shuffle + sym15 (1 block per class):")
    print("  cpu:", float(cp.mean(tw.cpu_times).item()), " gpu:", float(cp.mean(tw.gpu_times).item()))

    print("\nPersistent blocks + sym15 (atomic counter):")
    print("  cpu:", float(cp.mean(tp.cpu_times).item()), " gpu:", float(cp.mean(tp.gpu_times).item()))
