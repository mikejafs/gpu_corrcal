"""
Generalized Warp Reduction Kernel for Arbitrary n_eig
======================================================

This is a generalization of cov_reduce_r3_sym6_warp. The original kernel
hardcodes n_eig=3 (and therefore n_sym=6 unique symmetric entries). This
version templates the kernel source with n_eig as a compile-time constant,
so the compiler can still unroll loops and optimize register usage.

WHAT CHANGED FROM THE n_eig=3 VERSION:
---------------------------------------
1. `acc[6]` --> `acc[N_SYM]` where N_SYM = N_EIG*(N_EIG+1)/2
2. The inner accumulation loop now computes the upper triangle generically
   instead of manually writing out d0*d0, d0*d1, d1*d1, ...
3. The final write-back unpacks from packed upper-triangle to full n_eig x n_eig
4. The kernel source is a Python f-string with {N_EIG} and {N_SYM} substituted

WHAT STAYED THE SAME:
---------------------
- warp_sum (shuffle reduction) -- identical, it operates on one float at a time
- Two-level reduction (warp -> shared memory -> warp 0) -- identical structure
- Strided access pattern in the accumulation loop -- identical
- Launch geometry (one CUDA block per baseline group) -- identical

Requirements:
    cupy, numpy
    Needs access to the simulate_params / gridding modules from your project
    for the test harness, but the kernel itself is standalone.

Usage:
    python cov_reduce_general.py
"""

import cupy as cp
import numpy as np
import sys
import os

from utils.gridding import *
from utils.simulate_params import *
from utils.zp_puregpu_funcs_py import *
from cupyx.profiler import benchmark


# ============================================================================
# KERNEL SOURCE GENERATOR
# ============================================================================
#
# Instead of one hardcoded kernel string, we have a function that generates
# the kernel source for any n_eig. The key substitutions are:
#
#   N_EIG  = n_eig                          (e.g. 3, 5, 8)
#   N_SYM  = n_eig * (n_eig + 1) // 2      (e.g. 6, 15, 36)
#
# These appear as #define constants in the generated CUDA source, so the
# compiler treats them as compile-time constants -- just like the literal
# `6` in the original kernel.

def make_kernel_source(n_eig):
    """
    Generate CUDA source for the warp reduction kernel with a specific n_eig.

    The generated kernel computes, for each baseline group b:

        C_b = sum_{i in segment_b} (1/w_i) * d_i * d_i^T

    where d_i is a vector of length n_eig, and C_b is a symmetric n_eig x n_eig
    matrix stored in packed upper-triangle form internally, then written out as
    a full n_eig x n_eig matrix.

    Parameters
    ----------
    n_eig : int
        Number of eigenmodes (columns of D). Must be >= 1.

    Returns
    -------
    source : str
        CUDA C source code (pure ASCII, safe for NVRTC).
    """
    n_sym = n_eig * (n_eig + 1) // 2

    source = rf'''
// Compile-time constants -- the compiler sees these as literals,
// enabling full loop unrolling and optimal register allocation.
#define N_EIG {n_eig}
#define N_SYM {n_sym}

// -----------------------------------------------------------------------
// warp_sum: unchanged from the r3 version.
// Reduces a single float across 32 lanes using shuffle-down.
// -----------------------------------------------------------------------
__device__ __forceinline__ float warp_sum(float v) {{
    unsigned mask = 0xffffffffu;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}}

// -----------------------------------------------------------------------
// accumulate_sym_general: generalized version of accumulate_sym6_r3.
//
// WHAT CHANGED:
//   In the r3 version, we manually wrote:
//       float d0 = D[i*3+0], d1 = D[i*3+1], d2 = D[i*3+2];
//       acc[0] += wd0*d0;  acc[1] += wd0*d1;  ... (6 lines)
//
//   Now we use two nested loops over the upper triangle:
//       for col in 0..N_EIG:
//           for row in 0..col+1:
//               acc[idx++] += w_i * d[row] * d[col]
//
//   The #pragma unroll hint tells the compiler to unroll both loops
//   since N_EIG is a compile-time constant. The generated assembly
//   should be nearly identical to the hand-written version.
//
// PACKED INDEX ORDER (matches the original):
//   For n_eig=3: (0,0), (0,1),(1,1), (0,2),(1,2),(2,2)
//   For n_eig=4: (0,0), (0,1),(1,1), (0,2),(1,2),(2,2), (0,3),(1,3),(2,3),(3,3)
//   General: column-major upper triangle, i.e. for each col j, rows 0..j.
// -----------------------------------------------------------------------
__device__ __forceinline__
void accumulate_sym_general(const float* __restrict__ D,
                            const float* __restrict__ w,
                            int start, int stop,
                            float* __restrict__ acc)
{{
    // Zero the accumulator
    #pragma unroll
    for (int t = 0; t < N_SYM; ++t) acc[t] = 0.0f;

    int tid = (int)threadIdx.x;
    int T   = (int)blockDim.x;

    for (int i = start + tid; i < stop; i += T) {{
        float wi = 1.0f / w[i];

        // Load the n_eig-element vector for row i.
        // Using a local array lets the compiler keep these in registers.
        float d[N_EIG];
        #pragma unroll
        for (int e = 0; e < N_EIG; ++e) {{
            d[e] = D[i * N_EIG + e];
        }}

        // Accumulate upper triangle of the outer product.
        // idx walks through the packed symmetric storage in the same
        // order as the original kernel: column-major upper triangle.
        int idx = 0;
        #pragma unroll
        for (int col = 0; col < N_EIG; ++col) {{
            float wd_col = wi * d[col];
            #pragma unroll
            for (int row = 0; row <= col; ++row) {{
                acc[idx] += d[row] * wd_col;
                idx++;
            }}
        }}
    }}
}}

// -----------------------------------------------------------------------
// Main kernel: generalized version of cov_reduce_r3_sym6_warp.
//
// The structure is identical to the original:
//   1. Each block handles one baseline group (edges[b] .. edges[b+1])
//   2. Threads accumulate partial sums with strided access
//   3. Intra-warp reduction via warp_sum (shuffle)
//   4. Lane 0 of each warp writes to shared memory
//   5. Warp 0 does a final reduction and writes the full matrix
//
// The ONLY structural change is in step 5: instead of manually
// unpacking 6 values into a 3x3 matrix, we loop over the packed
// triangle and fill the full n_eig x n_eig output.
// -----------------------------------------------------------------------
extern "C" __global__
void cov_reduce_sym_warp(const float* __restrict__ D,
                         const float* __restrict__ w,
                         const int*   __restrict__ edges,
                         float*       __restrict__ C_out,
                         int nb)
{{
    int b = (int)blockIdx.x;
    if (b >= nb) return;

    int start = edges[b];
    int stop  = edges[b + 1];

    // -- Phase 1: accumulate partial sums in registers --
    float acc[N_SYM];
    accumulate_sym_general(D, w, start, stop, acc);

    // -- Phase 2: intra-warp shuffle reduction --
    // Each of the N_SYM accumulators is reduced independently.
    // This is identical to the r3 version, just looped over N_SYM.
    #pragma unroll
    for (int t = 0; t < N_SYM; ++t) acc[t] = warp_sum(acc[t]);

    // -- Phase 3: inter-warp reduction via shared memory --
    int lane = (int)(threadIdx.x & 31);
    int warp = (int)(threadIdx.x >> 5);

    extern __shared__ float sh[];
    int num_warps = (int)(blockDim.x >> 5);

    // Lane 0 of each warp writes its N_SYM partial sums to shared memory.
    // Layout: sh[sym_index * num_warps + warp_id]
    // This is the same layout as the original, just with N_SYM instead of 6.
    if (lane == 0) {{
        #pragma unroll
        for (int t = 0; t < N_SYM; ++t) {{
            sh[t * num_warps + warp] = acc[t];
        }}
    }}
    __syncthreads();

    // Warp 0 gathers the per-warp results and does a final reduction.
    if (warp == 0) {{
        float v[N_SYM];
        if (lane < num_warps) {{
            #pragma unroll
            for (int t = 0; t < N_SYM; ++t) v[t] = sh[t * num_warps + lane];
        }} else {{
            #pragma unroll
            for (int t = 0; t < N_SYM; ++t) v[t] = 0.0f;
        }}

        #pragma unroll
        for (int t = 0; t < N_SYM; ++t) v[t] = warp_sum(v[t]);

        // -- Phase 4: unpack upper triangle into full symmetric matrix --
        // The original kernel does this with 9 manual assignments.
        // We generalize with a loop that mirrors the packing order.
        if (lane == 0) {{
            float* out = C_out + (size_t)b * N_EIG * N_EIG;     //Note this is just computing an address
                                                                //C_out is the beginning and b * n_eig * n_eig
                                                                //is just moving to the beginning of block b 
                                                                //in memory

            /*
            Aside: Yes, float *out creates a pointer variable called out
                that is happening in a register, private to the individual
                thread. But this is not the same thing as memory initialization.
                There is no memory allocation, just the creation of a 
                variable for this individual thread that points to the
                desired location in memory that we want to write to.
                (size_t)b * n_eig * n_eig is literally just adding 
                b*n_eig*n_eig bytes to the C_out pointers original address. 
            */

            int idx = 0;
            #pragma unroll
            for (int col = 0; col < N_EIG; ++col) {{
                #pragma unroll
                for (int row = 0; row <= col; ++row) {{
                    out[row * N_EIG + col] = v[idx];  // upper triangle
                    out[col * N_EIG + row] = v[idx];  // mirror to lower
                    idx++;
                }}
            }}
        }}
    }}
}}
'''
    return source


# ============================================================================
# KERNEL CACHE
# ============================================================================
#
# We cache compiled kernels by n_eig so we don't recompile on every call.
# In practice you'd pick one n_eig at startup and use it for the whole run,
# but caching makes the interface cleaner.

_kernel_cache = {}

def get_kernel(n_eig):
    """Get (or compile) the warp reduction kernel for a given n_eig."""
    if n_eig not in _kernel_cache:
        source = make_kernel_source(n_eig)
        _kernel_cache[n_eig] = cp.RawKernel(source, "cov_reduce_sym_warp")
    return _kernel_cache[n_eig]


# ============================================================================
# PYTHON INTERFACE
# ============================================================================

def cov_reduce_sym(w, D, edges, threads=128):
    """
    Compute batched weighted outer-product reduction with arbitrary n_eig.

    For each baseline group b defined by edges[b]..edges[b+1]:
        C_b = sum_{i} (1/w_i) * D[i,:] * D[i,:]^T

    Parameters
    ----------
    w : cp.ndarray, shape (N,), float32
        Weights (the kernel computes 1/w internally).
    D : cp.ndarray, shape (N, n_eig), float32, C-contiguous
        Data matrix. n_eig is inferred from D.shape[1].
    edges : cp.ndarray, shape (nb+1,), int32
        Segment boundaries (CSR-style pointers).
    threads : int
        Threads per CUDA block. Must be a multiple of 32.

    Returns
    -------
    C : cp.ndarray, shape (nb, n_eig, n_eig), float32
        One symmetric matrix per baseline group.
    """
    n_eig = D.shape[1]
    n_sym = n_eig * (n_eig + 1) // 2

    assert D.dtype == cp.float32 and D.ndim == 2 and D.flags.c_contiguous
    assert w.dtype == cp.float32 and w.ndim == 1 and w.flags.c_contiguous
    assert w.shape[0] == D.shape[0]
    assert edges.dtype == cp.int32 and edges.ndim == 1
    assert threads % 32 == 0, "threads must be a multiple of 32 (warp size)"

    nb = edges.size - 1
    out = cp.empty((nb, n_eig, n_eig), dtype=cp.float32)

    num_warps = threads // 32
    # Shared memory: N_SYM floats per warp (one per symmetric entry)
    shared = n_sym * num_warps * 4  # bytes

    kernel = get_kernel(n_eig)
    kernel((nb,), (threads,), (D, w, edges, out, np.int32(nb)), shared_mem=shared)
    return out


# ============================================================================
# TESTS & DEMO
# ============================================================================

#Adding extra Cupy comparison the way it's actually done for good measure
def cupy_matmul(inv_noise, diffuse):
    temp = inv_noise[..., None] * diffuse
    out = cp.transpose(diffuse, [0, 2, 1]) @ temp
    cp.cuda.Stream.null.synchronize()
    return out

def test_against_numpy(n_eig, n_ant=400, rows=20, cols=20):
    """
    Verify the generalized kernel against a naive NumPy implementation.

    The NumPy reference computes each C_b by:
      1. Slicing D[start:stop] and w[start:stop]
      2. Computing W = diag(1/w)
      3. C_b = D^T @ W @ D

    This is O(n^2) per group and painfully slow, but unambiguously correct.
    """
    print(f"  n_eig={n_eig}  n_ant={n_ant}  ", end="", flush=True)

    cp.random.seed(42)

    spms = SimCorrcalParams(n_ant, n_eig, n_src=1, precision='float32', xp=cp)
    edges = spms.edges(rows, cols, use_random=False)
    edges_gpu = cp.asarray(edges)

    sim_data = spms.sim_data()
    w = sim_data[0]
    D = sim_data[1]
    # w = 1.0 / w  # match the convention in the original test script

    # --- GPU computation ---
    C_gpu = cov_reduce_sym(w, D, edges_gpu)
    C_gpu_np = cp.asnumpy(C_gpu)

    # --- CuPy Reference ---
    zp_w_inv, nb, lb = zeroPad(w, edges_gpu, return_inv=True, dtype=cp.float32)
    zp_D, nb, lb = zeroPad(D, edges_gpu, return_inv=False, dtype=cp.float32)

    CuPy_ref = cupy_matmul(zp_w_inv, zp_D)
    CuPy_ref_np = cp.asnumpy(CuPy_ref)

    # --- NumPy reference ---
    w_np = cp.asnumpy(w)
    D_np = cp.asnumpy(D)
    edges_np = cp.asnumpy(edges_gpu)
    nb = edges_np.size - 1
    C_ref = np.zeros((nb, n_eig, n_eig), dtype=np.float32)

    for b in range(nb):
        s, e = edges_np[b], edges_np[b + 1]
        if s >= e:
            continue
        Db = D_np[s:e]             # (seg_len, n_eig)
        wb = w_np[s:e]             # (seg_len,)
        inv_w = (1.0/wb).astype(np.float64)  # higher precision for reference
        Db64 = Db.astype(np.float64)
        C_ref[b] = (Db64.T * inv_w) @ Db64     # D^T @ diag(1/w) @ D


    # --- Compare ---
    # Use a relative tolerance since values scale with segment size
    max_abs_err = np.max(np.abs(C_gpu_np - C_ref))
    max_val = max(np.max(np.abs(C_ref)), 1e-8)
    rel_err = max_abs_err / max_val

    ok = rel_err < 1e-4  # float32 vs float64 reference
    status = "PASS" if ok else "FAIL"
    
    cupy_ok = np.allclose(C_gpu_np, CuPy_ref_np)
    cupy_status = "CuPy PASS" if cupy_ok else "CuPy FAIL"
    
    print(f"max_rel_err={rel_err:.2e}  [{status}] ~~ Against CuPy: [{cupy_status}]")
    
    print(f"")
    return ok


def test_matches_r3_kernel():
    """
    Verify that the generalized kernel with n_eig=3 produces identical
    output to the original hardcoded r3 kernel.
    """
    print("  Checking generalized n_eig=3 vs original r3 kernel... ", end="", flush=True)

    cp.random.seed(42)
    n_eig = 3
    rows, cols = 20, 20
    n_ant = rows * cols

    spms = SimCorrcalParams(n_ant, n_eig, n_src=1, precision='float32', xp=cp)
    edges = spms.edges(rows, cols, use_random=False)
    edges_gpu = cp.asarray(edges)

    sim_data = spms.sim_data()
    w = sim_data[0]
    D = sim_data[1]
    # w = 1.0 / w

    # Original r3 kernel (imported from your existing code or inlined here)
    kernel_r3_source = r'''
__device__ __forceinline__ float warp_sum(float v) {
    unsigned mask = 0xffffffffu;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}

__device__ __forceinline__
void accumulate_sym6_r3(const float* __restrict__ D,
                        const float* __restrict__ w,
                        int start, int stop,
                        float* __restrict__ acc)
{
    #pragma unroll
    for (int t=0;t<6;++t) acc[t] = 0.0f;

    int tid = (int)threadIdx.x;
    int T   = (int)blockDim.x;

    for (int i = start + tid; i < stop; i += T) {
        float wi = 1/w[i];
        float d0 = D[i*3 + 0];
        float d1 = D[i*3 + 1];
        float d2 = D[i*3 + 2];
        float wd0 = wi * d0;
        float wd1 = wi * d1;
        float wd2 = wi * d2;
        acc[0] += wd0*d0;
        acc[1] += wd0*d1;
        acc[2] += wd1*d1;
        acc[3] += wd0*d2;
        acc[4] += wd1*d2;
        acc[5] += wd2*d2;
    }
}

extern "C" __global__
void cov_reduce_r3_sym6_warp(const float* __restrict__ D,
                             const float* __restrict__ w,
                             const int* __restrict__ edges,
                             float* __restrict__ C6,
                             int nb)
{
    int b = (int)blockIdx.x;
    if (b >= nb) return;
    int start = edges[b];
    int stop  = edges[b+1];
    float acc[6];
    accumulate_sym6_r3(D, w, start, stop, acc);
    #pragma unroll
    for (int t=0;t<6;++t) acc[t] = warp_sum(acc[t]);
    int lane = (int)(threadIdx.x & 31);
    int warp = (int)(threadIdx.x >> 5);
    extern __shared__ float sh[];
    int num_warps = (int)(blockDim.x >> 5);
    if (lane == 0) {
        #pragma unroll
        for (int t=0;t<6;++t) sh[t * num_warps + warp] = acc[t];
    }
    __syncthreads();
    if (warp == 0) {
        float v[6];
        if (lane < num_warps) {
            #pragma unroll
            for (int t=0;t<6;++t) v[t] = sh[t * num_warps + lane];
        } else {
            #pragma unroll
            for (int t=0;t<6;++t) v[t] = 0.0f;
        }
        #pragma unroll
        for (int t=0;t<6;++t) v[t] = warp_sum(v[t]);
        if (lane == 0) {
            float* out = C6 + (size_t)b * 9;
            out[0] = v[0];  out[1] = v[1];  out[2] = v[3];
            out[3] = v[1];  out[4] = v[2];  out[5] = v[4];
            out[6] = v[3];  out[7] = v[4];  out[8] = v[5];
        }
    }
}
'''
    k_r3 = cp.RawKernel(kernel_r3_source, "cov_reduce_r3_sym6_warp")

    # Run original r3 kernel
    nb = edges_gpu.size - 1
    out_r3 = cp.empty((nb, 3, 3), dtype=cp.float32)
    threads = 128
    num_warps = threads // 32
    shared = 6 * num_warps * 4
    k_r3((nb,), (threads,), (D, w, edges_gpu, out_r3, np.int32(nb)), shared_mem=shared)

    # Run generalized kernel with n_eig=3
    out_gen = cov_reduce_sym(w, D, edges_gpu, threads=threads)

    # They should be bitwise identical (same math, same order of operations)
    match = cp.allclose(out_r3, out_gen, atol=0, rtol=0)
    # If not bitwise, check very close (floating point associativity edge cases)
    if not match:
        match = cp.allclose(out_r3, out_gen, atol=1e-6, rtol=1e-5)
        if match:
            print("PASS (within float32 tolerance, not bitwise)")
        else:
            max_diff = float(cp.max(cp.abs(out_r3 - out_gen)))
            print(f"FAIL (max_diff={max_diff:.2e})")
    else:
        print("PASS (bitwise identical)")
    return bool(match)


if __name__ == "__main__":
    print()
    print("Generalized Warp Reduction Kernel")
    print("=" * 60)

    # ---- Test 1: generalized kernel matches original r3 kernel ----
    print()
    print("Test 1: Generalized n_eig=3 vs original r3 kernel")
    print("-" * 60)
    test_matches_r3_kernel()

    # ---- Test 2: correctness for various n_eig values ----
    print()
    print("Test 2: Correctness vs NumPy reference and vs CuPy Reference")
    print("-" * 60)
    for n_eig in [1, 2, 3, 4, 5, 8, 15, 16, 17, 18]:
        test_against_numpy(n_eig)

    # ---- Benchmark across n_eig values ----
    print()
    print("Benchmark: varying n_eig")
    print("-" * 60)

    rows, cols = 44, 44
    n_ant = rows * cols
    print(f"For all tests, n_ant={n_ant}")

    for n_eig in [1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]:
        cp.random.seed(42)
        spms = SimCorrcalParams(n_ant, n_eig, n_src=1, precision='float32', xp=cp)
        edges = spms.edges(rows, cols, use_random=False)
        edges_gpu = cp.asarray(edges)
        sim_data = spms.sim_data()
        w = sim_data[0]
        D = sim_data[1]

        times = benchmark(cov_reduce_sym, (w, D, edges_gpu), n_repeat=100)
        avg_gpu = float(cp.mean(times.gpu_times)) * 1e6  # microseconds
        avg_cpu = float(cp.mean(times.cpu_times)) * 1e6
        n_sym = n_eig * (n_eig + 1) // 2
        print(f"  n_eig={n_eig}  n_sym={n_sym:>3}  "
              f"gpu={avg_gpu:>8.1f} us  cpu={avg_cpu:>8.1f} us")

    print()