"""
Warp Shuffle Sum Reduction — A Learning Example
=================================================

This script isolates the `warp_sum` pattern from the covariance kernel
(cov_reduce_r3_sym6_warp) and builds it up in two stages:

  1. Single-warp reduction  — 32 elements, one warp, pure shuffle ops
  2. Block-level reduction  — N elements, many warps, shared memory glue

Both kernels are written in raw CUDA C and launched via CuPy's RawKernel.
The goal is to make the mechanics of __shfl_down_sync completely transparent
before you see it embedded in a more complex kernel.

Requirements:
    pip install cupy-cuda12x   (match your CUDA version)

Usage:
    python warp_sum_reduction.py
"""

import numpy as np
import cupy as cp

# ═══════════════════════════════════════════════════════════════════════════════
# KERNEL 1 — Single-Warp Reduction (32 elements only)
# ═══════════════════════════════════════════════════════════════════════════════
#
# This is the simplest possible shuffle reduction. One warp (32 threads) each
# holds one value. After 5 rounds of __shfl_down_sync, thread 0 holds the sum.
#
# HOW __shfl_down_sync WORKS:
#
#   float __shfl_down_sync(unsigned mask, float var, int delta);
#
#   Each thread calls this simultaneously. Thread `t` receives the value of
#   `var` from thread `t + delta`. If `t + delta` is out of the warp (≥ 32),
#   the thread keeps its own value (effectively adding 0 in a sum context).
#
#   `mask` is a 32-bit bitmask saying which threads participate. 0xffffffff
#   means all 32 threads. The hardware requires all threads named in the mask
#   to actually execute the instruction (otherwise undefined behavior).
#
# VISUAL — what happens across 5 rounds for an 8-thread example (real warps
# are 32 threads, but the pattern is identical):
#
#   Thread:    0    1    2    3    4    5    6    7
#   Initial:  [a0] [a1] [a2] [a3] [a4] [a5] [a6] [a7]
#
#   Round 1 (delta=4):
#     t0 += t4 → a0+a4    t1 += t5 → a1+a5    t2 += t6    t3 += t7
#     t4..t7 also execute but their results don't matter anymore
#
#   Round 2 (delta=2):
#     t0 += t2 → (a0+a4)+(a2+a6)    t1 += t3 → (a1+a5)+(a3+a7)
#
#   Round 3 (delta=1):
#     t0 += t1 → sum of all 8 values  ← final answer in thread 0
#
#   For 32 threads: deltas are 16, 8, 4, 2, 1  (5 rounds = log2(32)).
#
# WHY THIS IS FAST:
#   - No shared memory reads/writes (~10 cycle latency each)
#   - No __syncthreads() barriers
#   - Pure register-to-register transfers within the warp's own hardware
#   - 5 instructions total to reduce 32 values

SINGLE_WARP_KERNEL = r"""
extern "C" __global__
void warp_sum_32(const float* input, float* output) {
    // -- Step 1: Each thread loads one value ----------------------------
    // threadIdx.x ranges from 0..31 (we launch exactly 32 threads).
    float val = input[threadIdx.x];

    // -- Step 2: Shuffle-down reduction across the warp ---------------
    // After each round, lower-numbered threads accumulate the sum of
    // a progressively larger slice of the original 32 values.
    //
    // 0xffffffff = all 32 lanes participate in the shuffle.

    val += __shfl_down_sync(0xffffffff, val, 16);
    // Now threads 0-15 each hold (their original) + (the value from 16 slots ahead).
    // e.g. thread 0 holds input[0] + input[16]

    val += __shfl_down_sync(0xffffffff, val, 8);
    // Threads 0-7 hold partial sums of 4 elements each.
    // e.g. thread 0 holds input[0] + input[16] + input[8] + input[24]

    val += __shfl_down_sync(0xffffffff, val, 4);
    // Threads 0-3 hold partial sums of 8 elements each.

    val += __shfl_down_sync(0xffffffff, val, 2);
    // Threads 0-1 hold partial sums of 16 elements each.

    val += __shfl_down_sync(0xffffffff, val, 1);
    // Thread 0 holds the sum of all 32 elements. Done!

    // -- Step 3: Thread 0 writes the result -----------------------------
    if (threadIdx.x == 0) {
        output[0] = val;
    }
}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# KERNEL 2 — Block-Level Reduction (arbitrary N, multiple warps)
# ═══════════════════════════════════════════════════════════════════════════════
#
# When N > 32 we need more than one warp. The problem: shuffle only works
# WITHIN a warp. So we need a two-phase approach:
#
#   Phase 1: Each warp reduces its 32 lanes → 1 partial sum  (shuffle)
#   Phase 2: Collect all partial sums into shared memory,
#            then warp 0 reduces those partial sums             (shuffle again)
#
# This is the exact same pattern used in cov_reduce_r3_sym6_warp, except
# there each thread accumulates 6 values (the symmetric matrix entries)
# instead of 1.
#
# SHARED MEMORY ROLE:
#   Shared memory is the bridge between warps. It's on-chip SRAM (~10 cycle
#   latency), much faster than global VRAM (~400-800 cycles), but it's
#   block-scoped: only threads in the same block can see it.
#
#   We allocate one float per warp. After phase 1, each warp's lane-0 writes
#   its partial sum to sh[warp_id]. After a __syncthreads() barrier (so all
#   warps have finished writing), warp 0 reads those values back and does a
#   final shuffle reduction.
#
# STRIDED ACCESS PATTERN:
#   For N > blockDim.x, each thread processes multiple elements with stride
#   equal to blockDim.x. This ensures coalesced memory access: on each loop
#   iteration, consecutive threads read consecutive addresses.
#
#   Example with 4 threads, 12 elements:
#     Iteration 0: t0→[0] t1→[1] t2→[2] t3→[3]   ← coalesced
#     Iteration 1: t0→[4] t1→[5] t2→[6] t3→[7]   ← coalesced
#     Iteration 2: t0→[8] t1→[9] t2→[10] t3→[11] ← coalesced

BLOCK_REDUCTION_KERNEL = r"""
extern "C" __global__
void block_sum(const float* input, float* output, int N) {

    // -- Shared memory: one slot per warp in this block -----------------
    // Allocated dynamically at launch time (the 3rd arg of the <<<>>> config).
    // Size = (blockDim.x / 32) * sizeof(float).
    extern __shared__ float sh[];

    int tid   = threadIdx.x;
    int gtid  = blockIdx.x * blockDim.x + threadIdx.x;  // global thread id
    int stride = blockDim.x * gridDim.x;                 // total threads in grid

    // -- Phase 0: Each thread accumulates its share ---------------------
    // If N > total thread count, each thread sums multiple elements.
    // The stride pattern keeps memory access coalesced.
    float val = 0.0f;
    for (int i = gtid; i < N; i += stride) {
        val += input[i];
    }

    // -- Phase 1: Intra-warp reduction (shuffle) ------------------------
    // Identical to the single-warp kernel above.
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);

    // -- Lane 0 of each warp writes to shared memory -------------------
    int lane_id = tid % 32;           // position within the warp (0..31)
    int warp_id = tid / 32;           // which warp within the block (0..)

    if (lane_id == 0) {
        sh[warp_id] = val;
    }

    // -- Barrier: wait for all warps to finish writing ------------------
    // __syncthreads() is a block-level barrier. Every thread in the block
    // must reach this point before any thread can proceed past it.
    // This guarantees that sh[] is fully populated before warp 0 reads it.
    __syncthreads();

    // -- Phase 2: Warp 0 reduces the per-warp partial sums -------------
    int num_warps = blockDim.x / 32;

    if (warp_id == 0) {
        // Lane `lane_id` reads the partial sum from warp `lane_id`.
        // If there are fewer warps than 32, pad with 0.
        val = (lane_id < num_warps) ? sh[lane_id] : 0.0f;

        // One more shuffle reduction -- same 5 instructions.
        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val, 8);
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);

        // Thread 0 of block 0 writes the final result.
        if (tid == 0) {
            // atomicAdd lets multiple blocks contribute to a single output.
            // If we launched only 1 block, a plain store would also work.
            atomicAdd(&output[0], val);
        }
    }
}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# PYTHON DRIVER — compile, run, verify
# ═══════════════════════════════════════════════════════════════════════════════

def demo_single_warp():
    """
    Demo 1: Reduce exactly 32 floats with one warp.

    This is the atomic building block. If you understand this, you understand
    __shfl_down_sync — everything else is scaffolding to handle more data.
    """
    print("=" * 70)
    print("DEMO 1: Single-Warp Reduction (32 elements)")
    print("=" * 70)

    # Compile the kernel. CuPy caches the compilation so this is fast after
    # the first run.
    kernel = cp.RawKernel(SINGLE_WARP_KERNEL, "warp_sum_32")

    # Create 32 floats on the GPU. We use simple values so you can verify
    # the sum by hand: 1 + 2 + ... + 32 = 528.
    data = cp.arange(1, 33, dtype=cp.float32)  # [1, 2, 3, ..., 32]
    result = cp.zeros(1, dtype=cp.float32)

    # Launch config:
    #   grid  = (1,)  → 1 block
    #   block = (32,) → 32 threads = exactly 1 warp
    #
    # No shared memory needed (shuffle only uses registers).
    kernel(
        grid=(1,),          # 1 block
        block=(32,),        # 32 threads (1 warp)
        args=(data, result)
    )

    gpu_sum = result.get().item()         # copy back to CPU (get), item converts to Python scalar
    cpu_sum = np.arange(1, 33, dtype=np.float32).sum()

    print(f"  Input:    [1, 2, 3, ..., 32]")
    print(f"  GPU sum:  {gpu_sum}")
    print(f"  CPU sum:  {cpu_sum}")
    print(f"  Match:    {'YES' if np.isclose(gpu_sum, cpu_sum) else 'NO'}")
    print()


def demo_block_reduction(N=10_000, threads_per_block=256):
    """
    Demo 2: Reduce N floats using the two-level (warp → block) pattern.

    This mirrors the full reduction strategy in cov_reduce_r3_sym6_warp.

    Parameters
    ----------
    N : int
        Number of elements to sum. Can be any positive integer.
    threads_per_block : int
        Must be a multiple of 32 (warp size). Common choices: 128, 256, 512.
        256 is a good default — it gives 8 warps per block, which keeps the
        SM (streaming multiprocessor) busy without using too many registers.
    """
    print("=" * 70)
    print(f"DEMO 2: Block-Level Reduction ({N:,} elements)")
    print("=" * 70)

    kernel = cp.RawKernel(BLOCK_REDUCTION_KERNEL, "block_sum")

    # Random floats so we exercise the full precision path
    np.random.seed(42)
    host_data = np.random.randn(N).astype(np.float32)
    data = cp.asarray(host_data)
    result = cp.zeros(1, dtype=cp.float32)

    # ── Launch geometry ───────────────────────────────────────────────
    # num_blocks: enough blocks to cover all elements, capped at 256.
    # Capping prevents launching thousands of tiny blocks; each block
    # handles ceil(N / total_threads) elements via the strided loop.
    num_blocks = min((N + threads_per_block - 1) // threads_per_block, 256)

    # Shared memory: one float per warp in the block.
    num_warps = threads_per_block // 32
    shared_mem_bytes = num_warps * 4  # 4 bytes per float

    print(f"  Threads/block:  {threads_per_block}  ({num_warps} warps)")
    print(f"  Blocks:         {num_blocks}")
    print(f"  Shared memory:  {shared_mem_bytes} bytes  "
          f"({num_warps} floats × 4 bytes)")
    print()

    kernel(
        grid=(num_blocks,),
        block=(threads_per_block,),
        args=(data, result, np.int32(N)),
        shared_mem=shared_mem_bytes
    )

    gpu_sum = result.get().item()
    # Use float64 on CPU for a more accurate reference (float32 accumulation
    # over 10k values loses some precision due to rounding).
    cpu_sum = host_data.astype(np.float64).sum()

    print(f"  GPU sum (float32 accum):  {gpu_sum:.6f}")
    print(f"  CPU sum (float64 ref):    {cpu_sum:.6f}")
    print(f"  Absolute error:           {abs(gpu_sum - cpu_sum):.6f}")
    # For 10k random normal values, float32 error is typically < 0.1
    print(f"  Acceptable:               "
          f"{'YES' if abs(gpu_sum - cpu_sum) < 1.0 else 'NO'}  "
          f"(float32 rounding is expected)")
    print()


def demo_correctness_sweep():
    """
    Demo 3: Sweep across different array sizes to verify the kernel works
    for edge cases — sizes that aren't multiples of 32, small arrays, etc.
    """
    print("=" * 70)
    print("DEMO 3: Correctness Sweep (various N)")
    print("=" * 70)

    kernel = cp.RawKernel(BLOCK_REDUCTION_KERNEL, "block_sum")
    threads_per_block = 256
    num_warps = threads_per_block // 32
    shared_mem_bytes = num_warps * 4

    test_sizes = [1, 7, 31, 32, 33, 64, 100, 255, 256, 1000, 50_000, 1_000_000]
    all_pass = True

    for N in test_sizes:
        np.random.seed(0)
        host_data = np.random.randn(N).astype(np.float32)
        data = cp.asarray(host_data)
        result = cp.zeros(1, dtype=cp.float32)

        num_blocks = min((N + threads_per_block - 1) // threads_per_block, 256)

        kernel(
            grid=(num_blocks,),
            block=(threads_per_block,),
            args=(data, result, np.int32(N)),
            shared_mem=shared_mem_bytes
        )

        gpu_sum = result.get().item()
        cpu_sum = host_data.astype(np.float64).sum()

        # Tolerance scales with N — more elements = more float32 rounding
        tol = max(1e-3, N * 1e-5)
        ok = abs(gpu_sum - cpu_sum) < tol
        if not ok:
            all_pass = False

        status = "PASS" if ok else "FAIL"
        print(f"  N={N:>10,}  GPU={gpu_sum:>14.4f}  "
              f"CPU={cpu_sum:>14.4f}  err={abs(gpu_sum - cpu_sum):.4f}  [{status}]")

    print()
    print(f"  Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print()
    print("Warp Shuffle Sum Reduction — Learning Examples")
    print("━" * 70)
    print()
    demo_single_warp()
    demo_block_reduction()
    demo_correctness_sweep()