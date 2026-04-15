```
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
```