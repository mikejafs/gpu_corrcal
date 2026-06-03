# Fused Cholesky Inverse Experiment Notes

Here's a summary of the bugs found and fixed in this session:

1. zeroPad2d thread block limit (silent kernel failure) The zeroPad2d CUDA kernel set threadsPerBlock.z = in_array_cols (i.e., n_eig). When n_eig > 16, the total threads per block exceeded the 1024 hardware limit (8 * 8 * 17 = 1088), causing a silent launch failure and all-zero output. The cupy reference returned zeros, making the warp reduction kernel appear wrong when it was actually correct. Fix: use a fixed threadsPerBlock.z = 8 and grid over columns instead. Same bug existed in undo_zeroPad2d and in both the float32 and float64 versions.

2. cupy_ref return value swap zeroPad returns (array, largest_block, nb) but cupy_ref was treating the second return value as nb when it was actually largest_block. This caused shape mismatches for non-square antenna grids (rows ≠ cols). Fix: unpack as lb, nb instead of nb, lb.

3. Warp reduction kernel register/stack boundary The warp shuffle reduction kernel works correctly up to ~n_eig=16 (with 128 threads/block). Beyond that, the compiler moves local arrays to the stack, and warp shuffles on stack-resident values produce incorrect results. This is a hardware-imposed boundary, not a software bug. For n_eig > 16, a shared-memory reduction kernel is needed. The #pragma unroll directive smooths the performance degradation curve but doesn't change the correctness boundary.

4. Fused Cholesky+inverse kernel missing identity The Woodbury formula requires cholesky(I + temp2) but the kernel was computing cholesky(temp2). Without the +I, small blocks produced non-positive-definite matrices causing sqrtf of negative values → NaN. Fix: add + 1.0f to the diagonal read in the Cholesky step.

5. SimCorrcalParams zero noise values cp.random.rand generates values in [0, 1), so zero is possible. At large antenna counts (millions of baselines), hitting exactly 0.0 becomes likely. This causes 1/noise = inf, propagating through the pipeline. Fix: clamp noise with noise + 1e-6 or cp.maximum(noise, 1e-10).







