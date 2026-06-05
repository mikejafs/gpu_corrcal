# Experiment: diffuse_bar kernel (multiply_temp_by_chol)

Computes Del_prime = (N_inv .* diffuse) @ L_del_inv_T per redundant block, where L_del_inv_T comes from the fused Cholesky+inverse kernel. This is the third kernel in the Woodbury inversion pipeline: warp reduction -> fused Cholesky+inverse -> diffuse_bar.

## Kernel design

One CUDA block per redundant block (not per matrix like the Cholesky kernel). Threads within a block cooperate: they load L_inv_T for that block into shared memory once, then each thread handles one or more rows from temp = N_inv .* diffuse, computing a small matrix-vector product per row. The noise inversion is fused inline (1/noise[i] * diffuse[i,:]) to avoid a separate temp buffer. Output shape matches diffuse: (n_total, n_eig).

## Bugs found

1. Shared memory partial load: the original load used if (threadIdx.x < N*N) to load L_inv_T into shared memory. With blockDim.x = 128 and N*N > 128 (i.e. n_eig >= 12), elements beyond index 127 were never loaded -- shared memory contained uninitialized values, producing garbage output. Same class of bug as the zeroPad threadsPerBlock issue. Fix: strided cooperative load for (int i = threadIdx.x; i < N*N; i += blockDim.x).

2. Launch configuration: initially used <<<(nb+128)/128, 128>>> (one thread per block like Cholesky), but this kernel requires one CUDA block per redundant block since threads share L_inv_T via shared memory. Fix: <<<nb, 128>>>.

3. ctypes argtypes: edges was declared as ctypes.c_int instead of ctypes.c_void_p -- it's a pointer to a GPU array, not a scalar.

4. cupy reference shape mismatch: undo-zeropadding temp before the multiply produced a flat (n_total, n_eig) array that couldn't broadcast with t3 at (nb, n_eig, n_eig). Fix: keep temp zero-padded for the cupy matmul, then undo-zeropad the result.

5. undo_zeroPad dtype bug: edges was cast to the data dtype (float32) instead of int64, corrupting the edge indices and causing illegal memory access.

## Accuracy

Max diff against cupy reference is ~5e-3 to 1e-2 across all n_eig (1-20). This accumulates float32 rounding from three different kernel paths vs cupy's single batched matmul path. No NaN or inf issues after fixes. When tested in isolation (same L_inv_T fed to both paths), precision is ~1e-5, confirming the kernel itself is correct and the end-to-end diff is from upstream path differences.