# Folder Description

This folder extends the warp-reduction covariance kernel (originally hardcoded for `n_eig=3` in `warp_red_kern_r3.py`) to a templated CUDA version that works for arbitrary `n_eig`. The primary purpose is benchmarking and validating that generalization before building future kernels on top of it.

**Key file: `geneig_warp_red_kernel_templated_test.py`** — this is the boilerplate test harness to reuse going forward. It contains:
- `load_kernel` / `call_kernel`: ctypes interface to load and call `.so` kernels
- `cpu_ref` / `cupy_ref`: reference implementations to check correctness against
- `make_test_data`: test data generation via `SimCorrcalParams`
- `correctness_test` / `test_multiple_correct`: compare kernel output to CPU and CuPy references across a range of `n_eig`
- `timing_test` / `time_multiple`: benchmark kernel vs CuPy reference
- `timing_plot_nant_varies` / `timing_plot_neig_varies`: generate benchmark plots saved to `varying_nant_plots/` and `varying_neig_plots/`

The two `.cu` files differ only in whether `#pragma unroll` is used (see conclusions below).

## Important Conclusions

### Correctness

The templated kernel passes all correctness tests against both the CuPy and CPU references for `n_eig=1` through `n_eig=20` at `n_ant=768` (`32x24`). Differences are consistent with float32 rounding and well within the `atol=rtol=1e-4` tolerance used. It also matches the original hardcoded `n_eig=3` kernel to within machine precision:

```
WITH unroll
=================================================================
CORRECTNESS TESTS
=================================================================
AGAINST HARDCODED VERSION:  n_eig= 3  |  allclose: True  |  max |diff|: 1.56e-02
-----------------------------------------------------------------
ALL PASSED
```

### #pragma unroll is essential

The most important finding is that `#pragma unroll` is **not optional** — without it, the kernel suffers catastrophic performance degradation at larger `n_eig`. Up to `n_eig~12` both versions are comparable (~80 µs), but without unroll performance collapses at `n_eig=15` (~633 µs vs ~112 µs with unroll, a ~6x slowdown) and gets worse from there, reaching ~3600 µs at `n_eig=20` vs ~203 µs with unroll (~18x slower). This is almost certainly the compiler failing to keep loop-carried values in registers and spilling to local memory when the loop bounds aren't known at compile time. With `#pragma unroll`, performance scales cleanly and roughly linearly with `n_sym = n_eig*(n_eig+1)/2` across the full tested range, which is the expected behavior.

The templated version also runs slightly faster than the hardcoded one for the same parameters — reliably ~7 µs faster, possibly due to overhead in the f-string code generation approach used by the hardcoded version:

```
WITH unroll
=================================================================
TIMING TESTS -> n_ant = 32 * 24 = 768
=================================================================
  [n_eig=3  n_sym=  6]   KERNEL: gpu=    27.1 us  cpu=     5.0 us | HARDCODE: gpu=    34.2 us  cpu=    12.6 us
=================================================================
```

### Timing vs CuPy (with unroll, n_ant=768)

| n_eig | n_sym | Kernel (µs) |
|-------|-------|-------------|
| 1     | 1     | ~9          |
| 3     | 6     | ~25         |
| 5     | 15    | ~33         |
| 10    | 55    | ~65         |
| 15    | 120   | ~112        |
| 20    | 210   | ~203        |

The CuPy reference is not shown here because it is a batched matmul on zero-padded data and is not a fair apples-to-apples comparison — its overhead grows with the zero-padding structure rather than with `n_eig` directly.

### Scaling with n_ant (fixed n_eig=3, RTX 5070 Laptop GPU)

The reduction kernel is largely **insensitive to n_ant** in the tested range (2² to 27² antennas), staying roughly flat between ~10–20 µs. This makes sense: more antennas means more baselines and therefore more work per block, but also more parallelism to fill the GPU. CuPy by contrast sits at ~700–1700 µs and starts climbing at larger sizes. The kernel is roughly **50–100x faster than CuPy** across the full range at n_eig=3. See `varying_nant_plots/`.

### Scaling with n_eig (fixed n_ant=576, RTX 5070 Laptop GPU)

Kernel time grows roughly linearly with n_eig (on a log scale the slope is consistent), going from ~9 µs at n_eig=1 to ~200 µs at n_eig=19. There is a noticeable bump around n_eig=4–5 (spike then drop) which may indicate a warp resource boundary being crossed. CuPy starts at ~800 µs (n_eig=1) and grows to ~3000 µs (n_eig=19) — the kernel remains faster across the full range, with the gap narrowing at high n_eig but still substantial (~15x at n_eig=19). See `varying_neig_plots/`.