# Short Summaries of Working Sessions

# Working Session Log 

- [April 15 2026](#april-15-2026)
- [April 29 2026](#april-29-2026)


## April 15 2026
- Chatted with Jon -- needed to get rid of hardcoded eigenmode in the reduction kernel
- re-wrote kernel for general eigenmodes; still awaiting detailed tests for correctness etc..

## April 29 2026

### Coding Work

- 

### Results



## June 9 2026

### Coding Work

- Implemented three-kernel Sig_prime (source-level Woodbury) chain: `cap_reduce` (capacitance reduction), `chol_inv_fused` (single n_src × n_src Cholesky + inverse), `apply_sig_prime` (assemble Gamma, apply triangular solve)
- Replaced 400-line explicit template instantiation block with X-macros (~25 lines). Each kernel's signature now written once in `INST_CAP`/`INST_APPLY`/`INST_CHOL`; preprocessor expands the 1–20 × 1–20 grid. Prevents signature drift between instantiation and definition
- Added `extern "C"` dispatch wrappers with function-pointer tables (`launch_cap_reduce`, `launch_chol_inv_fused`, `launch_apply_sig_prime`). Runtime `(n_src, n_eig)` indexes into table — no 400-way switch, ctypes-friendly, returns nonzero on out-of-range dispatch
- Fixed critical ctypes argument-order bug: C dispatch functions take `(n_src, n_eig, nb, pointers..., stream)` but Python bindings were passing `(pointers..., nb, n_src, n_eig)` with no stream. Pointers were being truncated into int slots and vice versa. `M_sig` stayed at zero because `cap_reduce` was either not launching or writing to garbage addresses. Fixed `_kernels.py` argtypes and `linalg.py` wrappers to match C signature; added `restype = c_int` and return-code checks on all three launchers
- Fixed `xp.eye(Sig.shape[2])` → `xp.eye(Sig.shape[2], dtype=cp.float32)` in CuPy reference (same fp64 promotion bug as the diffuse level)
- Built per-kernel timing harness (`benchmark_per_kernel.py`) using `cupyx.profiler.benchmark` to isolate which kernel owns runtime at each antenna count
- Ran Nsight Compute (`ncu`) profiling on `cap_reduce` at 35×35 antennas. Required running WSL terminal as Administrator for GPU counter permissions. Key findings:
  - Achieved occupancy: 99.2% — not the bottleneck
  - 58.5% of stall cycles: MIO scoreboard (shared-memory atomic serialization)
  - Only 11.6 / 32 threads active per warp — threads waiting for `atomicAdd` to shared memory
  - L1/TEX throughput at 97% — hardware busy serializing atomics, not doing useful compute
- Attempted per-warp tiled shared memory (8 private tiles, one per warp) — no improvement because the contention is intra-warp (32 threads hitting the same address), not cross-warp
- Rewrote `cap_reduce` to register-accumulate: each thread keeps private `rB[NB_TILE]` and `rS[NT]` arrays, accumulates with plain `+=` in the hot loop (zero shared atomics), then reduces via `__shfl_down_sync` within each warp, then lane 0 of each warp does one `atomicAdd` into shared. Total shared atomics per element: 8 (one per warp) instead of group_size (~2380 at 35×35). Awaiting benchmark results

### Results

- Correctness: `allclose` passes for all n_eig = 1–20 at n_src = 3, max |diff| ~ 1e-4 (float32 accumulation noise, no outliers or n_eig dependence)
- Timing (full `inv_cov`, A40, n_eig = 5, n_src = 15): ~4–5× speedup over float32 CuPy reference across antenna sweep, ratio growing at large antenna count
- Timing (full `inv_cov`, RTX 5070, n_eig = 5, n_src = 15): ~4× at 35×35, with `cap_reduce` dominating at 60–90% of pipeline time
- `cap_reduce` profiled as memory/atomic-bound, not occupancy-bound — register-accumulate rewrite targets the confirmed bottleneck (58% MIO stalls)
- Diffuse-level (`Del_prime`) speedup remains ~20× on A40 — combined `inv_cov` speedup is stronger than Sig_prime alone suggests since diffuse dominates at moderate n_eig

### Open Items

- Benchmark register-accumulate `cap_reduce` and re-profile with `ncu` to confirm MIO stall reduction
- Run n_src sweep (fixed n_eig, vary n_src 1–20) to validate the "any pair in 1–20" claim on both axes
- Benchmark full `inv_cov` (diffuse + source combined) as the number the CG solver sees — this is the figure that justifies the work
- Find the antenna count where CuPy OOMs due to dense matrix materialization — that point (where the orange line stops existing) is the strongest argument for the collaboration
- Push to A40 cluster for production-hardware numbers before the CHORD talk