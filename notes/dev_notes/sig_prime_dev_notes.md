# Dev Notes Concerning the Second half of Woodbury

## Overview

The second Woodbury level computes the whitened source factor `Sig_prime` (aka `Sig_bar`) such that:

```
C^{-1} = N^{-1} - Del_bar Del_bar^T - Sig_bar Sig_bar^T
```

for `C = N + Delta Delta^T + Sigma Sigma^T`. `Del_bar` (the diffuse level) is computed by the existing Del_prime pipeline. The Sig_prime chain takes `Del_prime` as input and produces the source-level correction.

The pipeline is three kernels launched in sequence:

1. **`cap_reduce`** — builds per-group B tile and accumulates the capacitance matrix Lambda into M_sig
2. **`chol_inv_fused`** — single-thread Cholesky + triangular inverse on the small (n_src × n_src) matrix I + Lambda
3. **`apply_sig_prime`** — per-row Gamma assembly and triangular solve to produce the final Sig_prime output

All three are templated on `<N_SRC, N_EIG>` (or `<N_SRC>` for chol_inv_fused) and use the ragged CSR `edges` layout — no zero-padding, one CUDA block per redundant group.


## Kernel Details

### Kernel 1: `cap_reduce<N_SRC, N_EIG>`

**Purpose:** For each redundant group b, compute:
- `B[b][s,e] = sum_i Sig[i,s] * Del'[i,e]` — stored to global, read by kernel 3
- `Lambda = sum_b (Sig^T A - B B^T)` — accumulated into M_sig via global atomicAdd (lower tri only)

where `A[i,s] = Sig[i,s] / N[i]`.

**Why B is stored, not recomputed:** B carries two independent row indices within a group (i and i') via `B B^T`, so it can't factor to a per-row scalar. At CHORD redundancy (group sizes ~1000+), recomputing B in kernel 3 would re-run a full reduction over all baselines. One small write + one small read is cheaper.

**Layout:** `B` is `(nb, N_SRC, N_EIG)` — fixed by sources/modes, does NOT grow with n_total.

**Accumulator pattern:** `M_sig` is zeroed by the caller (`ws.M_sig.fill(0.0)` in Python) before each `inv_cov` call. The kernel atomicAdds per-group contributions into it. This is the only accumulator in the pipeline that requires explicit zeroing.

**Original implementation:** Shared-memory atomics in the hot loop. Each thread atomicAdds into `Bsh[sidx*N_EIG+e]` and `SHA[k]` on every row it processes. At 256 threads and group_size=2380, this is ~2380 × 165 atomics per block.

**Profiling results (ncu, RTX 5070, `cap_reduce<10,5>` at 35×35):**
- Achieved occupancy: 99.2% — not the bottleneck
- 58.5% of stall cycles: MIO scoreboard (shared-memory atomic serialization)
- Avg active threads per warp: 11.6 / 32 — threads waiting for their turn at the same shared address
- L1/TEX throughput: 97% — hardware busy serializing atomics, not doing useful compute
- Compute and memory throughput both at 97% — the "busy" unit is L1 processing atomics

**Optimization attempt 1 — per-warp tiled shared memory:** Gave each of 8 warps its own private `Bsh[warp][...]` tile, reducing cross-warp contention. Result: no measurable improvement. The profiler showed the contention is *intra-warp* (32 threads within the same warp hitting the same address), which per-warp tiling doesn't fix. Reverted.

**Optimization attempt 2 — register-accumulate + warp shuffle:** Each thread keeps private `rB[NB_TILE]` and `rS[NT]` arrays, accumulates with plain `+=` in the hot loop (zero shared atomics). After all rows, intra-warp reduction via `__shfl_down_sync` combines 32 threads → lane 0. Then lane 0 of each warp does one `atomicAdd` into shared. Total shared atomics per element: 8 (one per warp) instead of group_size. At large N_SRC/N_EIG the register arrays spill to L1-cached local memory, but pipelined L1 loads are non-blocking unlike serialized shared atomics. Awaiting benchmark results.


### Kernel 2: `chol_inv_fused<N_SRC>`

**Purpose:** Compute `L_inv = (chol(I + Lambda))^{-1}`, where Lambda is the lower-triangular M_sig from kernel 1.

**Launch:** `<<<1, 1>>>` — single thread. The matrix is `(n_src × n_src)`, at most 20×20 = 400 floats. Negligible runtime (~50µs, <1% of pipeline at scale).

**Key correctness note:** The `+ I` (identity) is added here, not in kernel 1. Kernel 1 accumulates only Lambda; kernel 2 adds 1.0 to the diagonal before Cholesky. This is the same `+1.0f` on the diagonal that was a real correctness bug in the diffuse-level kernel — missing it gives `chol(Lambda)` instead of `chol(I + Lambda)`.

**Output:** Lower-triangular `L_inv`. Strictly-upper entries are explicitly zeroed (output buffer may be `cp.empty`).


### Kernel 3: `apply_sig_prime<N_SRC, N_EIG>`

**Purpose:** For each baseline row i in group b:
```
Gamma[i,s] = Sig[i,s]/N[i] - sum_e Del'[i,e] * B[b][s,e]
Sig_prime[i,s] = sum_{t<=s} Gamma[i,t] * L_inv[s,t]
```

**Design:** Reads the precomputed B tile (from kernel 1) and L_inv (from kernel 2) into shared memory once per block. Then each thread strides over its rows, computing Gamma and applying the triangular solve. No reduction, no atomics — purely per-row independent work.

**Shared memory:** `Bsh[N_SRC*N_EIG]` + `Ls[N_SRC*N_SRC]`. Both loaded with the standard strided `for (t = threadIdx.x; t < N; t += blockDim.x)` pattern.


## Template Instantiation

All three kernels are templated on compile-time integer parameters. Explicit instantiation is required for the `.so` because the specialization is selected at runtime from Python.

**X-macro approach** (replaces 400+ lines of hand-written instantiations):
```cpp
#define NSRC_LIST(F, ...) F(1,__VA_ARGS__) F(2,__VA_ARGS__) ... F(20,__VA_ARGS__)
#define GRID_2D(F)  EIG_ROW(1,F) EIG_ROW(2,F) ... EIG_ROW(20,F)

#define INST_CAP(NS, NE) \
    template __global__ void cap_reduce<NS,NE>( \
        const float*,const float*,const float*,const int*,float*,float*,int);

GRID_2D(INST_CAP)         // 400 cap_reduce specializations
GRID_2D(INST_APPLY)       // 400 apply_sig_prime specializations
NSRC_LIST(INST_CHOL, 0)   // 20  chol_inv_fused specializations
```

Each kernel's signature is written **once** in its `INST_*` macro. The same macros are reused to fill the dispatch function-pointer tables, so the instantiation set and dispatch table can never disagree.

**Critical lesson:** The original 400-line explicit instantiation block had `apply_sig_prime` with 7 args instead of 8 (missing `const float* L_inv`). This produced `"no instance of function template matches the specified type"` × 100 errors. The X-macro approach makes this class of bug structurally impossible.


## Dispatch (extern "C" Wrappers)

Three `extern "C"` functions callable from ctypes:
- `launch_cap_reduce(n_src, n_eig, nb, Sig, Del_prime, noise, edges, B, M_sig, stream)`
- `launch_chol_inv_fused(n_src, M_sig, L_inv, stream)`
- `launch_apply_sig_prime(n_src, n_eig, nb, Sig, Del_prime, noise, edges, B, L_inv, Sig_prime, stream)`

All return `int` (0 = success, nonzero = `(n_src, n_eig)` out of compiled 1–20 range).

Function-pointer tables (`cap_tab[NSRC_MAX][NEIG_MAX]`, etc.) are filled lazily on first call using the same `GRID_2D`/`NSRC_LIST` macros. Runtime `(n_src, n_eig)` indexes into the table directly — no switch/if chain.

**Argument order convention:** Scalars first `(n_src, n_eig, nb)`, then pointers, then `stream`. This differs from the older diffuse-level bindings which put pointers first. The convention split caused a critical ctypes bug (see below).


## Bugs Encountered and Fixed

### 1. Template instantiation signature mismatch
**Symptom:** 100 nvcc errors `"no instance of function template matches the specified type"`.
**Cause:** `apply_sig_prime` instantiations listed 7 parameters; the definition has 8 (missing `const float* L_inv`). The generator script reused the `cap_reduce` signature for all three kernels.
**Fix:** X-macro instantiation with per-kernel signature written once.

### 2. ctypes argument-order scramble
**Symptom:** `M_sig` always zero after `cap_reduce`, but `B` was nonzero. Kernel output `Sig_prime` had mean ≈ 0.5 (raw input), or was all zeros, or garbage — varying with n_eig.
**Cause:** C dispatch takes `(int n_src, int n_eig, int nb, pointers..., void* stream)`. Python `argtypes` declared `(pointers..., int nb, int n_src, int n_eig)` with no stream. ctypes was shoving 64-bit pointers into 32-bit int slots (truncating) and small ints into pointer slots (garbage addresses). The kernel either didn't launch (dispatch miss from truncated n_src/n_eig) or launched with garbage pointer args.
**Why B worked but M_sig didn't:** The pointer truncation happened to land B's address where the kernel expected an output (luck of memory layout), but M_sig's slot received a different garbage value. The `if (!in_range(n_src, n_eig))` guard sometimes returned early (no launch, M_sig stays zero), sometimes didn't (launch with garbage).
**Fix:** Match Python argtypes to C signature exactly (scalars first, pointers next, stream=0 last). Set `restype = c_int` and check return code on every call.
**Lesson:** When the convention for arg order differs between old and new bindings, pattern-matching the new on the old silently corrupts everything. `restype = c_int` + raise-on-nonzero catches dispatch failures immediately.

### 3. fp64 promotion in CuPy reference
**Symptom:** CuPy reference producing wrong-scale output, cuSOLVER using fp64 kernels.
**Cause:** `xp.eye(Sig.shape[2])` defaults to float64, promoting all downstream matmuls.
**Fix:** `xp.eye(Sig.shape[2], dtype=cp.float32)`. Same bug as the diffuse level, recurring in the source branch.

### 4. `undo_zeroPad` corrupting reference output
**Symptom:** CuPy reference output appeared frozen across n_eig (identical `[2.9675, 1.0514, ...]` vector), with occasional all-zeros, sign flips, or echoes of previous iterations.
**Cause:** `undo_zeroPad` with `ReImsplit=True` was aliasing buffers and introducing stale/uninitialized data. The reference was actually correct *before* un-padding.
**Fix:** Compared in padded space (before `undo_zeroPad`) to verify the reference was healthy, then identified the un-pad as the corruption source. Scale probes (`cp.abs(...).max()` and `.mean()`) before vs after un-pad isolated the problem.

### 5. n_src=1 masking real bugs
**Symptom:** Tests at n_src=1 showed weird patterns but were hard to interpret — frozen reference, collapsed Sig_prime.
**Cause:** With n_src=1, the entire second Woodbury level degenerates to scalar arithmetic. `M_sig` is 1×1, `L_sig = chol(1 + scalar) = scalar`, and `Sig_prime ≈ A = N_inv * Sig`. This can't exercise the lower-triangular solve, `B B^T` cross terms, or packed-index logic. The test was validating almost nothing.
**Fix:** Set n_src ≥ 3 for correctness tests. n_src=1 is a degenerate case, not a useful test point.


## Benchmark Methodology

### Per-kernel timing
`benchmark_per_kernel.py` uses `cupyx.profiler.benchmark` to time each of the 6 stages independently. Key design choices:
- Full chain runs once first so every stage gets physically correct inputs (not random data that could change branch/NaN behavior)
- `cap_reduce` stage re-zeros `M_sig` every repeat inside the timed callable (it's an accumulator; without re-zero you'd measure accumulate-on-top, wrong result and different atomic behavior)
- Reports `nb`, max group size, and mean group size per config — these are the numbers that test occupancy/contention hypotheses

**Per-kernel harness caveat:** The per-kernel `cap_reduce` numbers were ~3× higher than the same kernel timed inside the full pipeline (11044µs vs ~2800µs at 35×35). The discrepancy was caused by `benchmark`'s repeat loop interacting with the `.fill(0)` inside the timed callable. The full-pipeline manual-sync timer (`sync(); t0; make_cap_reduce(); sync(); t1`) gave the trustworthy number. Always reconcile per-kernel and full-pipeline numbers before optimizing.

### Full pipeline timing
`benchmark-inv_cov.py` times the complete `inv_cov` call (diffuse + source) using `cupyx.profiler.benchmark`. The `gpu` column from benchmark is the trustworthy number; `cpu` measures only launch/enqueue time and diverges from gpu at large N.


## Performance Summary (as of June 9 2026)

### Correctness
- `allclose` passes for n_eig = 1–20 at n_src = 3 against float32 CuPy reference
- Max |diff| ~ 1e-4, uniform across n_eig — float32 accumulation noise, no outliers

### Timing (full inv_cov, A40, n_eig = 5, n_src = 15)
- ~4–5× speedup over CuPy across antenna sweep, ratio growing at large antenna count
- Diffuse level alone: ~20× (unchanged from prior work)
- `cap_reduce` dominates source-level runtime at 60–90% depending on config

### Profiling-confirmed bottleneck
- `cap_reduce` is shared-atomic-bound (58% MIO stalls, 11.6/32 active threads)
- Occupancy is 99% — NOT the problem
- Register-accumulate + warp-shuffle rewrite targets the confirmed bottleneck directly


## File Structure

```
sig_prime_kernels.cu      — all three kernels + X-macro instantiations + dispatch
                            compile: nvcc -Xcompiler -fPIC -shared -o sig_prime_kernels.so sig_prime_kernels.cu
                            .so is .gitignored, recompiled per machine

_kernels.py               — ctypes bindings (argtypes, restype, dlsym)
                            convention: scalars first, then pointers, then stream
                            restype = c_int on all three launchers

linalg.py                 — thin Python wrappers (make_cap_reduce, fused_cholesky_inverse_sig,
                            apply_sig_prime_launch). Pass n_src/n_eig first, pointers next,
                            stream=0 last. Check return code, raise on nonzero.

inverse_covariance.py     — InvCovWorkspace (preallocates all buffers) + inv_cov() entry point.
                            M_sig.fill(0.0) before every cap_reduce call.
```