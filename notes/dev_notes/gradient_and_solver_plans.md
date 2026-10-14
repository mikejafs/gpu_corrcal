## Gradient + solver plan (2026-10-14)

### Optimizer
- Stay with `scipy.optimize.minimize(method="CG")`. CuPy has no scipy.optimize equivalent.
- Gains live on the host (scipy-owned NumPy array). Each evaluation uploads a temporary device copy and sends back only the nll scalar and the gradient (2·n_ant floats). The model (N, Δ, Σ, data, workspaces) stays in VRAM.
- Per-call host↔device sync is accepted: latency-bound, roughly 20–40 µs per eval pair vs ~416 µs nll alone at 512 ant.
- Custom GPU solver deferred. It would only matter for reducing iteration count (e.g. Fisher-diagonal preconditioning), not for removing syncs.

### Two gradient entry points
| | `grad_nll` (CPU mirror) | `nll_and_grad` (solver) |
|---|---|---|
| Purpose | accuracy test vs CorrCal; 4-line plot vs Pascua+ Fig. 2 | pass to `minimize(..., jac=True)` |
| apply_gains + inverse | own call, `inv(return_det=False)` | one call, `inv(return_det=True)` |
| χ² = d·p | no | yes, reusing p |
| Returns | gradient | `(nll, gradient)` |

The combined version saves one `apply_gains` + one full inverse per trial point. scipy caches the `(f, g)` tuple per x.

### Shared gradient helper (both versions)
1. p = C⁻¹d (gained inverse)
2. q = (C − N) Gᵀp: matvec on the **ungained** model with a preallocated zero-noise vector, `isinv=False`
3. s, t, P̄ (P̄ = row-sum of squares over Δ̄ and Σ̄, Re+Im pairs): CuPy elementwise to start
4. `accumulate_gradient`: **new CUDA kernel**, one thread per baseline, `atomicAdd` into the 2·n_ant output (4 entries per antenna per baseline). Fall back to a sort-by-antenna segmented reduction if contention shows in profiling.
5. Phase-prior term, then `/ scale`

### Gotchas when comparing to CPU CorrCal
- CPU `grad_nll` divides gains by `scale` twice (`gains = gains/scale` then `apply_gains(gains/scale, ...)`). Test at `scale=1`; GPU divides once.
- `accumulate_gradient` uses `noise[2*k]` (real-row noise only) in the trace prefactor. Mirror exactly; check against the noise factor-of-2 question from the Hessian derivation.

### Next steps
- [ ] Write `accumulate_gradient` kernel + launcher
- [ ] Shared helper, `grad_nll`, `nll_and_grad` in `optimize.py`
- [ ] Accuracy test vs `corrcal.optimize.grad_nll` (scale=1)
- [ ] `likelihood_stack` benchmark target: Inv, Inv+logdet, nll, grad → compare to Pascua+ Fig. 2
- [ ] CPU run at n_eig=2, n_src=1 to match the paper's settings (paper: nll ≈ 0.125 s, grad ≈ 0.19 s at 1000 ant, i7-8565U single core)
- [ ] With representative data: one full `minimize` solve, record wall time, n_evals, mean eval time → decide whether solver work (preconditioning) is worth it