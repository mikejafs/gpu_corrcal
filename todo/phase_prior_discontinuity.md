# TODO: phase prior is discontinuous at ±π

**Status:** open. Not blocking current validation or benchmarks.
**Fix when:** writing `nll_and_grad` (it touches the prior assembly anyway), and before any curvature, Fisher or Cramér–Rao work evaluated at the solution.
**Also report upstream:** r-pascua/corrcal has the same prior.

## The problem

The overall gain phase is unobservable, so CorrCal adds a prior to the negative log-likelihood:

```
prior = (mean_a arg g_a)^2 / phs_norm_fac^2,    arg g_a = arctan2(Im g_a, Re g_a) in (-pi, pi]
```

`arctan2` wraps at ±π. When one antenna's phase crosses the cut, its arg jumps by 2π, the mean jumps by 2π/N_ant, and the prior (and so f) has a step discontinuity along that direction.

## Observed

`curvature_study.ipynb`, 12x12 grid, SNR 1e4, `phs_norm_fac = 0.1`:

- CG stopped with antenna 21 at phase 3.141592, pinned against the step. Moving it across raised f by 0.75.
- Finite-difference curvature across the step gave a spurious eigenvalue that scales as 1/h: −6.9e4, −4.6e3 and −431 at h = 1e-6, 1e-5, 1e-4.

## Impact

1. **Derivatives at the solution.** Any Fisher matrix, Cramér–Rao comparison, Newton or trust-region step, or finite-difference check is invalid when an antenna sits near ±π. This is the main reason to fix it.
2. **Bias on a pinned antenna.** CG stays pinned when the antenna's optimum lies within about δ ≈ sqrt(2 · jump / H) of the cut, where H is the curvature of that antenna's phase. At 144 dishes and SNR 1e4 that is about 2e-3 rad, comparable to the noise error, on roughly one antenna occasionally. The jump scales as (2π/N_ant)² / phs_norm_fac², so it is about 15x smaller at 512 dishes: negligible at CHORD scale.
3. **Line searches** can stall against the step. Rare in practice.

CPU and GPU share the same prior, so CPU vs GPU comparisons are unaffected.

## Fix: prior on the phase of the circular mean

```
u_a = g_a / |g_a| = e^{i phi_a}
C = sum_a cos phi_a,   S = sum_a sin phi_a,   R^2 = C^2 + S^2
psi = atan2(S, C)                      # phase of sum_a u_a: continuous as each phi_a crosses ±pi
prior = psi^2 / phs_norm_fac^2
```

Gradient, with x = (Re g_a, Im g_a) interleaved:

```
d psi / d phi_a     = (cos phi_a * C + sin phi_a * S) / R^2
d phi_a / d Re g_a  = -Im g_a / |g_a|^2
d phi_a / d Im g_a  =  Re g_a / |g_a|^2

pref_a              = (2 psi / phs_norm_fac^2) * (d psi / d phi_a) / |g_a|^2
grad[Re g_a]       += -pref_a * Im g_a
grad[Im g_a]       +=  pref_a * Re g_a
```

Checked numerically: the gradient matches central differences to 7e-10 relative, and the prior is continuous through ±π.

The only singular case is R = 0 (unit phasors summing to zero), which does not occur for calibrated gains.

### Reference implementation (NumPy)

```python
def phase_prior(gains, phs_norm_fac):
    g = gains[0::2] + 1j * gains[1::2]
    ph = np.angle(g)
    C, S = np.cos(ph).sum(), np.sin(ph).sum()
    psi = np.arctan2(S, C)
    val = psi ** 2 / phs_norm_fac ** 2
    pref = 2 * psi / phs_norm_fac ** 2 * (np.cos(ph) * C + np.sin(ph) * S) / (C * C + S * S) / np.abs(g) ** 2
    grad = np.empty_like(gains)
    grad[0::2] = -pref * g.imag
    grad[1::2] = pref * g.real
    return val, grad
```

## Where to change it

| Code | Location | Change |
|---|---|---|
| gpu_corrcal `nll` | `src/gpu_corrcal/optimize.py`, `phases = cp.arctan2(...)` / `phs_norm = cp.mean(phases)**2 / ...` (lines ~69–70) | use `psi` above |
| gpu_corrcal `grad_nll` | `optimize.py`: `phs_coef = 2 / (n_ant**2 * phs_norm_fac**2)` (~line 163) | pass `2 / phs_norm_fac**2`; psi, C, S are computed in the kernel |
| GPU kernel, accumulation | `kernels/grad_nll_kernels.cu`, K1: `atomicAdd(phs_sum, ph)` (~line 224) | accumulate two sums, `cos(ph)` and `sin(ph)`: `phs_sum` becomes a (2,) buffer |
| GPU kernel, application | `grad_nll_kernels.cu`, K3: `pref = phs_coef * (*phs_sum) / (gr*gr + gi*gi)` (~line 440) | `psi = atan2(S, C)`, `pref = phs_coef * psi * (cos ph * C + sin ph * S) / (C*C + S*S) / (gr*gr + gi*gi)` |
| workspace | `linalg.py` `GradWorkspace`: `self.phs_sum = cp.empty(1, ...)` (~line 447) | size 2 |
| CPU reference | r-pascua/corrcal `corrcal/optimize.py`: `nll` (lines ~69–70), `grad_nll` (lines ~164–167) | same change, for the accuracy tests |

The kernel structure is unchanged: one reduction pass (now two sums instead of one), then a per-antenna application.

## Test plan

1. Unit test `phase_prior` against central differences, including an antenna placed at π ± 1e-7.
2. Update the CPU reference, then rerun `tests_accuracy_nll` and `tests_accuracy_grad_nll` with a finite `phs_norm_fac`.
3. Rerun the 12x12 case in `curvature_study.ipynb` with `phs_norm_fac = 0.1`: the reference optimum should no longer pin an antenna at ±π, and the smallest curvature eigenvalue should be positive.

## Note on meaning

For tightly clustered phases, psi ≈ mean(phi), so `phs_norm_fac` keeps its meaning. For phases spread over the circle (random gains), the two differ, but both only pin the overall phase, which is an exact degeneracy and removed by referencing anyway.