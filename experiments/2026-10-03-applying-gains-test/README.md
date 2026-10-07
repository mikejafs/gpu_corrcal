# Experiment: applying gains in the zero-padded CuPy path

## Question

Is the output of `cupy_utils.apply_gains` still zero-padded, and therefore which steps of the CuPy gain application have to run on every likelihood evaluation? The answer decides whether the CuPy reference benchmark should time `apply_gains` alone or `zeropad_gains + apply_gains`.

## What `apply_gains_ex.py` does

Runs the full zero-padded path on a small 3x4 antenna array (n_ant = 12, n_bl = 66, 17 redundant blocks) so every array can be printed:

1. Simulates noise, Δ, Σ, gains and antenna index arrays with `SimCorrcalParams`.
2. Zero-pads Δ with `zeroPad`.
3. Builds and zero-pads the per-baseline complex gain matrix g_p conj(g_q) with `zeropad_gains`.
4. Applies it with `apply_gains(G, zp_diffuse)`.
5. Prints the shape of each array, checks that every padded row is zero in Δ, G and the output, and prints the smallest block in full.
6. Un-pads the output with `undo_zeroPad` and compares it with CorrCal's CPU `apply_gains_to_mat`.

## Run

```bash
conda activate gpu_corrcal
python apply_gains_ex.py
```

## Expected shapes

| Array | Shape |
|---|---|
| `zp_diffuse` | (n_blocks, largest_block, n_eig) |
| `G` | (n_blocks, largest_block/2, 1), complex, one entry per baseline |
| `out` | (n_blocks, largest_block, n_eig) |

`out = xp.zeros_like(mat)` keeps the padded shape, and no reshape collapses it. Padded rows stay zero because both G and the padded matrix are zero there. The padding check should print 0 for all three arrays, and the comparison with CorrCal should agree to float32 precision.

## What this means for benchmarking

CorrCal's `nll` and `grad_nll` start each call from the ungained Δ and Σ and apply the current gains. The CG optimizer (`scipy.optimize.minimize(method="CG")`) changes the gains at every step, so:

- **Once per calibration:** `zeroPad` of Δ, Σ, N^-1 and data.
- **Every `nll` / `grad_nll` call:** `zeropad_gains` (once, shared by Δ and Σ), then `apply_gains` on Δ and on Σ.

The CuPy benchmark should time `zeropad_gains + apply_gains(Δ) + apply_gains(Σ)`. Timing `apply_gains` alone leaves out work that runs on every iteration.

## Fixes needed in `cupy_utils.py` to run this

1. Import `zeroPad` from its module: `from gpu_corrcal.utils.zp_puregpu_funcs_py import zeroPad`.
2. Pass `dtype` to `zeroPad` in `zeropad_gains` (and in `apply_gains_to_mat`), e.g. `dtype=cp.float64`. `zeroPad` has no default for it.