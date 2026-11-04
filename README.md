# gpu_corrcal

GPU-accelerated Correlation Calibration (CorrCal) for large radio interferometers, built for the 512-dish [CHORD](https://arxiv.org/abs/1911.01777) telescope.

CorrCal ([Sievers 2017](https://arxiv.org/abs/1701.01860); [Pascua et al. 2026](https://arxiv.org/abs/2602.06109)) solves for antenna gains by maximizing a Gaussian likelihood in which the sky is marginalized into the data covariance:

```
C(g) = N + G (Δ Δᵀ + Σ Σᵀ) Gᵀ
-log L(g) = dᵀ C⁻¹ d + log det C
```

`N` is diagonal noise, `Δ` is a low-rank, block-diagonal term for redundant baseline groups, `Σ` holds bright point sources, and `G` applies per-baseline gain products. `gpu_corrcal` evaluates the likelihood and its analytic gradient with custom CUDA kernels that act directly on this block-sparse structure through a two-level Woodbury inversion. The full covariance is never formed.

## Performance

On an NVIDIA A40 at CHORD's 512 antennas:

| Operation | GPU time | Speedup vs CPU CorrCal |
|---|---|---|
| Inverse covariance | ~100 µs | >100× |
| Gradient of −log L | [fill] | ~200× |

Results agree with the CPU reference to single precision.

![Benchmark](results/benchmark_plots/<FIGURE>.png)

## Status

- [x] Inverse covariance and log-determinant (two-level Woodbury, fused Cholesky kernels)
- [x] Covariance and inverse-covariance mat-vec
- [x] Gain application
- [x] Negative log-likelihood and analytic gradient
- [x] End-to-end calibration in simulation (`scipy.optimize` driving GPU nll/grad)
- [ ] Fused nll + gradient call
- [ ] Curvature (Fisher) preconditioning
- [ ] Application to CHORD Pathfinder data

## Installation

Requires an NVIDIA GPU, the CUDA toolkit (`nvcc`), and CuPy.

```bash
git clone https://github.com/mikejafs/gpu_corrcal
cd gpu_corrcal
pip install -e .

# Compile each kernel into a shared library next to its source
cd src/gpu_corrcal/kernels
for f in *.cu; do nvcc -O3 -Xcompiler -fPIC -shared -o "${f%.cu}.so" "$f"; done
```

## Usage

```python
from scipy.optimize import minimize
from gpu_corrcal.sparse import GpuSparseCov
from gpu_corrcal.optimize import nll, grad_nll

# noise: (n_row,), src_mat: (n_row, n_src), diff_mat: (n_row, n_eig),
# edges: (n_grp + 1,) redundant-group boundaries in Re/Im-split row order
cov = GpuSparseCov(noise, src_mat, diff_mat, edges, n_eig)

res = minimize(
    nll, gains0, jac=grad_nll,
    args=(cov, data, ant_1_inds, ant_2_inds),
    method="CG",
)
```

`data`, `ant_1_inds` and `ant_2_inds` should be on the device; see the docstrings in `optimize.py` for dtypes and conventions.

## Layout

```
src/gpu_corrcal/
  sparse.py        GpuSparseCov: covariance container, inverse, mat-vec, gains
  optimize.py      nll and grad_nll for use with scipy.optimize
  linalg.py        kernel launchers
  kernels/         CUDA sources
  tests/           accuracy tests against CPU CorrCal and benchmarks
```

## Author

Michael Jafs, McGill University and the CHORD Collaboration. Supervised by Jonathan Sievers.
