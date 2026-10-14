"""
GPU counterpart of corrcal.sparse.SparseCov.

GpuSparseCov has the same attributes (noise, diff_mat, src_mat, edges, n_eig,
n_src, n_bls, n_grp, isinv) and methods (copy, apply_gains, inv, __matmul__,
expand) as the CPU class, so the likelihood and gradient can be written once
against that interface and checked line by line against corrcal.optimize.

Every method only dispatches to the existing launchers in linalg.py and
inverse_covariance.py. No math lives here.

Buffer ownership
----------------
All scratch and output buffers live in one CovWorkspace, allocated once from
the problem shape. Every object derived from the same model (copy(), the
gained copy, inv()) shares it. Consequences:

- copy() is shallow by default. No method writes into the arrays an object
  was constructed with: apply_gains and inv write into workspace buffers and
  rebind attributes, and code like `cov.noise = cp.zeros_like(cov.noise)`
  also only rebinds. So a shallow copy behaves like CorrCal's deep copy for
  every call pattern in corrcal.optimize.
- apply_gains writes into ws.diff_g / ws.src_g. A second object that shares
  the workspace and calls apply_gains overwrites the first one's gained
  matrices.
- inv() returns an object whose noise, diff_mat and src_mat ARE workspace
  buffers (ws.ninv, ws.inv.diffuse_bar, ws.inv.sig_prime). The next inv() on
  the same workspace overwrites them.
- __matmul__ returns a freshly allocated vector (CuPy memory pool), so
  p = cinv @ d and q = cov @ x coexist, which grad_nll needs.

One likelihood or gradient evaluation uses each workspace buffer once, so
this is safe there. To keep a result across evaluations use copy(deep=True).
"""

import cupy as cp
import numpy as np

from gpu_corrcal.inverse_covariance import InvCovWorkspace, inv_cov
from gpu_corrcal.linalg import MatvecWorkspace, gpu_apply_gains, sparse_cov_vec_mul


class CovWorkspace:
    """Every preallocated buffer GpuSparseCov needs, sized once per problem."""

    def __init__(self, diff_mat, src_mat, edges):
        n_row, n_eig = diff_mat.shape
        n_src = src_mat.shape[1]
        nb = edges.shape[0] - 1
        self.shape = (n_row, n_eig, n_src, nb)

        self.inv = InvCovWorkspace(diff_mat, src_mat, edges)   # Del_bar, Sig_bar, logdet parts
        self.mv = MatvecWorkspace(nb, n_eig, n_src)            # matvec scratch
        self.diff_g = cp.empty((n_row, n_eig), dtype=cp.float32)  # G Delta
        self.src_g = cp.empty((n_row, n_src), dtype=cp.float32)   # G Sigma
        self.ninv = cp.empty(n_row, dtype=cp.float32)             # N^-1


class GpuSparseCov:
    """
    Doubly sparse CorrCal covariance C = N + Delta Delta^T + Sigma Sigma^T on
    the GPU, float32, block-diagonal Delta only.

    Parameters
    ----------
    noise : (n_row,) diagonal of N. Host or device; converted to float32.
    src_mat : (n_row, n_src) Sigma.
    diff_mat : (n_row, n_eig) block-diagonal entries of Delta.
    edges : (nb+1,) Re/Im-split group edges. Stored as device int32.
    n_eig : must equal diff_mat.shape[1].
    isinv : True if this object holds (N^-1, Delta_bar, Sigma_bar) and
        represents C^-1 = N^-1 - Delta_bar Delta_bar^T - Sigma_bar Sigma_bar^T.
    ws : CovWorkspace to share. Allocated from the shapes if None.
    check : keep the launchers' shape/dtype assertions on. Turn off once the
        pipeline is validated.
    """

    def __init__(self, noise, src_mat, diff_mat, edges, n_eig, isinv=False,
                 ws=None, check=True):
        self.noise = cp.ascontiguousarray(cp.asarray(noise, dtype=cp.float32))
        self.src_mat = cp.ascontiguousarray(cp.asarray(src_mat, dtype=cp.float32))
        self.diff_mat = cp.ascontiguousarray(cp.asarray(diff_mat, dtype=cp.float32))
        self.edges = cp.asarray(edges, dtype=cp.int32)
        # Host copy for expand() and validation; one transfer at construction.
        self.edges_host = np.asarray(cp.asnumpy(self.edges), dtype=np.int64)

        n_row = self.noise.shape[0]
        self.n_grp = self.edges.shape[0] - 1
        self.n_bls = n_row // 2
        self.n_src = self.src_mat.shape[1]
        self.n_eig = int(n_eig)
        self.isinv = bool(isinv)
        self.check = check

        if self.diff_mat.shape != (n_row, self.n_eig):
            raise ValueError(
                f"diff_mat shape {self.diff_mat.shape} != (n_row, n_eig) = "
                f"({n_row}, {self.n_eig}); only block-diagonal Delta is supported."
            )
        if self.src_mat.shape[0] != n_row:
            raise ValueError(f"src_mat has {self.src_mat.shape[0]} rows, noise has {n_row}.")
        if self.edges_host[-1] != n_row or np.any(self.edges_host % 2):
            raise ValueError("edges must be Re/Im-split (all even) and end at n_row.")

        self.ws = ws if ws is not None else CovWorkspace(self.diff_mat, self.src_mat, self.edges)
        if self.ws.shape != (n_row, self.n_eig, self.n_src, self.n_grp):
            raise ValueError(f"workspace shape {self.ws.shape} does not match this covariance.")

    # ------------------------------------------------------------------
    def _derive(self, **changes):
        """New object sharing everything with self except the given attributes.
        Skips __init__, so no validation, allocation or host transfer."""
        new = object.__new__(type(self))
        new.__dict__.update(self.__dict__)
        new.__dict__.update(changes)
        return new

    def copy(self, deep=False):
        """Shallow by default (see module docstring). deep=True copies the
        noise, diffuse and source arrays; the workspace is still shared."""
        if not deep:
            return self._derive()
        return self._derive(
            noise=self.noise.copy(), src_mat=self.src_mat.copy(), diff_mat=self.diff_mat.copy()
        )

    # ------------------------------------------------------------------
    def apply_gains(self, gains, ant_1_array, ant_2_array):
        """
        Delta <- G Delta, Sigma <- G Sigma, in place on this object (rebinds to
        ws.diff_g / ws.src_g), as SparseCov.apply_gains.

        gains : (2*n_ant,) alternating Re/Im. Converted to device float32.
        ant_1_array, ant_2_array : (n_bls,) device int32. Pass int32 to avoid
            a conversion copy on every call.
        """
        if self.diff_mat is self.ws.diff_g or self.src_mat is self.ws.src_g:
            raise RuntimeError(
                "apply_gains called on an already-gained covariance: the output "
                "buffer would alias the input. Apply gains to a copy of the "
                "ungained model instead."
            )
        gains = cp.asarray(gains, dtype=cp.float32)
        a1 = cp.asarray(ant_1_array, dtype=cp.int32)
        a2 = cp.asarray(ant_2_array, dtype=cp.int32)

        self.diff_mat = gpu_apply_gains(gains, self.diff_mat, a1, a2,
                                        out=self.ws.diff_g, check=self.check)
        self.src_mat = gpu_apply_gains(gains, self.src_mat, a1, a2,
                                       out=self.ws.src_g, check=self.check)

    # ------------------------------------------------------------------
    def inv(self, return_det=False):
        """
        Woodbury inverse, as SparseCov.inv.

        Returns
        -------
        cinv : GpuSparseCov with isinv=True holding (N^-1, Delta_bar, Sigma_bar).
            Its arrays are workspace buffers (see module docstring).
        logdet : 0-d device float32, log det C - log det N (same convention as
            CorrCal). Only if return_det. Left on device so no sync is forced;
            float(logdet) when you need it on the host.
        """
        if self.isinv:
            raise NotImplementedError(
                "inv_cov only maps (N, Delta, Sigma) -> (N^-1, Delta_bar, Sigma_bar)."
            )
        cp.reciprocal(self.noise, out=self.ws.ninv)
        res = inv_cov(self.noise, self.diff_mat, self.src_mat, self.edges,
                      self.ws.inv, compute_det=return_det)
        cinv = self._derive(noise=self.ws.ninv, diff_mat=res[0], src_mat=res[1], isinv=True)
        if return_det:
            return cinv, res[2]
        return cinv

    # ------------------------------------------------------------------
    def matvec(self, vec, out=None):
        """C @ vec (or C^-1 @ vec if isinv). Allocates out if not given."""
        vec = cp.asarray(vec, dtype=cp.float32)
        if out is None:
            out = cp.empty_like(vec)
        return sparse_cov_vec_mul(
            self.noise, self.diff_mat, self.src_mat, vec, self.edges, self.ws.mv,
            isinv=self.isinv, out=out, check=self.check,
        )

    def __matmul__(self, vec):
        return self.matvec(vec)

    # ------------------------------------------------------------------
    def expand(self):
        """Dense matrix in float64, as SparseCov.expand. Testing only:
        O(n_row^2) memory."""
        D = self.diff_mat.astype(cp.float64)
        S = self.src_mat.astype(cp.float64)
        n_row = D.shape[0]

        lowrank = S @ S.T
        for start, stop in zip(self.edges_host[:-1], self.edges_host[1:]):
            Db = D[start:stop]
            lowrank[start:stop, start:stop] += Db @ Db.T

        dense = -lowrank if self.isinv else lowrank
        dense[cp.arange(n_row), cp.arange(n_row)] += self.noise.astype(cp.float64)
        return dense