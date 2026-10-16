# Folder for storing various linear algebra functions written over top of cuda kernels

import ctypes
from gpu_corrcal._kernels import *

# warp_reduction_kernel_lib = load_warp_reduction_kernel()

def make_small_blocks(
        noise_gpu, diffuse_gpu, edges_gpu, out
        ):
    
    """Launch the kernel through ctypes."""
    n_eig = diffuse_gpu.shape[1]
    nb = len(edges_gpu) - 1
    threads_per_block = 128
    
    warp_reduction(
        diffuse_gpu.data.ptr,
        noise_gpu.data.ptr,
        edges_gpu.data.ptr,
        out.data.ptr,
        ctypes.c_int(nb),
        ctypes.c_int(n_eig),
        ctypes.c_int(threads_per_block),
    )
    # cp.cuda.Stream.null.synchronize()
    return out

def fused_cholesky_inverse(temp2, edges, out, out_diffuse_det=None):
    n_eig = temp2.shape[1]
    num_blocks = len(edges) - 1
    compute_det = out_diffuse_det is not None
    fused_chol_inv(
        temp2.data.ptr,
        out.data.ptr,
        out_diffuse_det.data.ptr if compute_det else 0,
        num_blocks,
        n_eig,
        int(compute_det),
    )
    # cp.cuda.Stream.null.synchronize()
    return out

def mul_temp_by_diffuse_chol(noise, diffuse, L_inv_T, out, edges):
    n_eig = diffuse.shape[1]
    nb = len(edges) - 1
    mul_temp_by_diffuse_chol_inv(
        noise.data.ptr,
        diffuse.data.ptr,
        L_inv_T.data.ptr,
        out.data.ptr,
        edges.data.ptr,
        nb,
        n_eig
    )
    # cp.cuda.Stream.null.synchronize()
    return out

# ============================================================================
# Additions to gpu_corrcal/linalg.py  --  Sig_prime thin launchers.
# Append after mul_temp_by_diffuse_chol. Same style: pull dims from shapes,
# nb = len(edges)-1, pass .data.ptr, return the out buffer.
# ============================================================================
 
# def make_cap_reduce(Sig, Del_prime, noise, edges, B, M_sig):
#     """Kernel 1: build per-group B and accumulate the capacitance Lambda into M_sig.
 
#     M_sig is an accumulator (atomicAdd across groups) and MUST be zeroed before
#     this call -- caller is responsible (see inv_cov / workspace reset).
#     """
#     n_src = Sig.shape[1]
#     n_eig = Del_prime.shape[1]
#     nb = len(edges) - 1
#     cap_reduce(
#         Sig.data.ptr,
#         Del_prime.data.ptr,
#         noise.data.ptr,
#         edges.data.ptr,
#         B.data.ptr,
#         M_sig.data.ptr,
#         nb,
#         n_src,
#         n_eig,
#     )
#     # cp.cuda.Stream.null.synchronize()
#     return B, M_sig
 
# def fused_cholesky_inverse_sig(M_sig, out):
#     """Kernel 2: L_inv = (chol(I + Lambda))^{-1} for the single (n_src x n_src) M_sig."""
#     n_src = M_sig.shape[0]
#     chol_inv_fused(
#         M_sig.data.ptr,
#         out.data.ptr,
#         n_src,
#     )
#     # cp.cuda.Stream.null.synchronize()
#     return out
 
# def apply_sig_prime_launch(Sig, Del_prime, noise, edges, B, L_inv, out):
#     """Kernel 3: Gamma = A - Del' B^T, then Sig_prime = Gamma (L_sig^T)^{-1}."""
#     n_src = Sig.shape[1]
#     n_eig = Del_prime.shape[1]
#     nb = len(edges) - 1
#     apply_sig_prime(
#         Sig.data.ptr,
#         Del_prime.data.ptr,
#         noise.data.ptr,
#         edges.data.ptr,
#         B.data.ptr,
#         L_inv.data.ptr,
#         out.data.ptr,
#         nb,
#         n_src,
#         n_eig,
#     )
#     # cp.cuda.Stream.null.synchronize()
#     return out


# ============================================================================
# Sig_prime thin launchers for linalg.py  --  CORRECTED.
#
# Pass scalars FIRST (n_src, n_eig, nb), then pointers, then stream=0, to match
# the C dispatch signatures. Each launcher returns int; check it and raise on
# nonzero so a bad (n_src, n_eig) surfaces immediately instead of silently
# leaving the output buffer untouched.
#
# Replace the existing make_cap_reduce / fused_cholesky_inverse_sig /
# apply_sig_prime_launch in linalg.py with these.
# ============================================================================
 
def make_cap_reduce(Sig, Del_prime, noise, edges, B, M_sig):
    """Kernel 1: build per-group B and accumulate the capacitance Lambda into M_sig.
 
    M_sig is an accumulator (atomicAdd across groups) and MUST be zeroed before
    this call -- caller is responsible (see inv_cov / workspace reset).
    """
    n_src = Sig.shape[1]
    n_eig = Del_prime.shape[1]
    nb = len(edges) - 1
    ret = cap_reduce(
        n_src, n_eig, nb,
        Sig.data.ptr,
        Del_prime.data.ptr,
        noise.data.ptr,
        edges.data.ptr,
        B.data.ptr,
        M_sig.data.ptr,
        0,                       # stream = default
    )
    if ret != 0:
        raise RuntimeError(
            f"cap_reduce dispatch failed: n_src={n_src}, n_eig={n_eig}, ret={ret}"
        )
    return B, M_sig
 
 
def fused_cholesky_inverse_sig(M_sig, out, out_source_det = None):
    """Kernel 2: L_inv = (chol(I + Lambda))^{-1} for the single (n_src x n_src) M_sig."""
    n_src = M_sig.shape[0]
    compute_det = out_source_det is not None
    ret = chol_inv_fused(
        n_src,
        M_sig.data.ptr,
        out.data.ptr,
        out_source_det.data.ptr if compute_det else 0,
        int(compute_det),
        0,                       # stream = default
    )
    if ret != 0:
        raise RuntimeError(
            f"chol_inv_fused dispatch failed: n_src={n_src}, ret={ret}"
        )
    return out
 
 
def apply_sig_prime_launch(Sig, Del_prime, noise, edges, B, L_inv, out):
    """Kernel 3: Gamma = A - Del' B^T, then Sig_prime = Gamma (L_sig^T)^{-1}."""
    n_src = Sig.shape[1]
    n_eig = Del_prime.shape[1]
    nb = len(edges) - 1
    ret = apply_sig_prime(
        n_src, n_eig, nb,
        Sig.data.ptr,
        Del_prime.data.ptr,
        noise.data.ptr,
        edges.data.ptr,
        B.data.ptr,
        L_inv.data.ptr,
        out.data.ptr,
        0,                       # stream = default
    )
    if ret != 0:
        raise RuntimeError(
            f"apply_sig_prime dispatch failed: n_src={n_src}, n_eig={n_eig}, ret={ret}"
        )
    return out


# ---------------------------------------------------------------------------
# workspace
# ---------------------------------------------------------------------------
class MatvecWorkspace:
    """
    Preallocated scratch for sparse_cov_times_vec.

    Instantiate once before the CG loop -- nb, n_eig and n_src are fixed across
    iterations, so nothing here needs to be reallocated.

    del_tmp is fully overwritten every call and is cp.empty-safe.
    sig_tmp is an atomic accumulator; it is zeroed inside the C launcher, so it
    does NOT need to be zeroed from Python.
    """

    def __init__(self, nb, n_eig, n_src, out_size=None):
        self.nb = int(nb)
        self.n_eig = int(n_eig)
        self.n_src = int(n_src)

        self.del_tmp = cp.empty(self.nb * self.n_eig, dtype=cp.float32)
        self.sig_tmp = cp.empty(self.n_src, dtype=cp.float32)

        # optional persistent output buffer for the CG loop
        self.out = (
            cp.empty(int(out_size), dtype=cp.float32)
            if out_size is not None
            else None
        )
        

def sparse_cov_vec_mul(
    noise,
    diffuse,
    source,
    vec,
    edges,
    ws,
    isinv=True,
    out=None,
    check=True,
):
    """
    Apply the sparse covariance (or its inverse) to a vector.

    Parameters
    ----------
    noise    : (n_row,) float32 cupy array.  Pass N^-1 when isinv=True.
    diffuse  : (n_row, n_eig) float32, C-contiguous.  Pass Del_bar when isinv=True.
    source   : (n_row, n_src) float32, C-contiguous.  Pass Sig_bar when isinv=True.
    vec      : (n_row,) float32
    edges    : (nb+1,) int32
    ws       : MatvecWorkspace sized for this (nb, n_eig, n_src)
    isinv    : sign convention.  True  -> N*vec - Del@.. - Sig@..
                                 False -> N*vec + Del@.. + Sig@..
    out      : optional preallocated (n_row,) float32 output
    check    : run the shape/dtype assertions (disable in the hot loop)

    Returns
    -------
    (n_row,) float32 cupy array
    """
    n_row = vec.shape[0]
    nb = edges.shape[0] - 1
    n_eig = diffuse.shape[1]
    n_src = source.shape[1]

    if out is None:
        out = ws.out if ws.out is not None else cp.empty(n_row, dtype=cp.float32)

    stream = cp.cuda.get_current_stream()

    sparse_cov_times_vec(
        noise.data.ptr,
        diffuse.data.ptr,
        source.data.ptr,
        vec.data.ptr,
        out.data.ptr,
        ws.del_tmp.data.ptr,
        ws.sig_tmp.data.ptr,
        edges.data.ptr,
        ctypes.c_int(nb),
        ctypes.c_int(n_src),
        ctypes.c_int(n_eig),
        ctypes.c_int(1 if isinv else 0),
        ctypes.c_void_p(stream.ptr),
    )

    return out


MAX_N_COL = 20

def gpu_apply_gains(gains, mat, ant_1, ant_2, out=None, check=True):
    """GPU port of corrcal.utils.apply_gains_to_mat. mat: (2*n_bl, n_col) float32."""
    
    "TODO: switch the internally allocated out initialization to a ws buffer when writing the full pipeline"
    "For now, just pass initialization outside of wrapper into the out parameter"
    n_bl, n_col = ant_1.shape[0], mat.shape[1]
    if check:
        assert gains.dtype == cp.float32 and mat.dtype == cp.float32
        assert ant_1.dtype == cp.int32 and ant_2.dtype == cp.int32
        assert mat.flags.c_contiguous and mat.shape[0] == 2 * n_bl
        assert 1 <= n_col <= MAX_N_COL, f"n_col={n_col} outside compiled grid"
        if out is not None:
            assert out.data.ptr != mat.data.ptr, "out must not alias mat"
    if out is None:
        out = cp.empty_like(mat)
    ret = apply_gains_launch(
        gains.data.ptr, ant_1.data.ptr, ant_2.data.ptr,
        mat.data.ptr, out.data.ptr,
        ctypes.c_int(n_bl), ctypes.c_int(n_col),
        ctypes.c_void_p(cp.cuda.get_current_stream().ptr),
    )
    if ret != 0:
        raise RuntimeError(f"apply_gains dispatch failed: n_col={n_col}")
    return out




# class GradWorkspace:
#     """
#     Preallocated scratch for launch_grad_nll, plus the antenna -> baseline
#     CSR used by the gather kernel.
 
#     ant_1 / ant_2 are fixed for a whole solve, so build this once before
#     scipy.optimize.minimize and pass it to every gradient call.
 
#     csr_idx entries are e = 2k + role (role 0: antenna is ant_1[k], role 1:
#     ant_2[k]). Viewed as float2[2 n_bl], contrib is indexed directly by e.
#     Entries are sorted by (antenna, e), so each warp's loads walk contrib in
#     increasing address order.
 
#     del_tmp, contrib, out: fully overwritten every call, cp.empty-safe.
#     sig_tmp, phs_sum: atomic accumulators, zeroed inside the C launcher.
#     """
 
#     def __init__(self, ant_1, ant_2, n_ant, nb, n_eig, n_src):
#         a1 = cp.asarray(ant_1, dtype=cp.int32)
#         a2 = cp.asarray(ant_2, dtype=cp.int32)
#         n_bl = a1.shape[0]
#         self.n_ant, self.n_bl, self.nb = int(n_ant), int(n_bl), int(nb)
#         self.n_eig, self.n_src = int(n_eig), int(n_src)
 
#         owner = cp.concatenate([a1, a2])
#         k = cp.arange(n_bl, dtype=cp.int32)
#         entry = cp.concatenate([2 * k, 2 * k + 1])
#         order = cp.lexsort(cp.stack([entry, owner]))       # primary key: owner
#         self.csr_idx = cp.ascontiguousarray(entry[order]).astype(cp.int32)
#         counts = cp.bincount(owner, minlength=self.n_ant)
#         self.csr_ptr = cp.zeros(self.n_ant + 1, dtype=cp.int32)
#         self.csr_ptr[1:] = cp.cumsum(counts).astype(cp.int32)
 
#         self.del_tmp = cp.empty((self.nb, self.n_eig), dtype=cp.float32)
#         self.sig_tmp = cp.empty(self.n_src, dtype=cp.float32)
#         self.phs_sum = cp.empty(1, dtype=cp.float32)
#         self.contrib = cp.empty((self.n_bl, 4), dtype=cp.float32)
#         self.out = cp.empty(2 * self.n_ant, dtype=cp.float32)
 
 
# def gpu_grad_nll(gains, p, gained, cinv, cov, ant_1, ant_2, ws,
#                  phs_coef=0.0, inv_scale=1.0, check=True):
#     """
#     Launch the fused gradient (K1 reduce, K2 per-baseline, K3 per-antenna gather).
 
#     gains  : (2 n_ant,) float32, ALREADY divided by scale
#     p      : (n_row,) float32, C^-1 d
#     gained : GpuSparseCov after apply_gains (diff_mat / src_mat = G Delta, G Sigma)
#     cinv   : gained.inv() result (diff_mat / src_mat = Delta_bar, Sigma_bar)
#     cov    : ungained GpuSparseCov (noise, Delta, Sigma)
#     ant_1, ant_2 : (n_bl,) int32 device
#     ws     : GradWorkspace
 
#     Returns ws.out, the (2 n_ant,) float32 gradient on device. Overwritten by
#     the next call; copy it if it must survive.
#     """
#     n_row = p.shape[0]
#     n_bl = n_row // 2
#     nb = cov.edges.shape[0] - 1
#     n_eig, n_src = cov.diff_mat.shape[1], cov.src_mat.shape[1]
#     n_ant = gains.shape[0] // 2
 
#     if check:
#         f32 = (gains, p, gained.diff_mat, gained.src_mat, cov.diff_mat, cov.src_mat,
#                cinv.diff_mat, cinv.src_mat, cov.noise)
#         assert all(a.dtype == cp.float32 and a.flags.c_contiguous for a in f32)
#         assert ant_1.dtype == cp.int32 and ant_2.dtype == cp.int32
#         assert cov.edges.dtype == cp.int32
#         assert (ws.n_bl, ws.n_ant, ws.nb, ws.n_eig, ws.n_src) == (n_bl, n_ant, nb, n_eig, n_src), \
#             "GradWorkspace shape does not match this problem"
#         assert 1 <= n_eig <= MAX_N_COL and 1 <= n_src <= MAX_N_COL
 
#     ret = grad_nll_launch(
#         n_src, n_eig, nb, n_bl, n_ant,
#         gained.diff_mat.data.ptr, gained.src_mat.data.ptr,
#         cov.diff_mat.data.ptr, cov.src_mat.data.ptr,
#         cinv.diff_mat.data.ptr, cinv.src_mat.data.ptr,
#         cov.noise.data.ptr, p.data.ptr, gains.data.ptr,
#         ant_1.data.ptr, ant_2.data.ptr, cov.edges.data.ptr,
#         ws.csr_ptr.data.ptr, ws.csr_idx.data.ptr,
#         ws.del_tmp.data.ptr, ws.sig_tmp.data.ptr, ws.phs_sum.data.ptr,
#         ws.contrib.data.ptr, ws.out.data.ptr,
#         ctypes.c_float(phs_coef), ctypes.c_float(inv_scale),
#         ctypes.c_void_p(cp.cuda.get_current_stream().ptr),
#     )
#     if ret != 0:
#         raise RuntimeError(f"grad_nll dispatch failed: n_src={n_src}, n_eig={n_eig}")
#     return ws.out
 

class GradWorkspace:
    """
    Preallocated scratch for launch_grad_nll, plus the antenna -> baseline
    CSR used by the gather kernel.
 
    ant_1 / ant_2 are fixed for a whole solve, so build this once before
    scipy.optimize.minimize and pass it to every gradient call.
 
    csr_idx entries are e = 2k + role (role 0: antenna is ant_1[k], role 1:
    ant_2[k]). Viewed as float2[2 n_bl], contrib is indexed directly by e.
    Entries are sorted by (antenna, e), so each warp's loads walk contrib in
    increasing address order.
 
    del_tmp, sig_tmp, contrib, out: fully overwritten every call, cp.empty-safe.
    u_acc (float64), phs_sum: atomic accumulators, zeroed inside the C launcher.
    """
 
    def __init__(self, ant_1, ant_2, n_ant, nb, n_eig, n_src):
        a1 = cp.asarray(ant_1, dtype=cp.int32)
        a2 = cp.asarray(ant_2, dtype=cp.int32)
        n_bl = a1.shape[0]
        self.n_ant, self.n_bl, self.nb = int(n_ant), int(n_bl), int(nb)
        self.n_eig, self.n_src = int(n_eig), int(n_src)
 
        owner = cp.concatenate([a1, a2])
        k = cp.arange(n_bl, dtype=cp.int32)
        entry = cp.concatenate([2 * k, 2 * k + 1])
        order = cp.lexsort(cp.stack([entry, owner]))       # primary key: owner
        self.csr_idx = cp.ascontiguousarray(entry[order]).astype(cp.int32)
        counts = cp.bincount(owner, minlength=self.n_ant)
        self.csr_ptr = cp.zeros(self.n_ant + 1, dtype=cp.int32)
        self.csr_ptr[1:] = cp.cumsum(counts).astype(cp.int32)
 
        self.del_tmp = cp.empty((self.nb, self.n_eig), dtype=cp.float32)
        self.sig_tmp = cp.empty(self.n_src, dtype=cp.float32)
        self.u_acc = cp.empty(self.n_src, dtype=cp.float64)
        self.phs_sum = cp.empty(1, dtype=cp.float32)
        self.contrib = cp.empty((self.n_bl, 4), dtype=cp.float32)
        self.out = cp.empty(2 * self.n_ant, dtype=cp.float32)
 
 
def gpu_grad_nll(gains, p, data, gained, cinv, cov, ant_1, ant_2, ws,
                 phs_coef=0.0, inv_scale=1.0, check=True):
    """
    Launch the fused gradient (K1 reduce, K2 per-baseline, K3 per-antenna gather).
 
    gains  : (2 n_ant,) float32, ALREADY divided by scale
    p      : (n_row,) float32, C^-1 d
    data   : (n_row,) float32, d
    gained : GpuSparseCov after apply_gains (diff_mat / src_mat = G Delta, G Sigma).
             Its workspace gained.ws.inv must still hold the factors from the
             inv() that produced cinv (L_del, B, L_sig_inv): no other inv() on
             that workspace in between.
    cinv   : gained.inv() result (diff_mat / src_mat = Delta_bar, Sigma_bar)
    cov    : ungained GpuSparseCov (noise, Delta, Sigma)
    ant_1, ant_2 : (n_bl,) int32 device
    ws     : GradWorkspace
 
    Returns ws.out, the (2 n_ant,) float32 gradient on device. Overwritten by
    the next call; copy it if it must survive.
    """
    n_row = p.shape[0]
    n_bl = n_row // 2
    nb = cov.edges.shape[0] - 1
    n_eig, n_src = cov.diff_mat.shape[1], cov.src_mat.shape[1]
    n_ant = gains.shape[0] // 2
 
    iw = gained.ws.inv
    if check:
        f32 = (gains, p, data, iw.L_del, iw.B, iw.L_sig_inv, gained.diff_mat, gained.src_mat, cov.diff_mat, cov.src_mat,
               cinv.diff_mat, cinv.src_mat, cov.noise)
        assert all(a.dtype == cp.float32 and a.flags.c_contiguous for a in f32)
        assert ant_1.dtype == cp.int32 and ant_2.dtype == cp.int32
        assert cov.edges.dtype == cp.int32
        assert (ws.n_bl, ws.n_ant, ws.nb, ws.n_eig, ws.n_src) == (n_bl, n_ant, nb, n_eig, n_src), \
            "GradWorkspace shape does not match this problem"
        assert 1 <= n_eig <= MAX_N_COL and 1 <= n_src <= MAX_N_COL
 
    ret = grad_nll_launch(
        n_src, n_eig, nb, n_bl, n_ant,
        gained.diff_mat.data.ptr, gained.src_mat.data.ptr,
        cov.diff_mat.data.ptr, cov.src_mat.data.ptr,
        cinv.diff_mat.data.ptr, cinv.src_mat.data.ptr,
        iw.L_del.data.ptr, iw.B.data.ptr, iw.L_sig_inv.data.ptr,
        cov.noise.data.ptr, p.data.ptr, data.data.ptr, gains.data.ptr,
        ant_1.data.ptr, ant_2.data.ptr, cov.edges.data.ptr,
        ws.csr_ptr.data.ptr, ws.csr_idx.data.ptr,
        ws.u_acc.data.ptr, ws.phs_sum.data.ptr,
        ws.del_tmp.data.ptr, ws.sig_tmp.data.ptr,
        ws.contrib.data.ptr, ws.out.data.ptr,
        ctypes.c_float(phs_coef), ctypes.c_float(inv_scale),
        ctypes.c_void_p(cp.cuda.get_current_stream().ptr),
    )
    if ret != 0:
        raise RuntimeError(f"grad_nll dispatch failed: n_src={n_src}, n_eig={n_eig}")
    return ws.out
 