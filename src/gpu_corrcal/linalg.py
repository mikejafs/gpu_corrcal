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