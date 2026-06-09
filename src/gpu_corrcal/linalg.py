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

def fused_cholesky_inverse(temp2, edges, out):
    n_eig = temp2.shape[1]
    num_blocks = len(edges) - 1
    fused_chol_inv(
        temp2.data.ptr,
        out.data.ptr,
        num_blocks,
        n_eig
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
 
 
def fused_cholesky_inverse_sig(M_sig, out):
    """Kernel 2: L_inv = (chol(I + Lambda))^{-1} for the single (n_src x n_src) M_sig."""
    n_src = M_sig.shape[0]
    ret = chol_inv_fused(
        n_src,
        M_sig.data.ptr,
        out.data.ptr,
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