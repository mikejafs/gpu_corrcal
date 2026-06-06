# Serves as the main reference for accuracy and time benchmarking as the 
# full GPU routine is built up
import cupy as cp
from gpu_corrcal.utils.zp_puregpu_funcs_py import *


def inverse_covariance(N, Del, Sig, xp, ret_det = False, N_is_inv = True):
    """
    CuPy inverse covariance.

    Given the components of the 2-level sparse covariance object, computes 
    the components of the inverse covariance object. Currectly does not 
    support the option to return the determinant of the covariance.


    Parameters
    ----------
    N: Noise 
    Del: \\Delta (diffuse) sky component matrix with shape n_bl x n_eig
    Sig: \\Sigma Source component matrix with shape n_bl x n_src
    edges: Array controlling the start and stop of the redundant blocks in the sparse diffuse matrix
    xp: Choice of running on the gpu (xp = cp) or cpu (xp = np)
    ret_det: Option to return the log(det(C)) along with the inverse covariance. Defaults to False

    Returns
    -------
    N^-1: Inverse noise matrix
    Del': The primed version of the diffuse sky matrix
    Sig': The primed version of the source component matrix
    """

    # eps = 1e-6


    if N_is_inv:
        N_inv = N
    else:
        N_inv = 1/N

    temp = N_inv[..., None] * Del  
    temp2 = xp.transpose(Del, [0, 2, 1]) @ temp

    #the current red kernel does up to this
    # return temp2

    L_del = xp.linalg.cholesky(xp.eye(Del.shape[2], dtype=cp.float32)[None, ...] + temp2)
    print(L_del.dtype)   
    t3 = xp.transpose(xp.linalg.inv(L_del), [0, 2, 1])
    print(t3.dtype)
    # return temp, t3

    # Del_prime = temp @ xp.transpose(xp.linalg.inv(L_del).conj(), [0, 2, 1]) 
    Del_prime = temp @ t3
        
    return Del_prime

    A = N_inv[..., None] * Sig
    B = xp.transpose(Sig.conj(), [0, 2, 1]) @ Del_prime
    W = A - Del_prime @ xp.transpose(B.conj(), [0, 2, 1])
    L_sig = xp.linalg.cholesky(
        xp.eye(Sig.shape[2]) + xp.sum(
            xp.transpose(A.conj(), [0, 2, 1]) @ Sig, axis = 0
        ) - xp.sum(
            B @ xp.transpose(B.conj(), [0, 2, 1]), axis = 0
        )
    )
    Sig_prime = W @ xp.linalg.inv(L_sig).T.conj()[None, ...]
    # Sig_prime = (A - Del_prime @ xp.transpose(B.conj(), [0, 2, 1])) @ xp.linalg.inv(L_sig).T.conj()[None, ...]


    if ret_det:
        # logdet = 2*(xp.sum(xp.diag(L_del)) + xp.sum(xp.diag(L_sig)))
        #the line should actually be -> Need to check why this works and how differ from xp.diag
        logdet = 2*(xp.sum(xp.diagonal(xp.log(L_del), axis2 = 1, axis1 = 2)) + xp.sum(xp.diagonal(xp.log(L_sig))))
        # cp.cuda.Stream.null.synchronize()
        return logdet, N_inv, Del_prime, Sig_prime 
    else:
        pass
    # cp.cuda.Stream.null.synchronize()
    return N_inv, Del_prime, Sig_prime


def setup_cupy_ref(noise, diffuse, edges):
    zp_inv_noise, lb, nb = zeroPad(noise, edges, return_inv=True, dtype=cp.float32)
    zp_diffuse, lb, nb = zeroPad(diffuse, edges, return_inv=False, dtype=cp.float32)
    # temp = zp_inv_noise[..., None] * zp_diffuse
    # out = cp.transpose(zp_diffuse, [0, 2, 1]) @ temp
    diffuse_bar = inverse_covariance(zp_inv_noise, zp_diffuse, 1, cp, ret_det=False, N_is_inv=True)

    cp.cuda.Stream.null.synchronize()
    return diffuse_bar


def cupy_ref(temp, edges, t3):
    # L_del = cp.linalg.cholesky(cp.eye(temp2.shape[1])[None, ...] + temp2)   
    # t3 = cp.linalg.inv(L_del)
    Del_prime = temp @ t3

    return Del_prime