import numpy as np
import cupy as cp
from typing import Any
# from zp_puregpu_funcs_py import *
from cupyx.profiler import benchmark

def sparse_cov_times_vec(N, Del, Sig, vec, isinv):
    """
    Multiplies a sparse covariance object by a vector from the right

    #TODO: Could be beneficial to separate this function (for the sake of too many arguments)
        into one function for cov_times_data and inv_cov_times_data -> NO DON'T BOTHER
    """
    if vec.ndim == 2:
        vec = vec.reshape(vec.shape[0], vec.shape[1], 1)
        N = N.reshape(vec.shape[0], vec.shape[1], 1)
    else:
        pass
    del_tmp = cp.transpose(Del, [0, 2, 1]) @ vec
    sig_tmp = cp.sum(cp.transpose(Sig, [0, 2, 1]) @ vec, axis=0)
    if isinv:
        out = N * vec - Del @ del_tmp - Sig @ sig_tmp
    else:
        out = N * vec + Del @ del_tmp + Sig @ sig_tmp
    # out = out.reshape(vec.shape[0], vec.shape[1])
    return out



"""
-----------------------------------------------------------------------------------------------------------------------
Separating the zeropadding of the gains matrix into its own funnction and making a new apply gains to mat function
-----------------------------------------------------------------------------------------------------------------------
"""

def zeropad_gains(gains, edges, ant_1_array, ant_2_array, xp=cp, return_inv=False):
    """
    Zeropads and constructs the gain matrix used to apply gains to the source and
    diffuse matrices.

    Params
    ------
    gains

    edges

    xp

    Returns
    _______
    cplex_gain_mat
        Zeropadded gain matrix
    """

    complex_gains = gains[::2] + 1j*gains[1::2]

    # NOTE: Not entirely sure why we construct the cplex gain mat in this way =>
    #   Need to think a bit more about why this is..
    tmp_gain_mat = (
            complex_gains[ant_1_array, None] * complex_gains[ant_2_array, None].conj()
    )

    # print(tmp_gain_mat)
    # print(complex_gains.shape)

    # initialize gain mat to be zeropadded
    # gain_mat = xp.zeros((len(complex_gains), 1))
    gain_mat = xp.zeros((2*len(tmp_gain_mat), 1))

    # Re/Im split the gain mat and zeropad using edges array
    gain_mat[::2] = tmp_gain_mat.real
    gain_mat[1::2] = tmp_gain_mat.imag
    zp_gain_mat, largest_block, n_blocks = zeroPad(gain_mat, edges, return_inv=False)
    # print(zp_gain_mat.shape)
    zp_gain_mat = zp_gain_mat.reshape(n_blocks*largest_block, 1)
    # print(zp_gai-n_mat.shape)

    # re-assemble and re-shape the (now zeropadded) complex gain mat
    re_zp_gain_mat = zp_gain_mat[::2]
    im_zp_gain_mat = zp_gain_mat[1::2]
    cplex_gain_mat = re_zp_gain_mat + 1j * im_zp_gain_mat
    cplex_gain_mat = cplex_gain_mat.reshape(n_blocks, largest_block // 2, 1)

    return cplex_gain_mat


def apply_gains(cplex_gain_mat, mat, xp=cp):
    """
    Same as the apply_gains_to_mat function below but with the zeropadding of the gain
    matrix done in a separate function above. Note that in the world where we used this
    function, the gain matrix has already been constructed elsewhere and so all we need
    to do here is apply the gains.

    TODO: Fill out rest of docstring later

    """
    #initialize output matrix that will have gains applied
    out = xp.zeros_like(mat)

    #apply the gains
    out[:, ::2] = (
        cplex_gain_mat.real * mat[:, ::2] - cplex_gain_mat.imag * mat[:, 1::2]
    )
    out[:, 1::2] = (
        cplex_gain_mat.imag * mat[:, ::2] + cplex_gain_mat.real * mat[:, 1::2]
    )

    return out



"""
-----------------------------------------------------------------------------------------------------------------------
Original way of doing things where the gain matrix is constructed and zeropadded and applied to the mat all
in the same function
-----------------------------------------------------------------------------------------------------------------------
"""

def apply_gains_to_mat(
    gains: cp.ndarray, 
    mat: cp.ndarray, 
    edges: cp.ndarray, 
    ant_1_array: cp.ndarray, #these have length = n_bls
    ant_2_array: cp.ndarray, 
    xp: Any, 
    is_zeropadded: bool = True
):
    """
    Apply a pair of complex gains to a matrix. Utilizes the Re/Im split.
    Only accounts for "one half" of the gain application, meaning the 
    function is really performing eg. (g_1g_2*\Delta_{1,2}), where it is 
    understood that antenna's 1 and 2 belong to the baseline sitting at
    the same row as that baseline row in the \Delta (\Sigma) matrix. Note that 
    although the matrix provided may be zeropadded, the gain matrix is zeropadded 
    here and as such, should always be provided in an un-zeropadded (original) format.

    NOTE: Could be smart in the future to wrap the zeropadding of the gain 
        matrix in a separate function.

    Params
    ------
    gains
        1D array of Re/Im alternating gains to be applied to the source or
        diffuse matrices. Contains one set of Re/Im gains for all antennas 
        in the array. 
    mat
        Gains are applied to this. Can be 2d as in original C-corrcal.
        If 3d, is_zeropadded must be set to True
    edges
        Indices of edges of redundant blocks in the diffuse matrix.
    ant_1_array
        Indices of the first antenna in each baseline
    ant_2_array
        Indices of the second antenna in each baseline
    xp
        np for cpu (Numpy), cp for gpu (CuPy)
    is_zeropadded
        Boolean. Indicate whether the provided matrix has been zeropadded 
        previously

    Returns
    -------
    out
        Matrix with applied gains (explain this a bit better)
    """

    complex_gains = gains[::2] + 1j*gains[1::2]
    out = xp.zeros_like(mat)

    if is_zeropadded:
        #construct gain mat in the original way

        #NOTE: Not entirely sure why we construct the cplex gain mat in this way =>
        #   Need to think a bit more about why this is..
        tmp_gain_mat = (
            complex_gains[ant_1_array, None] * complex_gains[ant_2_array, None].conj()
        )

        #initialize gain mat to be zeropadded
        gain_mat = xp.zeros((len(complex_gains),1))   

        #Re/Im split the gain mat and zeropad using edges array
        gain_mat[::2] = tmp_gain_mat.real
        gain_mat[1::2] = tmp_gain_mat.imag
        zp_gain_mat, largest_block, n_blocks = zeroPad(gain_mat, edges, return_inv=False)
        zp_gain_mat = zp_gain_mat.reshape(n_blocks * largest_block, 1)

        #re-assemble and re-shape the (now zeropadded) complex gain mat
        re_zp_gain_mat = zp_gain_mat[::2]
        im_zp_gain_mat = zp_gain_mat[1::2]
        cplex_gain_mat = re_zp_gain_mat + 1j*im_zp_gain_mat
        cplex_gain_mat = cplex_gain_mat.reshape(n_blocks, largest_block//2, 1)  #really shouldn't need the floor devision
        # if edges array has even number of entries for every block, but leaving in for now

        #apply the gains
        out[:, ::2] = (
            cplex_gain_mat.real * mat[:, ::2] - cplex_gain_mat.imag * mat[:, 1::2]
        )
        out[:, 1::2] = (
            cplex_gain_mat.imag * mat[:, ::2] + cplex_gain_mat.real * mat[:, 1::2]
        )
    else:
        #apply the gains using the original corrcal routine
        gain_mat = (
            complex_gains[ant_1_array,None] * complex_gains[ant_2_array,None].conj()
        )
        out[::2] = gain_mat.real * mat[::2] - gain_mat.imag * mat[1::2]
        out[1::2] = gain_mat.imag * mat[::2] + gain_mat.real * mat[1::2]
    return out


def summarize_benchmark_results(function, *args):
    """
    Use the CuPy benchmark function and nicely print out just the
    total time spent on both the gpu and the cpu.
    """

    # print(args)
    test_results = str(benchmark(function, (args), n_repeat=1000))
    test_results = test_results.split()
    cpu_t = float(test_results[3])/1e6
    gpu_t = float(test_results[14])/1e6
    print(f"Time on cpu: {cpu_t:.6f}s")
    print(f"Time on gpu: {gpu_t:.6f}s")


