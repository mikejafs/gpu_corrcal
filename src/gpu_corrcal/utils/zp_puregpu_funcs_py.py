import ctypes
import cupy as cp
import matplotlib.pyplot as plt

from pathlib import Path
current_dir = Path(__file__).resolve().parent.parent / "kernels"
print(current_dir)

# full_path = "/home/mike/corrcal_gpu_pipeline/pipeline/zp_puregpu_funcs.so"
# fp32_path = "/home/mike/corrcal_gpu_pipeline/pipeline/zp_puregpu_funcs_f32.so"
# full_path = current_dir / "zp_puregpu_funcs.so"
# fp32_path = current_dir / "zp_puregpu_funcs_f32.so"

full_path_f32 = "zp_puregpu_funcs_f32.so"
full_path_f64 = "zp_puregpu_funcs_f64.so"

zp_cuda_lib = ctypes.cdll.LoadLibrary(full_path_f64)
zp_cuda_lib_fp32 = ctypes.cdll.LoadLibrary(full_path_f32)

zp_cuda_lib.zeroPad1d.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_long),
    ctypes.c_int,
    ctypes.c_int
]
zp_cuda_lib.undo_zeroPad1d.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_long),
    ctypes.c_int,
    ctypes.c_int
]
zp_cuda_lib.zeroPad2d.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_long),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int
]
zp_cuda_lib.undo_zeroPad2d.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_long),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int
]

zp_cuda_lib_fp32.zeroPad1d.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_long),
    ctypes.c_int,
    ctypes.c_int
]

zp_cuda_lib_fp32.undo_zeroPad1d.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_long),
    ctypes.c_int,
    ctypes.c_int
]

zp_cuda_lib_fp32.zeroPad2d.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_long),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int
]

zp_cuda_lib_fp32.undo_zeroPad2d.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_long),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int
]

"""
Stand alone 1d and 2d versions were just for initial 
testing. They are both packaged into the full zeroPad
and undo_zeroPad funcs below.
"""
def zeroPad1d(array, edges):
    array = cp.array(array, dtype=cp.double)
    edges = cp.array(edges, dtype=cp.int64)
    largest_block = cp.array(cp.diff(edges).max(), dtype = cp.int64)
    n_blocks = cp.array(edges.size - 1, dtype = cp.int64)
    largest_block = int(largest_block.get())
    n_blocks = int(n_blocks.get())

    out_array = cp.zeros((n_blocks*largest_block), dtype = cp.double)

    zp_cuda_lib.zeroPad(
        #do not need cast here
        ctypes.cast(array.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(out_array.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(edges.data.ptr, ctypes.POINTER(ctypes.c_long)),
        n_blocks,
        largest_block
    )
    # cp.cuda.Stream.null.synchronize()
    return out_array, largest_block, n_blocks

def zeroPad2d(array, edges):
    array = cp.array(array, dtype=cp.double)
    edges = cp.array(edges, dtype=cp.int64)
    array_cols = array.shape[1]
    largest_block = cp.array(cp.diff(edges).max(), dtype = cp.int64)
    n_blocks = cp.array(edges.size - 1, dtype = cp.int64)
    largest_block = int(largest_block.get())
    n_blocks = int(n_blocks.get())

    out_array = cp.zeros((n_blocks*largest_block*array_cols), dtype = cp.double)

    zp_cuda_lib.zeroPad(
        ctypes.cast(array.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(out_array.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(edges.data.ptr, ctypes.POINTER(ctypes.c_long)),
        array_cols,
        n_blocks,
        largest_block
    )
    # cp.cuda.Stream.null.synchronize()
    return out_array, largest_block, n_blocks


def zeroPad(array, edges, return_inv, dtype):
    """
    Zeropads an input matrix according to the largest block in the diffuse
    sky covariance matrix. In other words, calculates the largest block using
    the edges array and adds rows of zeros in every other redundant block such
    that each block is the same size.

    Params
    ------
    array
        Input matrix to be zeropadded.
        CAUTION: Array must be of the 1d/2d form 

    edges
        Array of indices defining the beginning and end of each redundant block
        in the diffuse matrix
    return_inv
        Boolean. Set to True if want to calculate 1/noise_mat to avoid divide by
        zero issues after zeropadding

    Returns
    -------
    out_array
        Zeropadded array.
        CAUTION: Output array is of the 2d/3d form

    largest_block
        The size of the largest block in the diffuse matrix.
    n_blocks
        Total number of redundant blocks.
    """

    array = cp.array(array, dtype=dtype)
    edges = cp.array(edges, dtype=cp.int64)
    largest_block = cp.array(cp.diff(edges).max(), dtype = cp.int64)
    n_blocks = cp.array(edges.size - 1, dtype = cp.int64)
    largest_block = int(largest_block.get())
    n_blocks = int(n_blocks.get())

    if dtype == cp.float64:
        lib = zp_cuda_lib
        ctype = ctypes.c_double
    elif dtype == cp.float32: 
        lib = zp_cuda_lib_fp32
        ctype = ctypes.c_float


    if return_inv:
        array = 1/array

    # #--------------------------
    #     eps = 1e-8
    #     array = cp.where(cp.abs(array) < eps, eps, array)
    #     array = 1.0 / array
    # #--------------------------
    
    
    else:
        pass

    if array.ndim == 1: 
        out_array = cp.zeros((n_blocks*largest_block), dtype = dtype)
        lib.zeroPad1d(
            ctypes.cast(array.data.ptr, ctypes.POINTER(ctype)),
            ctypes.cast(out_array.data.ptr, ctypes.POINTER(ctype)),
            ctypes.cast(edges.data.ptr, ctypes.POINTER(ctypes.c_long)),
            n_blocks,
            largest_block
        )
        out_array = out_array.reshape(n_blocks, largest_block)
        cp.cuda.Stream.null.synchronize()
    else:
        array_cols = array.shape[1]
        out_array = cp.zeros((n_blocks*largest_block*array_cols), dtype = dtype)
        lib.zeroPad2d(
            ctypes.cast(array.data.ptr, ctypes.POINTER(ctype)),
            ctypes.cast(out_array.data.ptr, ctypes.POINTER(ctype)),
            ctypes.cast(edges.data.ptr, ctypes.POINTER(ctypes.c_long)),
            array_cols,
            n_blocks,
            largest_block
        )
        out_array = out_array.reshape(n_blocks, largest_block, array_cols)
        cp.cuda.Stream.null.synchronize()
    return out_array, largest_block, n_blocks

    
def undo_zeroPad(array, edges, ReImsplit=False):
    """
    "Undoes" the action of the zeroPad function. In other words, takes a matrix
    that has been padded with zeros according to the largest diffuse matrix block
    and returns the original matrix in its state prior to zeropadding.
    """

    array = cp.array(array, dtype=cp.double)
    edges = cp.array(edges, dtype=cp.int64)
    largest_block = cp.array(cp.diff(edges).max(), dtype = cp.int32)
    n_blocks = cp.array(edges.size - 1, dtype = cp.int32)
    largest_block = int(largest_block.get())
    n_blocks = int(n_blocks.get())

    if ReImsplit:
        largest_block = largest_block
        n_bl = int(edges[-1])
    else:
        largest_block = int(largest_block/2)
        n_bl = int(edges[-1]/2)
        # edges = cp.array([int(x/2) for x in edges])
        edges = edges//2

    if array.ndim == 2:
        array = array.reshape(n_blocks*largest_block)
        out_array = cp.zeros(n_bl, dtype = cp.double)
        zp_cuda_lib.undo_zeroPad1d(
            ctypes.cast(array.data.ptr, ctypes.POINTER(ctypes.c_double)),
            ctypes.cast(out_array.data.ptr, ctypes.POINTER(ctypes.c_double)),
            ctypes.cast(edges.data.ptr, ctypes.POINTER(ctypes.c_long)),
            n_blocks,
            largest_block
        )
        # out_array = out_array.reshape(n_blocks, largest_block, 1)
        cp.cuda.Stream.null.synchronize()
    else:
        array_cols = array.shape[2]
        array = array.reshape(n_blocks*largest_block*array_cols)
        # print(array)
        out_array = cp.zeros((int(edges[-1]), array_cols), dtype = cp.double)
        zp_cuda_lib.undo_zeroPad2d(
            ctypes.cast(array.data.ptr, ctypes.POINTER(ctypes.c_double)),
            ctypes.cast(out_array.data.ptr, ctypes.POINTER(ctypes.c_double)),
            ctypes.cast(edges.data.ptr, ctypes.POINTER(ctypes.c_long)),
            array_cols,
            n_blocks,
            largest_block
        )
        # out_array = out_array.reshape(n_blocks, largest_block, array_cols)
        cp.cuda.Stream.null.synchronize()

    return out_array

    