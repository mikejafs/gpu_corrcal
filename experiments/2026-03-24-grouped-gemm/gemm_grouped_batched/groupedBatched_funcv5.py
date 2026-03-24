"""
Attemping to zeropad according to three different sizes of blocks (small, med, large), for grouped batched gemm with only 3 groups.

CHANGES FROM V4:
- switched from fortraning the whole input arrays, to creating lists of col-major sub arrays
    -> This mitigated converting the entire 3-dim set of arrays to col-major simultaneously
        which I believe was creating problems by setting incorrect strides since the first
        index was just meant to separate blocks and not be involved in the fortran ordering
"""

import numpy as np
import cupy as cp
import ctypes
from simulate_params import *
from zp_puregpu_funcs_py import *
from cupyx.profiler import benchmark

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#add a single print line for better terminal readability
# print()

lib = ctypes.cdll.LoadLibrary("/home/mike/Thesis_rough_work/CUDA_programming/cuBLAS/gemm_grouped_batched/gemmGroupedBatched.so")

#define the python-side argtypes for the grouped batched cublas gemm
lib.gemmGroupedBatched.argtypes = [
    ctypes.c_int,                   #just a number (device/host irrelevant)
    ctypes.POINTER(ctypes.c_int),   #on host
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p,                #first array of pointers to sub matrices (on device)
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
]


def split_and_fortran_copy(batch_array):
    """
    Convert a (batch, m, n) array into a list of standalone
    Fortran-contiguous (m, n) matrices.
    """
    return [cp.asfortranarray(batch_array[i].copy())
            for i in range(batch_array.shape[0])]

#small wrapper used to define pointers from (CuPy) python arrays
def ptr(array):
    return ctypes.c_void_p(array.data.ptr)

def make_float_pp(blocks):
    arr = (ctypes.POINTER(ctypes.c_float) * len(blocks))()
    for i, blk in enumerate(blocks):
        ptr = ctypes.cast(blk.data.ptr, ctypes.POINTER(ctypes.c_float))
        arr[i] = ptr
    return arr

# -----------------------------------------------------------------------------
# Step 1 — Partition blocks into 3 super-groups
# -----------------------------------------------------------------------------

def partition_into_3_groups(edges):
    """Return {0:[small], 1:[medium], 2:[large]} block-ID groups."""
    edges = cp.asnumpy(edges)
    block_sizes = np.diff(edges)

    #sort the edges; doesn't matter since we just care about
    #relative shapes for the speed tests at this stage.
    #   Note if we wanted to do this in practice, we would simply
    #   have to sort the actual data into redundant groups from
    #   smallest to largest.

    block_order = np.argsort(block_sizes)
    sorted_sizes = block_sizes[block_order]
    edges_sorted = np.zeros(len(sorted_sizes) + 1, dtype = edges.dtype)
    edges_sorted[1:] = np.cumsum(sorted_sizes)

    # print(f"Sorted edges: {edges_sorted}")

    med = np.mean(block_sizes)
    max  = block_sizes.max()
    mid = 0.5 * (med + max)

    edges_small = []
    edges_med = []
    edges_large = []

    for start, stop in zip(edges_sorted[:], edges_sorted[1:]):
        block_size = stop - start
        if block_size < med:
            if start not in edges_small:
                edges_small.append(int(start))
            edges_small.append(int(stop))

        elif block_size < mid:
            if start not in edges_med:
                edges_med.append(int(start))
            edges_med.append(int(stop))

        else:
            if start not in edges_large:
                edges_large.append(int(start))
            edges_large.append(int(stop))

    return np.array(edges_small), np.array(edges_med), np.array(edges_large)    


def prepare_grouped_batched_params(A_array, B_array, edges):
    n_eig = A_array.shape[1]
    groupCount = 3

    edges_small, edges_med, edges_large = partition_into_3_groups(edges)

    #group 0: the 'small' blocks
    A0_array = cp.array(A_array[edges_small[0]:edges_small[-1]])
    B0_array = cp.array(B_array[edges_small[0]:edges_small[-1]])

    edges_small_local = edges_small - edges_small[0]

    A0_zp, nb, lb = zeroPad(A0_array, edges_small_local, return_inv=False, dtype=cp.float32)
    A0_zp = (A0_zp.transpose([0,2,1]))
    B0_zp, _, _ = zeroPad(B0_array, edges_small_local, return_inv=False, dtype=cp.float32)
    C0_zp = cp.zeros((len(edges_small_local)-1, A0_zp.shape[1], B0_zp.shape[2]), dtype = cp.float32)

    #group 1: the 'medium' blocks
    A1_array = cp.array(A_array[edges_med[0]:edges_med[-1]])
    B1_array = cp.array(B_array[edges_med[0]:edges_med[-1]])

    edges_med_local = edges_med - edges_med[0]
    
    A1_zp, _, _ = zeroPad(A1_array, edges_med_local, return_inv=False, dtype=cp.float32)
    A1_zp = (A1_zp.transpose([0,2,1]))
    B1_zp, _, _ = zeroPad(B1_array, edges_med_local, return_inv=False, dtype=cp.float32)
    C1_zp = cp.zeros((len(edges_med_local)-1, A1_zp.shape[1], B1_zp.shape[2]), dtype=cp.float32)

    #group 2: the 'large' blocks
    A2_array = cp.array(A_array[edges_large[0]:edges_large[-1]])
    B2_array = cp.array(B_array[edges_large[0]:edges_large[-1]])

    edges_large_local = edges_large - edges_large[0]

    A2_zp, _, _ = zeroPad(A2_array, edges_large_local, return_inv=False, dtype=cp.float32)
    A2_zp = (A2_zp.transpose([0,2,1]))

    B2_zp, _, _ = zeroPad(B2_array, edges_large_local, return_inv=False, dtype=cp.float32)
    C2_zp = cp.zeros((len(edges_large_local)-1, A2_zp.shape[1], B2_zp.shape[2]), dtype=cp.float32)


    all_A = (
        split_and_fortran_copy(A0_zp) +
        split_and_fortran_copy(A1_zp) +
        split_and_fortran_copy(A2_zp)
    )
    all_B = (
        split_and_fortran_copy(B0_zp) +
        split_and_fortran_copy(B1_zp) +
        split_and_fortran_copy(B2_zp)
    )
    all_C = (
        split_and_fortran_copy(C0_zp) +
        split_and_fortran_copy(C1_zp) +
        split_and_fortran_copy(C2_zp)
    )


    A_ptrs = cp.array([ar.data.ptr for ar in all_A], dtype=cp.uintp)
    B_ptrs = cp.array([br.data.ptr for br in all_B], dtype=cp.uintp)
    C_ptrs = cp.array([cr.data.ptr for cr in all_C], dtype=cp.uintp)

    A_ptrs_ct = ctypes.c_void_p(A_ptrs.data.ptr)
    B_ptrs_ct = ctypes.c_void_p(B_ptrs.data.ptr)
    C_ptrs_ct = ctypes.c_void_p(C_ptrs.data.ptr)

    # # Build host arrays of device pointers using ctypes
    m_row = np.array([A0_zp.shape[1], A1_zp.shape[1], A2_zp.shape[1]], np.int32)
    n_row = np.array([B0_zp.shape[2], B1_zp.shape[2], B2_zp.shape[2]], np.int32)
    k_row = np.array([B0_zp.shape[1], B1_zp.shape[1], B2_zp.shape[1]], np.int32)


    lda = m_row.copy().astype(np.int32, copy=False)
    ldb = k_row.copy().astype(np.int32, copy=False)
    ldc = m_row.copy().astype(np.int32, copy=False)

    transA = np.array([0]*groupCount, dtype=np.int32)
    transB = np.array([0]*groupCount, dtype=np.int32)
    
    alpha_arr = np.ones(groupCount, dtype=np.float32)
    beta_arr = np.zeros(groupCount, dtype=np.float32)

    group_sizes = np.array([len(A0_zp), len(A1_zp), len(A2_zp)], dtype = np.int32)
    
    return {
            "groupCount" : groupCount, 
            "transA": transA, 
            "transB": transB, 
            "ms": m_row, 
            "ns": n_row, 
            "ks": k_row, 
            "alphas": alpha_arr, 
            "Aptrs": A_ptrs_ct,
            # "Aptrs": A_ptrs,
            "ldas": lda,
            "Bptrs": B_ptrs_ct,
            # "Bptrs": B_ptrs,
            "ldbs": ldb,
            "betas": beta_arr,
            "Cptrs": C_ptrs_ct,
            # "Cptrs": C_ptrs,
            "ldcs": ldc,
            "groupSizes": group_sizes,
            "A_raw": A_ptrs,
            "B_raw": B_ptrs,
            "C_raw": C_ptrs,
            "A_blocks": all_A,
            "B_blocks": all_B,
            "C_blocks": all_C
    }


def groupedBatchedMatmul(param_dict):
    lib.gemmGroupedBatched(
        param_dict["groupCount"],
        param_dict["transA"].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        param_dict["transB"].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        param_dict["ms"].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        param_dict["ns"].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        param_dict["ks"].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        param_dict["alphas"].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        param_dict["Aptrs"],  
        param_dict["ldas"].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        param_dict["Bptrs"],
        param_dict["ldbs"].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        param_dict["betas"].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        param_dict["Cptrs"],
        param_dict["ldcs"].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        param_dict["groupSizes"].ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    )
    cp.cuda.Stream.null.synchronize()
    # return cp.stack(param_dict["C_blocks"], axis=0)  # ordered by block index


if __name__ == "__main__":
            

    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SET UP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

    #Parameter set up
    n_eig = 3
    n_src = 1
    rows = 3
    cols = 2
    n_ant = rows*cols
    # n_ant = 1000

    print('Parameters:')
    print('Number of antennas (Re Im split):', n_ant,
        '\nNumber of eigenmodes:', n_eig,
        '\nNumbe of sources:', n_src)
    print()


    cp.random.seed(0)
    spms = SimCorrcalParams(n_ant, n_eig, n_src, precision='float32', xp=cp)
    edges = spms.edges(grid_par_x=rows, grid_par_y=cols, use_random=False)

    edges = cp.asnumpy(edges)
    block_sizes = np.diff(edges)

    #sort the edges; doesn't matter since we just care about
    #relative shapes for the speed tests at this stage.
    #   Note if we wanted to do this in practice, we would simply
    #   have to sort the actual data into redundant groups from
    #   smallest to largest.


    block_order = np.argsort(block_sizes)
    sorted_sizes = block_sizes[block_order]

    edges_sorted = np.zeros(len(sorted_sizes) + 1, dtype = edges.dtype)
    edges_sorted[1:] = np.cumsum(sorted_sizes)

    #simulated matrices with correct shapes
    sim_data = spms.sim_data()
    noise = sim_data[0]
    diff = sim_data[1]

    #zeropad diff and noise 
    zp_noise, nb, lb = zeroPad(noise, edges_sorted, return_inv=True, dtype=cp.float32)
    zp_diff, nb, lb = zeroPad(diff, edges_sorted, return_inv=False, dtype=cp.float32)

    noise = 1/noise


    """~~~~~~~~~~~~~~~~~~~~~ COMPUTATION OF diff.T @ N^-1 @ diff ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

    """
    diff.T @ temp

    temp == N^-1 @ diff
    """

    #set up the temp mat just before the mat mul we are interested in
    temp = noise[..., None] * diff
    zp_temp = zp_noise[..., None] * zp_diff


    #cupy computation of diff.T @ N^-1 @ diff
    def cupy_block_mul(diff, tmp):
        res = cp.transpose(diff, [0, 2, 1]) @ tmp
        cp.cuda.Stream.null.synchronize()
        return res

    #return temp2 = diff.T @ N^-1 @ diff
    temp2 = cupy_block_mul(zp_diff, zp_temp)

    print(temp2)

    #cublas compuation of diff.T @ N^-1 @ diff
    params = prepare_grouped_batched_params(diff, temp, edges)
    C_array = groupedBatchedMatmul(params)
    # print(C_array.dtype)

    check_match = False

    if check_match:
        print("------------------------------------------------------------")
        #CHECK IF CUPY AND CUBLAS ARE COMPUTING THE SAME THING
        if np.allclose(temp2, C_array):
            print('Checking correctness with np.allclose (default params):' \
            '\nCuPy and cuBLAS match')
        else:
            print('Checking correctness with np.allclose (default params):' \
            '\nCuPy and cuBLAS DO NOT match')
        print("------------------------------------------------------------")


    """~~~~~~~~~~~~~~~~~~~~~ BENCHMARKING CUPY AND CUBLAS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

    #CUPY TIMES
    times = (benchmark(cupy_block_mul, (zp_diff, zp_temp), n_repeat = 100))
    # gpu_times = gpu_times.split()
    # gpu_cpu_t = float(gpu_times[3])/1e6
    # gpu_gpu_t = float(gpu_times[14])/1e6

    gpu_t_s = times.gpu_times
    cpu_t_s = times.cpu_times

    avg_gpu_t = cp.mean(gpu_t_s)
    avg_cpu_t = cp.mean(cpu_t_s)

    # print(gpu_cpu_t, gpu_gpu_t)
    print()
    print('CuPy times:')
    print('cpu:', avg_cpu_t, ' gpu:', avg_gpu_t)

    #CUBLAS TIMES
    times = (benchmark(groupedBatchedMatmul, (params,), n_repeat = 100))
    # gpu_times = gpu_times.split()
    # gpu_cpu_t = float(gpu_times[3])/1e6
    # gpu_gpu_t = float(gpu_times[14])/1e6

    gpu_t_s = times.gpu_times
    cpu_t_s = times.cpu_times

    avg_gpu_t = cp.mean(gpu_t_s)
    avg_cpu_t = cp.mean(cpu_t_s)

    # print(gpu_cpu_t, gpu_gpu_t)
    print()
    print('CuBLAS times:')
    print('cpu:', avg_cpu_t, ' gpu:', avg_gpu_t)

