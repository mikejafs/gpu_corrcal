"""
Pulled the grouped gem batched lib call outside of the prepare params function
in this file compared to groupedBatched_func
"""

import numpy as np
import cupy as cp
import ctypes
from simulate_params import *
from zp_puregpu_funcs_py import *

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


#small wrapper used to define pointers from (CuPy) python arrays
def ptr(array):
    return ctypes.c_void_p(array.data.ptr)


def group_blocks_by_size(edges):
    edges = cp.asnumpy(edges)
    block_sizes = np.diff(edges)
 
    groups = {}
    for i, height in enumerate(block_sizes):
        groups.setdefault(height, []).append(i)

    return block_sizes, groups


def prepare_grouped_batched_params(A_array, B_array, edges):
    bl_sizes, grps = group_blocks_by_size(edges)
    group_sizes = np.array([len(grp_size) for grp_size in grps.values()], dtype=np.int32)
    # print('the group size are:', group_sizes)
    # print('The block sizes are:', bl_sizes)
    # print('PRINTING GRPS:', sorted(grps.items()))

    groupCount = len(grps)  #total number of groups
    # print('The total number of groups is:', groupCount)

    transA = np.array([0]*groupCount, dtype=np.int32)
    transB = np.array([0]*groupCount, dtype=np.int32)
    
    alpha_arr = np.ones(groupCount, dtype=np.float32)
    beta_arr = np.zeros(groupCount, dtype=np.float32)

    m_array = np.zeros(groupCount, dtype=np.int32)
    n_array = np.zeros(groupCount, dtype=np.int32)
    k_array = np.zeros(groupCount, dtype=np.int32)

    #now build pointer arrays for each of the input and output matrices
    A_ptrs = []
    B_ptrs = []
    C_ptrs = []

    n_eig = A_array.shape[1]

    for g_i, (h, block_id) in enumerate(grps.items()):
        m = n_eig
        n = n_eig #above and this are equal only if we are performing A^T @ A
        k = h

        m_array[g_i] = m
        n_array[g_i] = n
        k_array[g_i] = k

        A_list = []
        B_list = []
        C_list = []
        
        for b_id in block_id:

            start = edges[b_id]
            stop = edges[b_id+1]

            # A_array_i = A_array[start:stop, :].T
            # B_array_i = B_array[start:stop, :]

            A_array_i = A_array[start:stop, :].astype(cp.float32, copy=False) # (n_eig, h)
            B_array_i = B_array[start:stop, :].astype(cp.float32, copy=False)    # (h, n_eig)
            # print(A_array_i.shape)

            A_cm = cp.asfortranarray(A_array_i.T)
            B_cm = cp.asfortranarray(B_array_i)
            C_array_i = cp.zeros((n_eig, n_eig), dtype=cp.float32, order='F')

            # print("Block", b_id)
            # print("A_block:\n", A_array_i)
            # print("B_block:\n", B_array_i)
            # print("Aᵀ @ B (CPU):\n", (A_array_i @ B_array_i).get())

            A_list.append(A_cm)
            B_list.append(B_cm)
            C_list.append(C_array_i)

        A_ptrs.append(A_list)
        B_ptrs.append(B_list)
        C_ptrs.append(C_list)

    # print(A_ptrs)

    # === Flatten pointer lists into ONE contiguous list (required by cuBLAS) ===
    flat_A = [arr for group in A_ptrs for arr in group]
    flat_B = [arr for group in B_ptrs for arr in group]
    flat_C = [arr for group in C_ptrs for arr in group]

    # print('flat A:', flat_A)

    # Create device arrays of raw pointers
    A_ptrs_raw = cp.asarray([a.data.ptr for a in flat_A], dtype=cp.uintp)
    B_ptrs_raw = cp.asarray([b.data.ptr for b in flat_B], dtype=cp.uintp)
    C_ptrs_raw = cp.asarray([c.data.ptr for c in flat_C], dtype=cp.uintp)

    # ctypes pointer to the device arrays
    A_ptrs_ct = ctypes.c_void_p(A_ptrs_raw.data.ptr)
    B_ptrs_ct = ctypes.c_void_p(B_ptrs_raw.data.ptr)
    C_ptrs_ct = ctypes.c_void_p(C_ptrs_raw.data.ptr)

    # lda_array = k_array.copy()
    # ldb_array = n_array.copy()
    # ldc_array = n_array.copy()

    # Correct row-major mapping for C_block = A_array_i @ B_array_i => changed to col order now
    lda_array = m_array.copy().astype(np.int32, copy=False)  # = n_eig
    ldb_array = k_array.copy().astype(np.int32, copy=False)  # = block height h
    ldc_array = m_array.copy().astype(np.int32, copy=False)  

    return {"groupCount" : groupCount, 
            "transA": transA, 
            "transB": transB, 
            "ms": m_array, 
            "ns": n_array, 
            "ks": k_array, 
            "alphas": alpha_arr, 
            "Aptrs": A_ptrs_ct,
            "ldas": lda_array,
            "Bptrs": B_ptrs_ct,
            "ldbs": ldb_array,
            "betas": beta_arr,
            "Cptrs": C_ptrs_ct,
            "ldcs": ldc_array,
            "groupSizes": group_sizes,
            "A_raw": A_ptrs_raw,
            "B_raw": B_ptrs_raw,
            "C_raw": C_ptrs_raw,
            "Carrays": C_ptrs
            }


#thin wrapper around the grouped batched matmul cublas function
def groupedBatchedMatmul(param_dict):
    # print('The edges array is:', edges)

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
    # return param_dict["Carrays"]  # ordered by block index


def reshape_out(C_ptrs, edges):
    bl_sizes, grps = group_blocks_by_size(edges)
    group_sizes = np.array([len(grp_size) for grp_size in grps.values()], dtype=np.int32)

    flat_C = [arr for group in C_ptrs for arr in group]
    C_out = [None] * len(bl_sizes)
    idx = 0
    for (h, block_ids), size in zip(grps.items(), group_sizes):
        for j, b in enumerate(block_ids):
            C_out[b] = flat_C[idx + j]
        idx += size

    # print("edges:", edges)
    # print("block_sizes:", bl_sizes)
    # print("groups:", grps)
    # print("group_sizes:", group_sizes)
    # print("N_blocks:", len(bl_sizes), "sum(group_sizes):", group_sizes.sum())

    C_stacked = cp.stack(C_out, axis=0)
    return C_stacked



if __name__ == "__main__":
            
    #Parameter set up
    n_ant = 500
    n_eig = 3
    n_src = 1

    cp.random.seed(10)
    spms = SimCorrcalParams(n_ant, n_eig, n_src, precision='float32', xp=cp)
    edges = spms.edges()

    #simulated matrices with correct shapes
    sim_data = spms.sim_data()
    noise = sim_data[0]
    diff = sim_data[1]

    #zeropad diff and noise 
    # zp_noise, nb, lb = zeroPad(noise, edges, return_inv=True)
    # zp_diff, nb, lb = zeroPad(diff, edges, return_inv=False)

    noise = 1/noise

    # print('zeropadded diff', zp_diff)
    # print('regular diff', diff)
    # print(zp_noise)

    #set up the temp mat just before the mat mul we are interested in
    temp = noise[..., None] * diff
    # zp_temp = zp_noise[..., None] * zp_diff

    # temp2 = cp.transpose(zp_diff, [0, 2, 1]) @ zp_temp
    # print(temp2)

    #running the batched grouped matmul
    params = prepare_grouped_batched_params(diff, temp, edges)
    C_array = groupedBatchedMatmul(params)
    # print(C_array)

    # noise = cp.asnumpy(noise)
    # diff = cp.asnumpy(diff)
    # edges = cp.asnumpy(edges)

    # out_cpu = make_small_blocks(noise, diff, edges)
    # print(out_cpu)



    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #also conclude with a single print line 
    # print()