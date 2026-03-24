"""
Attemping to zeropad according to three different sizes of blocks (small, med, large), for grouped batched gemm with only 3 groups.
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
    # ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_void_p,
    # ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p,
    # ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
]



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

    # print(f"Block sizes: {block_sizes}")
    # print(f"sorted block sizes: {sorted_sizes}")

    edges_sorted = np.zeros(len(sorted_sizes) + 1, dtype = edges.dtype)
    edges_sorted[1:] = np.cumsum(sorted_sizes)

    print(f"Sorted edges: {edges_sorted}")

    med = np.mean(block_sizes)
    # print(f"The mean block size is {med}")
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

    # A0_blocks = [A_array[start:stop] for start, stop in zip(edges_small[:-1], edges_small[1:])]
    # B0_blocks = [B_array[start:stop] for start, stop in zip(edges_small[:-1], edges_small[1:])]

    # A0_array = cp.concatenate(A0_blocks)
    # B0_array = cp.concatenate(B0_blocks)
    # print(f"The A0 arrays are: {A0_blocks}")
    # print(f"The A0 arrays are: {A0_array}")


    edges_small_local = edges_small - edges_small[0]
    print(f"the small local edges are: {edges_small_local}")

    A0_zp, nb, lb = zeroPad(A0_array, edges_small_local, return_inv=False, dtype=cp.float32)
    # print(A0_zp)
    A0_zp = cp.asfortranarray(A0_zp.transpose([0,2,1]))
    # print(f"transposed A0 array {A0_zp}")
    B0_zp, _, _ = zeroPad(B0_array, edges_small_local, return_inv=False, dtype=cp.float32)
    # print(B0_zp)
    B0_zp = cp.asfortranarray(B0_zp)
    # print(f"B0 zp array is {B0_zp}")
    C0_zp = cp.zeros((len(edges_small_local)-1, A0_zp.shape[1], B0_zp.shape[2]), dtype = cp.float32, order='F')
    # print(C0_zp)

    #group 1: the 'medium' blocks
    A1_array = cp.array(A_array[edges_med[0]:edges_med[-1]])
    B1_array = cp.array(B_array[edges_med[0]:edges_med[-1]])

    edges_med_local = edges_med - edges_med[0]
    
    """DEBUGGING"""
    # print(edges_med)
    print(f"the medium local edges are: {edges_med_local}")
    # print(f"the medium array is \n {A1_array}")
    """DEBUGGING"""

    A1_zp, _, _ = zeroPad(A1_array, edges_med_local, return_inv=False, dtype=cp.float32)
    # print(f"zeropadded medium array is \n {A1_zp}")

    A1_zp = cp.asfortranarray(A1_zp.transpose([0,2,1]))
    B1_zp, _, _ = zeroPad(B1_array, edges_med_local, return_inv=False, dtype=cp.float32)
    # print(B1_zp)
    B1_zp = cp.asfortranarray(B1_zp)
    C1_zp = cp.zeros((len(edges_med_local)-1, A1_zp.shape[1], B1_zp.shape[2]), dtype=cp.float32, order='F')

    #group 2: the 'large' blocks
    A2_array = cp.array(A_array[edges_large[0]:edges_large[-1]])
    B2_array = cp.array(B_array[edges_large[0]:edges_large[-1]])

    edges_large_local = edges_large - edges_large[0]
    print(f"the large local edges are: {edges_large_local}")

    A2_zp, _, _ = zeroPad(A2_array, edges_large_local, return_inv=False, dtype=cp.float32)
    # print(A2_zp)
    A2_zp = cp.asfortranarray(A2_zp.transpose([0,2,1]))
    B2_zp, _, _ = zeroPad(B2_array, edges_large_local, return_inv=False, dtype=cp.float32)
    print(B2_zp.strides)
    
    B2_zp = cp.asfortranarray(B2_zp)
    print(B2_zp.strides)
    
    C2_zp = cp.zeros((len(edges_large_local)-1, A2_zp.shape[1], B2_zp.shape[2]), dtype=cp.float32, order='F')

    # print(A0_zp)
    # print(B0_zp)
    # # print(C0_zp.shape)

    # print(A1_zp)
    # print(B1_zp)
    # # print(C1_zp.shape)

    # print(f" A2 array: {A2_zp}")
    # print(f" B2 array: {B2_zp}")
    # # print(C2_zp.shape)

    All_A = list(A0_zp) + list(A1_zp) + list(A2_zp)
    # print(f"length of all list A {len(All_A)}")
    # print(f"list of all A mats {[A for A in All_A]}")
    All_B = list(B0_zp) + list(B1_zp) + list(B2_zp)
    All_C = list(C0_zp) + list(C1_zp) + list(C2_zp)


    # -------------------------------------------------------------------------
    # FIX: Use Numpy (CPU) to hold the list of pointers, not CuPy (GPU)
    # -------------------------------------------------------------------------
    
    # 1. Collect the pointers into a standard Python list
    A_ptr_list = [ar.data.ptr for ar in All_A]
    B_ptr_list = [br.data.ptr for br in All_B]
    C_ptr_list = [cr.data.ptr for cr in All_C]

    # 2. Create Numpy arrays (Host Memory) 
    #    We use uint64 (uintp) to hold the 64-bit pointer addresses
    # A_ptrs_host = np.array(A_ptr_list, dtype=np.uintp)
    # B_ptrs_host = np.array(B_ptr_list, dtype=np.uintp)
    # C_ptrs_host = np.array(C_ptr_list, dtype=np.uintp)

    # # 3. Cast to ctypes pointers
    # #    This passes a CPU pointer (to the array) that contains GPU pointers (values)
    # A_ptrs_ct = A_ptrs_host.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))
    # B_ptrs_ct = B_ptrs_host.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))
    # C_ptrs_ct = C_ptrs_host.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))

    A_ptrs = cp.array([ar.data.ptr for ar in All_A], dtype=cp.uintp)
    # print(f"list of all A mat pointers: {A_ptrs}")
    B_ptrs = cp.array([br.data.ptr for br in All_B], dtype=cp.uintp)
    C_ptrs = cp.array([cr.data.ptr for cr in All_C], dtype=cp.uintp)

    # A_ptrs = (ctypes.c_void_p * len(All_A))()
    # for i, blk in enumerate(All_A):
    #     A_ptrs[i] = ctypes.c_void_p(blk.data.ptr)

    # B_ptrs = (ctypes.c_void_p * len(All_B))()
    # for i, blk in enumerate(All_B):
    #     B_ptrs[i] = ctypes.c_void_p(blk.data.ptr)

    # C_ptrs = (ctypes.c_void_p * len(All_C))()
    # for i, blk in enumerate(All_C):
    #     C_ptrs[i] = ctypes.c_void_p(blk.data.ptr)


    A_ptrs_ct = ctypes.c_void_p(A_ptrs.data.ptr)
    # print(f"cupy ctypes pointer to all A {A_ptrs_ct}")
    B_ptrs_ct = ctypes.c_void_p(B_ptrs.data.ptr)
    C_ptrs_ct = ctypes.c_void_p(C_ptrs.data.ptr)

    # # Build host arrays of device pointers using ctypes
    m_row = np.array([A0_zp.shape[1], A1_zp.shape[1], A2_zp.shape[1]], np.int32)
    n_row = np.array([B0_zp.shape[2], B1_zp.shape[2], B2_zp.shape[2]], np.int32)
    k_row = np.array([B0_zp.shape[1], B1_zp.shape[1], B2_zp.shape[1]], np.int32)
    # print(f"the ms are {m_row}")
    # print(f"the ns are {n_row}")
    # print(f"the ks are {k_row}")

    #In C-major ordering, so the number of rows in each matrix
    # lda = m_row.copy().astype(np.int32, copy=False)
    # ldb = k_row.copy().astype(np.int32, copy=False)
    # ldc = m_row.copy().astype(np.int32, copy=False)

    lda = m_row.copy().astype(np.int32, copy=False)
    ldb = k_row.copy().astype(np.int32, copy=False)
    ldc = m_row.copy().astype(np.int32, copy=False)

    transA = np.array([0]*groupCount, dtype=np.int32)
    transB = np.array([0]*groupCount, dtype=np.int32)
    
    alpha_arr = np.ones(groupCount, dtype=np.float32)
    beta_arr = np.zeros(groupCount, dtype=np.float32)

    group_sizes = np.array([len(A0_zp), len(A1_zp), len(A2_zp)], dtype = np.int32)
    print(f"the groups sizes are {group_sizes}")
    
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
            # "A_ptrs_host_ref": A_ptrs_host, 
            # "B_ptrs_host_ref": B_ptrs_host,
            # "C_ptrs_host_ref": C_ptrs_host,

            "A_blocks": All_A,
            "B_blocks": All_B,
            "C_blocks": All_C
    }


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
    return cp.stack(param_dict["C_blocks"], axis=0)  # ordered by block index
    # return param_dict["C_blocks"]  # ordered by block index



if __name__ == "__main__":
            
    #Parameter set up
    n_eig = 2
    n_src = 1
    rows = 3
    cols = 3
    n_ant = rows*cols

    print('Parameters:')
    print('Number of antennas (Re Im split):', n_ant,
        '\nNumber of eigenmodes:', n_eig,
        '\nNumbe of sources:', n_src)
    print()


    cp.random.seed(10)
    spms = SimCorrcalParams(n_ant, n_eig, n_src, precision='float32', xp=cp)
    edges = spms.edges(grid_par_x=rows, grid_par_y=cols, use_random=False)

    edges = cp.asnumpy(edges)
    block_sizes = np.diff(edges)

    #sort the edges; doesn't matter since we just care about
    #relative shapes for the speed tests at this stage.
    #   Note if we wanted to do this in practice, we would simply
    #   have to sort the actual data into redundant groups from
    #   smallest to largest.

    print("-------------")
    print(edges)
    print(block_sizes)
    block_order = np.argsort(block_sizes)
    sorted_sizes = block_sizes[block_order]
    print(block_order)
    print(sorted_sizes)

    # print(f"Block sizes: {block_sizes}")
    # print(f"sorted block sizes: {sorted_sizes}")

    edges_sorted = np.zeros(len(sorted_sizes) + 1, dtype = edges.dtype)
    edges_sorted[1:] = np.cumsum(sorted_sizes)
    print(edges_sorted)

    #simulated matrices with correct shapes
    sim_data = spms.sim_data()
    noise = sim_data[0]
    diff = sim_data[1]

    #zeropad diff and noise 
    zp_noise, nb, lb = zeroPad(noise, edges_sorted, return_inv=True, dtype=cp.float32)
    zp_diff, nb, lb = zeroPad(diff, edges_sorted, return_inv=False, dtype=cp.float32)

    noise = 1/noise

    # print('zeropadded diff', zp_diff)
    # print('regular diff', diff)
    # print(zp_noise)

    #set up the temp mat just before the mat mul we are interested in
    temp = noise[..., None] * diff
    zp_temp = zp_noise[..., None] * zp_diff


    #cupy computation of diff.T @ N^-1 @ diff
    def cupy_block_mul(diff, tmp):
        res = cp.transpose(diff, [0, 2, 1]) @ tmp
        cp.cuda.Stream.null.synchronize()
        return res

    temp2 = cupy_block_mul(zp_diff, zp_temp)
    # print(temp2)


    #---------------------------------------------------------------------------------------
    # TESTS FOR CORRECT EDGES SPLITTING
    #---------------------------------------------------------------------------------------    

    print(f"Original edges: {edges}")
    
    edg_s, edg_m, edg_l = partition_into_3_groups(edges)
    # print(grps.values())
    print(f"Small edges: {edg_s}")
    print(f"Med edges: {edg_m}")
    print(f"Large edges: {edg_l}")

    print(len(edg_s), len(edg_m), len(edg_l))


    #---------------------------------------------------------------------------------------
    # TESTS FOR CORRECT EDGES SPLITTING
    #---------------------------------------------------------------------------------------    

    # prepare_grouped_batched_params(diff, temp, edges)



    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #also conclude with a single print line 
    print()

    params = prepare_grouped_batched_params(diff, temp, edges)
    C_array = groupedBatchedMatmul(params)


    print()
    print("---------------")
    # print(diff)
    # print(zp_diff)
    # print(zp_temp)

    # # print(diff)
    print(temp2)

    # print()

    print(C_array)


    # params = prepare_grouped_batched_params(diff, temp, edges)
    # C_blocks = groupedBatchedMatmul(params)

    # A_blocks = params["A_blocks"]
    # B_blocks = params["B_blocks"]

    # for i, (A_blk, B_blk, C_blk) in enumerate(zip(A_blocks, B_blocks, C_blocks)):
    #     C_ref = A_blk @ B_blk
    #     print(f"\nBlock {i}")
    #     print("A shape, B shape:", A_blk.shape, B_blk.shape)
    #     print("CuPy A@B:\n", C_ref)
    #     print("cuBLAS C:\n", C_blk)