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


#small wrapper used to define pointers from (CuPy) python arrays
def ptr(array):
    return ctypes.c_void_p(array.data.ptr)


# -----------------------------------------------------------------------------
# Step 1 — Partition blocks into 3 super-groups
# -----------------------------------------------------------------------------
def partition_into_3_groups(edges):
    """Return {0:[small], 1:[medium], 2:[large]} block-ID groups."""
    edges = cp.asnumpy(edges)
    block_sizes = np.diff(edges)

    med = np.median(block_sizes)
    mx  = block_sizes.max()
    mid = 0.5 * (med + mx)

    groups = {0: [], 1: [], 2: []}

    for i, h in enumerate(block_sizes):
        if h < med:
            groups[0].append(i)
        elif h < mid:
            groups[1].append(i)
        else:
            groups[2].append(i)

    # print("\n=== GROUP PARTITION RESULTS ===")
    # print("Median block height:", med)
    # print("Midpoint height:", mid)
    # print("Max block height:", mx)
    # print("Groups:", groups)

    return groups, block_sizes


# -----------------------------------------------------------------------------
# Step 2 — Build per-group local edges arrays
# -----------------------------------------------------------------------------
def build_local_edges(block_ids, global_edges):
    """Construct a new edges array referring only to the chosen block IDs."""
    global_edges = cp.asnumpy(global_edges)  # FIX

    new_edges = [0]
    acc = 0
    for b in block_ids:
        h = global_edges[b+1] - global_edges[b]
        acc += h
        new_edges.append(acc)
    return np.asarray(new_edges, dtype=np.int64)


# -----------------------------------------------------------------------------
# Step 3 — Extract the blocks of A and B for each super-group and zero-pad them
# -----------------------------------------------------------------------------
def extract_and_zeropad_group(A, B, global_edges, block_ids, dtype):
    """
    Extract blocks belonging to one super-group, concatenate them,
    build a local edges array, and call your zeroPad().

    Returns:
        (zpA_blocks, zpB_blocks, local_edges, largest_block, n_blocks)
    """
    # Build local edges
    local_edges = build_local_edges(block_ids, global_edges)

    # Extract blocks belonging to the group
    A_slices = []
    B_slices = []

    for b in block_ids:
        s = global_edges[b]
        e = global_edges[b+1]
        A_slices.append(A[s:e])         # (h, n_eig)
        B_slices.append(B[s:e])         # (h, n_eig)

    # Concatenate blocks for zeroPad
    A_concat = cp.concatenate(A_slices, axis=0)
    B_concat = cp.concatenate(B_slices, axis=0)

    # Zero-pad using your CUDA kernels
    zpA, largest_block, n_blocks = zeroPad(A_concat, local_edges, return_inv=False, dtype=dtype)
    zpB, _, _ = zeroPad(B_concat, local_edges, return_inv=False, dtype=dtype)
    # print(zpA)

    # zpA and zpB now have shape:
    # (n_blocks, largest_block, n_eig)

    return zpA, zpB, local_edges, largest_block, n_blocks


# -----------------------------------------------------------------------------
# Step 4 — Prepare a GEMM group (Option A: one group per super-group)
# -----------------------------------------------------------------------------
def prepare_gemm_group(zpA, zpB):
    """
    zpA, zpB shapes: (n_blocks, H, n_eig)
    Returns:
        A_list, B_list, C_list for this group.
    """
    n_blocks, H, n_eig = zpA.shape

    A_list = []
    B_list = []
    C_list = []

    # For cuBLAS: Aᵀ @ B → shapes become (n_eig, H) @ (H, n_eig)
    for i in range(n_blocks):
        A_i = cp.asfortranarray(zpA[i].T)      # (n_eig, H)
        B_i = cp.asfortranarray(zpB[i])        # (H, n_eig)
        C_i = cp.zeros((n_eig, n_eig), dtype=zpA.dtype, order='F')

        A_list.append(A_i)
        B_list.append(B_i)
        C_list.append(C_i)

    return A_list, B_list, C_list, H


# -----------------------------------------------------------------------------
# Step 5 — Build ALL 3 GEMM groups and pass to your groupedBatched pipeline
# -----------------------------------------------------------------------------
def prepare_params_for_all_groups(A, B, edges, dtype):
    """
    Creates exactly 3 GEMM groups (Option A).
    Returns:
        param_dict ready for groupedBatchedMatmul()
        + a mapping from global block index → (group_id, local_block_index)
          used for reconstructing output.
    """
    # Partition blocks
    groups, block_sizes = partition_into_3_groups(edges)

    # Data structures to feed into groupedBatched
    all_A_lists = []
    all_B_lists = []
    all_C_lists = []
    k_array = []
    group_sizes = []
    block_map = {}  # global_block_index → (group_id, local_index)

    for g, block_ids in groups.items():
        if len(block_ids) == 0:
            # empty group — still append
            all_A_lists.append([])
            all_B_lists.append([])
            all_C_lists.append([])
            k_array.append(0)
            group_sizes.append(0)
            continue

        zpA, zpB, local_edges, largest_block, n_blocks = extract_and_zeropad_group(
            A, B, edges, block_ids, dtype
        )

        A_list, B_list, C_list, H = prepare_gemm_group(zpA, zpB)

        all_A_lists.append(A_list)
        all_B_lists.append(B_list)
        all_C_lists.append(C_list)

        k_array.append(H)
        group_sizes.append(n_blocks)

        # build map for reconstructing in global block order
        for j, b in enumerate(block_ids):
            block_map[b] = (g, j)

    # Now flatten pointer lists
    flat_A = [a for group in all_A_lists for a in group]
    flat_B = [b for group in all_B_lists for b in group]
    flat_C = [c for group in all_C_lists for c in group]

    # Device-side pointer arrays
    A_ptrs_raw = cp.asarray([a.data.ptr for a in flat_A], dtype=cp.uintp)
    B_ptrs_raw = cp.asarray([b.data.ptr for b in flat_B], dtype=cp.uintp)
    C_ptrs_raw = cp.asarray([c.data.ptr for c in flat_C], dtype=cp.uintp)

    A_ptrs_ct = ctypes.c_void_p(A_ptrs_raw.data.ptr)
    B_ptrs_ct = ctypes.c_void_p(B_ptrs_raw.data.ptr)
    C_ptrs_ct = ctypes.c_void_p(C_ptrs_raw.data.ptr)

    # m = n = n_eig
    if len(flat_A) > 0:
        n_eig = flat_A[0].shape[0]
    else:
        n_eig = 0

    m_array = np.array([n_eig]*3, dtype=np.int32)
    n_array = np.array([n_eig]*3, dtype=np.int32)
    k_array = np.array(k_array, dtype=np.int32)

    transA = np.array([0,0,0], dtype=np.int32)
    transB = np.array([0,0,0], dtype=np.int32)

    alpha_arr = np.ones(3, dtype=np.float32)
    beta_arr  = np.zeros(3, dtype=np.float32)

    lda_array = m_array.copy()
    ldb_array = k_array.copy()
    ldc_array = m_array.copy()

    group_sizes = np.array(group_sizes, dtype=np.int32)

    params = {
        "groupCount": 3,
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
        "C_lists": all_C_lists,
        "block_map": block_map,
        "block_sizes": block_sizes,
    }
    return params


# -----------------------------------------------------------------------------
# Step 6 — Run grouped GEMM, then reconstruct in global block order
# -----------------------------------------------------------------------------
def regroup_output(C_lists, block_map, n_blocks_total):
    """Undo grouping: return C_out in global block index order."""
    C_out = [None] * n_blocks_total

    for global_b, (g, j) in block_map.items():
        C_out[global_b] = C_lists[g][j]

    return cp.stack(C_out, axis=0)   # shape (N_blocks, n_eig, n_eig)



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
    return param_dict["C_lists"]  # ordered by block index


# def reshape_out(C_ptrs, edges):
#     bl_sizes, grps = group_blocks_by_size(edges)
#     group_sizes = np.array([len(grp_size) for grp_size in grps.values()], dtype=np.int32)

#     flat_C = [arr for group in C_ptrs for arr in group]
#     C_out = [None] * len(bl_sizes)
#     idx = 0
#     for (h, block_ids), size in zip(grps.items(), group_sizes):
#         for j, b in enumerate(block_ids):
#             C_out[b] = flat_C[idx + j]
#         idx += size

#     # print("edges:", edges)
#     # print("block_sizes:", bl_sizes)
#     # print("groups:", grps)
#     # print("group_sizes:", group_sizes)
#     # print("N_blocks:", len(bl_sizes), "sum(group_sizes):", group_sizes.sum())

#     C_stacked = cp.stack(C_out, axis=0)
#     return C_stacked

# -----------------------------------------------------------------------------
# Step 6 — Run grouped GEMM, then reconstruct in global block order
# -----------------------------------------------------------------------------
def regroup_output(C_lists, block_map, n_blocks_total):
    """Undo grouping: return C_out in global block index order."""
    C_out = [None] * n_blocks_total

    for global_b, (g, j) in block_map.items():
        C_out[global_b] = C_lists[g][j]

    return cp.stack(C_out, axis=0)   # shape (N_blocks, n_eig, n_eig)



if __name__ == "__main__":
            
    #Parameter set up
    n_ant = 5
    n_eig = 2
    n_src = 1

    print('Parameters:')
    print('Number of antennas (Re Im split):', n_ant,
        '\nNumber of eigenmodes:', n_eig,
        '\nNumbe of sources:', n_src)
    print()


    cp.random.seed(10)
    spms = SimCorrcalParams(n_ant, n_eig, n_src, precision='float32', xp=cp)
    edges = spms.edges(grid_par_x=3, grid_par_y=3, use_random=True)

    #simulated matrices with correct shapes
    sim_data = spms.sim_data()
    noise = sim_data[0]
    diff = sim_data[1]

    #zeropad diff and noise 
    zp_noise, nb, lb = zeroPad(noise, edges, return_inv=True, dtype=cp.float32)
    zp_diff, nb, lb = zeroPad(diff, edges, return_inv=False, dtype=cp.float32)

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

    #running the batched grouped matmul
    params = prepare_params_for_all_groups(diff, temp, edges, dtype=cp.float32)
    C_lists = groupedBatchedMatmul(params)
    

    # --- Regroup back to global order ---
    C_stacked = regroup_output(
        params["C_lists"], 
        params["block_map"], 
        len(params["block_sizes"])
    )

    # print(C_stacked)


    #CHECK IF CUPY AND CUBLAS ARE COMPUTING THE SAME THING
    if np.allclose(temp2, C_stacked):
        print('Checking correctness with np.allclose (default params):' \
        '\nCuPy and cuBLAS match')
    else:
        print('Checking correctness with np.allclose (default params):' \
        '\nCuPy and cuBLAS DO NOT match')


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

    def gemm_once():
        params = prepare_params_for_all_groups(diff, temp, edges, cp.float32)
        return groupedBatchedMatmul(params)

    #CUBLAS TIMES
    times = (benchmark(gemm_once, (), n_repeat = 100))
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




    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #also conclude with a single print line 
    # print()