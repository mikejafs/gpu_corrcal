import cupy as cp
import numpy as np
import ctypes

lib = ctypes.cdll.LoadLibrary("./gemmStridedBatched.so")

lib.gemmStridedBatched_matmul.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_long,
    ctypes.c_long,
    ctypes.c_long,
    ctypes.c_int,
]


#thin wrapper around batched matmul
def stridedBatched_matmul(A_array, B_array):
    m = A_array[0].shape[0]
    k = A_array[0].shape[1]
    n = B_array[0].shape[1]
    batchcount = A_array.shape[0]

    C_array = cp.zeros((batchcount, m, n), dtype=cp.float32)

    strideA, strideB, strideC = m*k, k*n, m*n

    lib.gemmStridedBatched_matmul(
        A_array.data.ptr,
        B_array.data.ptr,
        C_array.data.ptr,
        m, n, k,
        strideA,
        strideB,
        strideC,
        batchcount
    )
    return C_array


if __name__ == "__main__":
    check_against_numpy = True
    printing_results = True
    timing = False

    #print some new lines between saving results to an output file
    print('\n \nNEW RESULT~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    if printing_results:
        #test dimensions
        m, n, k = 5, 4, 3
        batchcount = 3

    elif timing:
        #dimensions for timing tests
        m, n, k = 3, 3, 3000
        batchcount = 1000
        print(f'\nDimensions of the matrices were: \n'
               f'A->({batchcount}, {m}, {k}), B->({batchcount}, {k}, {n}) \n')

    A_array = cp.random.rand(batchcount, m, k, dtype = cp.float32)
    B_array = cp.random.rand(batchcount, k, n, dtype = cp.float32)

    if printing_results:    
        print('\nInput matrices A and B: \n \n', A_array, '\n \n',
            B_array)
    
        result = stridedBatched_matmul(A_array, B_array)
        print('\nOutput matrix after strided batched: \n \n', result)

    if check_against_numpy:
        result = stridedBatched_matmul(A_array, B_array)
        #check against cupy matmul
        C = A_array@B_array

        if cp.allclose(C, result):
            print('\nGreat! The gemm result matches the numpy result\n')

    if timing:
        start_gemm = cp.cuda.Event()
        stop_gemm = cp.cuda.Event()

        start_cupy = cp.cuda.Event()
        stop_cupy = cp.cuda.Event()

        start_gemm.record()
        n_trials = 200

        for _ in range(n_trials):
            stridedBatched_matmul(A_array=A_array, B_array=B_array)

        stop_gemm.record()
        stop_gemm.synchronize()

        t_ms_gemm = cp.cuda.get_elapsed_time(start_gemm, stop_gemm) / n_trials
        print("average time for gemm (ms):", t_ms_gemm)

        start_cupy.record()

        for _ in range(n_trials):
            cp.matmul(A_array, B_array)

        stop_cupy.record()
        stop_cupy.synchronize()

        t_ms_cupy = cp.cuda.get_elapsed_time(start_cupy, stop_cupy) / n_trials
        print("average time for cupy (ms):", t_ms_cupy)

