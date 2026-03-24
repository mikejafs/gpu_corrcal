// nvcc -Xcompiler -fPIC -shared -o gemmStridedBatched.so gemmStridedBatched.cu -lcublas

#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

//create a global handle to not have to load everytime
cublasHandle_t global_handle;

// This runs once when the .so is loaded
__attribute__((constructor))
void init_cublas() {
    cublasCreate(&global_handle);

    // OPTIONAL: enable Tensor Cores for a big speedup
    cublasSetMathMode(global_handle, CUBLAS_TF32_TENSOR_OP_MATH);
}

// This runs when the .so is unloaded
__attribute__((destructor))
void destroy_cublas() {
    cublasDestroy(global_handle);
}

extern "C"
{
    void gemmStridedBatched_matmul(
        float *Aarray,
        float *Barray,
        float *Carray,
        int m, int n, int k,
        long long strideA,
        long long strideB,
        long long strideC,
        int batchcount

    ){
        // cublasHandle_t handle;
        // cublasCreate(&handle);

        const float alpha = 1.0f;
        const float beta = 0.0f;

        cublasSgemmStridedBatched(
            global_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, m, k,
            &alpha,
            Barray, n, strideB,
            Aarray, k, strideA,
            &beta,
            Carray, n, strideC,
            batchcount 
        );

        // cublasDestroy(handle);
    }
}