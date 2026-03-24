// nvcc -Xcompiler -fPIC -shared -o gemmGroupedBatched.so gemmGroupedBatched.cu -lcublas
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>


cublasHandle_t global_handle;

__attribute__((constructor))
void init_cublas(){
    cublasStatus_t status = cublasCreate(&global_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cublasCreate failed: %d\n", status);
    }

    cublasSetMathMode(global_handle, CUBLAS_TF32_TENSOR_OP_MATH);
    // cublasSetMathMode(global_handle, CUBLAS_DEFAULT_MATH);
}

__attribute__((destructor))
void destroy_cublas() {
    // printf(">>> Destroying cuBLAS handle now...\n");
    cublasDestroy(global_handle);
}

extern "C"
{
    void gemmGroupedBatched(
        int groupCount,
        const cublasOperation_t *transA,
        const cublasOperation_t *transB,
        const int *m,
        const int *n,
        const int *k,
        const float *alpha_array,
        const float *const *Aarray, const int *lda,
        const float *const *Barray, const int *ldb,
        const float *beta_array,
        float *const *Carray, const int *ldc,
        const int *groupsize
    ){

    // static cublasHandle_t handle = nullptr;
    // if (handle == nullptr) {
    //     cublasCreate(&handle);
    //     cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
    // }

    // cublasHandle_t handle;
    // cublasCreate(&handle);

    cublasStatus_t status = cublasSgemmGroupedBatched(
        global_handle,
        // handle,
        transA,
        transB,
        m, n, k,
        alpha_array,
        Aarray, lda,
        Barray, ldb,
        beta_array,
        Carray, ldc,
        groupCount, groupsize
    );

    if (status != CUBLAS_STATUS_SUCCESS){
        printf("groupedBatched GEMM failed: %d\n", status);
    }

    // cublasDestroy(handle);

    }

}