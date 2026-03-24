extern "C" __global__
void corrcal_fused_blocks(
    const float* __restrict__ D,
    const float* __restrict__ T,
    const int* __restrict__ edges,
    int n_eig,
    float* __restrict__ C_out
){
    int b = blockIdx.x;
    int tid = threadIdx.x;
    int TBLK = blockDim.x;

    int start = edges[b];
    int stop  = edges[b+1];
    int h = stop - start;

    int warp = tid >> 5;
    int lane = tid & 31;
    int num_warps = TBLK / 32;

    extern __shared__ float shmem[];

    for(int i = 0; i < n_eig; i++){
        for(int j = 0; j < n_eig; j++){

            float local_sum = 0.f;

            for(int k = tid; k < h; k += TBLK){
                int idx = start + k;
                float d = D[idx * n_eig + i];
                float t = T[idx * n_eig + j];
                local_sum += d * t;
            }

            unsigned mask = 0xffffffff;
            for(int offset = 16; offset > 0; offset >>= 1){
                local_sum += __shfl_down_sync(mask, local_sum, offset);
            }

            if(lane == 0){
                shmem[warp*(n_eig*n_eig) + i*n_eig + j] = local_sum;
            }

            __syncthreads();

            if(tid == 0){
                float block_sum = 0.f;
                for(int w = 0; w < num_warps; w++){
                    block_sum += shmem[w*(n_eig*n_eig) + i*n_eig + j];
                }
                int out_idx = b*(n_eig*n_eig) + i*n_eig + j;
                C_out[out_idx] = block_sum;
            }

            __syncthreads();
        }
    }
}
