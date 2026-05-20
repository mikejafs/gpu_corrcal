//nvcc -shared -o zp_puregpu_funcs.so zp_puregpu_funcs.cu -Xcompiler -fPIC
#include <stdio.h>

extern "C"
{
    __global__ void zeroPad1d_kernel(
        double* in_array,
        double* out_array,
        long* edges,
        int n_blocks,
        int largest_block
    ){
        int blockidx = blockIdx.x*blockDim.x + threadIdx.x;
        int idx = blockIdx.y*blockDim.y + threadIdx.y;
        if (blockidx < n_blocks){   
            long start = edges[blockidx];
            long stop = edges[blockidx + 1];
            long block_size = stop - start;
            if (idx < block_size){
                out_array[blockidx*largest_block + idx] = in_array[start + idx];
            }
        }
    }

    __global__ void undo_zeroPad1d_kernel(
        double* in_array,
        double* out_array,
        long* edges,
        int n_blocks,
        int largest_block
    ){
        int blockidx = blockIdx.x*blockDim.x + threadIdx.x;
        int idx = blockIdx.y*blockDim.y + threadIdx.y;
        if (blockidx < n_blocks){
            long start = edges[blockidx];
            long stop = edges[blockidx + 1];
            long block_size = stop - start;
            if (idx < block_size){
                out_array[start + idx] = in_array[blockidx*largest_block + idx];
            }
        } 
    }

    __global__ void zeroPad2d_kernel(
        double* in_array,
        double* out_array,
        long* edges,
        int in_array_cols,
        int n_blocks,
        int largest_block
    ){
        int blockidx = blockIdx.x*blockDim.x + threadIdx.x;
        int row_idx = blockIdx.y*blockDim.y + threadIdx.y;
        int col_idx = blockIdx.z*blockDim.z + threadIdx.z;

        if (blockidx < n_blocks){
            long start = edges[blockidx];
            long stop = edges[blockidx + 1];
            long block_size = stop - start;
            if (row_idx < block_size){
                if (col_idx < in_array_cols){
                    out_array[in_array_cols*(blockidx*largest_block + row_idx) + col_idx] 
                    = in_array[in_array_cols*(start + row_idx) + col_idx];
                }
            } 
        }
    }

    __global__ void undo_zeroPad2d_kernel(
        double* in_array,
        double* out_array,
        long* edges,
        int in_array_cols,
        int n_blocks,
        int largest_block
    ){
        int blockidx = blockIdx.x*blockDim.x + threadIdx.x;
        // int blockidx = blockIdx.x;
        int row_idx = blockIdx.y*blockDim.y + threadIdx.y;
        int col_idx = blockIdx.z*blockDim.z + threadIdx.z;

        if (blockidx < n_blocks){
            long start = edges[blockidx];
            long stop = edges[blockidx + 1];
            long block_size = stop - start;
            if (row_idx < block_size){
                if (col_idx < in_array_cols){
                    out_array[in_array_cols*(start + row_idx) + col_idx] 
                    = in_array[in_array_cols*(blockidx*largest_block + row_idx) + col_idx];
                }
            } 
        }
    }

    void zeroPad1d(
        double* in_array,
        double* out_array,
        long* edges,
        int n_blocks,
        int largest_block
    ){
        //define thread and threadblock sizes & launch kernel
        //Note the prblm is in 2D, so we need a grid of threads
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((n_blocks + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                        (largest_block + threadsPerBlock.y - 1) / threadsPerBlock.y);

    zeroPad1d_kernel<<<numBlocks, threadsPerBlock>>>(
        in_array,
        out_array,
        edges,
        n_blocks,
        largest_block
        );
    }

    void undo_zeroPad1d(
        double* in_array,
        double* out_array,
        long* edges,
        int n_blocks,
        int largest_block
    ){
        //define thread and threadblock sizes & launch kernel
        //Note the prblm is in 2D, so we need a grid of threads
        dim3 threadsPerBlock(8, 8);
        dim3 numBlocks((n_blocks + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (largest_block + threadsPerBlock.y - 1) / threadsPerBlock.y);

        undo_zeroPad1d_kernel<<<numBlocks, threadsPerBlock>>>(
            in_array,
            out_array,
            edges,
            n_blocks,
            largest_block
            );
    }


    void zeroPad2d(
        double* in_array,
        double* out_array,
        long* edges,
        int in_array_cols,
        int n_blocks,
        int largest_block
    ){
        //define thread and threadblock sizes & launch kernel
        //Note the prblm is in 2D, so we need a grid of threads

        //TODO: Understand my gpu architecture and per considerations well enough
        //to be able to choose the correct values for threadsperblock in each
        //dim & to know if this layout of numBlocks is the best way to do things
        dim3 threadsPerBlock(8, 8, in_array_cols);
        dim3 numBlocks((n_blocks + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                        (largest_block + threadsPerBlock.y - 1) / threadsPerBlock.y,
                        (in_array_cols + threadsPerBlock.z - 1) / threadsPerBlock.z);
         
        zeroPad2d_kernel<<<numBlocks, threadsPerBlock>>>(
            in_array,
            out_array,
            edges,
            in_array_cols,
            n_blocks,
            largest_block 
            );
        cudaDeviceSynchronize();
    }

    void undo_zeroPad2d(
        double* in_array,
        double* out_array,
        long* edges,
        int in_array_cols,
        int n_blocks,
        int largest_block
    ){
        //define thread and threadblock sizes & launch kernel
        //Note the prblm is in 2D, so we need a grid of threads

        //TODO: Understand my gpu architecture and per considerations well enough
        //to be able to choose the correct values for threadsperblock in each
        //dim & to know if this layout of numBlocks is the best way to do things
        dim3 threadsPerBlock(8, 8, in_array_cols);
        dim3 numBlocks((n_blocks + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                        (largest_block + threadsPerBlock.y - 1) / threadsPerBlock.y,
                        (in_array_cols + threadsPerBlock.z - 1) / threadsPerBlock.z);
         
        undo_zeroPad2d_kernel<<<numBlocks, threadsPerBlock>>>(
            in_array,
            out_array,
            edges,
            in_array_cols,
            n_blocks,
            largest_block 
            );
        cudaDeviceSynchronize();
    }
}
