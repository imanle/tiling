#include "common.h"
#include "timer.h"

#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(FILTER_RADIUS))

__constant__ float filter_c[FILTER_DIM][FILTER_DIM];

__global__ void convolution_tiled_kernel(float* input, float* output, unsigned int width, unsigned int height) {

     __shared__ float cov[IN_TILE_DIM][IN_TILE_DIM];
      int out_row = threadIdx.y + blockIdx.y * OUT_TILE_DIM;
      int out_col = threadIdx.x + blockIdx.x * OUT_TILE_DIM;
      int in_row = out_row - FILTER_DIM;
      int in_col = out_col - FILTER_DIM;
     
     float sum = 0.0f;
        if((in_row >= 0) && (in_row< height) && (in_col >= 0) && (in_col < width) ) {
        cov[threadIdx.y][threadIdx.x]= cov[in_row*width + in_col];
        }
    
    else{
     cov[threadIdx.y][threadIdx.x]=0;
     }
   __syncthreads();
     if(threadIdx.y < OUT_TILE_DIM && threadIdx.x < OUT_TILE_DIM){
        for(i = 0; i < FILTER_DIM; i++) {
            for(j = 0; j < FILTER_DIM; j++) { 
                sum += filter_c_[i][j] * cov[i+threadIdx.y][j+threadIdx.x];
} }
           __syncthreads();
          if(out_row < height && out_col < width){
    output[out_row*width + out_col] = sum;
          }
     }

}

void copyFilterToGPU(float filter[][FILTER_DIM]) {

    // Copy filter to constant memory

    cudaMemcpyToSymbol(filter_c, filter, FILTER_DIM*FILTER_DIM*sizeof(float));

}

void convolution_tiled_gpu(float* input_d, float* output_d, unsigned int width, unsigned int height) {

    // Call kernel

    dim3 numThreadsPerBlock(OUT_TILE_DIM, OUT_TILE_DIM);
    dim3 numBlocks((width + OUT_TILE_DIM - 1)/OUT_TILE_DIM, (height + OUT_TILE_DIM - 1)/OUT_TILE_DIM);
    convolution_tiled_kernel <<< numBlocks, numThreadsPerBlock >>> (input_d, output_d, width, height);



}
