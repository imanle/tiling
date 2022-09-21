
#include "common.h"
#include "timer.h"

#define TILE_DIM 32

__global__ void mm_tiled_kernel(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {
    __shared__ float A_s[TILE_DIM][TILE_DIM];
    __shared__ float B_s[TILE_DIM][TILE_DIM];
    unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for(unsigned int tile = 0; tile < (N + TILE_DIM -1 ) / TILE_DIM; ++tile) {

        if( row < M && tile*TILE_DIM+threadIdx.x < K){
        A_s[threadIdx.y][threadIdx.x] = A[row*K + tile*TILE_DIM + threadIdx.x];
        }
        else{
        A_s[threadIdx.y][threadIdx.x] = 0;
        }
        if( TILE_DIM*tile+threadIdx.y < K && col < N){
        
        B_s[threadIdx.y][threadIdx.x] = B[(tile*TILE_DIM + threadIdx.y)*K + col];
        }
        else{
        B_s[threadIdx.y][threadIdx.x] = 0;
        }
        
__syncthreads();
        for(unsigned int i = 0; i < TILE_DIM; ++i) {
            if(i+tile*TILE_DIM<K){
                sum += A_s[threadIdx.y][i]*B_s[i][threadIdx.x];
            }
            }
        }
            __syncthreads();
    if(row< M && col<N){
        C[row*N + col] = sum;   
} 


}

void mm_gpu(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {

    Timer timer;

    // Allocate GPU memory
    startTime(&timer);

    float *A_d, *B_d, *C_d; 
 cudaMalloc((void**) &A_d, M*K*sizeof(float));
 cudaMalloc((void**) &B_d, K*N*sizeof(float)); 
 cudaMalloc((void**) &C_d, M*N*sizeof(float));





    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    // Copy data to GPU
    startTime(&timer);

    cudaMemcpy(A_d, A, M*K*sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(B_d, B, K*N*sizeof(float), cudaMemcpyHostToDevice);







    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time");

    // Call kernel
    startTime(&timer);

   dim3 numThreadsPerBlock(32, 32);
    dim3 numBlocks((N + numThreadsPerBlock.x - 1)/numThreadsPerBlock.x ,
                   (M + numThreadsPerBlock.y - 1)/numThreadsPerBlock.y); 
    mm_tiled_kernel <<< numBlocks, numThreadsPerBlock >>> (A_d, B_d, C_d, M, N, K);







    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time", GREEN);

    // Copy data from GPU
    startTime(&timer);

    cudaMemcpy(C, C_d, M*N*sizeof(float), cudaMemcpyDeviceToHost);






    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");

    // Free GPU memory
    startTime(&timer);

    cudaFree(A_d); 
    cudaFree(B_d); 
    cudaFree(C_d);






    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");

}

