/**
 * Phase 1: Basic CUDA Kernel - Matrix Multiplication
 * 
 * This kernel demonstrates:
 * - 2D thread indexing for matrix operations
 * - Shared memory usage for performance optimization
 * - Memory coalescing (efficient global memory access)
 * 
 * Matrix Multiplication: C = A * B
 * where A is [M x K], B is [K x N], C is [M x N]
 * 
 * Optimization Strategy:
 * - Use shared memory to cache tiles of A and B
 * - Each thread block processes a tile of the output matrix
 * - Reduces global memory accesses by reusing loaded data
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

/**
 * Naive matrix multiplication kernel (for comparison)
 * Each thread computes one element of C
 * 
 * Memory access pattern: Not optimized - each thread reads entire row/column
 */
__global__ void naive_matmul_kernel(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    // Calculate 2D thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bounds check
    if (row < M && col < N) {
        float sum = 0.0f;
        // Compute dot product of row of A and column of B
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

/**
 * Optimized matrix multiplication using shared memory (tiled)
 * 
 * Key optimizations:
 * 1. Shared memory caching: Load tiles of A and B into shared memory
 * 2. Memory coalescing: Threads in a warp access contiguous memory
 * 3. Reduced global memory traffic: Each element loaded once per tile
 * 
 * TILE_SIZE: Size of the tile (typically 16 or 32)
 * - Larger tiles = more shared memory usage but better reuse
 * - Must fit in shared memory (typically 48KB per SM)
 */
#define TILE_SIZE 16

__global__ void tiled_matmul_kernel(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    // Shared memory for tiles
    // __shared__: Shared memory visible to all threads in the same block
    // Static allocation: Size known at compile time
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    // Calculate thread indices within block
    int tx = threadIdx.x;  // Thread x index in block
    int ty = threadIdx.y;  // Thread y index in block
    
    // Calculate global thread position (output matrix element)
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    // Accumulator for this thread's output element
    float sum = 0.0f;
    
    // Iterate over tiles
    // Each tile processes TILE_SIZE columns of A and TILE_SIZE rows of B
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tile of A into shared memory
        // Each thread loads one element
        int a_col = tile * TILE_SIZE + tx;
        if (row < M && a_col < K) {
            tile_A[ty][tx] = A[row * K + a_col];
        } else {
            tile_A[ty][tx] = 0.0f;  // Padding for out-of-bounds
        }
        
        // Load tile of B into shared memory
        int b_row = tile * TILE_SIZE + ty;
        if (b_row < K && col < N) {
            tile_B[ty][tx] = B[b_row * N + col];
        } else {
            tile_B[ty][tx] = 0.0f;  // Padding for out-of-bounds
        }
        
        // Synchronize: Wait for all threads in block to finish loading
        // Critical: Shared memory is only consistent after __syncthreads()
        __syncthreads();
        
        // Compute partial dot product using shared memory tiles
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_A[ty][k] * tile_B[k][tx];
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }
    
    // Write result to global memory
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * Host function to launch matrix multiplication kernel
 */
void matrix_multiply_cuda(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    bool use_tiled = true
) {
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    // Copy data from host to device
    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);
    
    // Configure kernel launch parameters
    dim3 threads_per_block(TILE_SIZE, TILE_SIZE);
    dim3 num_blocks(
        (N + TILE_SIZE - 1) / TILE_SIZE,  // Blocks in x dimension
        (M + TILE_SIZE - 1) / TILE_SIZE   // Blocks in y dimension
    );
    
    // Launch appropriate kernel
    if (use_tiled) {
        tiled_matmul_kernel<<<num_blocks, threads_per_block>>>(
            d_A, d_B, d_C, M, N, K
        );
    } else {
        naive_matmul_kernel<<<num_blocks, threads_per_block>>>(
            d_A, d_B, d_C, M, N, K
        );
    }
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
    
    cudaDeviceSynchronize();
    
    // Copy result back
    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

