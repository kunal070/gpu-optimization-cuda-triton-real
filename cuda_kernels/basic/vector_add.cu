/**
 * Phase 1: Basic CUDA Kernel - Vector Addition
 * 
 * This kernel demonstrates fundamental CUDA concepts:
 * - Thread indexing (threadIdx, blockIdx, blockDim, gridDim)
 * - Global memory access patterns
 * - Simple parallel computation
 * 
 * GPU Architecture Concepts:
 * - Each thread processes one element
 * - Threads are organized into blocks
 * - Blocks are organized into a grid
 * - All threads execute the same kernel code (SIMT - Single Instruction Multiple Threads)
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

/**
 * CUDA kernel for vector addition: c = a + b
 * 
 * @param a: Input vector A (device memory)
 * @param b: Input vector B (device memory)
 * @param c: Output vector C (device memory)
 * @param n: Number of elements
 * 
 * Thread Organization:
 * - Each thread computes one element: c[i] = a[i] + b[i]
 * - Thread ID is calculated as: blockIdx.x * blockDim.x + threadIdx.x
 * - This ensures all elements are processed in parallel
 */
__global__ void vector_add_kernel(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    // Calculate global thread index
    // blockIdx.x: which block this thread belongs to (0, 1, 2, ...)
    // blockDim.x: number of threads per block
    // threadIdx.x: thread index within the block (0 to blockDim.x-1)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bounds check: ensure we don't access out-of-bounds memory
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

/**
 * Host function to launch the vector addition kernel
 * 
 * @param a: Input vector A (host memory)
 * @param b: Input vector B (host memory)
 * @param c: Output vector C (host memory, will be filled)
 * @param n: Number of elements
 */
void vector_add_cuda(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    size_t size = n * sizeof(float);
    
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // Copy data from host to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    
    // Configure kernel launch parameters
    // threads_per_block: number of threads in each block (typically 256 or 512)
    // num_blocks: number of blocks needed to cover all elements
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;  // Ceiling division
    
    // Launch kernel
    // <<<num_blocks, threads_per_block>>>: execution configuration
    vector_add_kernel<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, n);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
    
    // Wait for kernel to complete
    cudaDeviceSynchronize();
    
    // Copy result back from device to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

