/**
 * Swish Activation Function CUDA Kernel
 * 
 * Swish activation: f(x) = x * sigmoid(x)
 * 
 * This is an alternative to ReLU that has been shown to work well
 * in deep networks. It's smooth and non-monotonic.
 * 
 * Implementation Strategy:
 * - Compute sigmoid(x) = 1 / (1 + exp(-x))
 * - Multiply by x
 * - Vectorized operations where possible
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

/**
 * Swish activation kernel
 * 
 * @param input: Input tensor
 * @param output: Output tensor
 * @param n: Total number of elements
 */
__global__ void swish_kernel(
    const float* input,
    float* output,
    int n
) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bounds check
    if (idx >= n) return;
    
    // Load input value
    float x = input[idx];
    
    // Compute Swish: x * sigmoid(x) = x / (1 + exp(-x))
    // For numerical stability, use: x * (1 / (1 + exp(-x)))
    float sigmoid_x = 1.0f / (1.0f + expf(-x));
    output[idx] = x * sigmoid_x;
}

/**
 * Optimized Swish using fast approximation
 * Uses fast exp approximation for better performance
 */
__global__ void swish_fast_kernel(
    const float* input,
    float* output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    float x = input[idx];
    
    // Fast sigmoid approximation: 1 / (1 + exp(-x))
    // Using __expf for faster computation
    float sigmoid_x = 1.0f / (1.0f + __expf(-x));
    output[idx] = x * sigmoid_x;
}

/**
 * Vectorized Swish kernel (processes multiple elements per thread)
 */
__global__ void swish_vectorized_kernel(
    const float* input,
    float* output,
    int n
) {
    // Process 4 elements per thread
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    // Load 4 elements at once (if available)
    if (idx + 3 < n) {
        float4 vec = *reinterpret_cast<const float4*>(&input[idx]);
        
        float4 result;
        result.x = vec.x / (1.0f + __expf(-vec.x));
        result.y = vec.y / (1.0f + __expf(-vec.y));
        result.z = vec.z / (1.0f + __expf(-vec.z));
        result.w = vec.w / (1.0f + __expf(-vec.w));
        
        *reinterpret_cast<float4*>(&output[idx]) = result;
    } else {
        // Handle remaining elements
        for (int i = 0; i < 4 && (idx + i) < n; i++) {
            float x = input[idx + i];
            output[idx + i] = x / (1.0f + __expf(-x));
        }
    }
}

/**
 * Host function to launch Swish kernel
 */
void swish_cuda(
    const float* input,
    float* output,
    int n,
    bool use_fast = true,
    bool vectorized = false
) {
    float *d_input, *d_output;
    size_t size = n * sizeof(float);
    
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    
    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);
    
    // Configure launch parameters
    int threads_per_block = 256;
    int num_blocks;
    
    if (vectorized) {
        num_blocks = (n + (threads_per_block * 4) - 1) / (threads_per_block * 4);
        swish_vectorized_kernel<<<num_blocks, threads_per_block>>>(
            d_input, d_output, n
        );
    } else {
        num_blocks = (n + threads_per_block - 1) / threads_per_block;
        if (use_fast) {
            swish_fast_kernel<<<num_blocks, threads_per_block>>>(
                d_input, d_output, n
            );
        } else {
            swish_kernel<<<num_blocks, threads_per_block>>>(
                d_input, d_output, n
            );
        }
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Swish kernel error: %s\n", cudaGetErrorString(err));
    }
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
}

