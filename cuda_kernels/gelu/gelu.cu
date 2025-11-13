/**
 * Phase 2: GELU Activation CUDA Kernel
 * 
 * GELU (Gaussian Error Linear Unit) activation function:
 *   GELU(x) = x * Φ(x)
 * 
 * where Φ(x) is the CDF of the standard normal distribution.
 * 
 * Common approximations:
 * 1. Exact (using erf): GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
 * 2. Fast approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
 * 3. Very fast (for inference): GELU(x) ≈ x * sigmoid(1.702 * x)
 * 
 * Implementation Strategy:
 * - Use fast approximation for good balance of accuracy and speed
 * - Vectorized operations where possible
 * - In-place and out-of-place variants
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

// Mathematical constants
#define M_SQRT2 1.41421356237309504880f
#define M_2_SQRTPI 1.12837916709551257390f
#define M_SQRT_2_OVER_PI 0.7978845608028654f

/**
 * Fast GELU approximation using tanh
 * 
 * Formula: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
 * 
 * This is the approximation used in many production systems (e.g., GPT models)
 * Accuracy: Very close to exact GELU, much faster than erf-based version
 */
__device__ __forceinline__ float gelu_fast(float x) {
    const float c1 = M_SQRT_2_OVER_PI;  // sqrt(2/π)
    const float c2 = 0.044715f;
    
    float x_cubed = x * x * x;
    float inner = c1 * (x + c2 * x_cubed);
    return 0.5f * x * (1.0f + tanhf(inner));
}

/**
 * Exact GELU using error function
 * 
 * Formula: GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
 * 
 * More accurate but slower due to erf computation
 */
__device__ __forceinline__ float gelu_exact(float x) {
    return 0.5f * x * (1.0f + erff(x / M_SQRT2));
}

/**
 * Very fast GELU approximation (for inference)
 * 
 * Formula: GELU(x) ≈ x * sigmoid(1.702 * x)
 * 
 * Fastest but less accurate
 */
__device__ __forceinline__ float gelu_very_fast(float x) {
    return x * (1.0f / (1.0f + expf(-1.702f * x)));
}

/**
 * GELU kernel - processes each element independently
 * 
 * @param input: Input tensor
 * @param output: Output tensor
 * @param n: Total number of elements
 * @param use_fast: If true, use fast approximation; else use exact
 */
__global__ void gelu_kernel(
    const float* input,
    float* output,
    int n,
    bool use_fast = true
) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bounds check
    if (idx >= n) return;
    
    // Load input value
    float x = input[idx];
    
    // Apply GELU activation
    if (use_fast) {
        output[idx] = gelu_fast(x);
    } else {
        output[idx] = gelu_exact(x);
    }
}

/**
 * In-place GELU kernel (modifies input directly)
 */
__global__ void gelu_inplace_kernel(
    float* data,
    int n,
    bool use_fast = true
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    if (use_fast) {
        data[idx] = gelu_fast(data[idx]);
    } else {
        data[idx] = gelu_exact(data[idx]);
    }
}

/**
 * Vectorized GELU kernel (processes multiple elements per thread)
 * 
 * This can improve performance by:
 * - Reducing thread launch overhead
 * - Better memory coalescing
 * - Better instruction-level parallelism
 */
__global__ void gelu_vectorized_kernel(
    const float* input,
    float* output,
    int n,
    bool use_fast = true
) {
    // Process multiple elements per thread
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    // Load 4 elements at once (if available)
    if (idx + 3 < n) {
        float4 vec = *reinterpret_cast<const float4*>(&input[idx]);
        
        float4 result;
        if (use_fast) {
            result.x = gelu_fast(vec.x);
            result.y = gelu_fast(vec.y);
            result.z = gelu_fast(vec.z);
            result.w = gelu_fast(vec.w);
        } else {
            result.x = gelu_exact(vec.x);
            result.y = gelu_exact(vec.y);
            result.z = gelu_exact(vec.z);
            result.w = gelu_exact(vec.w);
        }
        
        *reinterpret_cast<float4*>(&output[idx]) = result;
    } else {
        // Handle remaining elements
        for (int i = 0; i < 4 && (idx + i) < n; i++) {
            if (use_fast) {
                output[idx + i] = gelu_fast(input[idx + i]);
            } else {
                output[idx + i] = gelu_exact(input[idx + i]);
            }
        }
    }
}

/**
 * Host function to launch GELU kernel
 */
void gelu_cuda(
    const float* input,
    float* output,
    int n,
    bool use_fast = true,
    bool inplace = false,
    bool vectorized = false
) {
    float *d_input, *d_output;
    size_t size = n * sizeof(float);
    
    cudaMalloc(&d_input, size);
    
    if (inplace) {
        d_output = d_input;  // Same pointer for in-place
    } else {
        cudaMalloc(&d_output, size);
    }
    
    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);
    
    // Configure launch parameters
    int threads_per_block = 256;
    int num_blocks;
    
    if (vectorized) {
        // For vectorized kernel, we process 4 elements per thread
        num_blocks = (n + (threads_per_block * 4) - 1) / (threads_per_block * 4);
        gelu_vectorized_kernel<<<num_blocks, threads_per_block>>>(
            d_input, d_output, n, use_fast
        );
    } else {
        num_blocks = (n + threads_per_block - 1) / threads_per_block;
        if (inplace) {
            gelu_inplace_kernel<<<num_blocks, threads_per_block>>>(
                d_output, n, use_fast
            );
        } else {
            gelu_kernel<<<num_blocks, threads_per_block>>>(
                d_input, d_output, n, use_fast
            );
        }
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GELU kernel error: %s\n", cudaGetErrorString(err));
    }
    
    cudaDeviceSynchronize();
    
    if (!inplace) {
        cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);
    } else {
        cudaMemcpy(output, d_input, size, cudaMemcpyDeviceToHost);
    }
    
    cudaFree(d_input);
    if (!inplace) {
        cudaFree(d_output);
    }
}

