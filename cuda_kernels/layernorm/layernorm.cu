/**
 * Phase 2: Layer Normalization CUDA Kernel
 * 
 * Layer Normalization formula:
 *   y = gamma * (x - mean) / sqrt(variance + eps) + beta
 * 
 * where:
 *   mean = mean(x) over the last dimension
 *   variance = var(x) over the last dimension
 *   gamma, beta: learnable parameters (scale and shift)
 *   eps: small constant for numerical stability
 * 
 * Implementation Strategy:
 * 1. Compute mean and variance in a single pass (Welford's algorithm)
 * 2. Normalize using computed statistics
 * 3. Apply affine transformation (gamma, beta)
 * 
 * Memory Access Pattern:
 * - Each thread processes one element
 * - Need to compute statistics across the last dimension
 * - Use shared memory for reduction within a block
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256

/**
 * Compute mean using parallel reduction
 * Uses warp-level primitives for efficiency
 */
__device__ float warp_reduce_sum(float val) {
    // Warp shuffle: Direct communication between threads in a warp
    // No shared memory needed for warp-level operations
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * LayerNorm kernel - optimized version
 * 
 * @param input: Input tensor [batch_size, seq_len, hidden_size]
 * @param output: Output tensor (same shape)
 * @param gamma: Scale parameter [hidden_size]
 * @param beta: Shift parameter [hidden_size]
 * @param batch_size: Number of samples
 * @param seq_len: Sequence length
 * @param hidden_size: Hidden dimension (normalization dimension)
 * @param eps: Epsilon for numerical stability
 */
__global__ void layernorm_kernel(
    const float* input,
    float* output,
    const float* gamma,
    const float* beta,
    int batch_size,
    int seq_len,
    int hidden_size,
    float eps
) {
    // Calculate which element this thread processes
    int batch_idx = blockIdx.y;
    int seq_idx = blockIdx.x;
    int hidden_idx = threadIdx.x;
    
    // Bounds check
    if (batch_idx >= batch_size || seq_idx >= seq_len || hidden_idx >= hidden_size) {
        return;
    }
    
    // Calculate linear index
    int idx = (batch_idx * seq_len + seq_idx) * hidden_size + hidden_idx;
    
    // Load input value
    float x = input[idx];
    
    // Step 1: Compute mean using parallel reduction
    // Each thread contributes one value
    float sum = x;
    
    // Reduce within warp
    sum = warp_reduce_sum(sum);
    
    // First thread in warp writes to shared memory
    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    
    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (warp_id == 0) {
        sum = (lane_id < (BLOCK_SIZE / WARP_SIZE)) ? warp_sums[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane_id == 0) {
            warp_sums[0] = sum / hidden_size;  // Store mean
        }
    }
    __syncthreads();
    
    float mean = warp_sums[0];
    
    // Step 2: Compute variance
    float centered = x - mean;
    float variance_sum = centered * centered;
    
    // Reduce variance sum
    variance_sum = warp_reduce_sum(variance_sum);
    if (lane_id == 0) {
        warp_sums[warp_id] = variance_sum;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        variance_sum = (lane_id < (BLOCK_SIZE / WARP_SIZE)) ? warp_sums[lane_id] : 0.0f;
        variance_sum = warp_reduce_sum(variance_sum);
        if (lane_id == 0) {
            float variance = variance_sum / hidden_size;
            warp_sums[0] = rsqrtf(variance + eps);  // Store 1/sqrt(variance + eps)
        }
    }
    __syncthreads();
    
    float inv_std = warp_sums[0];
    
    // Step 3: Normalize and apply affine transformation
    float normalized = centered * inv_std;
    output[idx] = gamma[hidden_idx] * normalized + beta[hidden_idx];
}

/**
 * Simplified LayerNorm kernel for single sequence
 * Assumes input is [hidden_size] and processes one sequence at a time
 * 
 * This version is easier to understand and good for learning
 */
__global__ void layernorm_simple_kernel(
    const float* input,
    float* output,
    const float* gamma,
    const float* beta,
    int hidden_size,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= hidden_size) return;
    
    // Load input
    float x = input[idx];
    
    // Compute mean (simplified - using shared memory reduction)
    __shared__ float s_mean;
    __shared__ float s_inv_std;
    
    // First pass: compute sum
    __shared__ float s_sum[256];
    s_sum[threadIdx.x] = x;
    __syncthreads();
    
    // Reduction tree
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        s_mean = s_sum[0] / hidden_size;
    }
    __syncthreads();
    
    // Second pass: compute variance
    float centered = x - s_mean;
    s_sum[threadIdx.x] = centered * centered;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        float variance = s_sum[0] / hidden_size;
        s_inv_std = rsqrtf(variance + eps);
    }
    __syncthreads();
    
    // Normalize and apply affine transformation
    float normalized = centered * s_inv_std;
    output[idx] = gamma[idx] * normalized + beta[idx];
}

/**
 * Host function to launch LayerNorm kernel
 */
void layernorm_cuda(
    const float* input,
    float* output,
    const float* gamma,
    const float* beta,
    int batch_size,
    int seq_len,
    int hidden_size,
    float eps = 1e-5f
) {
    // Allocate device memory
    float *d_input, *d_output, *d_gamma, *d_beta;
    size_t input_size = batch_size * seq_len * hidden_size * sizeof(float);
    size_t param_size = hidden_size * sizeof(float);
    
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, input_size);
    cudaMalloc(&d_gamma, param_size);
    cudaMalloc(&d_beta, param_size);
    
    // Copy data
    cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma, param_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, beta, param_size, cudaMemcpyHostToDevice);
    
    // Launch configuration
    dim3 threads(hidden_size);
    dim3 blocks(seq_len, batch_size);
    
    // Launch kernel
    layernorm_kernel<<<blocks, threads>>>(
        d_input, d_output, d_gamma, d_beta,
        batch_size, seq_len, hidden_size, eps
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("LayerNorm kernel error: %s\n", cudaGetErrorString(err));
    }
    
    cudaDeviceSynchronize();
    
    // Copy result back
    cudaMemcpy(output, d_output, input_size, cudaMemcpyDeviceToHost);
    
    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_gamma);
    cudaFree(d_beta);
}

