/**
 * Phase 5: CUDA Kernel Launchers for PyTorch Extension
 * 
 * This file bridges our CUDA kernels with PyTorch tensors.
 * It handles:
 * - Extracting raw pointers from torch::Tensor
 * - Type dispatching (float, double, half)
 * - Launching kernels with appropriate configurations
 */

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

// Include our CUDA kernel implementations
// Note: In a real project, these would be in separate .cu files
// For now, we'll include the key kernel code here

#include <cuda_fp16.h>
#include <math.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256

// Forward declarations of device kernels
__global__ void layernorm_kernel_float(
    const float* input,
    float* output,
    const float* gamma,
    const float* beta,
    int batch_size,
    int seq_len,
    int hidden_size,
    float eps
);

__global__ void gelu_kernel_float(
    const float* input,
    float* output,
    int n,
    bool use_fast
);

__global__ void layernorm_gelu_fused_kernel_float(
    const float* input,
    float* output,
    const float* gamma,
    const float* beta,
    int batch_size,
    int seq_len,
    int hidden_size,
    float eps
);

__global__ void matrix_multiply_kernel_float(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
);

__global__ void vector_add_kernel_float(
    const float* a,
    const float* b,
    float* c,
    int n
);

// ========== LayerNorm Kernel Implementation ==========
__device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void layernorm_kernel_float(
    const float* input,
    float* output,
    const float* gamma,
    const float* beta,
    int batch_size,
    int seq_len,
    int hidden_size,
    float eps
) {
    int batch_idx = blockIdx.y;
    int seq_idx = blockIdx.x;
    int hidden_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len || hidden_idx >= hidden_size) {
        return;
    }
    
    int idx = (batch_idx * seq_len + seq_idx) * hidden_size + hidden_idx;
    float x = input[idx];
    
    // Compute mean
    float sum = x;
    sum = warp_reduce_sum(sum);
    
    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    
    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        sum = (lane_id < (BLOCK_SIZE / WARP_SIZE)) ? warp_sums[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane_id == 0) {
            warp_sums[0] = sum / hidden_size;
        }
    }
    __syncthreads();
    
    float mean = warp_sums[0];
    
    // Compute variance
    float centered = x - mean;
    float variance_sum = centered * centered;
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
            warp_sums[0] = rsqrtf(variance + eps);
        }
    }
    __syncthreads();
    
    float inv_std = warp_sums[0];
    float normalized = centered * inv_std;
    output[idx] = gamma[hidden_idx] * normalized + beta[hidden_idx];
}

// ========== GELU Kernel Implementation ==========
__device__ __forceinline__ float gelu_fast(float x) {
    const float c1 = 0.7978845608028654f;
    const float c2 = 0.044715f;
    float x_cubed = x * x * x;
    float inner = c1 * (x + c2 * x_cubed);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__global__ void gelu_kernel_float(
    const float* input,
    float* output,
    int n,
    bool use_fast
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    if (use_fast) {
        output[idx] = gelu_fast(input[idx]);
    } else {
        output[idx] = 0.5f * input[idx] * (1.0f + erff(input[idx] / 1.4142135623730951f));
    }
}

// ========== Fused LayerNorm + GELU Kernel ==========
__global__ void layernorm_gelu_fused_kernel_float(
    const float* input,
    float* output,
    const float* gamma,
    const float* beta,
    int batch_size,
    int seq_len,
    int hidden_size,
    float eps
) {
    int batch_idx = blockIdx.y;
    int seq_idx = blockIdx.x;
    int hidden_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len || hidden_idx >= hidden_size) {
        return;
    }
    
    int idx = (batch_idx * seq_len + seq_idx) * hidden_size + hidden_idx;
    float x = input[idx];
    
    // Mean
    float sum = x;
    sum = warp_reduce_sum(sum);
    
    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    
    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        sum = (lane_id < (BLOCK_SIZE / WARP_SIZE)) ? warp_sums[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane_id == 0) {
            warp_sums[0] = sum / hidden_size;
        }
    }
    __syncthreads();
    
    float mean = warp_sums[0];
    
    // Variance
    float centered = x - mean;
    float variance_sum = centered * centered;
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
            warp_sums[0] = rsqrtf(variance + eps);
        }
    }
    __syncthreads();
    
    float inv_std = warp_sums[0];
    float normalized = centered * inv_std;
    float affine = gamma[hidden_idx] * normalized + beta[hidden_idx];
    output[idx] = gelu_fast(affine);
}

// ========== Matrix Multiply Kernel ==========
#define TILE_SIZE 16

__global__ void matrix_multiply_kernel_float(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        int a_col = tile * TILE_SIZE + tx;
        if (row < M && a_col < K) {
            tile_A[ty][tx] = A[row * K + a_col];
        } else {
            tile_A[ty][tx] = 0.0f;
        }
        
        int b_row = tile * TILE_SIZE + ty;
        if (b_row < K && col < N) {
            tile_B[ty][tx] = B[b_row * N + col];
        } else {
            tile_B[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_A[ty][k] * tile_B[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ========== Vector Add Kernel ==========
__global__ void vector_add_kernel_float(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// ========== Host Launcher Functions ==========

void layernorm_cuda_forward(
    const torch::Tensor& input,
    torch::Tensor& output,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    int batch_size,
    int seq_len,
    int hidden_size,
    float eps
) {
    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    const float* gamma_ptr = gamma.data_ptr<float>();
    const float* beta_ptr = beta.data_ptr<float>();
    
    dim3 threads(hidden_size);
    dim3 blocks(seq_len, batch_size);
    
    layernorm_kernel_float<<<blocks, threads>>>(
        input_ptr, output_ptr, gamma_ptr, beta_ptr,
        batch_size, seq_len, hidden_size, eps
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gelu_cuda_forward(
    const torch::Tensor& input,
    torch::Tensor& output,
    int n_elements,
    bool use_fast
) {
    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    
    int threads_per_block = 256;
    int num_blocks = (n_elements + threads_per_block - 1) / threads_per_block;
    
    gelu_kernel_float<<<num_blocks, threads_per_block>>>(
        input_ptr, output_ptr, n_elements, use_fast
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void layernorm_gelu_fused_cuda_forward(
    const torch::Tensor& input,
    torch::Tensor& output,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    int batch_size,
    int seq_len,
    int hidden_size,
    float eps
) {
    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    const float* gamma_ptr = gamma.data_ptr<float>();
    const float* beta_ptr = beta.data_ptr<float>();
    
    dim3 threads(hidden_size);
    dim3 blocks(seq_len, batch_size);
    
    layernorm_gelu_fused_kernel_float<<<blocks, threads>>>(
        input_ptr, output_ptr, gamma_ptr, beta_ptr,
        batch_size, seq_len, hidden_size, eps
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void matrix_multiply_cuda_forward(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& C,
    int M, int N, int K,
    bool use_tiled
) {
    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();
    
    dim3 threads_per_block(TILE_SIZE, TILE_SIZE);
    dim3 num_blocks(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE
    );
    
    matrix_multiply_kernel_float<<<num_blocks, threads_per_block>>>(
        A_ptr, B_ptr, C_ptr, M, N, K
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void vector_add_cuda_forward(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& c,
    int n
) {
    const float* a_ptr = a.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    float* c_ptr = c.data_ptr<float>();
    
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    
    vector_add_kernel_float<<<num_blocks, threads_per_block>>>(
        a_ptr, b_ptr, c_ptr, n
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

