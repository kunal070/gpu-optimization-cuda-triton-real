/**
 * Custom Loss Functions CUDA Kernels
 * 
 * Implements common loss functions used in deep learning:
 * - Mean Squared Error (MSE)
 * - Cross Entropy
 * - Focal Loss
 * - Custom combinations
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

/**
 * Mean Squared Error Loss
 * MSE = mean((pred - target)^2)
 */
__global__ void mse_loss_kernel(
    const float* pred,
    const float* target,
    float* output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    float diff = pred[idx] - target[idx];
    output[idx] = diff * diff;
}

/**
 * Cross Entropy Loss (for classification)
 * CE = -sum(target * log(pred))
 */
__global__ void cross_entropy_kernel(
    const float* pred,
    const float* target,
    float* output,
    int n,
    float eps = 1e-8f
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    // Clamp prediction to avoid log(0)
    float p = fmaxf(pred[idx], eps);
    output[idx] = -target[idx] * __logf(p);
}

/**
 * Focal Loss (for handling class imbalance)
 * FL = -alpha * (1 - pred)^gamma * log(pred)
 */
__global__ void focal_loss_kernel(
    const float* pred,
    const float* target,
    float* output,
    int n,
    float alpha = 0.25f,
    float gamma = 2.0f,
    float eps = 1e-8f
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    float p = fmaxf(pred[idx], eps);
    float p_t = target[idx] * p + (1.0f - target[idx]) * (1.0f - p);
    float focal_weight = powf(1.0f - p_t, gamma);
    
    output[idx] = -alpha * focal_weight * __logf(p) * target[idx];
}

/**
 * Combined Loss (MSE + Regularization)
 */
__global__ void combined_loss_kernel(
    const float* pred,
    const float* target,
    float* output,
    int n,
    float lambda_reg = 0.01f
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    // MSE component
    float diff = pred[idx] - target[idx];
    float mse = diff * diff;
    
    // L2 regularization on predictions
    float reg = lambda_reg * pred[idx] * pred[idx];
    
    output[idx] = mse + reg;
}

/**
 * Host function to launch MSE loss
 */
float mse_loss_cuda(
    const float* pred,
    const float* target,
    int n
) {
    float *d_pred, *d_target, *d_output;
    size_t size = n * sizeof(float);
    
    cudaMalloc(&d_pred, size);
    cudaMalloc(&d_target, size);
    cudaMalloc(&d_output, size);
    
    cudaMemcpy(d_pred, pred, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target, size, cudaMemcpyHostToDevice);
    
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    
    mse_loss_kernel<<<num_blocks, threads_per_block>>>(
        d_pred, d_target, d_output, n
    );
    
    cudaDeviceSynchronize();
    
    // Reduce sum on device
    float* d_sum;
    cudaMalloc(&d_sum, sizeof(float));
    
    // Use thrust or manual reduction
    // For simplicity, copy back and reduce on host
    float* h_output = (float*)malloc(size);
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += h_output[i];
    }
    float loss = sum / n;
    
    free(h_output);
    cudaFree(d_pred);
    cudaFree(d_target);
    cudaFree(d_output);
    cudaFree(d_sum);
    
    return loss;
}

