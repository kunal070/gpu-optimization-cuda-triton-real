/**
 * Phase 5: PyTorch C++/CUDA Extension
 * 
 * This file provides the C++ interface between PyTorch and our CUDA kernels.
 * It uses PyTorch's ATen library for tensor operations and CUDA integration.
 * 
 * Key concepts:
 * - torch::Tensor: PyTorch tensor type
 * - AT_DISPATCH_FLOATING_TYPES: Dispatch based on scalar type
 * - CUDA_CHECK: Error checking for CUDA operations
 * - TORCH_CHECK: Input validation
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Forward declarations of CUDA kernel launchers
// These are defined in cuda_ops_kernel.cu

// LayerNorm
void layernorm_cuda_forward(
    const torch::Tensor& input,
    torch::Tensor& output,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    int batch_size,
    int seq_len,
    int hidden_size,
    float eps
);

// GELU
void gelu_cuda_forward(
    const torch::Tensor& input,
    torch::Tensor& output,
    int n_elements,
    bool use_fast
);

// Fused LayerNorm + GELU
void layernorm_gelu_fused_cuda_forward(
    const torch::Tensor& input,
    torch::Tensor& output,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    int batch_size,
    int seq_len,
    int hidden_size,
    float eps
);

// Matrix Multiply
void matrix_multiply_cuda_forward(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& C,
    int M, int N, int K,
    bool use_tiled
);

// Vector Add
void vector_add_cuda_forward(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& c,
    int n
);

/**
 * PyTorch function: LayerNorm
 * 
 * This function is callable from Python as:
 *   output = cuda_ops.layernorm(input, gamma, beta, eps)
 */
torch::Tensor layernorm(
    const torch::Tensor& input,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    double eps = 1e-5
) {
    // Input validation
    TORCH_CHECK(input.dim() == 3, "Input must be 3D tensor [batch, seq, hidden]");
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA device");
    TORCH_CHECK(gamma.is_cuda(), "Gamma must be on CUDA device");
    TORCH_CHECK(beta.is_cuda(), "Beta must be on CUDA device");
    
    int batch_size = input.size(0);
    int seq_len = input.size(1);
    int hidden_size = input.size(2);
    
    TORCH_CHECK(gamma.size(0) == hidden_size, "Gamma size mismatch");
    TORCH_CHECK(beta.size(0) == hidden_size, "Beta size mismatch");
    
    // Allocate output tensor (same shape as input)
    auto output = torch::empty_like(input);
    
    // Launch CUDA kernel
    layernorm_cuda_forward(
        input, output, gamma, beta,
        batch_size, seq_len, hidden_size,
        static_cast<float>(eps)
    );
    
    return output;
}

/**
 * PyTorch function: GELU
 */
torch::Tensor gelu(
    const torch::Tensor& input,
    bool use_fast = true
) {
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA device");
    
    auto output = torch::empty_like(input);
    int n_elements = input.numel();
    
    gelu_cuda_forward(input, output, n_elements, use_fast);
    
    return output;
}

/**
 * PyTorch function: Fused LayerNorm + GELU
 */
torch::Tensor layernorm_gelu_fused(
    const torch::Tensor& input,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    double eps = 1e-5
) {
    TORCH_CHECK(input.dim() == 3, "Input must be 3D tensor [batch, seq, hidden]");
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA device");
    
    int batch_size = input.size(0);
    int seq_len = input.size(1);
    int hidden_size = input.size(2);
    
    auto output = torch::empty_like(input);
    
    layernorm_gelu_fused_cuda_forward(
        input, output, gamma, beta,
        batch_size, seq_len, hidden_size,
        static_cast<float>(eps)
    );
    
    return output;
}

/**
 * PyTorch function: Matrix Multiply
 */
torch::Tensor matrix_multiply(
    const torch::Tensor& A,
    const torch::Tensor& B,
    bool use_tiled = true
) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be 2D tensor");
    TORCH_CHECK(A.is_cuda(), "A must be on CUDA device");
    TORCH_CHECK(B.is_cuda(), "B must be on CUDA device");
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    TORCH_CHECK(B.size(0) == K, "Matrix dimensions mismatch");
    
    auto C = torch::empty({M, N}, A.options());
    
    matrix_multiply_cuda_forward(A, B, C, M, N, K, use_tiled);
    
    return C;
}

/**
 * PyTorch function: Vector Add
 */
torch::Tensor vector_add(
    const torch::Tensor& a,
    const torch::Tensor& b
) {
    TORCH_CHECK(a.dim() == 1, "a must be 1D tensor");
    TORCH_CHECK(b.dim() == 1, "b must be 1D tensor");
    TORCH_CHECK(a.size(0) == b.size(0), "Vector sizes must match");
    TORCH_CHECK(a.is_cuda(), "a must be on CUDA device");
    TORCH_CHECK(b.is_cuda(), "b must be on CUDA device");
    
    int n = a.size(0);
    auto c = torch::empty_like(a);
    
    vector_add_cuda_forward(a, b, c, n);
    
    return c;
}

/**
 * Python module definition
 * 
 * This makes the functions available in Python as:
 *   import cuda_ops
 *   output = cuda_ops.layernorm(...)
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Custom CUDA operations for deep learning";
    
    m.def("layernorm", &layernorm, "LayerNorm (CUDA)",
          py::arg("input"), py::arg("gamma"), py::arg("beta"), py::arg("eps") = 1e-5);
    
    m.def("gelu", &gelu, "GELU activation (CUDA)",
          py::arg("input"), py::arg("use_fast") = true);
    
    m.def("layernorm_gelu_fused", &layernorm_gelu_fused, "Fused LayerNorm + GELU (CUDA)",
          py::arg("input"), py::arg("gamma"), py::arg("beta"), py::arg("eps") = 1e-5);
    
    m.def("matrix_multiply", &matrix_multiply, "Matrix multiplication (CUDA)",
          py::arg("A"), py::arg("B"), py::arg("use_tiled") = true);
    
    m.def("vector_add", &vector_add, "Vector addition (CUDA)",
          py::arg("a"), py::arg("b"));
}

