"""
Phase 4: LayerNorm Implementation using Triton

Triton is a Python-based GPU programming language that:
- Provides fine-grained control like CUDA
- Has a more readable, Python-like syntax
- Automatically handles many optimizations
- Supports auto-tuning for optimal performance

This implementation demonstrates:
- Triton kernel syntax
- Block-level programming model
- Automatic memory management
- Performance comparison with CUDA
"""

import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_kernel(
    input_ptr,      # Pointer to input tensor
    output_ptr,     # Pointer to output tensor
    gamma_ptr,      # Pointer to gamma (scale) parameter
    beta_ptr,       # Pointer to beta (shift) parameter
    n_elements,     # Number of elements in the normalization dimension
    eps: tl.constexpr,  # Epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,  # Block size (compile-time constant)
):
    """
    Triton kernel for LayerNorm
    
    Key differences from CUDA:
    - More Python-like syntax
    - Automatic memory management
    - Block-level programming (each program processes one block)
    - Compile-time constants using tl.constexpr
    """
    # Get program ID (which block this program is processing)
    row_id = tl.program_id(0)
    
    # Calculate offsets for this block
    row_start = row_id * n_elements
    
    # Load input data for this row
    # Block pointer: automatically handles memory coalescing
    block_ptr = input_ptr + row_start
    mask = tl.arange(0, BLOCK_SIZE) < n_elements
    
    x = tl.load(block_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    
    # Compute mean
    # Triton provides reduction operations
    mean = tl.sum(x) / n_elements
    
    # Center the data
    x_centered = x - mean
    
    # Compute variance
    variance = tl.sum(x_centered * x_centered) / n_elements
    
    # Compute inverse standard deviation
    inv_std = 1.0 / tl.sqrt(variance + eps)
    
    # Normalize
    normalized = x_centered * inv_std
    
    # Load gamma and beta
    gamma = tl.load(gamma_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=1.0)
    beta = tl.load(beta_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    
    # Apply affine transformation
    output = gamma * normalized + beta
    
    # Store result
    tl.store(output_ptr + row_start + tl.arange(0, BLOCK_SIZE), output, mask=mask)


def layernorm_triton(input_tensor, gamma, beta, eps=1e-5):
    """
    Python wrapper for Triton LayerNorm kernel
    
    Args:
        input_tensor: Input tensor of shape [batch_size, seq_len, hidden_size]
        gamma: Scale parameter of shape [hidden_size]
        beta: Shift parameter of shape [hidden_size]
        eps: Epsilon for numerical stability
    
    Returns:
        Normalized tensor of same shape as input
    """
    # Flatten batch and sequence dimensions
    original_shape = input_tensor.shape
    batch_size, seq_len, hidden_size = original_shape
    
    # Reshape to [batch_size * seq_len, hidden_size]
    input_flat = input_tensor.view(-1, hidden_size)
    n_rows = input_flat.shape[0]
    
    # Allocate output
    output = torch.empty_like(input_flat)
    
    # Choose block size (must be power of 2 and <= hidden_size)
    BLOCK_SIZE = triton.next_power_of_2(hidden_size)
    if BLOCK_SIZE > 1024:
        BLOCK_SIZE = 1024
    
    # Launch kernel
    # Grid: (n_rows,) - one program per row
    # Block: not used in Triton (each program is a block)
    layernorm_kernel[(n_rows,)](
        input_flat,
        output,
        gamma,
        beta,
        hidden_size,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape back to original shape
    return output.view(original_shape)


@triton.jit
def layernorm_kernel_optimized(
    input_ptr,
    output_ptr,
    gamma_ptr,
    beta_ptr,
    n_elements,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized Triton LayerNorm kernel
    
    Optimizations:
    - Better memory access patterns
    - Fused operations where possible
    - Vectorized loads/stores
    """
    row_id = tl.program_id(0)
    row_start = row_id * n_elements
    
    # Vectorized load (if BLOCK_SIZE is multiple of 4)
    block_ptr = input_ptr + row_start
    mask = tl.arange(0, BLOCK_SIZE) < n_elements
    
    x = tl.load(block_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    
    # Fused mean and variance computation
    mean = tl.sum(x) / n_elements
    x_centered = x - mean
    
    # Use fast math for variance
    variance = tl.sum(x_centered * x_centered) / n_elements
    inv_std = tl.math.rsqrt(variance + eps)  # Reciprocal square root (faster)
    
    # Load parameters
    gamma = tl.load(gamma_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=1.0)
    beta = tl.load(beta_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    
    # Fused normalization and affine transform
    output = gamma * (x_centered * inv_std) + beta
    
    tl.store(output_ptr + row_start + tl.arange(0, BLOCK_SIZE), output, mask=mask)


def layernorm_triton_optimized(input_tensor, gamma, beta, eps=1e-5):
    """Optimized version of Triton LayerNorm"""
    original_shape = input_tensor.shape
    batch_size, seq_len, hidden_size = original_shape
    
    input_flat = input_tensor.view(-1, hidden_size)
    n_rows = input_flat.shape[0]
    output = torch.empty_like(input_flat)
    
    BLOCK_SIZE = triton.next_power_of_2(hidden_size)
    if BLOCK_SIZE > 1024:
        BLOCK_SIZE = 1024
    
    layernorm_kernel_optimized[(n_rows,)](
        input_flat,
        output,
        gamma,
        beta,
        hidden_size,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output.view(original_shape)

