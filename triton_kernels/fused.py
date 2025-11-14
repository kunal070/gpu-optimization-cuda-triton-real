"""
Phase 4: Fused LayerNorm + GELU Implementation using Triton

Triton makes kernel fusion very natural - we can combine operations
in a single kernel with intermediate values staying in registers.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_gelu_fused_kernel(
    input_ptr,
    output_ptr,
    gamma_ptr,
    beta_ptr,
    n_elements,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    use_fast_gelu: tl.constexpr,
):
    """
    Fused LayerNorm + GELU kernel in Triton

    This demonstrates the power of Triton:
    - Clean, readable code
    - Automatic register management
    - Easy fusion of operations
    """
    row_id = tl.program_id(0)
    row_start = row_id * n_elements

    # Load input
    block_ptr = input_ptr + row_start
    mask = tl.arange(0, BLOCK_SIZE) < n_elements
    x = tl.load(block_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)

    # LayerNorm: Compute mean
    mean = tl.sum(x) / n_elements

    # LayerNorm: Center and compute variance
    x_centered = x - mean
    variance = tl.sum(x_centered * x_centered) / n_elements
    inv_std = tl.math.rsqrt(variance + eps)

    # LayerNorm: Normalize
    normalized = x_centered * inv_std

    # LayerNorm: Load parameters and apply affine transform
    gamma = tl.load(gamma_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=1.0)
    beta = tl.load(beta_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    affine = gamma * normalized + beta

    # GELU: Apply activation (still in registers!)
    # We use exact erf-based GELU in both branches for Triton 3.0 compatibility.
    sqrt_2 = 1.4142135623730951
    output = 0.5 * affine * (1.0 + tl.math.erf(affine / sqrt_2))

    # Store final result (only one write to global memory!)
    tl.store(output_ptr + row_start + tl.arange(0, BLOCK_SIZE), output, mask=mask)


def layernorm_gelu_fused_triton(input_tensor, gamma, beta, eps: float = 1e-5, use_fast_gelu: bool = True):
    """
    Python wrapper for fused LayerNorm + GELU kernel

    Args:
        input_tensor: Input tensor of shape [batch_size, seq_len, hidden_size]
        gamma: Scale parameter of shape [hidden_size]
        beta: Shift parameter of shape [hidden_size]
        eps: Epsilon for LayerNorm
        use_fast_gelu: Kept for API compatibility, but currently ignored
    Returns:
        Output tensor of same shape as input
    """
    original_shape = input_tensor.shape
    batch_size, seq_len, hidden_size = original_shape

    # Flatten batch and sequence dimensions
    input_flat = input_tensor.view(-1, hidden_size)
    n_rows = input_flat.shape[0]

    # Allocate output
    output = torch.empty_like(input_flat)

    # Choose block size
    BLOCK_SIZE = triton.next_power_of_2(hidden_size)
    if BLOCK_SIZE > 1024:
        BLOCK_SIZE = 1024

    # Launch kernel
    layernorm_gelu_fused_kernel[(n_rows,)](
        input_flat,
        output,
        gamma,
        beta,
        hidden_size,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        use_fast_gelu=use_fast_gelu,
    )

    # Reshape back
    return output.view(original_shape)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def layernorm_gelu_fused_autotuned_kernel(
    input_ptr,
    output_ptr,
    gamma_ptr,
    beta_ptr,
    n_elements,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    use_fast_gelu: tl.constexpr,
):
    """
    Auto-tuned version of fused kernel

    Triton's autotune feature automatically finds the best configuration
    for different input sizes by trying different block sizes and warp counts.
    """
    row_id = tl.program_id(0)
    row_start = row_id * n_elements

    block_ptr = input_ptr + row_start
    mask = tl.arange(0, BLOCK_SIZE) < n_elements
    x = tl.load(block_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)

    mean = tl.sum(x) / n_elements
    x_centered = x - mean
    variance = tl.sum(x_centered * x_centered) / n_elements
    inv_std = tl.math.rsqrt(variance + eps)

    normalized = x_centered * inv_std
    gamma = tl.load(gamma_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=1.0)
    beta = tl.load(beta_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    affine = gamma * normalized + beta

    # Same erf-based GELU here
    sqrt_2 = 1.4142135623730951
    output = 0.5 * affine * (1.0 + tl.math.erf(affine / sqrt_2))

    tl.store(output_ptr + row_start + tl.arange(0, BLOCK_SIZE), output, mask=mask)


def layernorm_gelu_fused_triton_autotuned(
    input_tensor,
    gamma,
    beta,
    eps: float = 1e-5,
    use_fast_gelu: bool = True,
):
    """Auto-tuned version of fused LayerNorm + GELU"""
    original_shape = input_tensor.shape
    batch_size, seq_len, hidden_size = original_shape

    input_flat = input_tensor.view(-1, hidden_size)
    n_rows = input_flat.shape[0]
    output = torch.empty_like(input_flat)

    BLOCK_SIZE = triton.next_power_of_2(hidden_size)
    if BLOCK_SIZE > 1024:
        BLOCK_SIZE = 1024

    layernorm_gelu_fused_autotuned_kernel[(n_rows,)](
        input_flat,
        output,
        gamma,
        beta,
        hidden_size,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        use_fast_gelu=use_fast_gelu,
    )

    return output.view(original_shape)
