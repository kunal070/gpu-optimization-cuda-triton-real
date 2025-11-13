"""
Phase 4: GELU Activation Implementation using Triton

Triton makes it easy to implement element-wise operations with:
- Automatic parallelization
- Vectorized memory operations
- Clean, readable code
"""

import torch
import triton
import triton.language as tl


@triton.jit
def gelu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    use_fast: tl.constexpr,
):
    """
    Triton kernel for GELU activation
    
    Args:
        use_fast: If True, use fast tanh approximation; else use exact erf
    """
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate block start
    block_start = pid * BLOCK_SIZE
    
    # Create mask for bounds checking
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    if use_fast:
        # Fast GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        sqrt_2_over_pi = 0.7978845608028654
        c = 0.044715
        x_cubed = x * x * x
        inner = sqrt_2_over_pi * (x + c * x_cubed)
        output = 0.5 * x * (1.0 + tl.math.tanh(inner))
    else:
        # Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        sqrt_2 = 1.4142135623730951
        output = 0.5 * x * (1.0 + tl.math.erf(x / sqrt_2))
    
    # Store output
    tl.store(output_ptr + offsets, output, mask=mask)


def gelu_triton(input_tensor, use_fast=True):
    """
    Python wrapper for Triton GELU kernel
    
    Args:
        input_tensor: Input tensor of any shape
        use_fast: If True, use fast approximation
    
    Returns:
        GELU-activated tensor of same shape
    """
    # Flatten for processing
    original_shape = input_tensor.shape
    input_flat = input_tensor.view(-1)
    n_elements = input_flat.numel()
    
    # Allocate output
    output = torch.empty_like(input_flat)
    
    # Choose block size
    BLOCK_SIZE = 1024  # Good default for element-wise ops
    
    # Calculate number of blocks
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Launch kernel
    gelu_kernel[(n_blocks,)](
        input_flat,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        use_fast=use_fast,
    )
    
    # Reshape back
    return output.view(original_shape)


@triton.jit
def gelu_kernel_vectorized(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    use_fast: tl.constexpr,
):
    """
    Vectorized GELU kernel (processes multiple elements per program)
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Vectorized load
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    if use_fast:
        sqrt_2_over_pi = 0.7978845608028654
        c = 0.044715
        x_cubed = x * x * x
        inner = sqrt_2_over_pi * (x + c * x_cubed)
        output = 0.5 * x * (1.0 + tl.math.tanh(inner))
    else:
        sqrt_2 = 1.4142135623730951
        output = 0.5 * x * (1.0 + tl.math.erf(x / sqrt_2))
    
    # Vectorized store
    tl.store(output_ptr + offsets, output, mask=mask)


def gelu_triton_vectorized(input_tensor, use_fast=True):
    """Vectorized version of Triton GELU"""
    original_shape = input_tensor.shape
    input_flat = input_tensor.view(-1)
    n_elements = input_flat.numel()
    output = torch.empty_like(input_flat)
    
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    gelu_kernel_vectorized[(n_blocks,)](
        input_flat,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        use_fast=use_fast,
    )
    
    return output.view(original_shape)

