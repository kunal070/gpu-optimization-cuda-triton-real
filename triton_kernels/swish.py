"""
Swish Activation Function Implementation using Triton

Swish: f(x) = x * sigmoid(x) = x / (1 + exp(-x))
"""

import torch
import triton
import triton.language as tl


@triton.jit
def swish_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for Swish activation
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
    
    # Compute Swish: x * sigmoid(x)
    sigmoid_x = 1.0 / (1.0 + tl.math.exp(-x))
    output = x * sigmoid_x
    
    # Store output
    tl.store(output_ptr + offsets, output, mask=mask)


def swish_triton(input_tensor):
    """
    Python wrapper for Triton Swish kernel
    
    Args:
        input_tensor: Input tensor of any shape
    
    Returns:
        Swish-activated tensor of same shape
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
    swish_kernel[(n_blocks,)](
        input_flat,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape back
    return output.view(original_shape)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def swish_kernel_autotuned(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Auto-tuned version of Swish kernel"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    sigmoid_x = 1.0 / (1.0 + tl.math.exp(-x))
    output = x * sigmoid_x
    
    tl.store(output_ptr + offsets, output, mask=mask)


def swish_triton_autotuned(input_tensor):
    """Auto-tuned version of Triton Swish"""
    original_shape = input_tensor.shape
    input_flat = input_tensor.view(-1)
    n_elements = input_flat.numel()
    output = torch.empty_like(input_flat)
    
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    swish_kernel_autotuned[(n_blocks,)](
        input_flat,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output.view(original_shape)

