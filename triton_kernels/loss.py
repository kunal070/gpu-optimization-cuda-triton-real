"""
Custom Loss Functions using Triton
"""

import torch
import triton
import triton.language as tl


@triton.jit
def mse_loss_kernel(
    pred_ptr,
    target_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Mean Squared Error Loss"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    
    diff = pred - target
    output = diff * diff
    
    tl.store(output_ptr + offsets, output, mask=mask)


def mse_loss_triton(pred, target):
    """
    Mean Squared Error Loss
    
    Args:
        pred: Predictions tensor
        target: Target tensor (same shape as pred)
    
    Returns:
        MSE loss value
    """
    original_shape = pred.shape
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    n_elements = pred_flat.numel()
    
    output = torch.empty_like(pred_flat)
    
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    mse_loss_kernel[(n_blocks,)](
        pred_flat,
        target_flat,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output.view(original_shape).mean()


@triton.jit
def cross_entropy_kernel(
    pred_ptr,
    target_ptr,
    output_ptr,
    n_elements,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Cross Entropy Loss"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    
    # Clamp to avoid log(0)
    p = tl.maximum(pred, eps)
    output = -target * tl.math.log(p)
    
    tl.store(output_ptr + offsets, output, mask=mask)


def cross_entropy_loss_triton(pred, target, eps=1e-8):
    """Cross Entropy Loss"""
    original_shape = pred.shape
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    n_elements = pred_flat.numel()
    
    output = torch.empty_like(pred_flat)
    
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    cross_entropy_kernel[(n_blocks,)](
        pred_flat,
        target_flat,
        output,
        n_elements,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output.view(original_shape).sum()

