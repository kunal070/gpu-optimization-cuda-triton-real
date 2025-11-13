"""
Simple Example: Using Custom GPU Kernels

This script demonstrates how to use the custom CUDA and Triton kernels
for common deep learning operations.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import CUDA extension
try:
    import cuda_ops
    CUDA_OPS_AVAILABLE = True
    print("[OK] CUDA ops extension loaded")
except ImportError:
    CUDA_OPS_AVAILABLE = False
    print("[SKIP] CUDA ops extension not available (run: python setup.py build_ext --inplace)")

# Try to import Triton kernels
try:
    from triton_kernels import (
        layernorm_triton,
        gelu_triton,
        layernorm_gelu_fused_triton,
    )
    TRITON_AVAILABLE = True
    print("[OK] Triton kernels loaded")
except ImportError:
    TRITON_AVAILABLE = False
    print("[SKIP] Triton kernels not available")


def example_layernorm():
    """Example: LayerNorm operation"""
    print("\n" + "="*60)
    print("Example: LayerNorm")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping example.")
        return
    
    device = 'cuda'
    batch_size, seq_len, hidden_size = 2, 4, 8
    
    # Create input
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, device=device)
    gamma = torch.ones(hidden_size, device=device)
    beta = torch.zeros(hidden_size, device=device)
    
    print(f"Input shape: {input_tensor.shape}")
    
    # PyTorch native
    pytorch_output = torch.nn.functional.layer_norm(
        input_tensor, (hidden_size,), gamma, beta
    )
    print(f"PyTorch output shape: {pytorch_output.shape}")
    
    # CUDA extension
    if CUDA_OPS_AVAILABLE:
        cuda_output = cuda_ops.layernorm(input_tensor, gamma, beta)
        max_diff = torch.max(torch.abs(cuda_output - pytorch_output)).item()
        print(f"CUDA Extension - Max difference: {max_diff:.2e}")
    
    # Triton
    if TRITON_AVAILABLE:
        triton_output = layernorm_triton(input_tensor, gamma, beta)
        max_diff = torch.max(torch.abs(triton_output - pytorch_output)).item()
        print(f"Triton - Max difference: {max_diff:.2e}")


def example_gelu():
    """Example: GELU activation"""
    print("\n" + "="*60)
    print("Example: GELU Activation")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping example.")
        return
    
    device = 'cuda'
    input_tensor = torch.randn(2, 4, 8, device=device)
    
    print(f"Input shape: {input_tensor.shape}")
    
    # PyTorch native
    pytorch_output = torch.nn.functional.gelu(input_tensor)
    print(f"PyTorch output shape: {pytorch_output.shape}")
    
    # CUDA extension
    if CUDA_OPS_AVAILABLE:
        cuda_output = cuda_ops.gelu(input_tensor, use_fast=True)
        max_diff = torch.max(torch.abs(cuda_output - pytorch_output)).item()
        print(f"CUDA Extension - Max difference: {max_diff:.2e}")
    
    # Triton
    if TRITON_AVAILABLE:
        triton_output = gelu_triton(input_tensor, use_fast=True)
        max_diff = torch.max(torch.abs(triton_output - pytorch_output)).item()
        print(f"Triton - Max difference: {max_diff:.2e}")


def example_fused():
    """Example: Fused LayerNorm + GELU"""
    print("\n" + "="*60)
    print("Example: Fused LayerNorm + GELU")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping example.")
        return
    
    device = 'cuda'
    batch_size, seq_len, hidden_size = 2, 4, 8
    
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, device=device)
    gamma = torch.ones(hidden_size, device=device)
    beta = torch.zeros(hidden_size, device=device)
    
    print(f"Input shape: {input_tensor.shape}")
    
    # PyTorch: Separate operations
    normalized = torch.nn.functional.layer_norm(
        input_tensor, (hidden_size,), gamma, beta
    )
    pytorch_output = torch.nn.functional.gelu(normalized)
    print(f"PyTorch (separate) output shape: {pytorch_output.shape}")
    
    # CUDA extension: Fused
    if CUDA_OPS_AVAILABLE:
        cuda_output = cuda_ops.layernorm_gelu_fused(input_tensor, gamma, beta)
        max_diff = torch.max(torch.abs(cuda_output - pytorch_output)).item()
        print(f"CUDA Extension (fused) - Max difference: {max_diff:.2e}")
    
    # Triton: Fused
    if TRITON_AVAILABLE:
        triton_output = layernorm_gelu_fused_triton(input_tensor, gamma, beta)
        max_diff = torch.max(torch.abs(triton_output - pytorch_output)).item()
        print(f"Triton (fused) - Max difference: {max_diff:.2e}")


def example_performance_comparison():
    """Example: Quick performance comparison"""
    print("\n" + "="*60)
    print("Example: Performance Comparison")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping example.")
        return
    
    device = 'cuda'
    batch_size, seq_len, hidden_size = 32, 512, 768
    
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, device=device)
    gamma = torch.ones(hidden_size, device=device)
    beta = torch.zeros(hidden_size, device=device)
    
    # Warmup
    for _ in range(5):
        _ = torch.nn.functional.layer_norm(input_tensor, (hidden_size,), gamma, beta)
    
    # Benchmark PyTorch
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(100):
        _ = torch.nn.functional.layer_norm(input_tensor, (hidden_size,), gamma, beta)
    end.record()
    torch.cuda.synchronize()
    pytorch_time = start.elapsed_time(end) / 100
    
    print(f"PyTorch Native: {pytorch_time:.3f} ms")
    
    # Benchmark CUDA extension
    if CUDA_OPS_AVAILABLE:
        for _ in range(5):
            _ = cuda_ops.layernorm(input_tensor, gamma, beta)
        
        torch.cuda.synchronize()
        start.record()
        for _ in range(100):
            _ = cuda_ops.layernorm(input_tensor, gamma, beta)
        end.record()
        torch.cuda.synchronize()
        cuda_time = start.elapsed_time(end) / 100
        
        speedup = pytorch_time / cuda_time
        print(f"CUDA Extension: {cuda_time:.3f} ms ({speedup:.2f}x speedup)")
    
    # Benchmark Triton
    if TRITON_AVAILABLE:
        for _ in range(5):
            _ = layernorm_triton(input_tensor, gamma, beta)
        
        torch.cuda.synchronize()
        start.record()
        for _ in range(100):
            _ = layernorm_triton(input_tensor, gamma, beta)
        end.record()
        torch.cuda.synchronize()
        triton_time = start.elapsed_time(end) / 100
        
        speedup = pytorch_time / triton_time
        print(f"Triton: {triton_time:.3f} ms ({speedup:.2f}x speedup)")


if __name__ == '__main__':
    print("="*60)
    print("GPU Optimization Project - Simple Examples")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("\nWARNING: CUDA is not available!")
        print("These examples require a CUDA-capable GPU.")
        sys.exit(1)
    
    print(f"\nDevice: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    
    example_layernorm()
    example_gelu()
    example_fused()
    example_performance_comparison()
    
    print("\n" + "="*60)
    print("Examples complete!")
    print("="*60)

