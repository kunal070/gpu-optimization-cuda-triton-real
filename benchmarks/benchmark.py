"""
Comprehensive Benchmark Script for GPU Optimization Project

This script benchmarks:
1. Native PyTorch operations
2. Custom CUDA kernels (via PyTorch extension)
3. Triton implementations

Compares performance across different implementations and input sizes.
"""

import torch
import time
import numpy as np
from typing import Dict, List, Tuple
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cuda_ops  # PyTorch CUDA extension
    CUDA_OPS_AVAILABLE = True
except ImportError:
    print("Warning: CUDA ops extension not available. Install with: python setup.py build_ext --inplace")
    CUDA_OPS_AVAILABLE = False

try:
    from triton_kernels import (
        layernorm_triton,
        gelu_triton,
        layernorm_gelu_fused_triton,
    )
    TRITON_AVAILABLE = True
except ImportError:
    print("Warning: Triton kernels not available")
    TRITON_AVAILABLE = False


def benchmark_function(
    func,
    *args,
    warmup=10,
    repeat=100,
    device='cuda'
) -> Tuple[float, float]:
    """
    Benchmark a function
    
    Returns:
        (mean_time_ms, std_time_ms)
    """
    # Warmup
    for _ in range(warmup):
        _ = func(*args)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(repeat):
        if device == 'cuda':
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            start_time = time.time()
        
        result = func(*args)
        
        if device == 'cuda':
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))  # milliseconds
        else:
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # convert to ms
    
    return np.mean(times), np.std(times)


def benchmark_layernorm(
    batch_size: int = 32,
    seq_len: int = 512,
    hidden_size: int = 768,
    device: str = 'cuda'
) -> Dict[str, Tuple[float, float]]:
    """Benchmark LayerNorm implementations"""
    print(f"\n{'='*60}")
    print(f"LayerNorm Benchmark: [{batch_size}, {seq_len}, {hidden_size}]")
    print(f"{'='*60}")
    
    results = {}
    
    # Create test data
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, device=device)
    gamma = torch.ones(hidden_size, device=device)
    beta = torch.zeros(hidden_size, device=device)
    eps = 1e-5
    
    # Native PyTorch
    def pytorch_layernorm():
        return torch.nn.functional.layer_norm(
            input_tensor, (hidden_size,), gamma, beta, eps
        )
    
    mean_time, std_time = benchmark_function(pytorch_layernorm, device=device)
    results['PyTorch Native'] = (mean_time, std_time)
    print(f"PyTorch Native:     {mean_time:.3f} ± {std_time:.3f} ms")
    
    # CUDA Extension
    if CUDA_OPS_AVAILABLE:
        def cuda_layernorm():
            return cuda_ops.layernorm(input_tensor, gamma, beta, eps)
        
        mean_time, std_time = benchmark_function(cuda_layernorm, device=device)
        results['CUDA Extension'] = (mean_time, std_time)
        speedup = results['PyTorch Native'][0] / mean_time
        print(f"CUDA Extension:     {mean_time:.3f} ± {std_time:.3f} ms ({speedup:.2f}x speedup)")
    
    # Triton
    if TRITON_AVAILABLE:
        def triton_layernorm():
            return layernorm_triton(input_tensor, gamma, beta, eps)
        
        mean_time, std_time = benchmark_function(triton_layernorm, device=device)
        results['Triton'] = (mean_time, std_time)
        speedup = results['PyTorch Native'][0] / mean_time
        print(f"Triton:             {mean_time:.3f} ± {std_time:.3f} ms ({speedup:.2f}x speedup)")
    
    return results


def benchmark_gelu(
    size: Tuple[int, ...] = (32, 512, 768),
    device: str = 'cuda'
) -> Dict[str, Tuple[float, float]]:
    """Benchmark GELU implementations"""
    print(f"\n{'='*60}")
    print(f"GELU Benchmark: {size}")
    print(f"{'='*60}")
    
    results = {}
    
    # Create test data
    input_tensor = torch.randn(*size, device=device)
    
    # Native PyTorch
    def pytorch_gelu():
        return torch.nn.functional.gelu(input_tensor)
    
    mean_time, std_time = benchmark_function(pytorch_gelu, device=device)
    results['PyTorch Native'] = (mean_time, std_time)
    print(f"PyTorch Native:     {mean_time:.3f} ± {std_time:.3f} ms")
    
    # CUDA Extension
    if CUDA_OPS_AVAILABLE:
        def cuda_gelu():
            return cuda_ops.gelu(input_tensor, use_fast=True)
        
        mean_time, std_time = benchmark_function(cuda_gelu, device=device)
        results['CUDA Extension'] = (mean_time, std_time)
        speedup = results['PyTorch Native'][0] / mean_time
        print(f"CUDA Extension:     {mean_time:.3f} ± {std_time:.3f} ms ({speedup:.2f}x speedup)")
    
    # Triton
    if TRITON_AVAILABLE:
        def triton_gelu():
            return gelu_triton(input_tensor, use_fast=True)
        
        mean_time, std_time = benchmark_function(triton_gelu, device=device)
        results['Triton'] = (mean_time, std_time)
        speedup = results['PyTorch Native'][0] / mean_time
        print(f"Triton:             {mean_time:.3f} ± {std_time:.3f} ms ({speedup:.2f}x speedup)")
    
    return results


def benchmark_fused_layernorm_gelu(
    batch_size: int = 32,
    seq_len: int = 512,
    hidden_size: int = 768,
    device: str = 'cuda'
) -> Dict[str, Tuple[float, float]]:
    """Benchmark fused LayerNorm + GELU"""
    print(f"\n{'='*60}")
    print(f"Fused LayerNorm+GELU Benchmark: [{batch_size}, {seq_len}, {hidden_size}]")
    print(f"{'='*60}")
    
    results = {}
    
    # Create test data
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, device=device)
    gamma = torch.ones(hidden_size, device=device)
    beta = torch.zeros(hidden_size, device=device)
    eps = 1e-5
    
    # PyTorch: Separate operations
    def pytorch_separate():
        normalized = torch.nn.functional.layer_norm(
            input_tensor, (hidden_size,), gamma, beta, eps
        )
        return torch.nn.functional.gelu(normalized)
    
    mean_time, std_time = benchmark_function(pytorch_separate, device=device)
    results['PyTorch Separate'] = (mean_time, std_time)
    print(f"PyTorch Separate:   {mean_time:.3f} ± {std_time:.3f} ms")
    
    # CUDA Extension: Fused
    if CUDA_OPS_AVAILABLE:
        def cuda_fused():
            return cuda_ops.layernorm_gelu_fused(input_tensor, gamma, beta, eps)
        
        mean_time, std_time = benchmark_function(cuda_fused, device=device)
        results['CUDA Fused'] = (mean_time, std_time)
        speedup = results['PyTorch Separate'][0] / mean_time    
        print(f"CUDA Fused:         {mean_time:.3f} ± {std_time:.3f} ms ({speedup:.2f}x speedup)")
    
    # Triton: Fused
    if TRITON_AVAILABLE:
        def triton_fused():
            return layernorm_gelu_fused_triton(input_tensor, gamma, beta, eps)
        
        mean_time, std_time = benchmark_function(triton_fused, device=device)
        results['Triton Fused'] = (mean_time, std_time)
        speedup = results['PyTorch Separate'][0] / mean_time
        print(f"Triton Fused:       {mean_time:.3f} ± {std_time:.3f} ms ({speedup:.2f}x speedup)")
    
    return results


def benchmark_matrix_multiply(
    M: int = 1024,
    N: int = 1024,
    K: int = 1024,
    device: str = 'cuda'
) -> Dict[str, Tuple[float, float]]:
    """Benchmark matrix multiplication"""
    print(f"\n{'='*60}")
    print(f"Matrix Multiply Benchmark: [{M}x{K}] @ [{K}x{N}] = [{M}x{N}]")
    print(f"{'='*60}")
    
    results = {}
    
    # Create test data
    A = torch.randn(M, K, device=device)
    B = torch.randn(K, N, device=device)
    
    # Native PyTorch
    def pytorch_matmul():
        return torch.matmul(A, B)
    
    mean_time, std_time = benchmark_function(pytorch_matmul, device=device)
    results['PyTorch Native'] = (mean_time, std_time)
    print(f"PyTorch Native:     {mean_time:.3f} ± {std_time:.3f} ms")
    
    # CUDA Extension
    if CUDA_OPS_AVAILABLE:
        def cuda_matmul():
            return cuda_ops.matrix_multiply(A, B, use_tiled=True)
        
        mean_time, std_time = benchmark_function(cuda_matmul, device=device)
        results['CUDA Extension'] = (mean_time, std_time)
        speedup = results['PyTorch Native'][0] / mean_time
        print(f"CUDA Extension:     {mean_time:.3f} ± {std_time:.3f} ms ({speedup:.2f}x speedup)")
    
    return results


def run_all_benchmarks():
    """Run all benchmarks with different sizes"""
    print("\n" + "="*60)
    print("GPU Optimization Benchmark Suite")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return
    
    device = 'cuda'
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    
    # Test different sizes
    test_configs = [
        (32, 512, 768),   # Small
        (64, 1024, 1024), # Medium
        (128, 2048, 2048), # Large
    ]
    
    all_results = {}
    
    for batch_size, seq_len, hidden_size in test_configs:
        print(f"\n{'#'*60}")
        print(f"Testing with batch={batch_size}, seq_len={seq_len}, hidden={hidden_size}")
        print(f"{'#'*60}")
        
        # LayerNorm
        all_results[f'layernorm_{batch_size}_{seq_len}_{hidden_size}'] = \
            benchmark_layernorm(batch_size, seq_len, hidden_size, device)
        
        # GELU
        all_results[f'gelu_{batch_size}_{seq_len}_{hidden_size}'] = \
            benchmark_gelu((batch_size, seq_len, hidden_size), device)
        
        # Fused
        all_results[f'fused_{batch_size}_{seq_len}_{hidden_size}'] = \
            benchmark_fused_layernorm_gelu(batch_size, seq_len, hidden_size, device)
    
    # Matrix Multiply
    all_results['matmul_1024'] = benchmark_matrix_multiply(1024, 1024, 1024, device)
    all_results['matmul_2048'] = benchmark_matrix_multiply(2048, 2048, 2048, device)
    
    return all_results


if __name__ == '__main__':
    results = run_all_benchmarks()
    print("\n" + "="*60)
    print("Benchmark Complete!")
    print("="*60)

