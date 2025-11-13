"""
Basic tests for CUDA kernels
Tests correctness by comparing with PyTorch native implementations
"""

import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cuda_ops
    CUDA_OPS_AVAILABLE = True
except ImportError:
    CUDA_OPS_AVAILABLE = False
    print("Warning: CUDA ops not available. Run: python setup.py build_ext --inplace")

try:
    from triton_kernels import layernorm_triton, gelu_triton, layernorm_gelu_fused_triton
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("Warning: Triton kernels not available")


def test_layernorm():
    """Test LayerNorm correctness"""
    print("Testing LayerNorm...")
    
    if not torch.cuda.is_available():
        print("  SKIP: CUDA not available")
        return
    
    device = 'cuda'
    batch_size, seq_len, hidden_size = 4, 8, 16
    
    # Create test data
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, device=device)
    gamma = torch.ones(hidden_size, device=device)
    beta = torch.zeros(hidden_size, device=device)
    eps = 1e-5
    
    # PyTorch reference
    ref_output = torch.nn.functional.layer_norm(
        input_tensor, (hidden_size,), gamma, beta, eps
    )
    
    # Test CUDA extension
    if CUDA_OPS_AVAILABLE:
        cuda_output = cuda_ops.layernorm(input_tensor, gamma, beta, eps)
        max_diff = torch.max(torch.abs(cuda_output - ref_output)).item()
        print(f"  CUDA Extension: max_diff = {max_diff:.2e}")
        assert max_diff < 1e-4, f"CUDA LayerNorm failed: max_diff = {max_diff}"
    
    # Test Triton
    if TRITON_AVAILABLE:
        triton_output = layernorm_triton(input_tensor, gamma, beta, eps)
        max_diff = torch.max(torch.abs(triton_output - ref_output)).item()
        print(f"  Triton: max_diff = {max_diff:.2e}")
        assert max_diff < 1e-4, f"Triton LayerNorm failed: max_diff = {max_diff}"
    
    print("  [PASS] LayerNorm tests passed")


def test_gelu():
    """Test GELU correctness"""
    print("Testing GELU...")
    
    if not torch.cuda.is_available():
        print("  SKIP: CUDA not available")
        return
    
    device = 'cuda'
    size = (4, 8, 16)
    
    # Create test data
    input_tensor = torch.randn(*size, device=device)
    
    # PyTorch reference
    ref_output = torch.nn.functional.gelu(input_tensor)
    
    # Test CUDA extension
    if CUDA_OPS_AVAILABLE:
        cuda_output = cuda_ops.gelu(input_tensor, use_fast=True)
        max_diff = torch.max(torch.abs(cuda_output - ref_output)).item()
        print(f"  CUDA Extension: max_diff = {max_diff:.2e}")
        assert max_diff < 1e-3, f"CUDA GELU failed: max_diff = {max_diff}"
    
    # Test Triton
    if TRITON_AVAILABLE:
        triton_output = gelu_triton(input_tensor, use_fast=True)
        max_diff = torch.max(torch.abs(triton_output - ref_output)).item()
        print(f"  Triton: max_diff = {max_diff:.2e}")
        assert max_diff < 1e-3, f"Triton GELU failed: max_diff = {max_diff}"
    
    print("  [PASS] GELU tests passed")


def test_fused_layernorm_gelu():
    """Test fused LayerNorm + GELU"""
    print("Testing Fused LayerNorm+GELU...")
    
    if not torch.cuda.is_available():
        print("  SKIP: CUDA not available")
        return
    
    device = 'cuda'
    batch_size, seq_len, hidden_size = 4, 8, 16
    
    # Create test data
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, device=device)
    gamma = torch.ones(hidden_size, device=device)
    beta = torch.zeros(hidden_size, device=device)
    eps = 1e-5
    
    # PyTorch reference (separate ops)
    normalized = torch.nn.functional.layer_norm(
        input_tensor, (hidden_size,), gamma, beta, eps
    )
    ref_output = torch.nn.functional.gelu(normalized)
    
    # Test CUDA extension
    if CUDA_OPS_AVAILABLE:
        cuda_output = cuda_ops.layernorm_gelu_fused(input_tensor, gamma, beta, eps)
        max_diff = torch.max(torch.abs(cuda_output - ref_output)).item()
        print(f"  CUDA Extension: max_diff = {max_diff:.2e}")
        assert max_diff < 1e-3, f"CUDA Fused failed: max_diff = {max_diff}"
    
    # Test Triton
    if TRITON_AVAILABLE:
        triton_output = layernorm_gelu_fused_triton(input_tensor, gamma, beta, eps)
        max_diff = torch.max(torch.abs(triton_output - ref_output)).item()
        print(f"  Triton: max_diff = {max_diff:.2e}")
        assert max_diff < 1e-3, f"Triton Fused failed: max_diff = {max_diff}"
    
    print("  [PASS] Fused LayerNorm+GELU tests passed")


def test_matrix_multiply():
    """Test matrix multiplication"""
    print("Testing Matrix Multiply...")
    
    if not torch.cuda.is_available():
        print("  SKIP: CUDA not available")
        return
    
    device = 'cuda'
    M, K, N = 64, 128, 32
    
    # Create test data
    A = torch.randn(M, K, device=device)
    B = torch.randn(K, N, device=device)
    
    # PyTorch reference
    ref_output = torch.matmul(A, B)
    
    # Test CUDA extension
    if CUDA_OPS_AVAILABLE:
        cuda_output = cuda_ops.matrix_multiply(A, B, use_tiled=True)
        max_diff = torch.max(torch.abs(cuda_output - ref_output)).item()
        print(f"  CUDA Extension: max_diff = {max_diff:.2e}")
        assert max_diff < 1e-3, f"CUDA MatMul failed: max_diff = {max_diff}"
    
    print("  [PASS] Matrix Multiply tests passed")


def test_vector_add():
    """Test vector addition"""
    print("Testing Vector Add...")
    
    if not torch.cuda.is_available():
        print("  SKIP: CUDA not available")
        return
    
    device = 'cuda'
    n = 1024
    
    # Create test data
    a = torch.randn(n, device=device)
    b = torch.randn(n, device=device)
    
    # PyTorch reference
    ref_output = a + b
    
    # Test CUDA extension
    if CUDA_OPS_AVAILABLE:
        cuda_output = cuda_ops.vector_add(a, b)
        max_diff = torch.max(torch.abs(cuda_output - ref_output)).item()
        print(f"  CUDA Extension: max_diff = {max_diff:.2e}")
        assert max_diff < 1e-5, f"CUDA VectorAdd failed: max_diff = {max_diff}"
    
    print("  [PASS] Vector Add tests passed")


if __name__ == '__main__':
    print("="*60)
    print("Running Basic Kernel Tests")
    print("="*60)
    
    test_vector_add()
    test_matrix_multiply()
    test_layernorm()
    test_gelu()
    test_fused_layernorm_gelu()
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)

