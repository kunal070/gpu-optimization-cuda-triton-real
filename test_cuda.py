"""
Quick test script to verify CUDA extension is working
Run this after building the CUDA extension
"""

import torch

print("="*60)
print("Testing CUDA Extension")
print("="*60)

# Check CUDA availability
if not torch.cuda.is_available():
    print("ERROR: CUDA not available!")
    exit(1)

print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"CUDA Version: {torch.version.cuda}")
print()

# Try to import CUDA ops
try:
    import cuda_ops
    print("[OK] CUDA ops module imported successfully")
except ImportError as e:
    print(f"[ERROR] Failed to import cuda_ops: {e}")
    print("\nMake sure you've built the extension:")
    print("  python setup.py build_ext --inplace")
    exit(1)

# Test LayerNorm
print("\nTesting LayerNorm...")
try:
    input_tensor = torch.randn(2, 4, 8, device='cuda')
    gamma = torch.ones(8, device='cuda')
    beta = torch.zeros(8, device='cuda')
    
    output = cuda_ops.layernorm(input_tensor, gamma, beta)
    print(f"  [OK] LayerNorm output shape: {output.shape}")
except Exception as e:
    print(f"  [ERROR] LayerNorm failed: {e}")
    exit(1)

# Test GELU
print("Testing GELU...")
try:
    input_tensor = torch.randn(2, 4, 8, device='cuda')
    output = cuda_ops.gelu(input_tensor, use_fast=True)
    print(f"  [OK] GELU output shape: {output.shape}")
except Exception as e:
    print(f"  [ERROR] GELU failed: {e}")
    exit(1)

# Test Vector Add
print("Testing Vector Add...")
try:
    a = torch.randn(1024, device='cuda')
    b = torch.randn(1024, device='cuda')
    c = cuda_ops.vector_add(a, b)
    print(f"  [OK] Vector Add output shape: {c.shape}")
except Exception as e:
    print(f"  [ERROR] Vector Add failed: {e}")
    exit(1)

# Test Matrix Multiply
print("Testing Matrix Multiply...")
try:
    A = torch.randn(64, 128, device='cuda')
    B = torch.randn(128, 32, device='cuda')
    C = cuda_ops.matrix_multiply(A, B, use_tiled=True)
    print(f"  [OK] Matrix Multiply output shape: {C.shape}")
except Exception as e:
    print(f"  [ERROR] Matrix Multiply failed: {e}")
    exit(1)

# Test Fused LayerNorm + GELU
print("Testing Fused LayerNorm + GELU...")
try:
    input_tensor = torch.randn(2, 4, 8, device='cuda')
    gamma = torch.ones(8, device='cuda')
    beta = torch.zeros(8, device='cuda')
    output = cuda_ops.layernorm_gelu_fused(input_tensor, gamma, beta)
    print(f"  [OK] Fused LayerNorm+GELU output shape: {output.shape}")
except Exception as e:
    print(f"  [ERROR] Fused LayerNorm+GELU failed: {e}")
    exit(1)

print("\n" + "="*60)
print("All CUDA extension tests passed!")
print("="*60)
print("\nYou can now run:")
print("  python tests/test_basic_kernels.py")
print("  python benchmarks/comprehensive_benchmark.py")

