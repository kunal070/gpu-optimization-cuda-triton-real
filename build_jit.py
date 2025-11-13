"""
JIT (Just-In-Time) compilation of CUDA kernels
This compiles your CUDA code on-the-fly without needing setup.py
"""
import torch
import os
import sys

print("="*60)
print("JIT Compilation of CUDA Kernels")
print("="*60 + "\n")

print("Checking prerequisites...")

# Check CUDA
if not torch.cuda.is_available():
    print("✗ CUDA not available")
    sys.exit(1)

print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA version: {torch.version.cuda}")

# Check CUDA toolkit
cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
if not cuda_home:
    cuda_home = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"

if not os.path.exists(cuda_home):
    print(f"✗ CUDA toolkit not found at: {cuda_home}")
    sys.exit(1)

print(f"✓ CUDA toolkit: {cuda_home}")

# Check if source files exist
cpp_file = 'pytorch_extensions/cuda_ops/cuda_ops.cpp'
cu_file = 'pytorch_extensions/cuda_ops/cuda_ops_kernel.cu'

if not os.path.exists(cpp_file):
    print(f"✗ Missing: {cpp_file}")
    sys.exit(1)

if not os.path.exists(cu_file):
    print(f"✗ Missing: {cu_file}")
    sys.exit(1)

print(f"✓ Source files found")

print("\n" + "="*60)
print("Compiling CUDA kernels (this may take a few minutes)...")
print("="*60 + "\n")

try:
    from torch.utils.cpp_extension import load
    
    # This will compile on-the-fly
    cuda_ops = load(
        name='cuda_ops',
        sources=[cpp_file, cu_file],
        extra_include_paths=['cuda_kernels', 'pytorch_extensions/cuda_ops'],
        extra_cflags=['-O3'],
        extra_cuda_cflags=[
            '-O3',
            '--use_fast_math',
            '-arch=sm_89',  # RTX 4060
            '--expt-relaxed-constexpr',
            '--expt-extended-lambda',
        ],
        verbose=True,
        with_cuda=True,
    )
    
    print("\n" + "="*60)
    print("✓ Compilation successful!")
    print("="*60 + "\n")
    
    # Quick test
    print("Running quick tests...\n")
    
    # Test 1: Vector Add
    print("1. Testing vector_add...")
    x = torch.randn(1000, device='cuda')
    y = torch.randn(1000, device='cuda')
    result = cuda_ops.vector_add(x, y)
    expected = x + y
    
    if torch.allclose(result, expected, rtol=1e-4):
        print("   ✓ vector_add works!")
    else:
        print("   ✗ vector_add failed")
    
    # Test 2: LayerNorm
    print("2. Testing layernorm...")
    x = torch.randn(2, 4, 8, device='cuda')
    gamma = torch.ones(8, device='cuda')
    beta = torch.zeros(8, device='cuda')
    result = cuda_ops.layernorm(x, gamma, beta)
    print(f"   ✓ layernorm works! Output shape: {result.shape}")
    
    # Test 3: GELU
    print("3. Testing gelu...")
    x = torch.randn(2, 4, 8, device='cuda')
    result = cuda_ops.gelu(x, use_fast=True)
    print(f"   ✓ gelu works! Output shape: {result.shape}")
    
    # Test 4: Fused operation
    print("4. Testing layernorm_gelu_fused...")
    x = torch.randn(2, 4, 8, device='cuda')
    gamma = torch.ones(8, device='cuda')
    beta = torch.zeros(8, device='cuda')
    result = cuda_ops.layernorm_gelu_fused(x, gamma, beta)
    print(f"   ✓ fused operation works! Output shape: {result.shape}")
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60 + "\n")
    
    print("Your CUDA kernels are now compiled and ready to use!")
    print("\nTo use them in your code:")
    print("  from torch.utils.cpp_extension import load")
    print("  cuda_ops = load(name='cuda_ops', sources=[...])")
    print("\nOr run: python test_custom_cuda.py")
    
except Exception as e:
    print(f"\n✗ Compilation failed: {e}")
    print("\nThis might be because:")
    print("  1. Visual Studio C++ tools are not fully installed")
    print("  2. CUDA toolkit path is incorrect")
    print("  3. Some source files have errors")
    print("\nRECOMMENDATION: Use WSL2 + Ubuntu for easier CUDA development")
    print("  wsl --install")
    print("  # Then in Ubuntu, everything will work smoothly")
    sys.exit(1)