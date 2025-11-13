# Quick Start Guide

## Installation

### Prerequisites
1. **NVIDIA GPU** with CUDA support (Compute Capability 7.5+)
2. **CUDA Toolkit 12.x** installed
3. **Python 3.10+**
4. **Visual Studio Build Tools** (Windows) or **GCC** (Linux)

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Build PyTorch CUDA Extensions

```bash
python setup.py build_ext --inplace
```

**Note**: On Windows, you may need to set environment variables:
```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0"
```

### Step 3: Verify Installation

```bash
python tests/test_basic_kernels.py
```

## Running Benchmarks

### Basic Benchmark

```bash
python benchmarks/benchmark.py
```

### Custom Benchmark

```python
import torch
from benchmarks.benchmark import benchmark_layernorm

# Test LayerNorm with custom size
results = benchmark_layernorm(
    batch_size=32,
    seq_len=512,
    hidden_size=768,
    device='cuda'
)
```

## Using the Kernels

### CUDA Extension (PyTorch)

```python
import torch
import cuda_ops  # After building

# LayerNorm
input_tensor = torch.randn(32, 512, 768, device='cuda')
gamma = torch.ones(768, device='cuda')
beta = torch.zeros(768, device='cuda')
output = cuda_ops.layernorm(input_tensor, gamma, beta)

# GELU
input_tensor = torch.randn(32, 512, 768, device='cuda')
output = cuda_ops.gelu(input_tensor, use_fast=True)

# Fused LayerNorm + GELU
output = cuda_ops.layernorm_gelu_fused(input_tensor, gamma, beta)
```

### Triton Kernels

```python
import torch
from triton_kernels import layernorm_triton, gelu_triton, layernorm_gelu_fused_triton

# LayerNorm
input_tensor = torch.randn(32, 512, 768, device='cuda')
gamma = torch.ones(768, device='cuda')
beta = torch.zeros(768, device='cuda')
output = layernorm_triton(input_tensor, gamma, beta)

# GELU
output = gelu_triton(input_tensor, use_fast=True)

# Fused
output = layernorm_gelu_fused_triton(input_tensor, gamma, beta)
```

## Troubleshooting

### Build Errors

**Error**: `CUDA not found`
- **Solution**: Set `CUDA_PATH` environment variable to your CUDA installation

**Error**: `MSVC not found` (Windows)
- **Solution**: Install Visual Studio Build Tools with C++ support

**Error**: `ninja not found`
- **Solution**: `pip install ninja`

### Runtime Errors

**Error**: `CUDA out of memory`
- **Solution**: Reduce batch size or sequence length in tests

**Error**: `Kernel launch failed`
- **Solution**: Check CUDA version compatibility and GPU compute capability

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'cuda_ops'`
- **Solution**: Run `python setup.py build_ext --inplace` to build the extension

**Error**: `ModuleNotFoundError: No module named 'triton'`
- **Solution**: `pip install triton`

## Next Steps

1. Read `docs/GPU_OPTIMIZATION_GUIDE.md` for detailed explanations
2. Explore the kernel implementations in `cuda_kernels/` and `triton_kernels/`
3. Run benchmarks to see performance improvements
4. Modify kernels to experiment with different optimizations

## Project Structure

```
yash-luli/
├── cuda_kernels/          # CUDA C++ implementations
│   ├── basic/             # Vector add, matrix multiply
│   ├── layernorm/         # LayerNorm kernel
│   ├── gelu/              # GELU activation
│   └── fused/             # Fused operations
├── triton_kernels/        # Triton Python implementations
├── pytorch_extensions/    # PyTorch C++/CUDA extensions
├── benchmarks/            # Benchmark scripts
├── tests/                 # Unit tests
└── docs/                  # Documentation
```

## Learning Path

1. **Start with basics**: Read `cuda_kernels/basic/vector_add.cu` to understand thread indexing
2. **Progress to optimizations**: Study `cuda_kernels/basic/matrix_multiply.cu` for shared memory usage
3. **Learn reductions**: Examine `cuda_kernels/layernorm/layernorm.cu` for parallel reductions
4. **Explore fusion**: See `cuda_kernels/fused/layernorm_gelu.cu` for kernel fusion
5. **Compare with Triton**: Compare CUDA and Triton implementations side-by-side

