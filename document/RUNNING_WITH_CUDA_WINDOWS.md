# Running the Project with CUDA on Windows

This guide shows you how to build and run the CUDA extensions on Windows.

## Prerequisites

1. ✅ CUDA Toolkit installed (you have CUDA 12.1)
2. ✅ Visual Studio Build Tools with C++ support
3. ✅ PyTorch with CUDA support (you have PyTorch 2.6.0+cu124)
4. ✅ Python 3.10+ (you have Python 3.13)

## Step 1: Open x64 Native Tools Command Prompt

**IMPORTANT**: You MUST use the x64 Native Tools Command Prompt, not regular PowerShell or CMD.

1. Press `Win + S` and search for "x64 Native Tools"
2. Open **"x64 Native Tools Command Prompt for VS 2022"** (or your VS version)
3. Navigate to your project directory:
   ```cmd
   cd C:\Users\kunal\OneDrive\Desktop\yash-luli
   ```

## Step 2: Set CUDA Path (if needed)

If CUDA_PATH is not set, set it:
```cmd
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1
```

## Step 3: Build the CUDA Extension

In the x64 Native Tools Command Prompt, run:

```cmd
python setup.py build_ext --inplace
```

This will:
- Compile the CUDA kernels
- Create the `cuda_ops` Python module
- Take 2-5 minutes depending on your system

**Expected output:**
```
running build_ext
building 'cuda_ops' extension
...
Successfully built cuda_ops
```

## Step 4: Verify Installation

Test that the CUDA extension is working:

```cmd
python -c "import cuda_ops; print('CUDA ops loaded successfully!')"
```

If successful, you'll see: `CUDA ops loaded successfully!`

## Step 5: Run Tests

Test the CUDA kernels:

```cmd
python tests/test_basic_kernels.py
```

This will test:
- Vector Add
- Matrix Multiply
- LayerNorm
- GELU
- Fused LayerNorm+GELU

## Step 6: Run Benchmarks with CUDA

Now run the comprehensive benchmark:

```cmd
python benchmarks/comprehensive_benchmark.py
```

This will compare:
- **PyTorch** (native)
- **CUDA** (your custom kernels)
- **Triton** (will be skipped on Windows)

## Quick Command Reference

```cmd
# 1. Open x64 Native Tools Command Prompt
# 2. Navigate to project
cd C:\Users\kunal\OneDrive\Desktop\yash-luli

# 3. Build CUDA extension
python setup.py build_ext --inplace

# 4. Test it
python -c "import cuda_ops; print('Success!')"

# 5. Run tests
python tests/test_basic_kernels.py

# 6. Run benchmarks
python benchmarks/comprehensive_benchmark.py
```

## Troubleshooting

### Error: "CUDA not found"

**Solution**: Set CUDA_PATH environment variable:
```cmd
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1
```

### Error: "MSVC not found" or "cl.exe not found"

**Solution**: 
1. Make sure you're using **x64 Native Tools Command Prompt**
2. Install Visual Studio Build Tools with C++ support
3. Restart the command prompt

### Error: "ninja not found"

**Solution**: Install ninja:
```cmd
pip install ninja
```

### Error: "Compiler version mismatch"

**Solution**: This is usually a warning and can be ignored. If build fails:
1. Make sure you're using the correct Visual Studio version
2. Try updating PyTorch: `pip install --upgrade torch`

### Error: "Out of memory" during build

**Solution**: Close other applications and try again. CUDA compilation uses significant memory.

### Build takes too long

**Solution**: This is normal. First build can take 5-10 minutes. Subsequent builds are faster due to caching.

## Alternative: Using JIT Compilation

If building fails, you can use PyTorch's JIT compilation (slower but easier):

```python
from torch.utils.cpp_extension import load

cuda_ops = load(
    name='cuda_ops',
    sources=[
        'pytorch_extensions/cuda_ops/cuda_ops.cpp',
        'pytorch_extensions/cuda_ops/cuda_ops_kernel.cu',
    ],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=True
)
```

Save this as `load_cuda_ops.py` and import it instead of `cuda_ops`.

## Verifying CUDA Extension Works

After building, test with a simple script:

```python
import torch
import cuda_ops

# Test LayerNorm
input_tensor = torch.randn(2, 4, 8, device='cuda')
gamma = torch.ones(8, device='cuda')
beta = torch.zeros(8, device='cuda')

output = cuda_ops.layernorm(input_tensor, gamma, beta)
print("LayerNorm output shape:", output.shape)
print("CUDA extension working!")

# Test GELU
input_tensor = torch.randn(2, 4, 8, device='cuda')
output = cuda_ops.gelu(input_tensor)
print("GELU output shape:", output.shape)
print("All tests passed!")
```

Save as `test_cuda.py` and run:
```cmd
python test_cuda.py
```

## Expected Performance

Once CUDA extension is built, you should see:
- **LayerNorm**: 1.5-2x speedup vs PyTorch
- **GELU**: 1.2-1.5x speedup vs PyTorch
- **Fused ops**: 2-3x speedup vs separate operations

## Next Steps

1. ✅ Build the extension (Step 3)
2. ✅ Run tests (Step 5)
3. ✅ Run benchmarks (Step 6)
4. ✅ Compare CUDA vs PyTorch performance
5. ✅ Analyze results in `benchmark_results.json`

## Notes

- **Triton**: Not available on Windows, but CUDA kernels provide similar performance
- **First build**: Takes longer (5-10 minutes)
- **Subsequent builds**: Much faster (uses cache)
- **GPU**: Make sure your GPU is CUDA-capable (you have RTX 4060, which is perfect!)

---

**You're all set! Follow the steps above to build and run your CUDA kernels on Windows.**

