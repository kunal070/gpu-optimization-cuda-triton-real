# How to Run Benchmarks - Step by Step Guide

This guide walks you through running the comprehensive benchmarking suite that matches your project presentation requirements.

## Prerequisites

1. ✅ CUDA-capable GPU
2. ✅ PyTorch with CUDA support installed
3. ✅ All Python dependencies installed (`pip install -r requirements.txt`)

## Step 1: Quick Test

First, verify everything is set up correctly:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

Should output: `CUDA available: True`

## Step 2: Run Basic Benchmark

Run the comprehensive benchmark with default settings:

```bash
python benchmarks/comprehensive_benchmark.py
```

This will:
- Test operations: LayerNorm, GELU, Swish, Loss
- Test batch sizes: 16, 32, 64
- Test sequence lengths: 256, 512, 1024
- Test tensor dimensions: 32, 64, 128, 256
- Save results to `benchmark_results.json`

**Expected output:**
```
============================================================
Comprehensive GPU Benchmark Suite
============================================================
Device: NVIDIA GeForce RTX 4060 Laptop GPU
CUDA Version: 12.4

Testing:
  Batch Sizes: [16, 32, 64]
  Sequence Lengths: [256, 512, 1024]
  Tensor Dimensions: [32, 64, 128, 256]
  Operations: ['LayerNorm', 'GELU', 'Swish', 'Loss']
============================================================

[1/144] Testing: batch=16, seq=256, dim=32
...
```

## Step 3: Customize Parameters

Edit `benchmarks/comprehensive_benchmark.py` to customize:

```python
# At the bottom of the file, modify:
results = run_comprehensive_benchmark(
    batch_sizes=[16, 32, 64, 128],        # Add more batch sizes
    sequence_lengths=[256, 512, 1024, 2048],  # Add more sequence lengths
    tensor_dimensions=[8, 16, 32, 64, 128, 256, 512],  # Add more dimensions
    operations=['LayerNorm', 'GELU', 'Swish', 'Loss'],
    output_file='my_custom_results.json'
)
```

## Step 4: View Results

### Option A: View in Terminal

The script automatically prints comparison tables:

```
================================================================================
Results for LayerNorm
================================================================================
Implementation  Time (ms)    Memory (MB)    Inference Speed    GPU Efficiency
--------------------------------------------------------------------------------
PyTorch         0.442        150.50         2262.44            85.30
CUDA            0.285        145.20         3508.77            92.15
Triton          0.320        148.30         3125.00            88.50
================================================================================
```

### Option B: Load JSON Results

```python
import json

with open('benchmark_results.json', 'r') as f:
    results = json.load(f)

# Filter by operation
layernorm = [r for r in results if r['operation'] == 'LayerNorm']

# Find fastest
fastest = min(layernorm, key=lambda x: x['time_ms'])
print(f"Fastest: {fastest['implementation']} ({fastest['time_ms']:.3f} ms)")
```

### Option C: Create Visualizations

```bash
python benchmarks/visualize_results.py
```

This creates:
- Comparison tables (CSV files)
- Performance plots (PNG files)
- Summary report (TXT file)

## Step 5: Analyze by Parameter

### Analyze by Batch Size

```python
import json
import pandas as pd

with open('benchmark_results.json', 'r') as f:
    results = json.load(f)

df = pd.DataFrame(results)

# Group by batch size and implementation
for batch_size in [16, 32, 64]:
    batch_data = df[df['batch_size'] == batch_size]
    print(f"\nBatch Size {batch_size}:")
    print(batch_data.groupby('implementation')['time_ms'].mean())
```

### Analyze by Sequence Length

```python
for seq_len in [256, 512, 1024]:
    seq_data = df[df['sequence_length'] == seq_len]
    print(f"\nSequence Length {seq_len}:")
    print(seq_data.groupby('implementation')['time_ms'].mean())
```

### Analyze by Tensor Dimension

```python
for dim in [32, 64, 128, 256]:
    dim_data = df[df['tensor_dimension'] == dim]
    print(f"\nTensor Dimension {dim}:")
    print(dim_data.groupby('implementation')['time_ms'].mean())
```

## Step 6: Compare CUDA vs Triton

```python
import json
import pandas as pd

with open('benchmark_results.json', 'r') as f:
    results = json.load(f)

df = pd.DataFrame(results)

# Compare CUDA vs Triton for each operation
for op in ['LayerNorm', 'GELU', 'Swish', 'Loss']:
    op_data = df[df['operation'] == op]
    
    cuda_data = op_data[op_data['implementation'] == 'CUDA']
    triton_data = op_data[op_data['implementation'] == 'Triton']
    
    if not cuda_data.empty and not triton_data.empty:
        cuda_avg = cuda_data['time_ms'].mean()
        triton_avg = triton_data['time_ms'].mean()
        speedup = triton_avg / cuda_avg
        
        print(f"\n{op}:")
        print(f"  CUDA:  {cuda_avg:.3f} ms")
        print(f"  Triton: {triton_avg:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x")
```

## Step 7: Generate Report for Presentation

Create a formatted report matching your presentation slides:

```python
import json
import pandas as pd

with open('benchmark_results.json', 'r') as f:
    results = json.load(f)

df = pd.DataFrame(results)

# Create tables matching presentation format
operations = ['LayerNorm', 'GELU', 'Swish', 'Loss']
metrics = ['time_ms', 'memory_mb', 'inference_speed', 'gpu_efficiency']

for op in operations:
    print(f"\n{'='*80}")
    print(f"{op} - Comparison Table")
    print(f"{'='*80}")
    
    op_data = df[df['operation'] == op]
    
    # Create pivot table
    table = op_data.pivot_table(
        index='implementation',
        values=metrics,
        aggfunc='mean'
    )
    
    table.columns = ['Time (ms)', 'Memory (MB)', 'Inference Speed', 'GPU Efficiency']
    print(table.round(2))
```

## Troubleshooting

### Issue: "CUDA ops not available"

**Solution**: Build the CUDA extension:
```bash
python setup.py build_ext --inplace
```

Note: This requires proper CUDA toolkit and compiler setup.

### Issue: "Triton kernels not available"

**Solution**: Install Triton (requires Python 3.10 or 3.11):
```bash
pip install triton
```

Or use Python 3.10/3.11 if you're on Python 3.13.

### Issue: Out of Memory

**Solution**: Reduce parameters:
```python
results = run_comprehensive_benchmark(
    batch_sizes=[16, 32],  # Smaller batch sizes
    sequence_lengths=[256, 512],  # Smaller sequences
    tensor_dimensions=[32, 64],  # Smaller dimensions
    operations=['LayerNorm'],  # Test one operation at a time
)
```

### Issue: Benchmark takes too long

**Solution**: Reduce the number of combinations:
- Test fewer batch sizes
- Test fewer sequence lengths
- Test fewer tensor dimensions
- Test one operation at a time

## Quick Reference

| Task | Command |
|------|---------|
| Run benchmark | `python benchmarks/comprehensive_benchmark.py` |
| Visualize results | `python benchmarks/visualize_results.py` |
| Load results | `python -c "import json; print(json.load(open('benchmark_results.json')))"` |
| Test single operation | Modify script to test only one operation |

## Expected Results Format

Your results will match the presentation slide format:

```
Operation: LayerNorm
Implementation: CUDA
Batch Size: 32
Sequence Length: 512
Tensor Dimension: 768
Time: 0.285 ms
Memory: 145.20 MB
Inference Speed: 3508.77 ops/s
GPU Efficiency: 92.15%
```

## Next Steps

1. ✅ Run the benchmark
2. ✅ Analyze results
3. ✅ Create visualizations
4. ✅ Compare CUDA vs Triton
5. ✅ Document findings for presentation

For detailed information, see `BENCHMARKING_GUIDE.md`.

