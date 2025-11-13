# Comprehensive Benchmarking Guide

This guide explains how to run the comprehensive benchmarking suite that matches your project goals.

## Overview

The benchmarking framework tests operations across:
- **Different batch sizes**: 16, 32, 64, 128, ...
- **Different sequence lengths**: 256, 512, 1024, 2048, ...
- **Different tensor dimensions**: 8, 16, 32, 64, 128, 256, 512, ...

And measures:
- **Time**: Execution time in milliseconds
- **Memory Usage**: Peak GPU memory in MB
- **Inference Speed**: Operations per second
- **GPU Efficiency**: Estimated GPU utilization percentage

## Operations Tested

1. **LayerNorm**: Layer Normalization
2. **GELU**: GELU activation function
3. **Swish**: Swish activation function (x * sigmoid(x))
4. **Loss**: Loss functions (MSE, Cross Entropy)

## Implementations Compared

- **PyTorch**: Native PyTorch implementations
- **CUDA**: Custom CUDA kernels (if extension is built)
- **Triton**: Triton implementations (if available)

## Running the Benchmark

### Basic Usage

```bash
python benchmarks/comprehensive_benchmark.py
```

This will run with default parameters and save results to `benchmark_results.json`.

### Custom Parameters

You can modify the parameters in the script:

```python
results = run_comprehensive_benchmark(
    batch_sizes=[16, 32, 64, 128],           # Custom batch sizes
    sequence_lengths=[256, 512, 1024],      # Custom sequence lengths
    tensor_dimensions=[32, 64, 128, 256],    # Custom tensor dimensions
    operations=['LayerNorm', 'GELU', 'Swish', 'Loss'],  # Operations to test
    output_file='my_results.json'            # Output file name
)
```

### Example: Testing Specific Configuration

```python
from benchmarks.comprehensive_benchmark import run_comprehensive_benchmark

# Test only specific values
results = run_comprehensive_benchmark(
    batch_sizes=[32],
    sequence_lengths=[512],
    tensor_dimensions=[768],
    operations=['LayerNorm', 'GELU'],
    output_file='specific_test.json'
)
```

## Understanding the Results

### Output Format

Results are saved as JSON with the following structure:

```json
{
  "operation": "LayerNorm",
  "implementation": "CUDA",
  "batch_size": 32,
  "sequence_length": 512,
  "tensor_dimension": 768,
  "time_ms": 0.442,
  "memory_mb": 150.5,
  "inference_speed": 2262.44,
  "gpu_efficiency": 85.3,
  "throughput_gbps": 12.5
}
```

### Metrics Explained

1. **time_ms**: Average execution time in milliseconds
2. **memory_mb**: Peak GPU memory usage in megabytes
3. **inference_speed**: Operations per second (1000 / time_ms)
4. **gpu_efficiency**: Estimated GPU utilization (0-100%)
5. **throughput_gbps**: Memory throughput in GB/s

## Analyzing Results

### Using Python

```python
import json

# Load results
with open('benchmark_results.json', 'r') as f:
    results = json.load(f)

# Filter by operation
layernorm_results = [r for r in results if r['operation'] == 'LayerNorm']

# Compare implementations
for impl in ['PyTorch', 'CUDA', 'Triton']:
    impl_results = [r for r in layernorm_results if r['implementation'] == impl]
    avg_time = sum(r['time_ms'] for r in impl_results) / len(impl_results)
    print(f"{impl} average time: {avg_time:.3f} ms")
```

### Creating Comparison Tables

The script automatically prints comparison tables for each operation showing:
- Implementation
- Time (ms)
- Memory (MB)
- Inference Speed
- GPU Efficiency

## Benchmarking by Parameter

### By Batch Size

To analyze how batch size affects performance:

```python
# Filter results for specific batch sizes
batch_16 = [r for r in results if r['batch_size'] == 16]
batch_32 = [r for r in results if r['batch_size'] == 32]
batch_64 = [r for r in results if r['batch_size'] == 64]

# Compare performance
for batch_results in [batch_16, batch_32, batch_64]:
    avg_time = sum(r['time_ms'] for r in batch_results) / len(batch_results)
    print(f"Batch {batch_results[0]['batch_size']}: {avg_time:.3f} ms")
```

### By Sequence Length

```python
seq_256 = [r for r in results if r['sequence_length'] == 256]
seq_512 = [r for r in results if r['sequence_length'] == 512]
seq_1024 = [r for r in results if r['sequence_length'] == 1024]
```

### By Tensor Dimension

```python
dim_128 = [r for r in results if r['tensor_dimension'] == 128]
dim_256 = [r for r in results if r['tensor_dimension'] == 256]
dim_512 = [r for r in results if r['tensor_dimension'] == 512]
```

## Performance Profiling

### Using PyTorch Profiler

For more detailed profiling, you can use PyTorch's built-in profiler:

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
    record_shapes=True,
    profile_memory=True
) as prof:
    with record_function("layernorm"):
        result = layernorm_triton(input_tensor, gamma, beta)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### Using Nsight Systems

For system-level profiling:

```bash
nsys profile --trace=cuda,nvtx python benchmarks/comprehensive_benchmark.py
```

This generates a `.nsys-rep` file that can be analyzed in Nsight Systems GUI.

## Tips for Accurate Benchmarking

1. **Warmup**: The benchmark includes warmup runs to ensure accurate timing
2. **Multiple Runs**: Results are averaged over 100 runs by default
3. **Memory Clearing**: GPU memory is cleared between tests
4. **Synchronization**: CUDA operations are properly synchronized

## Troubleshooting

### Out of Memory Errors

If you get CUDA out of memory errors:
- Reduce batch sizes
- Reduce sequence lengths
- Reduce tensor dimensions
- Test operations one at a time

### Slow Execution

- The comprehensive benchmark tests many combinations
- For quick tests, reduce the parameter ranges
- Use smaller tensor dimensions for initial testing

### Missing Implementations

- CUDA extension: Build with `python setup.py build_ext --inplace`
- Triton: Install with `pip install triton` (requires Python 3.10/3.11)

## Next Steps

1. Run the benchmark with your desired parameters
2. Analyze the results JSON file
3. Create visualizations (see visualization guide)
4. Compare CUDA vs Triton performance
5. Identify optimal configurations for your use case

## Example Workflow

```bash
# 1. Run benchmark
python benchmarks/comprehensive_benchmark.py

# 2. Analyze results
python -c "
import json
with open('benchmark_results.json') as f:
    results = json.load(f)
    
# Find fastest implementation for LayerNorm
layernorm = [r for r in results if r['operation'] == 'LayerNorm']
fastest = min(layernorm, key=lambda x: x['time_ms'])
print(f'Fastest LayerNorm: {fastest[\"implementation\"]} ({fastest[\"time_ms\"]:.3f} ms)')
"

# 3. Create visualizations (if visualization script exists)
# python benchmarks/visualize_results.py
```

