"""
Comprehensive Benchmarking Framework
Matches the project goals from the presentation slides

Measures:
- Time (execution time)
- Memory Usage
- Inference Speed (throughput)
- GPU Efficiency

Tests across:
- Different batch sizes: 16, 32, 64, 128, ...
- Different sequence lengths: 256, 512, 1024, 2048, ...
- Different tensor dimensions: 8, 16, 32, 64, 128, ...

Operations:
- LayerNorm
- GELU
- Swish
- Loss Functions
- Fused Operations
"""

import torch
import time
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cuda_ops
    CUDA_OPS_AVAILABLE = True
except ImportError:
    CUDA_OPS_AVAILABLE = False
    print("Warning: CUDA ops extension not available")

try:
    from triton_kernels import (
        layernorm_triton,
        gelu_triton,
        swish_triton,
        mse_loss_triton,
        layernorm_gelu_fused_triton,
    )
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("Warning: Triton kernels not available")


@dataclass
class BenchmarkResult:
    """Structure for benchmark results"""
    operation: str
    implementation: str  # 'PyTorch', 'CUDA', 'Triton'
    batch_size: int
    sequence_length: int
    tensor_dimension: int
    time_ms: float
    memory_mb: float
    inference_speed: float  # samples/second or ops/second
    gpu_efficiency: float  # percentage
    throughput_gbps: float  # memory throughput in GB/s


class GPUMonitor:
    """Monitor GPU usage during benchmarks"""
    
    def __init__(self):
        self.initial_memory = None
        self.peak_memory = None
        
    def start(self):
        """Start monitoring"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            self.initial_memory = torch.cuda.memory_allocated() / 1024**2  # MB
    
    def stop(self):
        """Stop monitoring and return stats"""
        if torch.cuda.is_available():
            self.peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            current_memory = torch.cuda.memory_allocated() / 1024**2  # MB
            return {
                'peak_memory_mb': self.peak_memory,
                'current_memory_mb': current_memory,
                'memory_delta_mb': self.peak_memory - (self.initial_memory or 0)
            }
        return {'peak_memory_mb': 0, 'current_memory_mb': 0, 'memory_delta_mb': 0}


def benchmark_operation(
    operation_name: str,
    func,
    *args,
    warmup: int = 10,
    repeat: int = 100,
    monitor: Optional[GPUMonitor] = None
) -> Dict:
    """
    Benchmark an operation and return comprehensive metrics
    
    Returns:
        Dictionary with time, memory, throughput, and efficiency metrics
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Warmup
    for _ in range(warmup):
        _ = func(*args)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Start monitoring
    if monitor:
        monitor.start()
    
    # Benchmark execution time
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
    
    # Stop monitoring
    memory_stats = {}
    if monitor:
        memory_stats = monitor.stop()
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    # Calculate throughput (operations per second)
    inference_speed = 1000.0 / mean_time if mean_time > 0 else 0
    
    # Estimate GPU efficiency (simplified - based on memory bandwidth utilization)
    # This is a placeholder - real efficiency requires detailed profiling
    gpu_efficiency = min(100.0, (inference_speed / 1000.0) * 100)  # Simplified metric
    
    # Calculate memory throughput (GB/s)
    # Estimate based on input/output sizes
    total_memory_gb = memory_stats.get('peak_memory_mb', 0) / 1024.0
    throughput_gbps = total_memory_gb / (mean_time / 1000.0) if mean_time > 0 else 0
    
    return {
        'time_ms': mean_time,
        'memory_mb': memory_stats.get('peak_memory_mb', 0),
        'inference_speed': inference_speed,  # ops/second
        'gpu_efficiency': gpu_efficiency,  # percentage (simplified)
        'throughput_gbps': throughput_gbps,
    }


def benchmark_layernorm(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    monitor: GPUMonitor
) -> List[BenchmarkResult]:
    """Benchmark LayerNorm across implementations"""
    results = []
    device = 'cuda'
    
    # Create test data
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, device=device)
    gamma = torch.ones(hidden_size, device=device)
    beta = torch.zeros(hidden_size, device=device)
    eps = 1e-5
    
    # PyTorch Native
    def pytorch_layernorm():
        return torch.nn.functional.layer_norm(
            input_tensor, (hidden_size,), gamma, beta, eps
        )
    
    metrics = benchmark_operation('LayerNorm', pytorch_layernorm, monitor=monitor)
    results.append(BenchmarkResult(
        operation='LayerNorm',
        implementation='PyTorch',
        batch_size=batch_size,
        sequence_length=seq_len,
        tensor_dimension=hidden_size,
        **metrics
    ))
    
    # CUDA Extension
    if CUDA_OPS_AVAILABLE:
        def cuda_layernorm():
            return cuda_ops.layernorm(input_tensor, gamma, beta, eps)
        
        metrics = benchmark_operation('LayerNorm', cuda_layernorm, monitor=monitor)
        results.append(BenchmarkResult(
            operation='LayerNorm',
            implementation='CUDA',
            batch_size=batch_size,
            sequence_length=seq_len,
            tensor_dimension=hidden_size,
            **metrics
        ))
    
    # Triton
    if TRITON_AVAILABLE:
        def triton_layernorm():
            return layernorm_triton(input_tensor, gamma, beta, eps)
        
        metrics = benchmark_operation('LayerNorm', triton_layernorm, monitor=monitor)
        results.append(BenchmarkResult(
            operation='LayerNorm',
            implementation='Triton',
            batch_size=batch_size,
            sequence_length=seq_len,
            tensor_dimension=hidden_size,
            **metrics
        ))
    
    return results


def benchmark_gelu(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    monitor: GPUMonitor
) -> List[BenchmarkResult]:
    """Benchmark GELU across implementations"""
    results = []
    device = 'cuda'
    
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, device=device)
    
    # PyTorch Native
    def pytorch_gelu():
        return torch.nn.functional.gelu(input_tensor)
    
    metrics = benchmark_operation('GELU', pytorch_gelu, monitor=monitor)
    results.append(BenchmarkResult(
        operation='GELU',
        implementation='PyTorch',
        batch_size=batch_size,
        sequence_length=seq_len,
        tensor_dimension=hidden_size,
        **metrics
    ))
    
    # CUDA Extension
    if CUDA_OPS_AVAILABLE:
        def cuda_gelu():
            return cuda_ops.gelu(input_tensor, use_fast=True)
        
        metrics = benchmark_operation('GELU', cuda_gelu, monitor=monitor)
        results.append(BenchmarkResult(
            operation='GELU',
            implementation='CUDA',
            batch_size=batch_size,
            sequence_length=seq_len,
            tensor_dimension=hidden_size,
            **metrics
        ))
    
    # Triton
    if TRITON_AVAILABLE:
        def triton_gelu():
            return gelu_triton(input_tensor, use_fast=True)
        
        metrics = benchmark_operation('GELU', triton_gelu, monitor=monitor)
        results.append(BenchmarkResult(
            operation='GELU',
            implementation='Triton',
            batch_size=batch_size,
            sequence_length=seq_len,
            tensor_dimension=hidden_size,
            **metrics
        ))
    
    return results


def benchmark_swish(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    monitor: GPUMonitor
) -> List[BenchmarkResult]:
    """Benchmark Swish across implementations"""
    results = []
    device = 'cuda'
    
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, device=device)
    
    # PyTorch Native (using x * sigmoid(x))
    def pytorch_swish():
        return input_tensor * torch.sigmoid(input_tensor)
    
    metrics = benchmark_operation('Swish', pytorch_swish, monitor=monitor)
    results.append(BenchmarkResult(
        operation='Swish',
        implementation='PyTorch',
        batch_size=batch_size,
        sequence_length=seq_len,
        tensor_dimension=hidden_size,
        **metrics
    ))
    
    # CUDA Extension (if available)
    if CUDA_OPS_AVAILABLE and hasattr(cuda_ops, 'swish'):
        def cuda_swish():
            return cuda_ops.swish(input_tensor)
        
        metrics = benchmark_operation('Swish', cuda_swish, monitor=monitor)
        results.append(BenchmarkResult(
            operation='Swish',
            implementation='CUDA',
            batch_size=batch_size,
            sequence_length=seq_len,
            tensor_dimension=hidden_size,
            **metrics
        ))
    
    # Triton
    if TRITON_AVAILABLE:
        def triton_swish():
            return swish_triton(input_tensor)
        
        metrics = benchmark_operation('Swish', triton_swish, monitor=monitor)
        results.append(BenchmarkResult(
            operation='Swish',
            implementation='Triton',
            batch_size=batch_size,
            sequence_length=seq_len,
            tensor_dimension=hidden_size,
            **metrics
        ))
    
    return results


def benchmark_loss(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    monitor: GPUMonitor
) -> List[BenchmarkResult]:
    """Benchmark Loss Functions"""
    results = []
    device = 'cuda'
    
    pred = torch.randn(batch_size, seq_len, hidden_size, device=device)
    target = torch.randn(batch_size, seq_len, hidden_size, device=device)
    
    # PyTorch MSE
    def pytorch_mse():
        return torch.nn.functional.mse_loss(pred, target)
    
    metrics = benchmark_operation('Loss_MSE', pytorch_mse, monitor=monitor)
    results.append(BenchmarkResult(
        operation='Loss',
        implementation='PyTorch',
        batch_size=batch_size,
        sequence_length=seq_len,
        tensor_dimension=hidden_size,
        **metrics
    ))
    
    # Triton MSE
    if TRITON_AVAILABLE:
        def triton_mse():
            return mse_loss_triton(pred, target)
        
        metrics = benchmark_operation('Loss_MSE', triton_mse, monitor=monitor)
        results.append(BenchmarkResult(
            operation='Loss',
            implementation='Triton',
            batch_size=batch_size,
            sequence_length=seq_len,
            tensor_dimension=hidden_size,
            **metrics
        ))
    
    return results


def run_comprehensive_benchmark(
    batch_sizes: List[int] = [16, 32, 64, 128],
    sequence_lengths: List[int] = [256, 512, 1024, 2048],
    tensor_dimensions: List[int] = [8, 16, 32, 64, 128, 256, 512],
    operations: List[str] = ['LayerNorm', 'GELU', 'Swish', 'Loss'],
    output_file: str = 'benchmark_results.json'
) -> List[BenchmarkResult]:
    """
    Run comprehensive benchmark across all parameter combinations
    
    Args:
        batch_sizes: List of batch sizes to test
        sequence_lengths: List of sequence lengths to test
        tensor_dimensions: List of tensor dimensions to test
        operations: List of operations to benchmark
        output_file: File to save results
    
    Returns:
        List of BenchmarkResult objects
    """
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return []
    
    print("="*80)
    print("Comprehensive GPU Benchmark Suite")
    print("="*80)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"\nTesting:")
    print(f"  Batch Sizes: {batch_sizes}")
    print(f"  Sequence Lengths: {sequence_lengths}")
    print(f"  Tensor Dimensions: {tensor_dimensions}")
    print(f"  Operations: {operations}")
    print("="*80)
    
    all_results = []
    monitor = GPUMonitor()
    
    total_combinations = len(batch_sizes) * len(sequence_lengths) * len(tensor_dimensions) * len(operations)
    current = 0
    
    for batch_size in batch_sizes:
        for seq_len in sequence_lengths:
            for hidden_size in tensor_dimensions:
                # Skip if tensor is too large
                if batch_size * seq_len * hidden_size > 50_000_000:  # ~200MB
                    continue
                
                print(f"\n[{current+1}/{total_combinations}] Testing: batch={batch_size}, seq={seq_len}, dim={hidden_size}")
                
                for op in operations:
                    try:
                        if op == 'LayerNorm':
                            results = benchmark_layernorm(batch_size, seq_len, hidden_size, monitor)
                        elif op == 'GELU':
                            results = benchmark_gelu(batch_size, seq_len, hidden_size, monitor)
                        elif op == 'Swish':
                            results = benchmark_swish(batch_size, seq_len, hidden_size, monitor)
                        elif op == 'Loss':
                            results = benchmark_loss(batch_size, seq_len, hidden_size, monitor)
                        else:
                            continue
                        
                        all_results.extend(results)
                        current += 1
                        
                    except Exception as e:
                        print(f"  Error testing {op}: {e}")
                        continue
    
    # Save results
    results_dict = [asdict(r) for r in all_results]
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Benchmark Complete! Results saved to {output_file}")
    print(f"Total results: {len(all_results)}")
    print(f"{'='*80}")
    
    return all_results


def print_results_table(results: List[BenchmarkResult], operation: str):
    """Print results in a table format matching the presentation slides"""
    print(f"\n{'='*80}")
    print(f"Results for {operation}")
    print(f"{'='*80}")
    print(f"{'Implementation':<15} {'Time (ms)':<12} {'Memory (MB)':<15} {'Inference Speed':<18} {'GPU Efficiency':<15}")
    print("-" * 80)
    
    for result in results:
        if result.operation == operation:
            print(f"{result.implementation:<15} {result.time_ms:<12.3f} {result.memory_mb:<15.2f} "
                  f"{result.inference_speed:<18.2f} {result.gpu_efficiency:<15.2f}")
    
    print("="*80)


if __name__ == '__main__':
    # Run benchmark with default parameters
    # You can customize these based on your needs
    
    results = run_comprehensive_benchmark(
        batch_sizes=[16, 32, 64],
        sequence_lengths=[256, 512, 1024],
        tensor_dimensions=[32, 64, 128, 256],
        operations=['LayerNorm', 'GELU', 'Swish', 'Loss'],
        output_file='benchmark_results.json'
    )
    
    # Print summary tables
    for op in ['LayerNorm', 'GELU', 'Swish', 'Loss']:
        print_results_table(results, op)

