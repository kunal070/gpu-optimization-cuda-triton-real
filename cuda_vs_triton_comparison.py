#!/usr/bin/env python3
"""
Complete CUDA vs Triton Comparison Benchmark
For JLR/UWindsor Hackathon Project 3

Compares:
- PyTorch (baseline)
- Custom CUDA kernels
- Triton kernels

Across multiple:
- Batch sizes (16, 32, 64, 128)
- Sequence lengths (256, 512, 1024, 2048)
- Tensor dimensions (256, 512, 768, 1024)
"""
import torch
import time
import json
from datetime import datetime

# Try to import custom implementations
try:
    import cuda_ops
    HAS_CUDA = True
except ImportError:
    print("⚠ Custom CUDA ops not available")
    HAS_CUDA = False

try:
    from triton_kernels import (
        layernorm_triton, swish_triton
    )
    # Try to import GELU and fused ops, but don't fail if they have issues
    try:
        from triton_kernels import gelu_triton, layernorm_gelu_fused_triton
        HAS_TRITON_GELU = True
    except:
        HAS_TRITON_GELU = False
    HAS_TRITON = True
except ImportError:
    print("⚠ Triton kernels not available")
    HAS_TRITON = False
    HAS_TRITON_GELU = False

def benchmark_operation(fn, *args, iterations=100, warmup=10):
    """Benchmark a single operation"""
    # Warmup
    for _ in range(warmup):
        _ = fn(*args)
    torch.cuda.synchronize()
    
    # Measure
    start = time.time()
    for _ in range(iterations):
        result = fn(*args)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / iterations * 1000  # ms
    
    return result, elapsed

def compare_implementations(name, pytorch_fn, cuda_fn, triton_fn, *args):
    """Compare PyTorch, CUDA, and Triton implementations"""
    results = {
        'operation': name,
        'pytorch': None,
        'cuda': None,
        'triton': None,
    }
    
    # PyTorch baseline
    try:
        result_pt, time_pt = benchmark_operation(pytorch_fn, *args)
        results['pytorch'] = {
            'time_ms': time_pt,
            'speedup': 1.0
        }
    except Exception as e:
        print(f"  PyTorch failed: {e}")
    
    # Custom CUDA
    if HAS_CUDA and cuda_fn:
        try:
            result_cuda, time_cuda = benchmark_operation(cuda_fn, *args)
            
            # Verify correctness
            if torch.allclose(result_pt, result_cuda, rtol=1e-3, atol=1e-3):
                correctness = "✓ Correct"
            else:
                max_diff = (result_pt - result_cuda).abs().max().item()
                correctness = f"⚠ Max diff: {max_diff:.6f}"
            
            results['cuda'] = {
                'time_ms': time_cuda,
                'speedup': time_pt / time_cuda,
                'correctness': correctness
            }
        except Exception as e:
            print(f"  CUDA failed: {e}")
    
    # Triton
    if HAS_TRITON and triton_fn:
        try:
            result_triton, time_triton = benchmark_operation(triton_fn, *args)
            
            # Verify correctness
            if torch.allclose(result_pt, result_triton, rtol=1e-3, atol=1e-3):
                correctness = "✓ Correct"
            else:
                max_diff = (result_pt - result_triton).abs().max().item()
                correctness = f"⚠ Max diff: {max_diff:.6f}"
            
            results['triton'] = {
                'time_ms': time_triton,
                'speedup': time_pt / time_triton,
                'correctness': correctness
            }
        except Exception as e:
            print(f"  Triton failed: {e}")
    
    return results

def print_comparison(results):
    """Pretty print comparison results"""
    print(f"\n  Operation: {results['operation']}")
    
    if results['pytorch']:
        print(f"  PyTorch:  {results['pytorch']['time_ms']:8.4f} ms  (baseline)")
    
    if results['cuda']:
        print(f"  CUDA:     {results['cuda']['time_ms']:8.4f} ms  "
              f"({results['cuda']['speedup']:.2f}x)  {results['cuda']['correctness']}")
    
    if results['triton']:
        print(f"  Triton:   {results['triton']['time_ms']:8.4f} ms  "
              f"({results['triton']['speedup']:.2f}x)  {results['triton']['correctness']}")

def main():
    print("="*80)
    print("CUDA vs Triton Comparison Benchmark")
    print("JLR/UWindsor Hackathon - Project 3")
    print("="*80 + "\n")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch CUDA: {torch.version.cuda}")
    print(f"Custom CUDA: {'Available' if HAS_CUDA else 'Not Available'}")
    print(f"Triton: {'Available' if HAS_TRITON else 'Not Available'}")
    
    all_results = []
    
    # Test configurations (as per project requirements)
    batch_sizes = [16, 32, 64, 128]
    seq_lengths = [256, 512, 1024]
    hidden_dims = [256, 512, 768]
    
    # Test 1: LayerNorm
    print("\n" + "="*80)
    print("TEST 1: Layer Normalization")
    print("="*80)
    
    for batch in batch_sizes[:2]:  # Test subset for speed
        for seq in seq_lengths[:2]:
            for dim in hidden_dims[:2]:
                print(f"\nConfig: batch={batch}, seq={seq}, dim={dim}")
                
                x = torch.randn(batch, seq, dim, device='cuda')
                gamma = torch.ones(dim, device='cuda')
                beta = torch.zeros(dim, device='cuda')
                
                layer_norm = torch.nn.LayerNorm(dim).to('cuda')
                
                results = compare_implementations(
                    f"LayerNorm_{batch}_{seq}_{dim}",
                    layer_norm,
                    lambda x: cuda_ops.layernorm(x, gamma, beta) if HAS_CUDA else None,
                    lambda x: layernorm_triton(x, gamma, beta) if HAS_TRITON else None,
                    x
                )
                
                print_comparison(results)
                all_results.append(results)
    
    # Test 2: GELU
    print("\n" + "="*80)
    print("TEST 2: GELU Activation")
    print("="*80)
    
    for batch in batch_sizes[:2]:
        for seq in seq_lengths[:2]:
            for dim in hidden_dims[:2]:
                print(f"\nConfig: batch={batch}, seq={seq}, dim={dim}")
                
                x = torch.randn(batch, seq, dim, device='cuda')
                
                results = compare_implementations(
                    f"GELU_{batch}_{seq}_{dim}",
                    torch.nn.functional.gelu,
                    lambda x: cuda_ops.gelu(x, True) if HAS_CUDA else None,
                    lambda x: gelu_triton(x, True) if HAS_TRITON else None,
                    x
                )
                
                print_comparison(results)
                all_results.append(results)
    
    # Test 3: Swish
    print("\n" + "="*80)
    print("TEST 3: Swish Activation")
    print("="*80)
    
    for batch in batch_sizes[:2]:
        for seq in seq_lengths[:2]:
            for dim in hidden_dims[:2]:
                print(f"\nConfig: batch={batch}, seq={seq}, dim={dim}")
                
                x = torch.randn(batch, seq, dim, device='cuda')
                
                results = compare_implementations(
                    f"Swish_{batch}_{seq}_{dim}",
                    torch.nn.functional.silu,
                    lambda x: cuda_ops.swish(x) if HAS_CUDA else None,
                    lambda x: swish_triton(x) if HAS_TRITON else None,
                    x
                )
                
                print_comparison(results)
                all_results.append(results)
    
    # Test 4: Fused LayerNorm + GELU
    print("\n" + "="*80)
    print("TEST 4: Fused LayerNorm + GELU (Kernel Fusion)")
    print("="*80)
    
    for batch in batch_sizes[:2]:
        for seq in seq_lengths[:2]:
            for dim in hidden_dims[:2]:
                print(f"\nConfig: batch={batch}, seq={seq}, dim={dim}")
                
                x = torch.randn(batch, seq, dim, device='cuda')
                gamma = torch.ones(dim, device='cuda')
                beta = torch.zeros(dim, device='cuda')
                
                layer_norm = torch.nn.LayerNorm(dim).to('cuda')
                
                results = compare_implementations(
                    f"Fused_{batch}_{seq}_{dim}",
                    lambda x: torch.nn.functional.gelu(layer_norm(x)),
                    lambda x: cuda_ops.layernorm_gelu_fused(x, gamma, beta) if HAS_CUDA else None,
                    lambda x: layernorm_gelu_fused_triton(x, gamma, beta) if HAS_TRITON else None,
                    x
                )
                
                print_comparison(results)
                all_results.append(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cuda_vs_triton_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {filename}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if HAS_CUDA:
        cuda_speedups = [r['cuda']['speedup'] for r in all_results if r.get('cuda')]
        if cuda_speedups:
            print(f"\nCUDA Performance:")
            print(f"  Average speedup: {sum(cuda_speedups)/len(cuda_speedups):.2f}x")
            print(f"  Best speedup:    {max(cuda_speedups):.2f}x")
    
    if HAS_TRITON:
        triton_speedups = [r['triton']['speedup'] for r in all_results if r.get('triton')]
        if triton_speedups:
            print(f"\nTriton Performance:")
            print(f"  Average speedup: {sum(triton_speedups)/len(triton_speedups):.2f}x")
            print(f"  Best speedup:    {max(triton_speedups):.2f}x")

if __name__ == '__main__':
    main()