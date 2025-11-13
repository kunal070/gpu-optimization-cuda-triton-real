#!/usr/bin/env python3
"""
Visualize CUDA vs Triton benchmark results
Creates charts for presentation
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results(filename):
    """Load benchmark results from JSON"""
    with open(filename, 'r') as f:
        return json.load(f)

def plot_speedup_comparison(results, output_file='speedup_comparison.png'):
    """Plot speedup comparison bar chart"""
    operations = []
    cuda_speedups = []
    triton_speedups = []
    
    for r in results:
        if r.get('cuda') and r.get('triton'):
            operations.append(r['operation'][:20])  # Truncate for readability
            cuda_speedups.append(r['cuda']['speedup'])
            triton_speedups.append(r['triton']['speedup'])
    
    if not operations:
        print("No data with both CUDA and Triton results")
        return
    
    x = np.arange(len(operations))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width/2, cuda_speedups, width, label='CUDA', color='green', alpha=0.8)
    ax.bar(x + width/2, triton_speedups, width, label='Triton', color='blue', alpha=0.8)
    
    ax.set_xlabel('Operation', fontsize=12)
    ax.set_ylabel('Speedup vs PyTorch', fontsize=12)
    ax.set_title('CUDA vs Triton Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(operations, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.axhline(y=1.0, color='r', linestyle='--', label='Baseline (PyTorch)')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def plot_operation_breakdown(results, output_file='operation_breakdown.png'):
    """Plot breakdown by operation type"""
    operations = ['LayerNorm', 'GELU', 'Swish', 'Fused']
    cuda_avg = []
    triton_avg = []
    
    for op_type in operations:
        cuda_speeds = [r['cuda']['speedup'] for r in results 
                      if op_type in r['operation'] and r.get('cuda')]
        triton_speeds = [r['triton']['speedup'] for r in results 
                        if op_type in r['operation'] and r.get('triton')]
        
        cuda_avg.append(np.mean(cuda_speeds) if cuda_speeds else 0)
        triton_avg.append(np.mean(triton_speeds) if triton_speeds else 0)
    
    x = np.arange(len(operations))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, cuda_avg, width, label='CUDA', color='green', alpha=0.8)
    ax.bar(x + width/2, triton_avg, width, label='Triton', color='blue', alpha=0.8)
    
    ax.set_xlabel('Operation Type', fontsize=12)
    ax.set_ylabel('Average Speedup', fontsize=12)
    ax.set_title('Average Speedup by Operation Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(operations)
    ax.legend()
    ax.axhline(y=1.0, color='r', linestyle='--', label='Baseline')
    ax.grid(axis='y', alpha=0.3)
    
    for i, (c, t) in enumerate(zip(cuda_avg, triton_avg)):
        if c > 0:
            ax.text(i - width/2, c + 0.05, f'{c:.2f}x', ha='center', fontsize=10)
        if t > 0:
            ax.text(i + width/2, t + 0.05, f'{t:.2f}x', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def plot_execution_time_heatmap(results, output_file='execution_time_heatmap.png'):
    """Plot execution time heatmap"""
    import pandas as pd
    
    # Extract LayerNorm results
    layernorm_results = [r for r in results if 'LayerNorm' in r['operation'] and not 'Fused' in r['operation']]
    
    if not layernorm_results:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (impl, title) in enumerate([('pytorch', 'PyTorch'), ('cuda', 'CUDA'), ('triton', 'Triton')]):
        data = []
        for r in layernorm_results:
            if r.get(impl):
                parts = r['operation'].split('_')
                if len(parts) >= 4:
                    batch, seq, dim = int(parts[1]), int(parts[2]), int(parts[3])
                    data.append((batch, seq, dim, r[impl]['time_ms']))
        
        if data:
            df = pd.DataFrame(data, columns=['batch', 'seq', 'dim', 'time'])
            pivot = df.pivot_table(values='time', index='seq', columns='batch', aggfunc='mean')
            
            im = axes[idx].imshow(pivot, cmap='YlOrRd', aspect='auto')
            axes[idx].set_title(f'{title} Execution Time', fontweight='bold')
            axes[idx].set_xlabel('Batch Size')
            axes[idx].set_ylabel('Sequence Length')
            axes[idx].set_xticks(range(len(pivot.columns)))
            axes[idx].set_xticklabels(pivot.columns)
            axes[idx].set_yticks(range(len(pivot.index)))
            axes[idx].set_yticklabels(pivot.index)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[idx])
            cbar.set_label('Time (ms)', rotation=270, labelpad=15)
            
            # Add values
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    text = axes[idx].text(j, i, f'{pivot.iloc[i, j]:.3f}',
                                         ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def plot_kernel_fusion_benefit(results, output_file='kernel_fusion_benefit.png'):
    """Show benefit of kernel fusion"""
    fused_results = [r for r in results if 'Fused' in r['operation']]
    
    if not fused_results:
        return
    
    configs = []
    cuda_speedups = []
    
    for r in fused_results:
        if r.get('cuda'):
            parts = r['operation'].split('_')
            if len(parts) >= 4:
                config = f"B{parts[1]}_S{parts[2]}_D{parts[3]}"
                configs.append(config)
                cuda_speedups.append(r['cuda']['speedup'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(configs)), cuda_speedups, color='purple', alpha=0.7)
    
    ax.set_xlabel('Configuration (Batch_Seq_Dim)', fontsize=12)
    ax.set_ylabel('Speedup vs PyTorch (Separate Ops)', fontsize=12)
    ax.set_title('Kernel Fusion Benefit: LayerNorm + GELU', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.axhline(y=1.0, color='r', linestyle='--', label='No Speedup')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, speedup) in enumerate(zip(bars, cuda_speedups)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.2f}x',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add average line
    avg_speedup = np.mean(cuda_speedups)
    ax.axhline(y=avg_speedup, color='green', linestyle=':', linewidth=2, 
               label=f'Average: {avg_speedup:.2f}x')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def generate_summary_table(results, output_file='summary_table.txt'):
    """Generate text summary table"""
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CUDA vs Triton Benchmark Summary\n")
        f.write("JLR/UWindsor Hackathon - Project 3\n")
        f.write("="*80 + "\n\n")
        
        # Overall statistics
        cuda_speedups = [r['cuda']['speedup'] for r in results if r.get('cuda')]
        triton_speedups = [r['triton']['speedup'] for r in results if r.get('triton')]
        
        f.write("Overall Performance:\n")
        f.write("-" * 40 + "\n")
        if cuda_speedups:
            f.write(f"CUDA Average Speedup:   {np.mean(cuda_speedups):.2f}x\n")
            f.write(f"CUDA Best Speedup:      {np.max(cuda_speedups):.2f}x\n")
            f.write(f"CUDA Worst Speedup:     {np.min(cuda_speedups):.2f}x\n\n")
        
        if triton_speedups:
            f.write(f"Triton Average Speedup: {np.mean(triton_speedups):.2f}x\n")
            f.write(f"Triton Best Speedup:    {np.max(triton_speedups):.2f}x\n")
            f.write(f"Triton Worst Speedup:   {np.min(triton_speedups):.2f}x\n\n")
        
        # Operation breakdown
        f.write("\nPerformance by Operation Type:\n")
        f.write("-" * 40 + "\n")
        
        for op_type in ['LayerNorm', 'GELU', 'Fused']:
            cuda_ops = [r['cuda']['speedup'] for r in results 
                       if op_type in r['operation'] and r.get('cuda')]
            triton_ops = [r['triton']['speedup'] for r in results 
                         if op_type in r['operation'] and r.get('triton')]
            
            f.write(f"\n{op_type}:\n")
            if cuda_ops:
                f.write(f"  CUDA:   Avg {np.mean(cuda_ops):.2f}x, Best {np.max(cuda_ops):.2f}x\n")
            if triton_ops:
                f.write(f"  Triton: Avg {np.mean(triton_ops):.2f}x, Best {np.max(triton_ops):.2f}x\n")
    
    print(f"✓ Saved: {output_file}")

def main():
    print("="*80)
    print("Generating Visualization for CUDA vs Triton Results")
    print("="*80 + "\n")
    
    # Find most recent results file
    results_files = list(Path('.').glob('cuda_vs_triton_results_*.json'))
    if not results_files:
        print("❌ No results files found!")
        print("Run cuda_vs_triton_comparison.py first")
        return
    
    latest_file = max(results_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading results from: {latest_file}\n")
    
    results = load_results(latest_file)
    
    # Generate all visualizations
    plot_speedup_comparison(results)
    plot_operation_breakdown(results)
    plot_execution_time_heatmap(results)
    plot_kernel_fusion_benefit(results)
    generate_summary_table(results)
    
    print("\n" + "="*80)
    print("✓ All visualizations generated!")
    print("="*80)
    print("\nGenerated files:")
    print("  - speedup_comparison.png")
    print("  - operation_breakdown.png")
    print("  - execution_time_heatmap.png")
    print("  - kernel_fusion_benefit.png")
    print("  - summary_table.txt")
    print("\nUse these for your presentation!")

if __name__ == '__main__':
    main()