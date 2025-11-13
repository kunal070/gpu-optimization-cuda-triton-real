"""
Visualize benchmark results
Creates charts and tables matching the presentation format
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict
import os


def load_results(filename: str = 'benchmark_results.json') -> List[Dict]:
    """Load benchmark results from JSON file"""
    if not os.path.exists(filename):
        print(f"Error: {filename} not found. Run comprehensive_benchmark.py first.")
        return []
    
    with open(filename, 'r') as f:
        return json.load(f)


def create_comparison_table(results: List[Dict], operation: str, save_path: str = None):
    """Create a comparison table for an operation"""
    df = pd.DataFrame(results)
    df_op = df[df['operation'] == operation]
    
    if df_op.empty:
        print(f"No results found for {operation}")
        return
    
    # Group by implementation
    table_data = []
    for impl in df_op['implementation'].unique():
        impl_data = df_op[df_op['implementation'] == impl]
        table_data.append({
            'Implementation': impl,
            'Avg Time (ms)': impl_data['time_ms'].mean(),
            'Avg Memory (MB)': impl_data['memory_mb'].mean(),
            'Avg Inference Speed': impl_data['inference_speed'].mean(),
            'Avg GPU Efficiency': impl_data['gpu_efficiency'].mean(),
        })
    
    table_df = pd.DataFrame(table_data)
    print(f"\n{'='*80}")
    print(f"Comparison Table for {operation}")
    print(f"{'='*80}")
    print(table_df.to_string(index=False))
    print(f"{'='*80}\n")
    
    if save_path:
        table_df.to_csv(save_path, index=False)
        print(f"Table saved to {save_path}")


def plot_by_batch_size(results: List[Dict], operation: str, save_path: str = None):
    """Plot performance vs batch size"""
    df = pd.DataFrame(results)
    df_op = df[df['operation'] == operation]
    
    if df_op.empty:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{operation} Performance vs Batch Size', fontsize=16)
    
    for impl in df_op['implementation'].unique():
        impl_data = df_op[df_op['implementation'] == impl]
        grouped = impl_data.groupby('batch_size').mean()
        
        axes[0, 0].plot(grouped.index, grouped['time_ms'], marker='o', label=impl)
        axes[0, 1].plot(grouped.index, grouped['memory_mb'], marker='s', label=impl)
        axes[1, 0].plot(grouped.index, grouped['inference_speed'], marker='^', label=impl)
        axes[1, 1].plot(grouped.index, grouped['gpu_efficiency'], marker='d', label=impl)
    
    axes[0, 0].set_xlabel('Batch Size')
    axes[0, 0].set_ylabel('Time (ms)')
    axes[0, 0].set_title('Execution Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].set_xlabel('Batch Size')
    axes[0, 1].set_ylabel('Memory (MB)')
    axes[0, 1].set_title('Memory Usage')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].set_xlabel('Batch Size')
    axes[1, 0].set_ylabel('Inference Speed (ops/s)')
    axes[1, 0].set_title('Inference Speed')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].set_xlabel('Batch Size')
    axes[1, 1].set_ylabel('GPU Efficiency (%)')
    axes[1, 1].set_title('GPU Efficiency')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_by_sequence_length(results: List[Dict], operation: str, save_path: str = None):
    """Plot performance vs sequence length"""
    df = pd.DataFrame(results)
    df_op = df[df['operation'] == operation]
    
    if df_op.empty:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{operation} Performance vs Sequence Length', fontsize=16)
    
    for impl in df_op['implementation'].unique():
        impl_data = df_op[df_op['implementation'] == impl]
        grouped = impl_data.groupby('sequence_length').mean()
        
        axes[0, 0].plot(grouped.index, grouped['time_ms'], marker='o', label=impl)
        axes[0, 1].plot(grouped.index, grouped['memory_mb'], marker='s', label=impl)
        axes[1, 0].plot(grouped.index, grouped['inference_speed'], marker='^', label=impl)
        axes[1, 1].plot(grouped.index, grouped['gpu_efficiency'], marker='d', label=impl)
    
    axes[0, 0].set_xlabel('Sequence Length')
    axes[0, 0].set_ylabel('Time (ms)')
    axes[0, 0].set_title('Execution Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].set_xlabel('Sequence Length')
    axes[0, 1].set_ylabel('Memory (MB)')
    axes[0, 1].set_title('Memory Usage')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].set_xlabel('Sequence Length')
    axes[1, 0].set_ylabel('Inference Speed (ops/s)')
    axes[1, 0].set_title('Inference Speed')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].set_xlabel('Sequence Length')
    axes[1, 1].set_ylabel('GPU Efficiency (%)')
    axes[1, 1].set_title('GPU Efficiency')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_by_tensor_dimension(results: List[Dict], operation: str, save_path: str = None):
    """Plot performance vs tensor dimension"""
    df = pd.DataFrame(results)
    df_op = df[df['operation'] == operation]
    
    if df_op.empty:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{operation} Performance vs Tensor Dimension', fontsize=16)
    
    for impl in df_op['implementation'].unique():
        impl_data = df_op[df_op['implementation'] == impl]
        grouped = impl_data.groupby('tensor_dimension').mean()
        
        axes[0, 0].plot(grouped.index, grouped['time_ms'], marker='o', label=impl)
        axes[0, 1].plot(grouped.index, grouped['memory_mb'], marker='s', label=impl)
        axes[1, 0].plot(grouped.index, grouped['inference_speed'], marker='^', label=impl)
        axes[1, 1].plot(grouped.index, grouped['gpu_efficiency'], marker='d', label=impl)
    
    axes[0, 0].set_xlabel('Tensor Dimension')
    axes[0, 0].set_ylabel('Time (ms)')
    axes[0, 0].set_title('Execution Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].set_xlabel('Tensor Dimension')
    axes[0, 1].set_ylabel('Memory (MB)')
    axes[0, 1].set_title('Memory Usage')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].set_xlabel('Tensor Dimension')
    axes[1, 0].set_ylabel('Inference Speed (ops/s)')
    axes[1, 0].set_title('Inference Speed')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].set_xlabel('Tensor Dimension')
    axes[1, 1].set_ylabel('GPU Efficiency (%)')
    axes[1, 1].set_title('GPU Efficiency')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def create_summary_report(results: List[Dict], save_path: str = 'benchmark_summary.txt'):
    """Create a text summary report"""
    df = pd.DataFrame(results)
    
    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BENCHMARK SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        
        for operation in df['operation'].unique():
            f.write(f"\n{operation}\n")
            f.write("-"*80 + "\n")
            
            df_op = df[df['operation'] == operation]
            
            for impl in df_op['implementation'].unique():
                impl_data = df_op[df_op['implementation'] == impl]
                f.write(f"\n{impl}:\n")
                f.write(f"  Average Time: {impl_data['time_ms'].mean():.3f} ms\n")
                f.write(f"  Average Memory: {impl_data['memory_mb'].mean():.2f} MB\n")
                f.write(f"  Average Inference Speed: {impl_data['inference_speed'].mean():.2f} ops/s\n")
                f.write(f"  Average GPU Efficiency: {impl_data['gpu_efficiency'].mean():.2f}%\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"Summary report saved to {save_path}")


if __name__ == '__main__':
    results = load_results()
    
    if not results:
        print("No results to visualize. Run comprehensive_benchmark.py first.")
        exit(1)
    
    operations = ['LayerNorm', 'GELU', 'Swish', 'Loss']
    
    # Create comparison tables
    print("\n" + "="*80)
    print("BENCHMARK RESULTS VISUALIZATION")
    print("="*80)
    
    for op in operations:
        create_comparison_table(results, op, f'{op.lower()}_comparison.csv')
    
    # Create plots
    print("\nGenerating plots...")
    for op in operations:
        plot_by_batch_size(results, op, f'{op.lower()}_by_batch_size.png')
        plot_by_sequence_length(results, op, f'{op.lower()}_by_sequence_length.png')
        plot_by_tensor_dimension(results, op, f'{op.lower()}_by_tensor_dimension.png')
    
    # Create summary report
    create_summary_report(results)
    
    print("\n" + "="*80)
    print("Visualization complete!")
    print("="*80)

