#!/usr/bin/env python3
"""
FULL Visualization Suite for CUDA/Triton/PyTorch Benchmark Results
Generates:
 - Speedup bars (STAR RESULT style)
 - Speedup heatmaps
 - Per-operation comparison tables
 - Summary report
 - Trend plots (batch, seq, dim)
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
import os


# -----------------------------------------------------------------------------
# LOAD RESULTS
# -----------------------------------------------------------------------------
def load_results(filename: str = 'benchmark_results.json') -> List[Dict]:
    if not os.path.exists(filename):
        print(f"❌ ERROR: {filename} not found. Run comprehensive_benchmark.py first.")
        return []
    with open(filename, 'r') as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# STAR RESULT BAR CHART: Triton speedup vs PyTorch (your favorite chart)
# -----------------------------------------------------------------------------
def plot_speedup_bar(results: List[Dict], operation: str, impl: str = 'Triton',
                     save_path: str = None):
    """
    Creates the famous STAR RESULT bar chart:
    - Purple bars
    - 1.0 baseline line
    - Green average speedup line
    - Labels on bars
    """

    df = pd.DataFrame(results)
    df_op = df[df['operation'] == operation]

    if df_op.empty:
        print(f"[BAR] No results for {operation}")
        return

    if impl not in df_op['implementation'].unique():
        print(f"[BAR] No {impl} results for {operation}")
        return

    # Split PyTorch vs Impl
    df_pt = df_op[df_op['implementation'] == 'PyTorch']
    df_impl = df_op[df_op['implementation'] == impl]

    # Merge on (batch, seq, dim)
    merged = pd.merge(
        df_pt, df_impl,
        on=['batch_size', 'sequence_length', 'tensor_dimension'],
        suffixes=('_pt', f'_{impl.lower()}')
    )

    if merged.empty:
        print(f"[BAR] No matching PyTorch/{impl} pairs.")
        return

    # Build config name like B32_S512_D256
    merged['config'] = (
        'B' + merged['batch_size'].astype(str) +
        '_S' + merged['sequence_length'].astype(str) +
        '_D' + merged['tensor_dimension'].astype(str)
    )

    # Compute speedup
    merged['speedup'] = merged['time_ms_pt'] / merged[f'time_ms_{impl.lower()}']

    # Sort cleanly
    merged = merged.sort_values(['tensor_dimension', 'sequence_length', 'batch_size'])

    configs = merged['config'].tolist()
    speedups = merged['speedup'].values
    avg_speedup = np.mean(speedups)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(15, 6))

    x = np.arange(len(configs))
    bars = ax.bar(x, speedups, color='#7e57c2')

    # 1.0 baseline
    ax.axhline(1.0, linestyle='--', color='red', linewidth=2, label='No Speedup (1.0x)')

    # Average speedup line
    ax.axhline(avg_speedup, linestyle='--', color='green',
               linewidth=2, label=f'Average Speedup ({avg_speedup:.2f}x)')

    # Labels on bars
    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width()/2,
                speedups[i] + 0.03,
                f"{speedups[i]:.2f}x",
                ha='center', fontsize=9)

    ax.set_title(f"{operation} – {impl} Speedup vs PyTorch", fontsize=16, fontweight='bold')
    ax.set_ylabel("Speedup (x)")
    ax.set_xlabel("Configuration (Batch_Seq_Dim)")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[BAR] Saved: {save_path}")
    else:
        plt.show()


# -----------------------------------------------------------------------------
# HEATMAP: Speedup vs Batch + Sequence
# -----------------------------------------------------------------------------
def plot_speedup_heatmap(results: List[Dict], operation: str,
                         impl: str = 'Triton', tensor_dim: int = 256,
                         save_path: str = None):

    df = pd.DataFrame(results)
    df_op = df[df['operation'] == operation]

    df_pt = df_op[df_op['implementation'] == 'PyTorch']
    df_impl = df_op[df_op['implementation'] == impl]

    merged = pd.merge(
        df_pt, df_impl,
        on=['batch_size', 'sequence_length', 'tensor_dimension'],
        suffixes=('_pt', f'_{impl.lower()}')
    )

    merged = merged[merged['tensor_dimension'] == tensor_dim]

    if merged.empty:
        print(f"[HEATMAP] No results for {operation} {impl} dim={tensor_dim}")
        return

    merged['speedup'] = merged['time_ms_pt'] / merged[f'time_ms_{impl.lower()}']

    pivot = merged.pivot_table(
        index='batch_size',
        columns='sequence_length',
        values='speedup'
    ).sort_index().sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, cmap='viridis', aspect='auto')

    ax.set_title(f"{operation} – {impl} Speedup Heatmap (D={tensor_dim})",
                 fontsize=16, fontweight='bold')
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Batch Size")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)

    # Annotate values
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i, j]:.2f}",
                    ha='center', va='center',
                    color="white" if pivot.values[i, j] > np.mean(pivot.values) else "black",
                    fontsize=9)

    plt.colorbar(im, ax=ax, label="Speedup (x)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[HEATMAP] Saved: {save_path}")
    else:
        plt.show()


# -----------------------------------------------------------------------------
# SUMMARY REPORT
# -----------------------------------------------------------------------------
def create_summary_report(results: List[Dict], save_path: str = 'benchmark_summary.txt'):
    df = pd.DataFrame(results)

    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(" BENCHMARK SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")

        for op in df['operation'].unique():
            f.write(f"\n{op}\n")
            f.write("-"*80 + "\n")

            df_op = df[df['operation'] == op]

            for impl in df_op['implementation'].unique():
                impl_df = df_op[df_op['implementation'] == impl]
                f.write(f"\n{impl}:\n")
                f.write(f"   Avg Time:   {impl_df['time_ms'].mean():.3f} ms\n")
                f.write(f"   Avg Memory: {impl_df['memory_mb'].mean():.2f} MB\n")
                f.write(f"   Avg Speed:  {impl_df['inference_speed'].mean():.2f} ops/s\n")

    print(f"[SUMMARY] Saved: {save_path}")


# -----------------------------------------------------------------------------
# MAIN VISUALIZATION PIPELINE
# -----------------------------------------------------------------------------
def main():

    print("="*80)
    print("📊 Generating All Visualization Figures")
    print("="*80)

    results = load_results('benchmark_results.json')
    if not results:
        return

    operations = ['LayerNorm', 'GELU', 'Swish', 'Loss']
    impls = ['Triton', 'CUDA']  # CUDA graphs appear only if extension is built

    # ---- GENERATE ALL GRAPH TYPES ----
    for op in operations:
        for impl in impls:
            # skip if no such impl exists
            if not any(r['operation'] == op and r['implementation'] == impl for r in results):
                continue

            # STAR RESULT BAR
            plot_speedup_bar(
                results, op, impl,
                f"{op}_{impl}_speedup_bar.png"
            )

            # HEATMAP
            plot_speedup_heatmap(
                results, op, impl, tensor_dim=256,
                save_path=f"{op}_{impl}_heatmap_D256.png"
            )

    # Summary text file
    create_summary_report(results)

    print("\n🎉 Visualization Complete!")
    print("Generated files:")
    print(" • *_speedup_bar.png")
    print(" • *_heatmap_D256.png")
    print(" • benchmark_summary.txt")
    print("="*80)


if __name__ == '__main__':
    main()
