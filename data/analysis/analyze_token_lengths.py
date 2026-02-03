#!/usr/bin/env python3
"""
Analyze token lengths in answers across different datasets.

Compares:
- CQA Eval Coarse (answer-level)
- CQA Eval Fine (sentence-level)
- MedInfo2019 (answer-level)
"""

import json
import os
import sys
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, os.path.join(project_root, 'code'))


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text using tokenizer."""
    return len(tokenizer.encode(text))


def analyze_dataset(data: List[Dict], dataset_name: str, tokenizer, text_field='answer') -> Dict:
    """Analyze token lengths in a dataset."""
    token_counts = []

    for item in data:
        if text_field in item:
            text = item[text_field]
            token_count = count_tokens(text, tokenizer)
            token_counts.append(token_count)

    if not token_counts:
        print(f"Warning: No data found for {dataset_name}")
        return {}

    stats = {
        'dataset': dataset_name,
        'n_samples': len(token_counts),
        'mean': np.mean(token_counts),
        'median': np.median(token_counts),
        'std': np.std(token_counts),
        'min': np.min(token_counts),
        'max': np.max(token_counts),
        'p25': np.percentile(token_counts, 25),
        'p75': np.percentile(token_counts, 75),
        'p90': np.percentile(token_counts, 90),
        'p95': np.percentile(token_counts, 95),
        'p99': np.percentile(token_counts, 99),
        'token_counts': token_counts
    }

    return stats


def print_stats(stats: Dict):
    """Print statistics for a dataset."""
    print(f"\n{'='*80}")
    print(f"{stats['dataset']}")
    print(f"{'='*80}")
    print(f"Samples: {stats['n_samples']}")
    print(f"\nToken Count Statistics:")
    print(f"  Mean:   {stats['mean']:.1f}")
    print(f"  Median: {stats['median']:.1f}")
    print(f"  Std:    {stats['std']:.1f}")
    print(f"  Min:    {stats['min']}")
    print(f"  Max:    {stats['max']}")
    print(f"\nPercentiles:")
    print(f"  25th:   {stats['p25']:.1f}")
    print(f"  75th:   {stats['p75']:.1f}")
    print(f"  90th:   {stats['p90']:.1f}")
    print(f"  95th:   {stats['p95']:.1f}")
    print(f"  99th:   {stats['p99']:.1f}")


def create_histogram(all_stats: List[Dict], output_dir: str):
    """Create histogram comparing token lengths across datasets."""
    fig, axes = plt.subplots(len(all_stats), 1, figsize=(12, 4 * len(all_stats)))

    if len(all_stats) == 1:
        axes = [axes]

    for idx, stats in enumerate(all_stats):
        ax = axes[idx]
        token_counts = stats['token_counts']

        ax.hist(token_counts, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {stats["mean"]:.1f}')
        ax.axvline(stats['median'], color='green', linestyle='--', linewidth=2, label=f'Median: {stats["median"]:.1f}')

        ax.set_xlabel('Token Count', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{stats["dataset"]} - Answer Token Length Distribution (n={stats["n_samples"]})', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'token_length_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved histogram to: {output_path}")
    plt.close()


def create_box_plot(all_stats: List[Dict], output_dir: str):
    """Create box plot comparing token lengths across datasets."""
    fig, ax = plt.subplots(figsize=(10, 6))

    data_to_plot = [stats['token_counts'] for stats in all_stats]
    labels = [stats['dataset'] for stats in all_stats]

    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)

    # Color the boxes
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)

    ax.set_ylim(-5, 400)  # Set x-axis limits
    ax.set_ylabel('Token Count', fontsize=12)
    ax.set_title('Answer Token Length Comparison Across Datasets', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Rotate labels if needed
    if len(labels) > 2:
        plt.xticks(rotation=15, ha='right')

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'token_length_boxplot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved box plot to: {output_path}")
    plt.close()


def save_summary_report(all_stats: List[Dict], output_dir: str):
    """Save summary statistics to text file."""
    output_path = os.path.join(output_dir, 'token_length_summary.txt')

    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Token Length Analysis Summary\n")
        f.write("="*80 + "\n\n")

        # Summary table
        f.write(f"{'Dataset':<30} {'N':<8} {'Mean':<10} {'Median':<10} {'Min':<8} {'Max':<8}\n")
        f.write("-"*80 + "\n")

        for stats in all_stats:
            f.write(f"{stats['dataset']:<30} "
                   f"{stats['n_samples']:<8} "
                   f"{stats['mean']:<10.1f} "
                   f"{stats['median']:<10.1f} "
                   f"{stats['min']:<8} "
                   f"{stats['max']:<8}\n")

        f.write("\n" + "="*80 + "\n\n")

        # Detailed statistics for each dataset
        for stats in all_stats:
            f.write(f"\n{stats['dataset']}\n")
            f.write("-"*80 + "\n")
            f.write(f"Samples:        {stats['n_samples']}\n")
            f.write(f"Mean:           {stats['mean']:.1f}\n")
            f.write(f"Median:         {stats['median']:.1f}\n")
            f.write(f"Std Dev:        {stats['std']:.1f}\n")
            f.write(f"Min:            {stats['min']}\n")
            f.write(f"Max:            {stats['max']}\n")
            f.write(f"25th percentile: {stats['p25']:.1f}\n")
            f.write(f"75th percentile: {stats['p75']:.1f}\n")
            f.write(f"90th percentile: {stats['p90']:.1f}\n")
            f.write(f"95th percentile: {stats['p95']:.1f}\n")
            f.write(f"99th percentile: {stats['p99']:.1f}\n")
            f.write("\n")

    print(f"✓ Saved summary report to: {output_path}")


def main():
    print("="*80)
    print("Token Length Analysis")
    print("="*80)

    # Setup paths
    data_dir = os.path.dirname(script_dir)
    output_dir = script_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer (using GPT-2 tokenizer as standard reference)
    print("\nLoading tokenizer (gpt2)...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    # Dataset paths
    datasets = [
        {
            'path': os.path.join(data_dir, 'coarse_5pt_expert+llm_consolidated.jsonl'),
            'name': 'CQA Eval - Coarse (Answer-level)',
            'text_field': 'answer'
        },
        {
            'path': os.path.join(data_dir, 'fine_5pt_expert+llm_consolidated.jsonl'),
            'name': 'CQA Eval - Fine (Sentence-level)',
            'text_field': 'answer'
        },
        {
            'path': os.path.join(data_dir, 'medinfo2019_medications_qa.jsonl'),
            'name': 'MedInfo2019 (Answer-level)',
            'text_field': 'answer'
        }
    ]

    # Analyze each dataset
    all_stats = []

    for dataset_info in datasets:
        if not os.path.exists(dataset_info['path']):
            print(f"\n⚠ Dataset not found: {dataset_info['path']}")
            continue

        print(f"\nAnalyzing {dataset_info['name']}...")
        data = load_jsonl(dataset_info['path'])

        stats = analyze_dataset(
            data,
            dataset_info['name'],
            tokenizer,
            text_field=dataset_info['text_field']
        )

        if stats:
            all_stats.append(stats)
            print_stats(stats)

    if not all_stats:
        print("\n⚠ No datasets were successfully analyzed")
        return

    # Create visualizations
    print(f"\n{'='*80}")
    print("Generating visualizations...")
    print(f"{'='*80}")

    create_histogram(all_stats, output_dir)
    create_box_plot(all_stats, output_dir)
    save_summary_report(all_stats, output_dir)

    print(f"\n{'='*80}")
    print("Analysis Complete!")
    print(f"{'='*80}")
    print(f"\nOutput files saved to: {output_dir}")


if __name__ == "__main__":
    main()
