"""
Analyze error detection experiment results.

This script analyzes how well models can detect errors in perturbed answers.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_detection_results(results_dir):
    """Load all error detection result files."""
    results = defaultdict(lambda: defaultdict(list))

    results_path = Path(results_dir)

    # Find all detection result files
    for perturbation_dir in results_path.iterdir():
        if not perturbation_dir.is_dir():
            continue

        perturbation_name = perturbation_dir.name

        for result_file in perturbation_dir.glob('detection_*.jsonl'):
            # Parse filename: detection_{perturbation}_{level}_{model}.jsonl
            filename = result_file.stem
            parts = filename.split('_')

            # Extract level and model
            if 'coarse' in filename:
                level = 'coarse'
                model_start_idx = filename.index('coarse') + len('coarse') + 1
            elif 'fine' in filename:
                level = 'fine'
                model_start_idx = filename.index('fine') + len('fine') + 1
            else:
                continue

            model = filename[model_start_idx:]

            # Load results
            with open(result_file, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    results[perturbation_name][(level, model)].append(entry)

    return results


def calculate_detection_stats(results):
    """Calculate detection statistics."""
    stats = []

    for perturbation, level_model_results in results.items():
        for (level, model), entries in level_model_results.items():
            total = len(entries)
            detected_yes = sum(1 for e in entries if e.get('detection_result', {}).get('detected') == 'yes')
            detected_no = sum(1 for e in entries if e.get('detection_result', {}).get('detected') == 'no')
            invalid = total - detected_yes - detected_no

            stats.append({
                'perturbation': perturbation,
                'level': level,
                'model': model,
                'total': total,
                'detected_yes': detected_yes,
                'detected_no': detected_no,
                'invalid': invalid,
                'detection_rate': detected_yes / total if total > 0 else 0,
                'miss_rate': detected_no / total if total > 0 else 0
            })

    return pd.DataFrame(stats)


def plot_detection_rates(stats_df, output_dir):
    """Plot detection rates by perturbation type."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Group by perturbation and level
    for level in stats_df['level'].unique():
        level_data = stats_df[stats_df['level'] == level]

        fig, ax = plt.subplots(figsize=(12, 6))

        perturbations = level_data['perturbation'].unique()
        x = np.arange(len(perturbations))
        width = 0.35

        # Get models
        models = level_data['model'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))

        for i, model in enumerate(models):
            model_data = level_data[level_data['model'] == model]
            detection_rates = [
                model_data[model_data['perturbation'] == p]['detection_rate'].values[0]
                if len(model_data[model_data['perturbation'] == p]) > 0 else 0
                for p in perturbations
            ]

            offset = width * (i - len(models)/2 + 0.5)
            bars = ax.bar(x + offset, detection_rates, width,
                         label=model, color=colors[i], alpha=0.8)

            # Add percentage labels on bars
            for j, (bar, rate) in enumerate(zip(bars, detection_rates)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{rate*100:.1f}%',
                       ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Perturbation Type', fontsize=12)
        ax.set_ylabel('Detection Rate', fontsize=12)
        ax.set_title(f'Error Detection Rates - {level.capitalize()} Level', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(perturbations, rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'detection_rates_{level}.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved plot: detection_rates_{level}.png")


def plot_confusion_matrix(stats_df, output_dir):
    """Plot detection vs miss rates."""
    output_dir = Path(output_dir)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, level in enumerate(stats_df['level'].unique()):
        ax = axes[idx]
        level_data = stats_df[stats_df['level'] == level]

        # Create grouped bar chart
        perturbations = level_data['perturbation'].unique()
        x = np.arange(len(perturbations))
        width = 0.35

        for model in level_data['model'].unique():
            model_data = level_data[level_data['model'] == model]

            detected = [
                model_data[model_data['perturbation'] == p]['detected_yes'].values[0]
                if len(model_data[model_data['perturbation'] == p]) > 0 else 0
                for p in perturbations
            ]
            missed = [
                model_data[model_data['perturbation'] == p]['detected_no'].values[0]
                if len(model_data[model_data['perturbation'] == p]) > 0 else 0
                for p in perturbations
            ]

            ax.bar(x, detected, width, label=f'{model} (Detected)', alpha=0.8)
            ax.bar(x, missed, width, bottom=detected, label=f'{model} (Missed)', alpha=0.8)

        ax.set_xlabel('Perturbation Type', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'{level.capitalize()} Level', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(perturbations, rotation=45, ha='right')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Error Detection: Detected vs Missed', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'detection_confusion.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved plot: detection_confusion.png")


def generate_summary_report(stats_df, output_dir):
    """Generate text summary report."""
    output_dir = Path(output_dir)
    report_path = output_dir / 'detection_summary_report.txt'

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ERROR DETECTION EXPERIMENT SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        for level in sorted(stats_df['level'].unique()):
            f.write(f"\n{level.upper()} LEVEL\n")
            f.write("-" * 80 + "\n\n")

            level_data = stats_df[stats_df['level'] == level]

            for model in sorted(level_data['model'].unique()):
                f.write(f"Model: {model}\n")
                f.write("-" * 40 + "\n")

                model_data = level_data[level_data['model'] == model]

                for _, row in model_data.iterrows():
                    f.write(f"\n  {row['perturbation']}:\n")
                    f.write(f"    Total examples: {row['total']}\n")
                    f.write(f"    Detected (yes): {row['detected_yes']} ({row['detection_rate']*100:.1f}%)\n")
                    f.write(f"    Missed (no):    {row['detected_no']} ({row['miss_rate']*100:.1f}%)\n")
                    if row['invalid'] > 0:
                        f.write(f"    Invalid:        {row['invalid']}\n")

                # Overall stats for this model
                total_all = model_data['total'].sum()
                detected_all = model_data['detected_yes'].sum()
                f.write(f"\n  OVERALL:\n")
                f.write(f"    Total examples: {total_all}\n")
                f.write(f"    Detection rate: {detected_all/total_all*100:.1f}%\n")
                f.write("\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("CROSS-PERTURBATION COMPARISON\n")
        f.write("=" * 80 + "\n\n")

        # Compare detection rates across perturbations
        pivot = stats_df.pivot_table(
            values='detection_rate',
            index='perturbation',
            columns=['level', 'model'],
            aggfunc='mean'
        )

        f.write(pivot.to_string())
        f.write("\n\n")

    print(f"Saved report: {report_path}")


def main():
    """Main analysis function."""
    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    results_dir = project_root / 'output' / 'cqa_eval' / 'experiment_results' / 'error_detection'
    output_dir = project_root / 'output' / 'cqa_eval' / 'analysis'

    print("=" * 80)
    print("ERROR DETECTION ANALYSIS")
    print("=" * 80)
    print(f"\nResults directory: {results_dir}")
    print(f"Output directory: {output_dir}\n")

    # Load results
    print("Loading detection results...")
    results = load_detection_results(results_dir)

    if not results:
        print("No detection results found!")
        return

    print(f"Found {len(results)} perturbation types")
    for pert, data in results.items():
        print(f"  - {pert}: {len(data)} level-model combinations")

    # Calculate statistics
    print("\nCalculating statistics...")
    stats_df = calculate_detection_stats(results)

    # Save statistics to CSV
    stats_csv = output_dir / 'detection_statistics.csv'
    stats_df.to_csv(stats_csv, index=False)
    print(f"Saved statistics: {stats_csv}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_detection_rates(stats_df, output_dir)

    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(stats_df, output_dir)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
