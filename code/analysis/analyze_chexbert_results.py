"""
Analyze CheXbert evaluation results from RadEval experiments.

Compares original vs perturbed CheXbert scores across different perturbation types.
Generates statistics and visualizations for:
- Accuracy (TOP5 conditions)
- Micro F1 (chexbert_5 and chexbert_all)
- Macro F1 (chexbert_5 and chexbert_all)
- Weighted F1 (chexbert_5 and chexbert_all)
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict

# Path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
output_base = os.path.join(project_root, 'output', 'radeval', 'experiment_results', 'baseline')
analysis_output_dir = os.path.join(project_root, 'output', 'radeval', 'analysis')
os.makedirs(analysis_output_dir, exist_ok=True)

# CheXbert metrics to analyze (all metrics for statistics, subset for plots)
ALL_METRICS = [
    'accuracy',
    'chexbert_5_micro_f1',
    'chexbert_all_micro_f1',
    'chexbert_5_macro_f1',
    'chexbert_all_macro_f1',
    'chexbert_5_weighted_f1',
    'chexbert_all_weighted_f1'
]

# Metrics to plot (only weighted F1 for "all 14 conditions")
PLOT_METRICS = [
    'chexbert_all_weighted_f1'
]

METRIC_LABELS = {
    'accuracy': 'Accuracy (TOP5)',
    'chexbert_5_micro_f1': 'Micro F1 (TOP5)',
    'chexbert_all_micro_f1': 'Micro F1 (All 14)',
    'chexbert_5_macro_f1': 'Macro F1 (TOP5)',
    'chexbert_all_macro_f1': 'Macro F1 (All 14)',
    'chexbert_5_weighted_f1': 'Weighted F1 (TOP5)',
    'chexbert_all_weighted_f1': 'Weighted F1 (All 14)'
}


def load_chexbert_results(filepath):
    """Load CheXbert results from JSONL file."""
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            entry = json.loads(line)
            results.append(entry)
    return results


def extract_scores(results, metric):
    """Extract original and perturbed scores for a specific metric."""
    original_scores = []
    perturbed_scores = []

    for entry in results:
        if 'original_chexbert_rating' in entry and 'perturbed_chexbert_rating' in entry:
            orig = entry['original_chexbert_rating'].get(metric)
            pert = entry['perturbed_chexbert_rating'].get(metric)

            if orig is not None and pert is not None:
                original_scores.append(orig)
                perturbed_scores.append(pert)

    return np.array(original_scores), np.array(perturbed_scores)


def compute_statistics(original_scores, perturbed_scores):
    """Compute statistical measures comparing original vs perturbed."""
    # Basic statistics
    orig_mean = np.mean(original_scores)
    orig_std = np.std(original_scores)
    pert_mean = np.mean(perturbed_scores)
    pert_std = np.std(perturbed_scores)

    # Compute differences (degradation)
    differences = original_scores - perturbed_scores
    mean_degradation = np.mean(differences)
    std_degradation = np.std(differences)

    # Percentage of cases where score decreased
    pct_decreased = np.mean(differences > 0) * 100
    pct_increased = np.mean(differences < 0) * 100
    pct_unchanged = np.mean(differences == 0) * 100

    # Statistical test (Wilcoxon signed-rank test for paired samples)
    if len(original_scores) > 0:
        statistic, p_value = stats.wilcoxon(original_scores, perturbed_scores, alternative='greater')
    else:
        statistic, p_value = None, None

    return {
        'original_mean': orig_mean,
        'original_std': orig_std,
        'perturbed_mean': pert_mean,
        'perturbed_std': pert_std,
        'mean_degradation': mean_degradation,
        'std_degradation': std_degradation,
        'pct_decreased': pct_decreased,
        'pct_increased': pct_increased,
        'pct_unchanged': pct_unchanged,
        'n_samples': len(original_scores),
        'wilcoxon_statistic': statistic,
        'wilcoxon_p_value': p_value
    }


def plot_comparison(perturbation_results, metric, output_path):
    """Create bar plot with average weighted F1 and 95% CI error bars."""
    if len(perturbation_results) == 0:
        return

    # Define ordering for perturbations
    perturbation_order = [
        ('swap_organs', 'Swap Organs'),
        ('swap_qualifiers', 'Swap Qualifiers'),
        ('remove_sentences_30', 'Remove Sent. 30%'),
        ('remove_sentences_50', 'Remove Sent. 50%'),
        ('remove_sentences_70', 'Remove Sent. 70%'),
        ('add_typos_03', 'Add Typos p=0.3'),
        ('add_typos_05', 'Add Typos p=0.5'),
        ('add_typos_07', 'Add Typos p=0.7'),
    ]

    # Collect all original scores (combined)
    all_original_scores = []
    for key, data in perturbation_results.items():
        all_original_scores.extend(data['original'][metric].tolist())

    # Compute mean and 95% CI for original
    orig_mean = np.mean(all_original_scores)
    orig_ci = 1.96 * np.std(all_original_scores) / np.sqrt(len(all_original_scores))  # 95% CI

    means = [orig_mean]
    cis = [orig_ci]
    labels = ['Original']

    # Collect perturbed scores for each perturbation (in order)
    for pert_key, pert_label in perturbation_order:
        # Find matching perturbation in results
        for key, data in perturbation_results.items():
            perturbation_dir = data['perturbation']

            # Match perturbation type
            matched = False
            if pert_key == 'swap_organs' and perturbation_dir == 'swap_organs':
                matched = True
            elif pert_key == 'swap_qualifiers' and perturbation_dir == 'swap_qualifiers':
                matched = True
            elif pert_key.startswith('remove_sentences'):
                pct = pert_key.split('_')[2]
                if perturbation_dir == 'remove_sentences' and f'{pct}pct' in data['filename']:
                    matched = True
            elif pert_key.startswith('add_typos'):
                prob = pert_key.split('_')[2]
                if perturbation_dir == 'add_typos' and f'{prob}prob' in data['filename']:
                    matched = True

            if matched:
                scores = data['perturbed'][metric]
                mean = np.mean(scores)
                ci = 1.96 * np.std(scores) / np.sqrt(len(scores))  # 95% CI
                means.append(mean)
                cis.append(ci)
                labels.append(pert_label)
                break

    # Create bar plot
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(means))
    colors = ['#1f77b4'] + ['#ff7f0e'] * (len(means) - 1)

    bars = ax.bar(x, means, yerr=cis, color=colors, alpha=0.7,
                  capsize=5, error_kw={'linewidth': 2})

    # Customize plot
    ax.set_ylabel(METRIC_LABELS[metric], fontsize=12)
    ax.set_title(f'{METRIC_LABELS[metric]} - Original vs Perturbed (Mean ± 95% CI)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(bottom=0, top=1.05)  # F1 scores range [0, 1]

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved plot: {os.path.basename(output_path)}")


def plot_degradation_distribution(perturbation_results, metric, output_path):
    """Plot distribution of score degradations for a specific metric."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    perturbations = list(perturbation_results.keys())

    for idx, perturbation in enumerate(perturbations[:4]):  # Limit to 4 for 2x2 grid
        if idx >= len(axes):
            break

        ax = axes[idx]
        orig_scores = perturbation_results[perturbation]['original'][metric]
        pert_scores = perturbation_results[perturbation]['perturbed'][metric]
        degradations = orig_scores - pert_scores

        # Histogram
        ax.hist(degradations, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
        ax.axvline(np.mean(degradations), color='green', linestyle='--', linewidth=2,
                  label=f'Mean: {np.mean(degradations):.3f}')

        ax.set_xlabel('Score Degradation (Original - Perturbed)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f"{perturbation_results[perturbation]['label']}", fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    # Hide unused subplots
    for idx in range(len(perturbations), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'{METRIC_LABELS[metric]} - Degradation Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved plot: {os.path.basename(output_path)}")


def generate_summary_report(perturbation_results, output_path):
    """Generate text summary report of all results."""
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CheXbert Evaluation Results Summary\n")
        f.write("=" * 80 + "\n\n")

        for perturbation, data in perturbation_results.items():
            f.write(f"\n{'='*80}\n")
            f.write(f"Perturbation: {data['label']}\n")
            f.write(f"{'='*80}\n")
            f.write(f"File: {data['filename']}\n")
            f.write(f"Number of samples: {data['stats']['accuracy']['n_samples']}\n\n")

            for metric in ALL_METRICS:
                stats_data = data['stats'][metric]
                f.write(f"\n{'-'*80}\n")
                f.write(f"{METRIC_LABELS[metric]}\n")
                f.write(f"{'-'*80}\n")
                f.write(f"  Original:  {stats_data['original_mean']:.4f} ± {stats_data['original_std']:.4f}\n")
                f.write(f"  Perturbed: {stats_data['perturbed_mean']:.4f} ± {stats_data['perturbed_std']:.4f}\n")
                f.write(f"  Mean degradation: {stats_data['mean_degradation']:.4f} ± {stats_data['std_degradation']:.4f}\n")
                f.write(f"\n")
                f.write(f"  Score decreased: {stats_data['pct_decreased']:.1f}%\n")
                f.write(f"  Score increased: {stats_data['pct_increased']:.1f}%\n")
                f.write(f"  Score unchanged: {stats_data['pct_unchanged']:.1f}%\n")

                if stats_data['wilcoxon_p_value'] is not None:
                    significance = "***" if stats_data['wilcoxon_p_value'] < 0.001 else \
                                 "**" if stats_data['wilcoxon_p_value'] < 0.01 else \
                                 "*" if stats_data['wilcoxon_p_value'] < 0.05 else "ns"
                    f.write(f"\n  Wilcoxon signed-rank test:\n")
                    f.write(f"    Statistic: {stats_data['wilcoxon_statistic']:.2f}\n")
                    f.write(f"    p-value: {stats_data['wilcoxon_p_value']:.4e} {significance}\n")
                    f.write(f"    (*** p<0.001, ** p<0.01, * p<0.05, ns p>=0.05)\n")

            f.write("\n")

    print(f"  Saved summary: {os.path.basename(output_path)}")


def main():
    print("\n" + "="*80)
    print("CheXbert Results Analysis")
    print("="*80)

    # Find all CheXbert result files
    perturbation_results = {}

    for perturbation_dir in os.listdir(output_base):
        perturbation_path = os.path.join(output_base, perturbation_dir)
        if not os.path.isdir(perturbation_path):
            continue

        # Look for chexbert rating files
        for filename in os.listdir(perturbation_path):
            if filename.endswith('_chexbert_rating.jsonl'):
                filepath = os.path.join(perturbation_path, filename)

                # Parse filename for label
                if perturbation_dir == 'add_typos':
                    prob = filename.split('_')[2].replace('prob', '')
                    label = f"Add Typos (p={float(prob)/10})"
                elif perturbation_dir == 'remove_sentences':
                    pct = filename.split('_')[2].replace('pct', '')
                    label = f"Remove Sentences ({pct}%)"
                elif perturbation_dir == 'swap_qualifiers':
                    label = "Swap Qualifiers"
                elif perturbation_dir == 'swap_organs':
                    label = "Swap Organs"
                else:
                    label = perturbation_dir.replace('_', ' ').title()

                print(f"\nLoading: {perturbation_dir}/{filename}")

                # Load results
                results = load_chexbert_results(filepath)
                print(f"  Loaded {len(results)} entries")

                # Extract scores for all metrics
                metric_data = {}
                for metric in ALL_METRICS:
                    orig, pert = extract_scores(results, metric)
                    metric_data[metric] = {
                        'original': orig,
                        'perturbed': pert
                    }

                # Compute statistics for all metrics
                stats_data = {}
                for metric in ALL_METRICS:
                    stats_data[metric] = compute_statistics(
                        metric_data[metric]['original'],
                        metric_data[metric]['perturbed']
                    )

                # Store results
                key = f"{perturbation_dir}_{filename}"
                perturbation_results[key] = {
                    'perturbation': perturbation_dir,
                    'filename': filename,
                    'label': label,
                    'original': {m: metric_data[m]['original'] for m in ALL_METRICS},
                    'perturbed': {m: metric_data[m]['perturbed'] for m in ALL_METRICS},
                    'stats': stats_data
                }

    if not perturbation_results:
        print("\nNo CheXbert result files found!")
        return

    print(f"\n{'='*80}")
    print(f"Generating Analysis Outputs")
    print(f"{'='*80}")

    # Generate comparison bar plot (only weighted F1 for all 14 conditions)
    print("\nGenerating comparison bar plot (weighted F1 only)...")
    for metric in PLOT_METRICS:
        output_path = os.path.join(analysis_output_dir, f'chexbert_{metric}_barplot.png')
        plot_comparison(perturbation_results, metric, output_path)

    # Generate summary report
    print("\nGenerating summary report...")
    report_path = os.path.join(analysis_output_dir, 'chexbert_summary_report.txt')
    generate_summary_report(perturbation_results, report_path)

    print(f"\n{'='*80}")
    print(f"Analysis Complete!")
    print(f"{'='*80}")
    print(f"Output directory: {analysis_output_dir}")
    print()


if __name__ == "__main__":
    main()
