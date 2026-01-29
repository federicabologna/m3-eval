"""
Analyze variance in model ratings across the 5 runs.

Compares rating consistency between models (Qwen3-8B vs GPT-4.1) and
between original vs perturbed answers.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
output_base = os.path.join(project_root, 'output', 'cqa_eval', 'experiment_results', 'baseline')
analysis_output_dir = os.path.join(project_root, 'output', 'cqa_eval', 'analysis')
os.makedirs(analysis_output_dir, exist_ok=True)

# Metrics to analyze
METRICS = ['correctness', 'relevance', 'safety']
METRIC_LABELS = {
    'correctness': 'Correctness',
    'relevance': 'Relevance',
    'safety': 'Safety'
}


def load_rating_file(filepath):
    """Load ratings from JSONL file."""
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            entry = json.loads(line)
            results.append(entry)
    return results


def extract_individual_rating_variances(results, rating_type='original'):
    """
    Extract variance of individual ratings for each metric.

    Args:
        results: List of rating entries
        rating_type: 'original' or 'perturbed'

    Returns:
        Dict mapping metric -> list of standard deviations (one per example)
    """
    variances = {metric: [] for metric in METRICS}

    for entry in results:
        rating_key = f'{rating_type}_rating'
        if rating_key not in entry:
            continue

        rating_data = entry[rating_key]

        # Check if individual_ratings exists
        if 'individual_ratings' not in rating_data:
            continue

        individual_ratings = rating_data['individual_ratings']

        # Extract scores for each metric across the 5 runs
        for metric in METRICS:
            scores = []
            for run in individual_ratings:
                if metric in run and 'score' in run[metric]:
                    scores.append(run[metric]['score'])

            # Calculate standard deviation if we have ratings
            if len(scores) > 1:
                std = np.std(scores)
                variances[metric].append(std)

    return variances


def collect_all_variances():
    """
    Collect variance data from all rating files for Qwen3-8B and GPT-4.1.

    Returns:
        Dict with structure: {model_name: {rating_type: {metric: [stds]}}}
    """
    data = defaultdict(lambda: {'original': {m: [] for m in METRICS},
                                 'perturbed': {m: [] for m in METRICS}})

    # Models to analyze
    target_models = {
        'Qwen3-8B': 'qwen3-8b',
        'GPT-4.1': 'gpt-4_1-2025-04-14'
    }

    # Scan all perturbation directories
    if not os.path.exists(output_base):
        print(f"Output directory not found: {output_base}")
        return data

    for perturbation_dir_name in os.listdir(output_base):
        perturbation_path = os.path.join(output_base, perturbation_dir_name)
        if not os.path.isdir(perturbation_path):
            continue

        print(f"\nScanning: {perturbation_dir_name}")

        # Look for rating files
        for filename in os.listdir(perturbation_path):
            if not filename.endswith('_rating.jsonl'):
                continue

            # Check if file is for one of our target models
            model_key = None
            model_name = None
            for name, key in target_models.items():
                if key.lower() in filename.lower():
                    model_key = key
                    model_name = name
                    break

            if not model_name:
                continue

            print(f"  Processing: {filename} ({model_name})")

            filepath = os.path.join(perturbation_path, filename)
            results = load_rating_file(filepath)

            # Extract variances for original ratings
            orig_variances = extract_individual_rating_variances(results, 'original')
            for metric in METRICS:
                data[model_name]['original'][metric].extend(orig_variances[metric])

            # Extract variances for perturbed ratings
            pert_variances = extract_individual_rating_variances(results, 'perturbed')
            for metric in METRICS:
                data[model_name]['perturbed'][metric].extend(pert_variances[metric])

    return data


def plot_variance_comparison(variance_data, output_path):
    """
    Create bar plot comparing rating variance between models and rating types.

    Shows mean standard deviation across all examples for each model/type combination.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    models = ['Qwen3-8B', 'GPT-4.1']
    rating_types = ['original', 'perturbed']

    # Colors: blue for Qwen3-8B, orange for GPT-4.1
    # Solid for original, hatched for perturbed
    colors = {
        'Qwen3-8B': '#1f77b4',
        'GPT-4.1': '#ff7f0e'
    }

    width = 0.35
    x = np.arange(len(models))

    for idx, metric in enumerate(METRICS):
        ax = axes[idx]

        # Calculate mean standard deviations
        orig_means = []
        orig_cis = []
        pert_means = []
        pert_cis = []

        for model in models:
            # Original ratings
            if model in variance_data and len(variance_data[model]['original'][metric]) > 0:
                stds = variance_data[model]['original'][metric]
                mean_std = np.mean(stds)
                ci = 1.96 * np.std(stds) / np.sqrt(len(stds))
                orig_means.append(mean_std)
                orig_cis.append(ci)
            else:
                orig_means.append(0)
                orig_cis.append(0)

            # Perturbed ratings
            if model in variance_data and len(variance_data[model]['perturbed'][metric]) > 0:
                stds = variance_data[model]['perturbed'][metric]
                mean_std = np.mean(stds)
                ci = 1.96 * np.std(stds) / np.sqrt(len(stds))
                pert_means.append(mean_std)
                pert_cis.append(ci)
            else:
                pert_means.append(0)
                pert_cis.append(0)

        # Plot bars
        for i, model in enumerate(models):
            # Original (solid)
            ax.bar(i - width/2, orig_means[i], width, yerr=orig_cis[i],
                   label='Original' if i == 0 else '',
                   capsize=3, alpha=0.8, color=colors[model])

            # Perturbed (hatched)
            ax.bar(i + width/2, pert_means[i], width, yerr=pert_cis[i],
                   label='Perturbed' if i == 0 else '',
                   capsize=3, alpha=0.8, color=colors[model], hatch='//')

        # Customize subplot
        ax.set_ylabel('Mean Standard Deviation', fontsize=11)
        ax.set_title(f'{METRIC_LABELS[metric]}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        if idx == 0:  # Only show legend on first subplot
            ax.legend(fontsize=10, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(bottom=0)

    # Overall title
    fig.suptitle('Rating Variance Across 5 Runs - Model Comparison',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nSaved variance comparison plot: {os.path.basename(output_path)}")


def print_summary_statistics(variance_data):
    """Print summary statistics about rating variance."""
    print("\n" + "="*80)
    print("RATING VARIANCE SUMMARY STATISTICS")
    print("="*80)

    for model in ['Qwen3-8B', 'GPT-4.1']:
        if model not in variance_data:
            continue

        print(f"\n{model}:")
        print("-" * 80)

        for rating_type in ['original', 'perturbed']:
            print(f"\n  {rating_type.title()} Ratings:")

            for metric in METRICS:
                stds = variance_data[model][rating_type][metric]
                if len(stds) == 0:
                    print(f"    {METRIC_LABELS[metric]}: No data")
                    continue

                mean_std = np.mean(stds)
                median_std = np.median(stds)
                min_std = np.min(stds)
                max_std = np.max(stds)
                n_examples = len(stds)

                # Calculate percentage of examples with zero variance (perfect agreement)
                pct_zero_var = (np.sum(np.array(stds) == 0) / len(stds)) * 100

                print(f"    {METRIC_LABELS[metric]}:")
                print(f"      Mean SD: {mean_std:.3f}")
                print(f"      Median SD: {median_std:.3f}")
                print(f"      Min SD: {min_std:.3f}, Max SD: {max_std:.3f}")
                print(f"      % Perfect Agreement (SD=0): {pct_zero_var:.1f}%")
                print(f"      N examples: {n_examples}")


def main():
    print("\n" + "="*80)
    print("Rating Variance Analysis")
    print("="*80)

    # Collect variance data from all files
    print("\nCollecting variance data from rating files...")
    variance_data = collect_all_variances()

    # Print summary statistics
    print_summary_statistics(variance_data)

    # Generate comparison plot
    print("\n" + "="*80)
    print("Generating Variance Comparison Plot")
    print("="*80)
    plot_path = os.path.join(analysis_output_dir, 'rating_variance_comparison.png')
    plot_variance_comparison(variance_data, plot_path)

    print(f"\n{'='*80}")
    print(f"Analysis Complete!")
    print(f"{'='*80}")
    print(f"Output directory: {analysis_output_dir}")
    print()


if __name__ == "__main__":
    main()
