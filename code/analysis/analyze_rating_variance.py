"""
Analyze variance in model ratings across the 5 runs.

Compares rating consistency between models (Qwen3-8B vs GPT-4.1) and
between original vs perturbed answers.
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
analysis_output_dir = os.path.join(project_root, 'output', 'analysis')
os.makedirs(analysis_output_dir, exist_ok=True)

# Dataset directories to scan
DATASET_DIRS = {
    'CQA': os.path.join(project_root, 'output', 'cqa_eval', 'experiment_results', 'baseline'),
    'MedInfo': os.path.join(project_root, 'output', 'medinfo', 'experiment_results', 'baseline'),
    'RadEval': os.path.join(project_root, 'output', 'radeval', 'experiment_results', 'baseline'),
    'WoundCare': os.path.join(project_root, 'output', 'woundcare', 'experiment_results', 'baseline'),
}

# Metrics to analyze
METRICS = ['correctness', 'relevance', 'safety']
METRIC_LABELS = {
    'correctness': 'Correctness',
    'relevance': 'Relevance',
    'safety': 'Safety'
}


def get_significance_marker(p_value):
    """Return significance marker based on p-value."""
    if p_value is None:
        return ''
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'


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
    Handles multiple data formats: CQA/MedInfo (nested metrics), GREEN (individual_scores), WoundCare (individual_ratings).

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

        # Format 1: CQA/MedInfo - nested metrics with individual_ratings
        if 'individual_ratings' in rating_data and isinstance(rating_data['individual_ratings'], list):
            individual_ratings = rating_data['individual_ratings']

            # Check if this is CQA/MedInfo format (metrics nested in each rating)
            if len(individual_ratings) > 0 and isinstance(individual_ratings[0], dict):
                # Check if first rating has metric keys
                first_rating = individual_ratings[0]
                if any(metric in first_rating for metric in METRICS):
                    # CQA/MedInfo format
                    for metric in METRICS:
                        scores = []
                        for run in individual_ratings:
                            if metric in run and 'score' in run[metric]:
                                scores.append(run[metric]['score'])

                        # Calculate standard deviation if we have ratings
                        if len(scores) > 1:
                            std = np.std(scores)
                            variances[metric].append(std)
                # Check if this is WoundCare format (rating field directly in dict)
                elif 'rating' in first_rating:
                    # WoundCare format - only has one metric (avg_rating)
                    # We'll use 'correctness' as the key for single-metric datasets
                    scores = [run['rating'] for run in individual_ratings if 'rating' in run]
                    if len(scores) > 1:
                        std = np.std(scores)
                        variances['correctness'].append(std)

        # Format 2: GREEN/RadEval - flat individual_scores list
        elif 'individual_scores' in rating_data:
            individual_scores = rating_data['individual_scores']
            if isinstance(individual_scores, list) and len(individual_scores) > 1:
                std = np.std(individual_scores)
                # Use 'correctness' as the key for single-metric datasets
                variances['correctness'].append(std)

    return variances


def collect_all_variances():
    """
    Collect variance data from all rating files for Qwen3-8B and GPT-4.1 across all datasets.

    Returns:
        Dict with structure: {dataset_name: {model_name: {rating_type: {metric: [stds]}}}}
    """
    data_by_dataset = {}

    # Models to analyze (use patterns that will match filename variations)
    target_models = {
        'Qwen3-8B': ['qwen3-8b', 'qwen3_8b'],
        'GPT-4.1': ['gpt-4.1-2025-04-14', 'gpt-4_1-2025-04-14']
    }

    # Scan all dataset directories
    for dataset_name, output_base in DATASET_DIRS.items():
        if not os.path.exists(output_base):
            print(f"\n{dataset_name} directory not found: {output_base}")
            continue

        print(f"\n{'='*80}")
        print(f"Scanning {dataset_name} dataset")
        print(f"{'='*80}")

        # Initialize data structure for this dataset
        data_by_dataset[dataset_name] = defaultdict(lambda: {'original': {m: [] for m in METRICS},
                                                               'perturbed': {m: [] for m in METRICS}})

        # Scan all perturbation directories
        for perturbation_dir_name in os.listdir(output_base):
            perturbation_path = os.path.join(output_base, perturbation_dir_name)
            if not os.path.isdir(perturbation_path):
                continue

            print(f"\n  {perturbation_dir_name}")

            # Look for rating files
            for filename in os.listdir(perturbation_path):
                if not filename.endswith('_rating.jsonl'):
                    continue

                # Check if file is for one of our target models
                model_name = None
                for name, patterns in target_models.items():
                    if any(pattern.lower() in filename.lower() for pattern in patterns):
                        model_name = name
                        break

                if not model_name:
                    continue

                print(f"    Processing: {filename} ({model_name})")

                filepath = os.path.join(perturbation_path, filename)
                results = load_rating_file(filepath)

                # Extract variances for original ratings
                orig_variances = extract_individual_rating_variances(results, 'original')
                for metric in METRICS:
                    data_by_dataset[dataset_name][model_name]['original'][metric].extend(orig_variances[metric])

                # Extract variances for perturbed ratings
                pert_variances = extract_individual_rating_variances(results, 'perturbed')
                for metric in METRICS:
                    data_by_dataset[dataset_name][model_name]['perturbed'][metric].extend(pert_variances[metric])

    return data_by_dataset


def plot_variance_comparison_by_dataset(variance_data_by_dataset, output_path):
    """
    Create 1x4 grid comparing rating variance across datasets.

    Each column represents one dataset (CQA, MedInfo, RadEval, WoundCare).
    For datasets with multiple metrics, data is aggregated across all metrics.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    models = ['Qwen3-8B', 'GPT-4.1']
    datasets = ['CQA', 'MedInfo', 'RadEval', 'WoundCare']

    # Colors: blue for Qwen3-8B, orange for GPT-4.1
    colors = {
        'Qwen3-8B': '#1f77b4',
        'GPT-4.1': '#ff7f0e'
    }

    width = 0.35
    x = np.arange(len(models))

    for dataset_idx, dataset in enumerate(datasets):
        variance_data = variance_data_by_dataset.get(dataset, {})
        ax = axes[dataset_idx]

        # Aggregate data across all metrics
        orig_means = []
        orig_cis = []
        pert_means = []
        pert_cis = []

        has_data = False

        for model in models:
            # Collect all standard deviations across all metrics for original ratings
            all_orig_stds = []
            for metric in METRICS:
                if model in variance_data and len(variance_data[model]['original'][metric]) > 0:
                    all_orig_stds.extend(variance_data[model]['original'][metric])
                    has_data = True

            if len(all_orig_stds) > 0:
                mean_std = np.mean(all_orig_stds)
                ci = 1.96 * np.std(all_orig_stds) / np.sqrt(len(all_orig_stds))
                orig_means.append(mean_std)
                orig_cis.append(ci)
            else:
                orig_means.append(0)
                orig_cis.append(0)

            # Collect all standard deviations across all metrics for perturbed ratings
            all_pert_stds = []
            for metric in METRICS:
                if model in variance_data and len(variance_data[model]['perturbed'][metric]) > 0:
                    all_pert_stds.extend(variance_data[model]['perturbed'][metric])
                    has_data = True

            if len(all_pert_stds) > 0:
                mean_std = np.mean(all_pert_stds)
                ci = 1.96 * np.std(all_pert_stds) / np.sqrt(len(all_pert_stds))
                pert_means.append(mean_std)
                pert_cis.append(ci)
            else:
                pert_means.append(0)
                pert_cis.append(0)

        if not has_data:
            # No data for this dataset
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            # Plot bars
            for i, model in enumerate(models):
                # Original (solid)
                if orig_means[i] > 0:
                    ax.bar(i - width/2, orig_means[i], width, yerr=orig_cis[i],
                           label='Original' if dataset_idx == 0 and i == 0 else '',
                           capsize=5, alpha=0.8, color=colors[model])

                # Perturbed (hatched)
                if pert_means[i] > 0:
                    ax.bar(i + width/2, pert_means[i], width, yerr=pert_cis[i],
                           label='Perturbed' if dataset_idx == 0 and i == 0 else '',
                           capsize=5, alpha=0.8, color=colors[model], hatch='//')

            # Add difference and significance markers
            for i, model in enumerate(models):
                # Collect all data for statistical test
                all_orig_data = []
                all_pert_data = []
                for metric in METRICS:
                    if model in variance_data:
                        if len(variance_data[model]['original'][metric]) > 0:
                            all_orig_data.extend(variance_data[model]['original'][metric])
                        if len(variance_data[model]['perturbed'][metric]) > 0:
                            all_pert_data.extend(variance_data[model]['perturbed'][metric])

                if len(all_orig_data) > 0 and len(all_pert_data) > 0 and pert_means[i] > 0:
                    diff = pert_means[i] - orig_means[i]

                    try:
                        _, p_val = stats.mannwhitneyu(all_orig_data, all_pert_data, alternative='two-sided')
                    except:
                        p_val = None

                    sig_marker = get_significance_marker(p_val)

                    y_pos = pert_means[i] + pert_cis[i] + 0.02
                    ax.text(i + width/2, y_pos, f'{diff:.2f}{sig_marker}',
                           ha='center', va='bottom', fontsize=9, fontweight='bold')

            ax.set_xticks(x)
            ax.set_xticklabels(models, fontsize=11)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(bottom=0)

        # Column title and y-label
        ax.set_title(f'{dataset}', fontsize=13, fontweight='bold')
        if dataset_idx == 0:
            ax.set_ylabel('Mean Standard Deviation', fontsize=12)
            ax.legend(fontsize=10, loc='upper right')

    # Overall title
    fig.suptitle('Rating Variance Across 5 Runs - By Dataset',
                 fontsize=15, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nSaved variance comparison plot: {os.path.basename(output_path)}")


def print_summary_statistics(variance_data_by_dataset):
    """Print summary statistics about rating variance by dataset."""
    print("\n" + "="*80)
    print("RATING VARIANCE SUMMARY STATISTICS BY DATASET")
    print("="*80)

    for dataset in ['CQA', 'MedInfo', 'RadEval', 'WoundCare']:
        if dataset not in variance_data_by_dataset:
            continue

        variance_data = variance_data_by_dataset[dataset]

        print(f"\n{'='*80}")
        print(f"{dataset} Dataset")
        print(f"{'='*80}")

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
    plot_path = os.path.join(analysis_output_dir, 'rating_variance_by_dataset.png')
    plot_variance_comparison_by_dataset(variance_data, plot_path)

    print(f"\n{'='*80}")
    print(f"Analysis Complete!")
    print(f"{'='*80}")
    print(f"Output directory: {analysis_output_dir}")
    print()


if __name__ == "__main__":
    main()
