"""
Analyze confidence scores in model ratings across the 5 runs.

Compares rating confidence between models (Qwen3-8B vs GPT-4.1) and
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

# Dataset directories to scan (only those with confidence scores)
DATASET_DIRS = {
    'CQA': os.path.join(project_root, 'output', 'cqa_eval', 'experiment_results', 'baseline'),
    'MedInfo': os.path.join(project_root, 'output', 'medinfo', 'experiment_results', 'baseline'),
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


def extract_individual_rating_confidences(results, rating_type='original'):
    """
    Extract confidence scores from individual ratings for each metric.

    Args:
        results: List of rating entries
        rating_type: 'original' or 'perturbed'

    Returns:
        Dict mapping metric -> list of confidence scores (flattened across all runs)
    """
    confidences = {metric: [] for metric in METRICS}

    for entry in results:
        rating_key = f'{rating_type}_rating'
        if rating_key not in entry:
            continue

        rating_data = entry[rating_key]

        # Check if individual_ratings exists
        if 'individual_ratings' not in rating_data:
            continue

        individual_ratings = rating_data['individual_ratings']

        # Extract confidence for each metric across the 5 runs
        for metric in METRICS:
            for run in individual_ratings:
                if metric in run and 'confidence' in run[metric]:
                    confidence = run[metric]['confidence']
                    confidences[metric].append(confidence)

    return confidences


def collect_all_confidences():
    """
    Collect confidence data from all rating files for Qwen3-8B and GPT-4.1 across all datasets.

    Returns:
        Dict with structure: {dataset_name: {model_name: {rating_type: {metric: [confidences]}}}}
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

                # Extract confidences for original ratings
                orig_confidences = extract_individual_rating_confidences(results, 'original')
                for metric in METRICS:
                    data_by_dataset[dataset_name][model_name]['original'][metric].extend(orig_confidences[metric])

                # Extract confidences for perturbed ratings
                pert_confidences = extract_individual_rating_confidences(results, 'perturbed')
                for metric in METRICS:
                    data_by_dataset[dataset_name][model_name]['perturbed'][metric].extend(pert_confidences[metric])

    return data_by_dataset


def plot_confidence_comparison_by_dataset(confidence_data_by_dataset, output_path):
    """
    Create 1x4 grid comparing rating confidence across datasets.

    Each column represents one dataset (CQA, MedInfo, RadEval, WoundCare).
    For datasets with multiple metrics, data is aggregated across all metrics.
    Note: Only CQA and MedInfo have confidence scores.
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
        confidence_data = confidence_data_by_dataset.get(dataset, {})
        ax = axes[dataset_idx]

        # Aggregate data across all metrics
        orig_means = []
        orig_cis = []
        pert_means = []
        pert_cis = []

        has_data = False

        for model in models:
            # Collect all confidence scores across all metrics for original ratings
            all_orig_confs = []
            for metric in METRICS:
                if model in confidence_data and len(confidence_data[model]['original'][metric]) > 0:
                    all_orig_confs.extend(confidence_data[model]['original'][metric])
                    has_data = True

            if len(all_orig_confs) > 0:
                mean_conf = np.mean(all_orig_confs)
                ci = 1.96 * np.std(all_orig_confs) / np.sqrt(len(all_orig_confs))
                orig_means.append(mean_conf)
                orig_cis.append(ci)
            else:
                orig_means.append(0)
                orig_cis.append(0)

            # Collect all confidence scores across all metrics for perturbed ratings
            all_pert_confs = []
            for metric in METRICS:
                if model in confidence_data and len(confidence_data[model]['perturbed'][metric]) > 0:
                    all_pert_confs.extend(confidence_data[model]['perturbed'][metric])
                    has_data = True

            if len(all_pert_confs) > 0:
                mean_conf = np.mean(all_pert_confs)
                ci = 1.96 * np.std(all_pert_confs) / np.sqrt(len(all_pert_confs))
                pert_means.append(mean_conf)
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
                    if model in confidence_data:
                        if len(confidence_data[model]['original'][metric]) > 0:
                            all_orig_data.extend(confidence_data[model]['original'][metric])
                        if len(confidence_data[model]['perturbed'][metric]) > 0:
                            all_pert_data.extend(confidence_data[model]['perturbed'][metric])

                if len(all_orig_data) > 0 and len(all_pert_data) > 0 and pert_means[i] > 0:
                    diff = pert_means[i] - orig_means[i]

                    try:
                        _, p_val = stats.mannwhitneyu(all_orig_data, all_pert_data, alternative='two-sided')
                    except:
                        p_val = None

                    sig_marker = get_significance_marker(p_val)

                    y_pos = pert_means[i] + pert_cis[i] + 0.15
                    ax.text(i + width/2, y_pos, f'{diff:.2f}{sig_marker}',
                           ha='center', va='bottom', fontsize=9, fontweight='bold')

            ax.set_xticks(x)
            ax.set_xticklabels(models, fontsize=11)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(bottom=0, top=5.5)  # Confidence is on 1-5 scale

        # Column title and y-label
        ax.set_title(f'{dataset}', fontsize=13, fontweight='bold')
        if dataset_idx == 0:
            ax.set_ylabel('Mean Confidence Score', fontsize=12)
            ax.legend(fontsize=10, loc='lower right')

    # Overall title
    fig.suptitle('Rating Confidence Across 5 Runs - By Dataset',
                 fontsize=15, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nSaved confidence comparison plot: {os.path.basename(output_path)}")


def print_summary_statistics(confidence_data_by_dataset):
    """Print summary statistics about rating confidence by dataset."""
    print("\n" + "="*80)
    print("RATING CONFIDENCE SUMMARY STATISTICS BY DATASET")
    print("="*80)

    for dataset in ['CQA', 'MedInfo', 'RadEval', 'WoundCare']:
        if dataset not in confidence_data_by_dataset:
            continue

        confidence_data = confidence_data_by_dataset[dataset]

        print(f"\n{'='*80}")
        print(f"{dataset} Dataset")
        print(f"{'='*80}")

        for model in ['Qwen3-8B', 'GPT-4.1']:
            if model not in confidence_data:
                continue

            print(f"\n{model}:")
            print("-" * 80)

            for rating_type in ['original', 'perturbed']:
                print(f"\n  {rating_type.title()} Ratings:")

                for metric in METRICS:
                    confidences = confidence_data[model][rating_type][metric]
                    if len(confidences) == 0:
                        print(f"    {METRIC_LABELS[metric]}: No data")
                        continue

                    mean_conf = np.mean(confidences)
                    median_conf = np.median(confidences)
                    std_conf = np.std(confidences)
                    min_conf = np.min(confidences)
                    max_conf = np.max(confidences)
                    n_ratings = len(confidences)

                    # Calculate distribution of confidence levels
                    conf_dist = {}
                    for c in sorted(set(confidences)):
                        count = np.sum(np.array(confidences) == c)
                        pct = (count / len(confidences)) * 100
                        conf_dist[c] = pct

                    print(f"    {METRIC_LABELS[metric]}:")
                    print(f"      Mean: {mean_conf:.3f}, Median: {median_conf:.1f}")
                    print(f"      Std: {std_conf:.3f}")
                    print(f"      Min: {min_conf:.1f}, Max: {max_conf:.1f}")
                    print(f"      N ratings: {n_ratings}")
                    print(f"      Distribution: {', '.join([f'{k}:{v:.1f}%' for k, v in conf_dist.items()])}")


def main():
    print("\n" + "="*80)
    print("Rating Confidence Analysis")
    print("="*80)

    # Collect confidence data from all files
    print("\nCollecting confidence data from rating files...")
    confidence_data = collect_all_confidences()

    # Print summary statistics
    print_summary_statistics(confidence_data)

    # Generate comparison plot
    print("\n" + "="*80)
    print("Generating Confidence Comparison Plot")
    print("="*80)
    plot_path = os.path.join(analysis_output_dir, 'rating_confidence_by_dataset.png')
    plot_confidence_comparison_by_dataset(confidence_data, plot_path)

    print(f"\n{'='*80}")
    print(f"Analysis Complete!")
    print(f"{'='*80}")
    print(f"Output directory: {analysis_output_dir}")
    print()


if __name__ == "__main__":
    main()
