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
    Collect confidence data from all rating files for Qwen3-8B and GPT-4.1.

    Returns:
        Dict with structure: {model_name: {rating_type: {metric: [confidences]}}}
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

            # Extract confidences for original ratings
            orig_confidences = extract_individual_rating_confidences(results, 'original')
            for metric in METRICS:
                data[model_name]['original'][metric].extend(orig_confidences[metric])

            # Extract confidences for perturbed ratings
            pert_confidences = extract_individual_rating_confidences(results, 'perturbed')
            for metric in METRICS:
                data[model_name]['perturbed'][metric].extend(pert_confidences[metric])

    return data


def plot_confidence_comparison(confidence_data, output_path):
    """
    Create bar plot comparing rating confidence between models and rating types.

    Shows mean confidence scores across all individual ratings for each model/type combination.
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

        # Calculate mean confidence scores
        orig_means = []
        orig_cis = []
        pert_means = []
        pert_cis = []

        for model in models:
            # Original ratings
            if model in confidence_data and len(confidence_data[model]['original'][metric]) > 0:
                confidences = confidence_data[model]['original'][metric]
                mean_conf = np.mean(confidences)
                ci = 1.96 * np.std(confidences) / np.sqrt(len(confidences))
                orig_means.append(mean_conf)
                orig_cis.append(ci)
            else:
                orig_means.append(0)
                orig_cis.append(0)

            # Perturbed ratings
            if model in confidence_data and len(confidence_data[model]['perturbed'][metric]) > 0:
                confidences = confidence_data[model]['perturbed'][metric]
                mean_conf = np.mean(confidences)
                ci = 1.96 * np.std(confidences) / np.sqrt(len(confidences))
                pert_means.append(mean_conf)
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

        # Add difference and significance markers
        for i, model in enumerate(models):
            if model in confidence_data:
                orig_data = confidence_data[model]['original'][metric]
                pert_data = confidence_data[model]['perturbed'][metric]

                if len(orig_data) > 0 and len(pert_data) > 0 and pert_means[i] > 0:
                    # Calculate difference (perturbed - original)
                    diff = pert_means[i] - orig_means[i]

                    # Perform Wilcoxon test (if we have paired data)
                    # Note: confidence data may not be naturally paired, so we use unpaired test
                    try:
                        _, p_val = stats.mannwhitneyu(orig_data, pert_data, alternative='two-sided')
                    except:
                        p_val = None

                    sig_marker = get_significance_marker(p_val)

                    # Add annotation
                    y_pos = pert_means[i] + pert_cis[i] + 0.15
                    ax.text(i + width/2, y_pos, f'{diff:.2f}{sig_marker}',
                           ha='center', va='bottom', fontsize=8, fontweight='bold')

        # Customize subplot
        ax.set_ylabel('Mean Confidence Score', fontsize=11)
        ax.set_title(f'{METRIC_LABELS[metric]}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        if idx == 0:  # Only show legend on first subplot
            ax.legend(fontsize=10, loc='lower right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(bottom=0, top=5.5)  # Confidence is on 1-5 scale

    # Overall title
    fig.suptitle('Rating Confidence Across 5 Runs - Model Comparison',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nSaved confidence comparison plot: {os.path.basename(output_path)}")


def print_summary_statistics(confidence_data):
    """Print summary statistics about rating confidence."""
    print("\n" + "="*80)
    print("RATING CONFIDENCE SUMMARY STATISTICS")
    print("="*80)

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
    plot_path = os.path.join(analysis_output_dir, 'rating_confidence_comparison.png')
    plot_confidence_comparison(confidence_data, plot_path)

    print(f"\n{'='*80}")
    print(f"Analysis Complete!")
    print(f"{'='*80}")
    print(f"Output directory: {analysis_output_dir}")
    print()


if __name__ == "__main__":
    main()
