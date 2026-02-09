"""
Analyze WoundCare evaluation results from baseline experiments.

Compares original vs perturbed ratings across different perturbation types and models.
Generates statistics and visualizations for medical advice quality scores.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
output_base = os.path.join(project_root, 'output', 'woundcare', 'experiment_results', 'baseline')
analysis_output_dir = os.path.join(project_root, 'output', 'woundcare', 'analysis')
os.makedirs(analysis_output_dir, exist_ok=True)


def load_woundcare_results(filepath):
    """Load WoundCare results from JSONL file."""
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            entry = json.loads(line)
            results.append(entry)
    return results


def merge_original_and_perturbed(perturbed_results, original_filepath):
    """Merge perturbed results with original ratings from a separate file."""
    # Load original ratings
    original_results = load_woundcare_results(original_filepath)

    # Create a lookup dictionary for original ratings
    # Key: question_id
    original_lookup = {}
    for entry in original_results:
        key = entry['question_id']
        original_lookup[key] = entry.get('original_rating')

    # Merge with perturbed results
    merged_results = []
    for entry in perturbed_results:
        key = entry['question_id']

        if key in original_lookup and original_lookup[key] is not None:
            # Create a new entry with both ratings
            merged_entry = entry.copy()
            merged_entry['original_rating'] = original_lookup[key]
            merged_results.append(merged_entry)

    return merged_results


def extract_scores(results):
    """Extract original and perturbed avg_rating scores."""
    original_scores = []
    perturbed_scores = []

    for entry in results:
        if 'original_rating' in entry and 'perturbed_rating' in entry:
            orig = entry['original_rating'].get('avg_rating')
            pert = entry['perturbed_rating'].get('avg_rating')

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


def plot_comparison(results_by_model, level, perturbation, output_path):
    """Create bar plot comparing original vs perturbed ratings across models."""
    if len(results_by_model) == 0:
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get unique models
    models = sorted(list(results_by_model.keys()))
    x = np.arange(len(models))
    width = 0.35

    # Extract data for each model
    orig_means = []
    orig_cis = []
    pert_means = []
    pert_cis = []
    differences = []
    p_values = []

    for model in models:
        data = results_by_model[model]

        # Calculate means and confidence intervals
        orig_scores = data['original']
        pert_scores = data['perturbed']

        orig_mean = np.mean(orig_scores) if len(orig_scores) > 0 else 0
        orig_ci = 1.96 * np.std(orig_scores) / np.sqrt(len(orig_scores)) if len(orig_scores) > 0 else 0
        pert_mean = np.mean(pert_scores) if len(pert_scores) > 0 else 0
        pert_ci = 1.96 * np.std(pert_scores) / np.sqrt(len(pert_scores)) if len(pert_scores) > 0 else 0

        orig_means.append(orig_mean)
        orig_cis.append(orig_ci)
        pert_means.append(pert_mean)
        pert_cis.append(pert_ci)

        # Calculate difference and get p-value
        diff = pert_mean - orig_mean  # perturbed - original
        differences.append(diff)
        p_values.append(data['stats']['wilcoxon_p_value'])

    # Plot bars
    ax.bar(x - width/2, orig_means, width, yerr=orig_cis,
           label='Original', capsize=5, alpha=0.8, color='#1f77b4')
    ax.bar(x + width/2, pert_means, width, yerr=pert_cis,
           label='Perturbed', capsize=5, alpha=0.8, color='#ff7f0e')

    # Add difference and significance markers
    for i, (diff, p_val, pert_mean, pert_ci) in enumerate(zip(differences, p_values, pert_means, pert_cis)):
        if pert_mean > 0:
            y_pos = max(orig_means[i] + orig_cis[i], pert_mean + pert_ci) + 0.05
            sig_marker = get_significance_marker(p_val)
            text = f'{diff:.3f}{sig_marker}'
            ax.text(x[i], y_pos, text,
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Customize plot
    ax.set_ylabel('Avg Rating (0-1 scale)', fontsize=12)
    ax.set_title(f'{perturbation.replace("_", " ").title()} - Original vs Perturbed ({level.title()} Level)',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(bottom=0, top=1.1)  # Ratings are 0-1 scale

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved plot: {os.path.basename(output_path)}")


def generate_summary_report(results_by_model, level, perturbation, output_path):
    """Generate text summary report with Wilcoxon signed-rank test results."""
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"WoundCare Evaluation Results - Wilcoxon Signed-Rank Test\n")
        f.write(f"Level: {level.title()}\n")
        f.write(f"Perturbation: {perturbation}\n")
        f.write("=" * 80 + "\n\n")

        for model, data in results_by_model.items():
            f.write(f"\n{'='*80}\n")
            f.write(f"Model: {model}\n")
            f.write(f"{'='*80}\n")

            stats_data = data['stats']
            f.write(f"Number of samples: {stats_data['n_samples']}\n\n")

            f.write(f"Original Mean: {stats_data['original_mean']:.4f} ± {stats_data['original_std']:.4f}\n")
            f.write(f"Perturbed Mean: {stats_data['perturbed_mean']:.4f} ± {stats_data['perturbed_std']:.4f}\n")
            f.write(f"Mean Degradation: {stats_data['mean_degradation']:.4f} ± {stats_data['std_degradation']:.4f}\n\n")

            f.write(f"Percentage Decreased: {stats_data['pct_decreased']:.2f}%\n")
            f.write(f"Percentage Increased: {stats_data['pct_increased']:.2f}%\n")
            f.write(f"Percentage Unchanged: {stats_data['pct_unchanged']:.2f}%\n\n")

            if stats_data['wilcoxon_p_value'] is not None:
                significance = "***" if stats_data['wilcoxon_p_value'] < 0.001 else \
                             "**" if stats_data['wilcoxon_p_value'] < 0.01 else \
                             "*" if stats_data['wilcoxon_p_value'] < 0.05 else "ns"

                f.write(f"Wilcoxon Test: ")
                f.write(f"W={stats_data['wilcoxon_statistic']:.2f}, ")
                f.write(f"p={stats_data['wilcoxon_p_value']:.4e} {significance}\n")

            f.write("\n")

    print(f"  Saved summary: {os.path.basename(output_path)}")


def main():
    print("\n" + "="*80)
    print("WoundCare Results Analysis")
    print("="*80)

    # Process each perturbation
    perturbations = ['swap_infection', 'swap_time_frequency']
    levels = ['coarse']  # WoundCare only has coarse level

    for level in levels:
        for perturbation in perturbations:
            perturbation_dir = os.path.join(output_base, perturbation)
            if not os.path.exists(perturbation_dir):
                continue

            print(f"\n{'='*80}")
            print(f"Processing: {perturbation} ({level})")
            print(f"{'='*80}")

            results_by_model = {}

            # Look for rating files for this level
            for filename in os.listdir(perturbation_dir):
                if not filename.endswith('_rating.jsonl'):
                    continue
                if level not in filename:
                    continue

                filepath = os.path.join(perturbation_dir, filename)

                # Extract model name from filename
                # Format: perturbation_level_model_rating.jsonl
                parts = filename.replace('_rating.jsonl', '').split('_')
                # Skip perturbation name and level
                level_idx = parts.index(level)
                model = '_'.join(parts[level_idx + 1:])

                print(f"\nLoading: {filename}")
                results = load_woundcare_results(filepath)
                print(f"  Loaded {len(results)} entries")

                # Check if we need to merge with original ratings from a separate file
                if len(results) > 0 and 'original_rating' not in results[0]:
                    # Look for corresponding original ratings file
                    original_ratings_dir = os.path.join(project_root, 'output', 'woundcare', 'original_ratings')
                    original_filename = f'original_{level}_{model}_rating.jsonl'
                    original_filepath = os.path.join(original_ratings_dir, original_filename)

                    if os.path.exists(original_filepath):
                        print(f"  Merging with original ratings from: {original_filename}")
                        results = merge_original_and_perturbed(results, original_filepath)
                        print(f"  Merged {len(results)} entries with original ratings")
                    else:
                        print(f"  Warning: No original ratings found at {original_filename}")

                # Extract scores
                orig_scores, pert_scores = extract_scores(results)
                print(f"  Extracted {len(orig_scores)} valid score pairs")

                if len(orig_scores) == 0:
                    print(f"  No valid scores found, skipping...")
                    continue

                # Compute statistics
                stats_data = compute_statistics(orig_scores, pert_scores)

                # Store results
                results_by_model[model] = {
                    'filename': filename,
                    'original': orig_scores,
                    'perturbed': pert_scores,
                    'stats': stats_data
                }

                # Print quick summary
                print(f"  Original: {stats_data['original_mean']:.4f} ± {stats_data['original_std']:.4f}")
                print(f"  Perturbed: {stats_data['perturbed_mean']:.4f} ± {stats_data['perturbed_std']:.4f}")
                print(f"  Degradation: {stats_data['mean_degradation']:.4f}")
                sig_marker = get_significance_marker(stats_data['wilcoxon_p_value'])
                print(f"  Wilcoxon p-value: {stats_data['wilcoxon_p_value']:.4e} {sig_marker}")

            if not results_by_model:
                print(f"  No result files found for {perturbation} ({level})")
                continue

            print(f"\n{'='*80}")
            print(f"Generating Analysis Outputs")
            print(f"{'='*80}")

            # Generate comparison plot
            print("\nGenerating comparison plot...")
            plot_path = os.path.join(analysis_output_dir,
                                    f'{perturbation}_{level}_comparison.png')
            plot_comparison(results_by_model, level, perturbation, plot_path)

            # Generate summary report
            print("\nGenerating summary report...")
            report_path = os.path.join(analysis_output_dir,
                                      f'{perturbation}_{level}_summary_report.txt')
            generate_summary_report(results_by_model, level, perturbation, report_path)

    print(f"\n{'='*80}")
    print(f"Analysis Complete!")
    print(f"{'='*80}")
    print(f"Output directory: {analysis_output_dir}")
    print()


if __name__ == "__main__":
    main()
