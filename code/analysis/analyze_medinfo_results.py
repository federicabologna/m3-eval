"""
Analyze CQA evaluation results from baseline experiments.

Compares original vs perturbed ratings across different perturbation types and models.
Generates statistics and visualizations for correctness, relevance, and safety scores.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
output_base = os.path.join(project_root, 'output', 'medinfo', 'experiment_results', 'baseline')
analysis_output_dir = os.path.join(project_root, 'output', 'medinfo', 'analysis')
os.makedirs(analysis_output_dir, exist_ok=True)

# Metrics to analyze
METRICS = ['correctness', 'relevance', 'safety']

METRIC_LABELS = {
    'correctness': 'Correctness',
    'relevance': 'Relevance',
    'safety': 'Safety'
}


def load_cqa_results(filepath):
    """Load CQA results from JSONL file."""
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
    original_results = load_cqa_results(original_filepath)

    # Create a lookup dictionary for original ratings
    # Key: (question_id, answer_id, sentence_id) for fine-grained or (question_id, answer_id) for coarse
    original_lookup = {}
    for entry in original_results:
        if 'sentence_id' in entry:
            # Fine-grained
            key = (entry['question_id'], entry['answer_id'], entry['sentence_id'])
        else:
            # Coarse-grained
            key = (entry['question_id'], entry['answer_id'])
        original_lookup[key] = entry.get('original_rating')

    # Merge with perturbed results
    merged_results = []
    for entry in perturbed_results:
        if 'sentence_id' in entry:
            key = (entry['question_id'], entry['answer_id'], entry['sentence_id'])
        else:
            key = (entry['question_id'], entry['answer_id'])

        if key in original_lookup and original_lookup[key] is not None:
            # Create a new entry with both ratings
            merged_entry = entry.copy()
            merged_entry['original_rating'] = original_lookup[key]
            merged_results.append(merged_entry)

    return merged_results


def extract_scores(results, metric):
    """Extract original and perturbed scores for a specific metric."""
    original_scores = []
    perturbed_scores = []

    for entry in results:
        if 'original_rating' in entry and 'perturbed_rating' in entry:
            orig = entry['original_rating'].get(metric, {}).get('score')
            pert = entry['perturbed_rating'].get(metric, {}).get('score')

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


def plot_medinfo_comparison(results_by_model, level, perturbation, output_path):
    """Create 1x3 grid for model answers only (no physician data in medinfo)."""
    if len(results_by_model) == 0:
        return

    # Create 1x3 subplot grid (1 row, 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Get unique evaluation models
    eval_models = set()
    for data in results_by_model.values():
        eval_models.add(data['eval_model'])
    eval_models = sorted(list(eval_models))
    x = np.arange(len(eval_models))

    width = 0.35
    offsets = [-width/2, width/2]  # Original, Perturbed

    # Plot model answers (single row)
    for idx, metric in enumerate(METRICS):
        ax = axes[idx]

        # Original scores
        orig_means = []
        orig_cis = []
        for eval_model in eval_models:
            data = None
            for key, d in results_by_model.items():
                if d['eval_model'] == eval_model:
                    data = d
                    break

            if data is None:
                orig_means.append(0)
                orig_cis.append(0)
                continue

            orig_scores = data['original'][metric]
            mean = np.mean(orig_scores) if len(orig_scores) > 0 else 0
            ci = 1.96 * np.std(orig_scores) / np.sqrt(len(orig_scores)) if len(orig_scores) > 0 else 0
            orig_means.append(mean)
            orig_cis.append(ci)

        # Plot original scores
        ax.bar(x + offsets[0], orig_means, width, yerr=orig_cis,
               label='Original', capsize=3, alpha=0.8, color='#1f77b4')

        # Perturbed scores
        pert_means = []
        pert_cis = []
        differences = []
        p_values = []

        for i, eval_model in enumerate(eval_models):
            data = None
            for key, d in results_by_model.items():
                if d['eval_model'] == eval_model:
                    data = d
                    break

            if data:
                pert_scores = data['perturbed'][metric]
                mean = np.mean(pert_scores) if len(pert_scores) > 0 else 0
                ci = 1.96 * np.std(pert_scores) / np.sqrt(len(pert_scores)) if len(pert_scores) > 0 else 0
                pert_means.append(mean)
                pert_cis.append(ci)

                # Calculate difference and get p-value
                diff = mean - orig_means[i]  # perturbed - original
                differences.append(diff)
                p_values.append(data['stats'][metric]['wilcoxon_p_value'])
            else:
                pert_means.append(0)
                pert_cis.append(0)
                differences.append(0)
                p_values.append(None)

        ax.bar(x + offsets[1], pert_means, width, yerr=pert_cis,
               label='Perturbed', capsize=3, alpha=0.8, color='#ff7f0e')

        # Add difference and significance markers
        for i, (diff, p_val, pert_mean, pert_ci) in enumerate(zip(differences, p_values, pert_means, pert_cis)):
            if pert_mean > 0:
                y_pos = pert_mean + pert_ci + 0.15
                sig_marker = get_significance_marker(p_val)
                text = f'{diff:.2f}{sig_marker}'
                ax.text(x[i] + offsets[1], y_pos, text,
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Customize subplot
        ax.set_ylabel(f'{METRIC_LABELS[metric]} Score', fontsize=11)
        ax.set_title(f'{METRIC_LABELS[metric]}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(eval_models, rotation=15, ha='right')
        if idx == 0:  # Only show legend on first subplot
            ax.legend(fontsize=10, loc='lower left')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(bottom=0, top=5.5)  # Scores are 1-5 scale

    # Overall title
    fig.suptitle(f'{perturbation.replace("_", " ").title()} - Original vs Perturbed ({level.title()} Level)',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved plot: {os.path.basename(output_path)}")


def generate_summary_report(results_by_model, level, perturbation, output_path):
    """Generate text summary report with Wilcoxon signed-rank test results."""
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"CQA Evaluation Results - Wilcoxon Signed-Rank Test\n")
        f.write(f"Level: {level.title()}\n")
        f.write(f"Perturbation: {perturbation}\n")
        f.write("=" * 80 + "\n\n")

        for key, data in results_by_model.items():
            f.write(f"\n{'='*80}\n")
            eval_model = data['eval_model']
            prob = data.get('prob')
            if prob:
                f.write(f"Model: {eval_model} (p={float(prob)/10})\n")
            else:
                f.write(f"Model: {eval_model}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Number of samples: {data['stats']['correctness']['n_samples']}\n\n")

            for metric in METRICS:
                stats_data = data['stats'][metric]

                if stats_data['wilcoxon_p_value'] is not None:
                    significance = "***" if stats_data['wilcoxon_p_value'] < 0.001 else \
                                 "**" if stats_data['wilcoxon_p_value'] < 0.01 else \
                                 "*" if stats_data['wilcoxon_p_value'] < 0.05 else "ns"

                    f.write(f"{METRIC_LABELS[metric]}: ")
                    f.write(f"W={stats_data['wilcoxon_statistic']:.2f}, ")
                    f.write(f"p={stats_data['wilcoxon_p_value']:.4e} {significance}\n")

            f.write("\n")

    print(f"  Saved summary: {os.path.basename(output_path)}")


def main():
    print("\n" + "="*80)
    print("CQA Results Analysis")
    print("="*80)

    # Process each perturbation
    perturbations = ['change_dosage']#, 'remove_sentences', 'add_typos', 'add_confusion']
    levels = ['coarse', 'fine']

    for level in levels:
        for perturbation in perturbations:
            perturbation_dir = os.path.join(output_base, perturbation)
            if not os.path.exists(perturbation_dir):
                continue

            print(f"\n{'='*80}")
            print(f"Processing: {perturbation} ({level})")
            print(f"{'='*80}")

            results_by_model = {}
            results_by_model_physician = {}

            # Look for rating files for this level
            for filename in os.listdir(perturbation_dir):
                if not filename.endswith('_rating.jsonl'):
                    continue
                if level not in filename:
                    continue

                filepath = os.path.join(perturbation_dir, filename)

                # Extract model name and probability (for add_typos) from filename
                # Format: perturbation_level_model_rating.jsonl
                parts = filename.replace('_rating.jsonl', '').split('_')
                prob = None

                # Skip perturbation name and level
                if perturbation == 'change_dosage':
                    model = '_'.join(parts[3:])  # change_dosage_level_model
                elif perturbation == 'remove_sentences':
                    # May have percentage: remove_sentences_0.3removed_level_model
                    # Skip to after level
                    level_idx = parts.index(level)
                    model = '_'.join(parts[level_idx + 1:])
                elif perturbation == 'add_typos':
                    # May have prob: add_typos_05prob_level_model
                    # Extract probability
                    for part in parts:
                        if 'prob' in part:
                            prob = part.replace('prob', '')
                            break
                    level_idx = parts.index(level)
                    model = '_'.join(parts[level_idx + 1:])
                else:  # add_confusion
                    model = '_'.join(parts[3:])

                print(f"\nLoading: {filename}")
                results = load_cqa_results(filepath)
                print(f"  Loaded {len(results)} entries")

                # Check if we need to merge with original ratings from a separate file
                if len(results) > 0 and 'original_rating' not in results[0]:
                    # Look for corresponding original ratings file
                    original_ratings_dir = os.path.join(project_root, 'output', 'cqa_eval', 'original_ratings')
                    original_filename = f'original_{level}_{model}_rating.jsonl'
                    original_filepath = os.path.join(original_ratings_dir, original_filename)

                    if os.path.exists(original_filepath):
                        print(f"  Merging with original ratings from: {original_filename}")
                        results = merge_original_and_perturbed(results, original_filepath)
                        print(f"  Merged {len(results)} entries with original ratings")
                    else:
                        print(f"  Warning: No original ratings found at {original_filename}")

                # Separate physician and model answers
                physician_results = [r for r in results if r.get('answer_type') == 'physician']
                model_results = [r for r in results if r.get('answer_type') != 'physician']

                print(f"  Physician answers: {len(physician_results)}")
                print(f"  Model answers: {len(model_results)}")

                # Process model answers
                if len(model_results) > 0:
                    # Extract scores for all metrics
                    metric_data = {}
                    for metric in METRICS:
                        orig, pert = extract_scores(model_results, metric)
                        metric_data[metric] = {
                            'original': orig,
                            'perturbed': pert
                        }

                    # Compute statistics for all metrics
                    stats_data = {}
                    for metric in METRICS:
                        stats_data[metric] = compute_statistics(
                            metric_data[metric]['original'],
                            metric_data[metric]['perturbed']
                        )

                    # Store results (key by model_prob for add_typos, just model otherwise)
                    key = f"{model}_{prob}" if prob else model
                    results_by_model[key] = {
                        'filename': filename,
                        'eval_model': model,
                        'prob': prob,
                        'original': {m: metric_data[m]['original'] for m in METRICS},
                        'perturbed': {m: metric_data[m]['perturbed'] for m in METRICS},
                        'stats': stats_data
                    }

                # Process physician answers
                if len(physician_results) > 0:
                    # Extract scores for all metrics
                    metric_data = {}
                    for metric in METRICS:
                        orig, pert = extract_scores(physician_results, metric)
                        metric_data[metric] = {
                            'original': orig,
                            'perturbed': pert
                        }

                    # Compute statistics for all metrics
                    stats_data = {}
                    for metric in METRICS:
                        stats_data[metric] = compute_statistics(
                            metric_data[metric]['original'],
                            metric_data[metric]['perturbed']
                        )

                    # Store results (key by model_prob for add_typos, just model otherwise)
                    key = f"{model}_{prob}" if prob else model
                    results_by_model_physician[key] = {
                        'filename': filename,
                        'eval_model': model,
                        'prob': prob,
                        'original': {m: metric_data[m]['original'] for m in METRICS},
                        'perturbed': {m: metric_data[m]['perturbed'] for m in METRICS},
                        'stats': stats_data
                    }

            if not results_by_model and not results_by_model_physician:
                print(f"  No result files found for {perturbation} ({level})")
                continue

            print(f"\n{'='*80}")
            print(f"Generating Analysis Outputs")
            print(f"{'='*80}")

            # Generate comparison plot (models only)
            if perturbation == 'change_dosage':
                print("\nGenerating comparison plot...")
                plot_path = os.path.join(analysis_output_dir,
                                        f'{perturbation}_{level}_comparison.png')
                plot_medinfo_comparison(results_by_model, level, perturbation, plot_path)

            # Generate summary reports
            if results_by_model:
                print("\nGenerating summary report (models)...")
                report_path = os.path.join(analysis_output_dir,
                                          f'{perturbation}_{level}_models_summary_report.txt')
                generate_summary_report(results_by_model, level, perturbation + ' (Models)', report_path)

            if results_by_model_physician:
                print("\nGenerating summary report (physician)...")
                report_path = os.path.join(analysis_output_dir,
                                          f'{perturbation}_{level}_physician_summary_report.txt')
                generate_summary_report(results_by_model_physician, level, perturbation + ' (Physician)', report_path)

    # Generate coarse vs fine comparison for change_dosage and qwen3-8b
    # Disabled: no physician answers in medinfo dataset
    # print(f"\n{'='*80}")
    # print(f"Generating Coarse vs Fine Comparison for change_dosage (Qwen3-8B)")
    # print(f"{'='*80}")
    #
    # # Load data for both levels
    # coarse_physician = None
    # coarse_model = None
    # fine_physician = None
    # fine_model = None
    #
    # perturbation = 'change_dosage'
    # perturbation_dir = os.path.join(output_base, perturbation)
    # target_model = 'Qwen3-8B'
    #
    # if os.path.exists(perturbation_dir):
    #     for level in ['coarse', 'fine']:
    #         for filename in os.listdir(perturbation_dir):
    #             if not filename.endswith('_rating.jsonl'):
    #                 continue
    #             if level not in filename:
    #                 continue
    #
    #             # Check if it's for Qwen3-8B (case-insensitive)
    #             if 'qwen3' not in filename.lower() or '8b' not in filename.lower():
    #                 continue
    #
    #             filepath = os.path.join(perturbation_dir, filename)
    #             print(f"\nLoading {level} data: {filename}")
    #             results = load_cqa_results(filepath)
    #
    #             # Separate physician and model answers
    #             physician_results = [r for r in results if r.get('answer_type') == 'physician']
    #             model_results = [r for r in results if r.get('answer_type') != 'physician']
    #
    #             # Process model answers
    #             if len(model_results) > 0:
    #                 metric_data = {}
    #                 for metric in METRICS:
    #                     orig, pert = extract_scores(model_results, metric)
    #                     metric_data[metric] = {
    #                         'original': orig,
    #                         'perturbed': pert
    #                     }
    #
    #                 data = {
    #                     'original': {m: metric_data[m]['original'] for m in METRICS},
    #                     'perturbed': {m: metric_data[m]['perturbed'] for m in METRICS}
    #                 }
    #
    #                 if level == 'coarse':
    #                     coarse_model = data
    #                 else:
    #                     fine_model = data
    #
    #             # Process physician answers
    #             if len(physician_results) > 0:
    #                 metric_data = {}
    #                 for metric in METRICS:
    #                     orig, pert = extract_scores(physician_results, metric)
    #                     metric_data[metric] = {
    #                         'original': orig,
    #                         'perturbed': pert
    #                     }
    #
    #                 data = {
    #                     'original': {m: metric_data[m]['original'] for m in METRICS},
    #                     'perturbed': {m: metric_data[m]['perturbed'] for m in METRICS}
    #                 }
    #
    #                 if level == 'coarse':
    #                     coarse_physician = data
    #                 else:
    #                     fine_physician = data
    #
    #     # Generate the comparison plot
    #     if any([coarse_physician, coarse_model, fine_physician, fine_model]):
    #         plot_path = os.path.join(analysis_output_dir,
    #                                 'change_dosage_coarse_vs_fine_qwen3_8b.png')
    #         plot_coarse_vs_fine_comparison(coarse_physician, coarse_model,
    #                                       fine_physician, fine_model,
    #                                       target_model, plot_path)
    #     else:
    #         print("  No data found for Qwen3-8B")
    # else:
    #     print(f"  Directory not found: {perturbation_dir}")

    print(f"\n{'='*80}")
    print(f"Analysis Complete!")
    print(f"{'='*80}")
    print(f"Output directory: {analysis_output_dir}")
    print()


if __name__ == "__main__":
    main()
