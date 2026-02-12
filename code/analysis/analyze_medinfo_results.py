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
from difflib import SequenceMatcher

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


def load_detection_results(results_file):
    """Load error detection results from file."""
    results = []
    with open(results_file, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def compute_similarity(str1, str2):
    """Compute similarity between two strings using SequenceMatcher (0-1 scale)."""
    return SequenceMatcher(None, str1.lower().strip(), str2.lower().strip()).ratio()


def analyze_detection_accuracy(results, is_original=False, similarity_threshold=0.6):
    """
    Analyze error detection accuracy for MedInfo with sentence matching.

    Args:
        results: List of detection results
        is_original: If True, analyzing original (clean) reports
        similarity_threshold: Minimum similarity to consider a sentence match (0-1)

    Returns:
        Dictionary with accuracy metrics
    """
    stats = {
        'total': 0,
        'detected': 0,
        'not_detected': 0,
        'detection_errors': 0,
        'reports_with_0_errors': 0,
        'reports_with_1_error': 0,
        'reports_with_2plus_errors': 0,
        # Sentence matching stats (only for perturbed)
        'correct_sentence_identified': 0,
        'wrong_sentence_identified': 0,
        'detected_but_no_match': 0,
    }

    for result in results:
        stats['total'] += 1

        # Check if error was detected
        detection_result = result.get('detection_result', {})
        detected = detection_result.get('detected', 'no').lower()

        if detected == 'yes':
            stats['detected'] += 1
        elif detected == 'no':
            stats['not_detected'] += 1
        else:
            stats['detection_errors'] += 1

        # Count number of errors claimed
        errors_list = detection_result.get('errors', [])
        num_errors = len(errors_list)

        if num_errors == 0:
            stats['reports_with_0_errors'] += 1
        elif num_errors == 1:
            stats['reports_with_1_error'] += 1
        else:
            stats['reports_with_2plus_errors'] += 1

        # For perturbed reports, check if the correct sentence was identified
        if not is_original and detected == 'yes':
            changes_detail = result.get('changes_detail', [])
            if changes_detail and errors_list:
                # Get ground truth error sentences (the modified sentences)
                gt_sentences = [change.get('modified', '') for change in changes_detail]

                # Get predicted error sentences
                pred_sentences = [error.get('incorrect_sentence', '') for error in errors_list]

                # Check if any predicted sentence matches any ground truth sentence
                best_match_score = 0
                for pred_sent in pred_sentences:
                    if pred_sent:  # Skip empty predictions
                        for gt_sent in gt_sentences:
                            if gt_sent:  # Skip empty ground truth
                                similarity = compute_similarity(pred_sent, gt_sent)
                                best_match_score = max(best_match_score, similarity)

                # Classify based on best match
                if best_match_score >= similarity_threshold:
                    stats['correct_sentence_identified'] += 1
                else:
                    stats['wrong_sentence_identified'] += 1
            elif detected == 'yes' and not errors_list:
                # Detected but provided no error details
                stats['detected_but_no_match'] += 1

    return stats


def create_error_detection_plot(all_stats, output_dir):
    """Create combined error detection plot for MedInfo showing detection rates and sentence matching."""
    if not all_stats:
        return

    perturbation_order = ['inject_critical_error', 'inject_noncritical_error']
    perturbation_labels = {
        'inject_critical_error': 'Critical Error',
        'inject_noncritical_error': 'Non-Critical Error'
    }

    # Filter to only perturbations we have
    perturbations = [p for p in perturbation_order if p in all_stats]

    if not perturbations:
        return

    # Create figure with 2 subplots: perturbed detection rates and original false positives
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('MedInfo Error Detection Analysis',
                 fontsize=16, fontweight='bold')

    # Colors with edges
    color_correct = '#4A7C59'        # Green - detected + correct sentence
    color_wrong = '#CC8844'          # Orange - detected + wrong sentence
    color_not_detected = '#A85252'   # Red - not detected
    edge_color = 'white'
    edge_width = 2

    x = np.arange(len(perturbations))
    width = 0.6

    # LEFT PLOT: Perturbed reports (detection + sentence matching)
    for idx, perturbation in enumerate(perturbations):
        if perturbation in all_stats:
            stats = all_stats[perturbation]['perturbed']
            total = stats['total']
            if total > 0:
                correct_pct = stats['correct_sentence_identified'] / total * 100
                wrong_pct = stats['wrong_sentence_identified'] / total * 100
                not_detected_pct = stats['not_detected'] / total * 100
            else:
                correct_pct = wrong_pct = not_detected_pct = 0

            # Stack bars with edges
            ax1.bar(x[idx], correct_pct, width,
                   label='Correct Sentence' if idx == 0 else '',
                   color=color_correct, edgecolor=edge_color, linewidth=edge_width)
            ax1.bar(x[idx], wrong_pct, width, bottom=correct_pct,
                   label='Wrong Sentence' if idx == 0 else '',
                   color=color_wrong, edgecolor=edge_color, linewidth=edge_width)
            ax1.bar(x[idx], not_detected_pct, width, bottom=correct_pct + wrong_pct,
                   label='Not Detected' if idx == 0 else '',
                   color=color_not_detected, edgecolor=edge_color, linewidth=edge_width)

            # Add percentage labels on bars
            if correct_pct > 5:
                ax1.text(x[idx], correct_pct/2, f'{correct_pct:.1f}%',
                        ha='center', va='center', fontweight='bold', fontsize=10, color='white')
            if wrong_pct > 5:
                ax1.text(x[idx], correct_pct + wrong_pct/2, f'{wrong_pct:.1f}%',
                        ha='center', va='center', fontweight='bold', fontsize=10)

    ax1.set_ylabel('Percentage of Reports (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Perturbed Reports (Detection + Sentence Matching)', fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels([perturbation_labels.get(p, p) for p in perturbations], fontsize=11)
    ax1.set_ylim(0, 115)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # RIGHT PLOT: Original reports (false positives)
    color_0 = '#4A7C59'    # Green - correct (0 errors)
    color_1 = '#CC8844'    # Orange - 1 error claimed
    color_2plus = '#A85252'  # Red - 2+ errors claimed

    for idx, perturbation in enumerate(perturbations):
        if perturbation in all_stats and all_stats[perturbation].get('original'):
            stats_orig = all_stats[perturbation]['original']
            total = stats_orig['total']
            if total > 0:
                pct_0 = stats_orig['reports_with_0_errors'] / total * 100
                pct_1 = stats_orig['reports_with_1_error'] / total * 100
                pct_2plus = stats_orig['reports_with_2plus_errors'] / total * 100
            else:
                pct_0 = pct_1 = pct_2plus = 0

            ax2.bar(x[idx], pct_0, width,
                   label='0 Errors (Correct)' if idx == 0 else '',
                   color=color_0, edgecolor=edge_color, linewidth=edge_width)
            ax2.bar(x[idx], pct_1, width, bottom=pct_0,
                   label='1 Error Claimed' if idx == 0 else '',
                   color=color_1, edgecolor=edge_color, linewidth=edge_width)
            ax2.bar(x[idx], pct_2plus, width, bottom=pct_0 + pct_1,
                   label='2+ Errors Claimed' if idx == 0 else '',
                   color=color_2plus, edgecolor=edge_color, linewidth=edge_width)

            # Add percentage label for correct (0 errors)
            if pct_0 > 5:
                ax2.text(x[idx], pct_0/2, f'{pct_0:.1f}%',
                        ha='center', va='center', fontweight='bold', fontsize=11)

    ax2.set_ylabel('Percentage of Reports (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Original Reports (False Positives)', fontsize=13, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels([perturbation_labels.get(p, p) for p in perturbations], fontsize=11)
    ax2.set_ylim(0, 115)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'medinfo_error_detection_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved error detection plot: {output_path}")
    plt.close()


def main():
    print("\n" + "="*80)
    print("CQA Results Analysis")
    print("="*80)

    # Process each perturbation (LLM-based only)
    perturbations = ['inject_critical_error', 'inject_noncritical_error']
    levels = ['coarse']  # Only coarse level for LLM-based injections

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

                # Extract model name from filename
                # Format: inject_critical_error_coarse_model_rating.jsonl
                # or: inject_noncritical_error_coarse_model_rating.jsonl
                parts = filename.replace('_rating.jsonl', '').split('_')

                # For LLM-based injections: perturbation_type_error_level_model
                # inject_critical_error = 3 parts, inject_noncritical_error = 3 parts
                if perturbation in ['inject_critical_error', 'inject_noncritical_error']:
                    # Skip perturbation name (3 parts), level (1 part), rest is model
                    model = '_'.join(parts[4:])  # inject_critical/noncritical_error_level_model
                    prob = None  # No probability parameter for LLM-based injections
                else:
                    model = '_'.join(parts[3:])
                    prob = None  # Default to None if not extracted

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
            if results_by_model:
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

    # Analyze error detection results
    print(f"\n{'='*80}")
    print(f"Error Detection Analysis")
    print(f"{'='*80}")

    error_detection_dir = os.path.join(project_root, 'output', 'medinfo', 'experiment_results', 'error_detection')
    error_detection_stats = {}

    if os.path.exists(error_detection_dir):
        for perturbation_dir in os.listdir(error_detection_dir):
            perturbation_path = os.path.join(error_detection_dir, perturbation_dir)
            if not os.path.isdir(perturbation_path):
                continue

            print(f"\nLoading: {perturbation_dir}")

            # Look for error detection files
            for filename in os.listdir(perturbation_path):
                if not filename.endswith('_error_detection.jsonl'):
                    continue

                filepath = os.path.join(perturbation_path, filename)
                print(f"  Loading: {filename}")

                # Analyze perturbed
                results_perturbed = load_detection_results(filepath)
                stats_perturbed = analyze_detection_accuracy(results_perturbed, is_original=False, similarity_threshold=0.7)
                print(f"    Perturbed: {len(results_perturbed)} reports")
                print(f"      Detected: {stats_perturbed['detected']}/{stats_perturbed['total']} ({stats_perturbed['detected']/stats_perturbed['total']*100:.1f}%)")
                if stats_perturbed['detected'] > 0:
                    print(f"      Correct sentence: {stats_perturbed['correct_sentence_identified']}/{stats_perturbed['detected']} ({stats_perturbed['correct_sentence_identified']/stats_perturbed['detected']*100:.1f}% of detected)")
                    print(f"      Wrong sentence: {stats_perturbed['wrong_sentence_identified']}/{stats_perturbed['detected']} ({stats_perturbed['wrong_sentence_identified']/stats_perturbed['detected']*100:.1f}% of detected)")

                # Load original if exists (for false positives)
                stats_original = None
                original_dir = os.path.join(project_root, 'output', 'medinfo', 'original_ratings')
                if os.path.exists(original_dir):
                    # Look for original file with matching model
                    parts = filename.replace('_error_detection.jsonl', '').split('_')
                    model = '_'.join(parts[4:])  # Extract model name
                    original_file = os.path.join(original_dir, f'original_coarse_{model}_error_detection.jsonl')

                    if os.path.exists(original_file):
                        print(f"    Loading original: {os.path.basename(original_file)}")
                        results_original = load_detection_results(original_file)
                        stats_original = analyze_detection_accuracy(results_original, is_original=True)
                        print(f"    Original: {len(results_original)} reports")
                        print(f"      False positives: {stats_original['detected']}/{stats_original['total']} ({stats_original['detected']/stats_original['total']*100:.1f}%)")

                error_detection_stats[perturbation_dir] = {
                    'perturbed': stats_perturbed,
                    'original': stats_original
                }

        # Create combined plot
        if error_detection_stats:
            print("\nCreating error detection plot...")
            create_error_detection_plot(error_detection_stats, analysis_output_dir)
    else:
        print(f"  No error detection results found at: {error_detection_dir}")

    print(f"\n{'='*80}")
    print(f"Analysis Complete!")
    print(f"{'='*80}")
    print(f"Output directory: {analysis_output_dir}")
    print()


if __name__ == "__main__":
    main()
