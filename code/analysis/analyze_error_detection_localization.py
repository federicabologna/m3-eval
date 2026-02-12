"""
Analyze error detection localization accuracy - COMBINED PLOT ONLY.

Creates a single comparison plot showing with vs without reference detection.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def load_detection_results(results_file):
    """Load error detection results from file."""
    results = []
    with open(results_file, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def analyze_localization_accuracy(results, skip_localization=False):
    """
    Analyze how well the model localized errors to specific sentences.

    Args:
        results: List of detection results
        skip_localization: If True, only analyze detection rate (for original reports)

    Returns:
        Dictionary with accuracy metrics
    """
    stats = {
        'total': 0,
        'detected': 0,
        'not_detected': 0,
        'has_ground_truth': 0,
        'has_prediction': 0,
        'exact_match': 0,
        'off_by_one': 0,
        'off_by_more': 0,
        'no_ground_truth': 0,
        'no_prediction': 0,
        'detection_errors': 0,
        'found_all_errors': 0,
        'found_partial_errors': 0,
        'found_none_errors': 0,
        'reports_with_0_errors': 0,
        'reports_with_1_error': 0,
        'reports_with_2plus_errors': 0,
    }

    detailed_results = []

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

        # Get ground truth error indices
        if skip_localization:
            # For original reports, just count how many errors were claimed
            errors_list = detection_result.get('errors', [])
            num_errors = len(errors_list)

            if num_errors == 0:
                stats['reports_with_0_errors'] += 1
            elif num_errors == 1:
                stats['reports_with_1_error'] += 1
            else:
                stats['reports_with_2plus_errors'] += 1
            continue

        # For perturbed reports, analyze localization
        changes_detail = result.get('changes_detail', [])
        ground_truth_indices = set()
        for change in changes_detail:
            if 'sentence_index' in change:
                ground_truth_indices.add(change['sentence_index'])

        # Get predicted error indices
        errors_list = detection_result.get('errors', [])
        predicted_indices = set()
        for error in errors_list:
            if 'sentence_index' in error and error['sentence_index'] != -1:
                predicted_indices.add(error['sentence_index'])

        num_gt_sentences = len(ground_truth_indices)
        has_ground_truth = num_gt_sentences > 0

        if has_ground_truth:
            stats['has_ground_truth'] += 1

        has_prediction = len(predicted_indices) > 0

        if has_prediction:
            stats['has_prediction'] += 1

        # Calculate coverage
        if has_ground_truth and has_prediction:
            num_correct = len(ground_truth_indices & predicted_indices)

            if num_correct == num_gt_sentences:
                stats['found_all_errors'] += 1
            elif num_correct > 0:
                stats['found_partial_errors'] += 1
            else:
                stats['found_none_errors'] += 1

            detailed_results.append({
                'id': result.get('id'),
                'detected': detected,
                'ground_truth_indices': sorted(ground_truth_indices),
                'predicted_indices': sorted(predicted_indices),
                'num_ground_truth_errors': num_gt_sentences,
                'num_predicted': len(predicted_indices),
                'num_correct': num_correct,
            })
        elif has_ground_truth and not has_prediction:
            stats['found_none_errors'] += 1
            detailed_results.append({
                'id': result.get('id'),
                'detected': detected,
                'ground_truth_indices': sorted(ground_truth_indices),
                'predicted_indices': [],
                'num_ground_truth_errors': num_gt_sentences,
                'num_predicted': 0,
                'num_correct': 0,
            })

    return stats, detailed_results


def create_combined_plot(all_stats, all_stats_with_ref, output_dir, dataset_name):
    """Create combined stacked bar plot comparing with/without reference and original."""
    if not all_stats:
        return

    perturbation_order = [
        'inject_false_prediction',
        'inject_contradiction',
        'inject_false_negation'
    ]

    perturbation_labels = {
        'inject_false_prediction': 'False Prediction',
        'inject_contradiction': 'Contradiction',
        'inject_false_negation': 'False Negation'
    }

    # Filter to only perturbations we have
    perturbations = [p for p in perturbation_order if p in all_stats]

    if not perturbations:
        return

    # Create figure with 2 subplots: perturbed and original
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Error Detection: With vs Without Reference ({dataset_name.upper()})',
                 fontsize=16, fontweight='bold')

    # Softer colors with edges
    color_all = '#4A7C59'       # Muted green - found all
    color_partial = '#CC8844'   # Muted orange - found partial
    color_none = '#A85252'      # Muted red - found none
    edge_color = 'white'
    edge_width = 2

    x = np.arange(len(perturbations))
    width = 0.35

    # LEFT PLOT: Perturbed reports
    for idx, perturbation in enumerate(perturbations):
        # Without reference
        if perturbation in all_stats:
            stats = all_stats[perturbation]['perturbed']
            total = stats['has_prediction']
            if total > 0:
                found_all_pct = stats['found_all_errors'] / total * 100
                found_partial_pct = stats['found_partial_errors'] / total * 100
                found_none_pct = 100 - found_all_pct - found_partial_pct
            else:
                found_all_pct = found_partial_pct = found_none_pct = 0

            # Stack bars with edges
            ax1.bar(x[idx] - width/2, found_all_pct, width,
                   label='Found All' if idx == 0 else '',
                   color=color_all, edgecolor=edge_color, linewidth=edge_width)
            ax1.bar(x[idx] - width/2, found_partial_pct, width, bottom=found_all_pct,
                   label='Found Partial' if idx == 0 else '',
                   color=color_partial, edgecolor=edge_color, linewidth=edge_width)
            ax1.bar(x[idx] - width/2, found_none_pct, width, bottom=found_all_pct + found_partial_pct,
                   label='Found None' if idx == 0 else '',
                   color=color_none, edgecolor=edge_color, linewidth=edge_width)

            # Add "Without Reference" label
            if idx == 0:
                ax1.text(x[idx] - width/2, -8, 'Without\nReference',
                        ha='center', va='top', fontsize=9, fontweight='bold')

        # With reference
        if perturbation in all_stats_with_ref:
            stats_ref = all_stats_with_ref[perturbation]['perturbed']
            total_ref = stats_ref['has_prediction']
            if total_ref > 0:
                found_all_pct_ref = stats_ref['found_all_errors'] / total_ref * 100
                found_partial_pct_ref = stats_ref['found_partial_errors'] / total_ref * 100
                found_none_pct_ref = 100 - found_all_pct_ref - found_partial_pct_ref
            else:
                found_all_pct_ref = found_partial_pct_ref = found_none_pct_ref = 0

            # Stack bars with edges
            ax1.bar(x[idx] + width/2, found_all_pct_ref, width,
                   color=color_all, edgecolor=edge_color, linewidth=edge_width)
            ax1.bar(x[idx] + width/2, found_partial_pct_ref, width, bottom=found_all_pct_ref,
                   color=color_partial, edgecolor=edge_color, linewidth=edge_width)
            ax1.bar(x[idx] + width/2, found_none_pct_ref, width,
                   bottom=found_all_pct_ref + found_partial_pct_ref,
                   color=color_none, edgecolor=edge_color, linewidth=edge_width)

            # Add "With Reference" label
            if idx == 0:
                ax1.text(x[idx] + width/2, -8, 'With\nReference',
                        ha='center', va='top', fontsize=9, fontweight='bold')

    ax1.set_ylabel('Percentage of Reports (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Perturbed Reports', fontsize=13, fontweight='bold', pad=15)
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
        # Without reference
        if perturbation in all_stats and all_stats[perturbation].get('original'):
            stats_orig = all_stats[perturbation]['original']
            total = stats_orig['total']
            if total > 0:
                pct_0 = stats_orig['reports_with_0_errors'] / total * 100
                pct_1 = stats_orig['reports_with_1_error'] / total * 100
                pct_2plus = stats_orig['reports_with_2plus_errors'] / total * 100
            else:
                pct_0 = pct_1 = pct_2plus = 0

            ax2.bar(x[idx] - width/2, pct_0, width,
                   label='0 Errors (Correct)' if idx == 0 else '',
                   color=color_0, edgecolor=edge_color, linewidth=edge_width)
            ax2.bar(x[idx] - width/2, pct_1, width, bottom=pct_0,
                   label='1 Error Claimed' if idx == 0 else '',
                   color=color_1, edgecolor=edge_color, linewidth=edge_width)
            ax2.bar(x[idx] - width/2, pct_2plus, width, bottom=pct_0 + pct_1,
                   label='2+ Errors Claimed' if idx == 0 else '',
                   color=color_2plus, edgecolor=edge_color, linewidth=edge_width)

        # With reference
        if perturbation in all_stats_with_ref and all_stats_with_ref[perturbation].get('original'):
            stats_orig_ref = all_stats_with_ref[perturbation]['original']
            total_ref = stats_orig_ref['total']
            if total_ref > 0:
                pct_0_ref = stats_orig_ref['reports_with_0_errors'] / total_ref * 100
                pct_1_ref = stats_orig_ref['reports_with_1_error'] / total_ref * 100
                pct_2plus_ref = stats_orig_ref['reports_with_2plus_errors'] / total_ref * 100
            else:
                pct_0_ref = pct_1_ref = pct_2plus_ref = 0

            ax2.bar(x[idx] + width/2, pct_0_ref, width,
                   color=color_0, edgecolor=edge_color, linewidth=edge_width)
            ax2.bar(x[idx] + width/2, pct_1_ref, width, bottom=pct_0_ref,
                   color=color_1, edgecolor=edge_color, linewidth=edge_width)
            ax2.bar(x[idx] + width/2, pct_2plus_ref, width, bottom=pct_0_ref + pct_1_ref,
                   color=color_2plus, edgecolor=edge_color, linewidth=edge_width)

    ax2.set_ylabel('Percentage of Reports (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Original Reports (Control - False Positives)', fontsize=13, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels([perturbation_labels.get(p, p) for p in perturbations], fontsize=11)
    ax2.set_ylim(0, 115)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    output_path = output_dir / f'localization_accuracy_{dataset_name}_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot: {output_path}")
    plt.close()


def main():
    """Main analysis function - creates only the combined comparison plot."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    print("=" * 80)
    print("ERROR DETECTION LOCALIZATION ANALYSIS - COMBINED PLOT ONLY")
    print("=" * 80)

    # Analyze RadEval
    radeval_dir = project_root / 'output' / 'radeval' / 'experiment_results' / 'error_detection'
    radeval_dir_with_ref = project_root / 'output' / 'radeval' / 'experiment_results' / 'error_detection_with_reference'
    radeval_original = project_root / 'output' / 'radeval' / 'original_ratings'
    radeval_output = project_root / 'output' / 'radeval' / 'analysis'
    radeval_output.mkdir(parents=True, exist_ok=True)

    radeval_stats = {}
    radeval_stats_with_ref = {}

    print("\nLoading RadEval error detection results...")

    # Load without reference results
    if radeval_dir.exists():
        for perturbation_dir in radeval_dir.iterdir():
            if perturbation_dir.is_dir():
                for result_file in perturbation_dir.glob('*_error_detection.jsonl'):
                    perturbation_name = perturbation_dir.name
                    print(f"  Loading: {perturbation_name} (without reference)")

                    # Analyze perturbed
                    results_perturbed = load_detection_results(result_file)
                    stats_perturbed, _ = analyze_localization_accuracy(results_perturbed)

                    # Load original if exists
                    stats_original = None
                    original_file = radeval_original / 'original_gpt-4_1-2025-04-14_error_detection.jsonl'
                    if original_file.exists():
                        results_original = load_detection_results(original_file)
                        stats_original, _ = analyze_localization_accuracy(results_original, skip_localization=True)

                    radeval_stats[perturbation_name] = {
                        'perturbed': stats_perturbed,
                        'original': stats_original
                    }

    # Load with reference results
    if radeval_dir_with_ref.exists():
        for perturbation_dir in radeval_dir_with_ref.iterdir():
            if perturbation_dir.is_dir():
                for result_file in perturbation_dir.glob('*_error_detection_with_reference.jsonl'):
                    perturbation_name = perturbation_dir.name
                    print(f"  Loading: {perturbation_name} (with reference)")

                    # Analyze perturbed
                    results_perturbed = load_detection_results(result_file)
                    stats_perturbed, _ = analyze_localization_accuracy(results_perturbed)

                    # Load original if exists
                    stats_original = None
                    original_file = radeval_original / 'original_gpt-4_1-2025-04-14_error_detection_with_reference.jsonl'
                    if original_file.exists():
                        results_original = load_detection_results(original_file)
                        stats_original, _ = analyze_localization_accuracy(results_original, skip_localization=True)

                    radeval_stats_with_ref[perturbation_name] = {
                        'perturbed': stats_perturbed,
                        'original': stats_original
                    }

    # Create combined plot
    if radeval_stats or radeval_stats_with_ref:
        print("\nCreating combined comparison plot...")
        create_combined_plot(radeval_stats, radeval_stats_with_ref, radeval_output, 'radeval')

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
