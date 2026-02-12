"""
Analyze RexErr baseline evaluation results.

Compares model predictions to ground truth acceptability labels,
analyzes error distributions, and computes classification metrics.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def load_results(filepath):
    """Load RexErr evaluation results from JSONL file."""
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def analyze_acceptability(results, threshold=0.5):
    """
    Analyze acceptability classification performance.

    Args:
        results: List of evaluation results
        threshold: Score threshold for acceptable classification

    Returns:
        Dictionary with classification metrics
    """
    ground_truth = []
    predictions = []
    scores = []

    for result in results:
        # Ground truth: acceptable field
        gt_acceptable = result.get('acceptable', False)
        ground_truth.append(1 if gt_acceptable else 0)

        # Prediction: score >= threshold means acceptable
        score = result.get('green_rating', {}).get('score', 0)
        scores.append(score)
        pred_acceptable = score >= threshold
        predictions.append(1 if pred_acceptable else 0)

    # Calculate metrics
    tp = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == 1 and pred == 1)
    fp = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == 0 and pred == 1)
    tn = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == 0 and pred == 0)
    fn = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == 1 and pred == 0)

    accuracy = (tp + tn) / len(ground_truth) if len(ground_truth) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'total': len(ground_truth),
        'scores': scores,
        'ground_truth': ground_truth,
        'predictions': predictions
    }


def analyze_error_distribution(results):
    """
    Analyze distribution of error types.

    Returns:
        Dictionary with error type statistics
    """
    error_types = {
        '(a) False report of a finding in the candidate': [],
        '(b) Missing a finding present in the reference': [],
        "(c) Misidentification of a finding's anatomic location/position": [],
        '(d) Misassessment of the severity of a finding': [],
        "(e) Mentioning a comparison that isn't in the reference": [],
        '(f) Omitting a comparison detailing a change from a prior study': [],
        'Matched Findings': []
    }

    for result in results:
        error_counts = result.get('green_rating', {}).get('error_counts', {})
        for error_type, count in error_counts.items():
            if error_type in error_types:
                error_types[error_type].append(count)

    # Calculate statistics
    stats = {}
    for error_type, counts in error_types.items():
        if counts:
            stats[error_type] = {
                'mean': np.mean(counts),
                'std': np.std(counts),
                'total': np.sum(counts),
                'nonzero': sum(1 for c in counts if c > 0)
            }

    return stats


def plot_score_distribution(metrics, output_dir):
    """Plot average GREEN score with confidence interval."""
    scores = metrics['scores']

    # Calculate mean and 95% CI
    mean_score = np.mean(scores)
    sem = np.std(scores) / np.sqrt(len(scores))  # Standard error of mean
    ci_95 = 1.96 * sem  # 95% confidence interval

    fig, ax = plt.subplots(figsize=(8, 6))

    # Single bar with error bar
    ax.bar(0, mean_score, width=0.6, color='steelblue', edgecolor='black',
           linewidth=2, alpha=0.8)
    ax.errorbar(0, mean_score, yerr=ci_95, fmt='none', color='black',
                linewidth=2, capsize=10, capthick=2)

    # Add value label
    ax.text(0, mean_score + ci_95 + 0.05, f'{mean_score:.3f}\n±{ci_95:.3f}',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Add threshold line
    ax.axhline(y=metrics['threshold'], color='red', linestyle='--',
               linewidth=2, label=f"Threshold = {metrics['threshold']}")

    ax.set_ylabel('GREEN Score', fontsize=13, fontweight='bold')
    ax.set_title('Average GREEN Score with 95% CI', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks([0])
    ax.set_xticklabels(['RexErr\nEvaluation'], fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    output_path = output_dir / 'rexerr_score_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_error_distribution(error_stats, output_dir):
    """Plot distribution of error types."""
    # Short names for plotting
    error_labels = {
        '(a) False report of a finding in the candidate': 'False Finding',
        '(b) Missing a finding present in the reference': 'Missing Finding',
        "(c) Misidentification of a finding's anatomic location/position": 'Wrong Location',
        '(d) Misassessment of the severity of a finding': 'Wrong Severity',
        "(e) Mentioning a comparison that isn't in the reference": 'Extra Comparison',
        '(f) Omitting a comparison detailing a change from a prior study': 'Missing Comparison'
    }

    # Filter to only error types (not Matched Findings)
    error_types = [k for k in error_stats.keys() if k in error_labels]
    labels = [error_labels[k] for k in error_types]
    totals = [error_stats[k]['total'] for k in error_types]
    nonzero = [error_stats[k]['nonzero'] for k in error_types]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Total errors
    x = np.arange(len(labels))
    ax1.bar(x, totals, color='steelblue', edgecolor='black', alpha=0.8)
    ax1.set_ylabel('Total Error Count', fontsize=12, fontweight='bold')
    ax1.set_title('Total Errors by Type', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)

    # Number of reports with each error type
    ax2.bar(x, nonzero, color='coral', edgecolor='black', alpha=0.8)
    ax2.set_ylabel('Number of Reports', fontsize=12, fontweight='bold')
    ax2.set_title('Reports Containing Each Error Type', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'rexerr_error_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_confusion_matrix(metrics, output_dir):
    """Plot confusion matrix for acceptability classification."""
    tp, fp, tn, fn = metrics['tp'], metrics['fp'], metrics['tn'], metrics['fn']

    confusion = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(confusion, cmap='Blues', alpha=0.8)

    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, confusion[i, j],
                          ha="center", va="center", color="black",
                          fontsize=20, fontweight='bold')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted\nUnacceptable', 'Predicted\nAcceptable'], fontsize=11)
    ax.set_yticklabels(['Actual\nUnacceptable', 'Actual\nAcceptable'], fontsize=11)
    ax.set_title(f'Confusion Matrix (Threshold = {metrics["threshold"]})',
                 fontsize=14, fontweight='bold', pad=15)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    output_path = output_dir / 'rexerr_confusion_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def generate_report(metrics, error_stats, output_dir):
    """Generate text summary report."""
    output_path = output_dir / 'rexerr_baseline_report.txt'

    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("RexErr Baseline Evaluation Analysis\n")
        f.write("=" * 80 + "\n\n")

        # Classification metrics
        f.write("ACCEPTABILITY CLASSIFICATION METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Threshold: {metrics['threshold']}\n")
        f.write(f"Total reports: {metrics['total']}\n\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.3f} ({metrics['tp'] + metrics['tn']}/{metrics['total']})\n")
        f.write(f"Precision: {metrics['precision']:.3f}\n")
        f.write(f"Recall:    {metrics['recall']:.3f}\n")
        f.write(f"F1 Score:  {metrics['f1']:.3f}\n\n")

        f.write("Confusion Matrix:\n")
        f.write(f"  True Positives:  {metrics['tp']}\n")
        f.write(f"  False Positives: {metrics['fp']}\n")
        f.write(f"  True Negatives:  {metrics['tn']}\n")
        f.write(f"  False Negatives: {metrics['fn']}\n\n")

        # Score statistics
        f.write("SCORE STATISTICS\n")
        f.write("-" * 80 + "\n")
        scores = metrics['scores']
        f.write(f"Mean score: {np.mean(scores):.3f}\n")
        f.write(f"Std dev:    {np.std(scores):.3f}\n")
        f.write(f"Min score:  {np.min(scores):.3f}\n")
        f.write(f"Max score:  {np.max(scores):.3f}\n")
        f.write(f"Median:     {np.median(scores):.3f}\n\n")

        # Error distribution
        f.write("ERROR TYPE DISTRIBUTION\n")
        f.write("-" * 80 + "\n")
        for error_type, stats in error_stats.items():
            if 'Matched' not in error_type:  # Skip matched findings
                f.write(f"\n{error_type}:\n")
                f.write(f"  Total errors: {stats['total']:.1f}\n")
                f.write(f"  Reports with this error: {stats['nonzero']}\n")
                f.write(f"  Mean per report: {stats['mean']:.2f}\n")
                f.write(f"  Std dev: {stats['std']:.2f}\n")

        # Matched findings
        if 'Matched Findings' in error_stats:
            f.write(f"\nMatched Findings:\n")
            f.write(f"  Total: {error_stats['Matched Findings']['total']:.1f}\n")
            f.write(f"  Mean per report: {error_stats['Matched Findings']['mean']:.2f}\n")

    print(f"  Saved: {output_path}")


def main():
    """Main analysis function."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    print("\n" + "=" * 80)
    print("RexErr Baseline Evaluation Analysis")
    print("=" * 80)

    # Find baseline result file
    baseline_dir = project_root / 'output' / 'rexerr' / 'experiment_results' / 'baseline'
    result_files = list(baseline_dir.glob('*_green.jsonl'))

    if not result_files:
        print("No baseline result files found!")
        return

    print(f"\nFound {len(result_files)} result file(s)")

    for result_file in result_files:
        print(f"\nAnalyzing: {result_file.name}")

        # Load results
        results = load_results(result_file)
        print(f"  Loaded {len(results)} evaluations")

        # Create output directory
        analysis_dir = project_root / 'output' / 'rexerr' / 'analysis'
        analysis_dir.mkdir(parents=True, exist_ok=True)

        # Analyze acceptability classification
        print("\nAnalyzing acceptability classification...")
        metrics = analyze_acceptability(results, threshold=0.5)

        # Analyze error distribution
        print("Analyzing error distribution...")
        error_stats = analyze_error_distribution(results)

        # Generate plot
        print("\nGenerating plot...")
        plot_score_distribution(metrics, analysis_dir)

        # Generate report
        print("\nGenerating report...")
        generate_report(metrics, error_stats, analysis_dir)

        print(f"\n{'=' * 80}")
        print("SUMMARY")
        print(f"{'=' * 80}")
        print(f"Accuracy:  {metrics['accuracy']:.1%}")
        print(f"Precision: {metrics['precision']:.1%}")
        print(f"Recall:    {metrics['recall']:.1%}")
        print(f"F1 Score:  {metrics['f1']:.1%}")

    print(f"\n{'=' * 80}")
    print("Analysis Complete!")
    print(f"{'=' * 80}")
    print(f"Output directory: {analysis_dir}")
    print()


if __name__ == "__main__":
    main()
