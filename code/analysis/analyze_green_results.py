"""
Analyze GREEN evaluation results from RadEval experiments.

Compares original vs perturbed GREEN scores across different perturbation types.
Generates statistics and visualizations.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
output_base = os.path.join(project_root, 'output', 'radeval', 'experiment_results', 'baseline')
analysis_output_dir = os.path.join(project_root, 'output', 'radeval', 'analysis')
os.makedirs(analysis_output_dir, exist_ok=True)


def load_green_results(filepath):
    """Load GREEN results from JSONL file."""
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            entry = json.loads(line)
            results.append(entry)
    return results


def extract_green_scores(results):
    """Extract original and perturbed GREEN scores.

    Supports both old format (green_score) and new format (score).
    """
    original_scores = []
    perturbed_scores = []
    original_individual_scores = []  # List of lists: each entry contains 5 individual scores
    perturbed_individual_scores = []

    for entry in results:
        if 'original_rating' in entry and 'perturbed_rating' in entry:
            # Try new format first (score), fall back to old format (green_score)
            orig = entry['original_rating'].get('score') or entry['original_rating'].get('green_score')
            pert = entry['perturbed_rating'].get('score') or entry['perturbed_rating'].get('green_score')

            if orig is not None and pert is not None:
                original_scores.append(orig)
                perturbed_scores.append(pert)

                # Extract individual scores if available (new format only)
                orig_individual = entry['original_rating'].get('individual_scores', [])
                pert_individual = entry['perturbed_rating'].get('individual_scores', [])
                original_individual_scores.append(orig_individual)
                perturbed_individual_scores.append(pert_individual)

    return (np.array(original_scores), np.array(perturbed_scores),
            original_individual_scores, perturbed_individual_scores)


def compute_statistics(original_scores, perturbed_scores):
    """Compute statistical measures comparing original vs perturbed."""
    orig_mean = np.mean(original_scores)
    orig_std = np.std(original_scores)
    pert_mean = np.mean(perturbed_scores)
    pert_std = np.std(perturbed_scores)

    differences = original_scores - perturbed_scores
    mean_degradation = np.mean(differences)
    std_degradation = np.std(differences)

    pct_decreased = np.mean(differences > 0) * 100
    pct_increased = np.mean(differences < 0) * 100
    pct_unchanged = np.mean(differences == 0) * 100

    # Wilcoxon signed-rank test for paired samples
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


def plot_comparison(perturbation_results, output_path):
    """Create bar plot with average GREEN score and 95% CI error bars."""
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
        all_original_scores.extend(data['original'].tolist())

    # Compute mean and 95% CI for original
    orig_mean = np.mean(all_original_scores)
    orig_ci = 1.96 * np.std(all_original_scores) / np.sqrt(len(all_original_scores))  # 95% CI

    means = [orig_mean]
    cis = [orig_ci]
    labels = ['Original']
    degradations = [None]  # No degradation for original
    p_values = [None]  # No p-value for original

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
                orig_scores = data['original']
                pert_scores = data['perturbed']
                mean = np.mean(pert_scores)
                ci = 1.96 * np.std(pert_scores) / np.sqrt(len(pert_scores))  # 95% CI

                # Calculate degradation
                degradation = np.mean(orig_scores) - mean

                # Perform Wilcoxon signed-rank test
                _, p_value = stats.wilcoxon(orig_scores, pert_scores, alternative='greater')

                means.append(mean)
                cis.append(ci)
                labels.append(pert_label)
                degradations.append(degradation)
                p_values.append(p_value)
                break

    # Create bar plot
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(means))
    colors = ['#1f77b4'] + ['#ff7f0e'] * (len(means) - 1)

    bars = ax.bar(x, means, yerr=cis, color=colors, alpha=0.7,
                  capsize=5, error_kw={'linewidth': 2})

    # Add degradation values and significance stars above bars
    for i in range(1, len(means)):  # Skip original (index 0)
        # Determine significance stars
        if p_values[i] < 0.001:
            stars = '***'
        elif p_values[i] < 0.01:
            stars = '**'
        elif p_values[i] < 0.05:
            stars = '*'
        else:
            stars = ''

        # Add text annotation (show perturbed - original)
        y_pos = means[i] + cis[i] + 0.02
        diff = -degradations[i]  # degradation is (orig - pert), so negate for (pert - orig)
        text = f'{diff:.3f}{stars}'
        ax.text(x[i], y_pos, text, ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Customize plot
    ax.set_ylabel('GREEN Score', fontsize=12)
    ax.set_title('GREEN Score - Original vs Perturbed (Mean ± 95% CI)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(bottom=0, top=1.15)  # Increased top to accommodate annotations

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved plot: {os.path.basename(output_path)}")


def plot_severity_effect(perturbation_results, output_dir):
    """
    Create plots showing how perturbation severity affects score degradation.

    Creates two separate plots:
    1. Add Typos: degradation vs probability (0.3, 0.5, 0.7)
    2. Remove Sentences: degradation vs percentage (30%, 50%, 70%)
    """
    # Collect data for add_typos
    typos_data = []
    for prob in ['03', '05', '07']:
        for key, data in perturbation_results.items():
            if data['perturbation'] == 'add_typos' and f'{prob}prob' in data['filename']:
                orig_scores = data['original']
                pert_scores = data['perturbed']
                degradation = orig_scores - pert_scores
                mean_deg = np.mean(degradation)
                ci_deg = 1.96 * np.std(degradation) / np.sqrt(len(degradation))
                typos_data.append({
                    'prob': float(prob) / 10,
                    'mean': mean_deg,
                    'ci': ci_deg
                })
                break

    # Collect data for remove_sentences
    remove_data = []
    for pct in ['30', '50', '70']:
        for key, data in perturbation_results.items():
            if data['perturbation'] == 'remove_sentences' and f'{pct}pct' in data['filename']:
                orig_scores = data['original']
                pert_scores = data['perturbed']
                degradation = orig_scores - pert_scores
                mean_deg = np.mean(degradation)
                ci_deg = 1.96 * np.std(degradation) / np.sqrt(len(degradation))
                remove_data.append({
                    'pct': int(pct),
                    'mean': mean_deg,
                    'ci': ci_deg
                })
                break

    # Extract model name from first result
    model_name = None
    for key, data in perturbation_results.items():
        filename = data['filename']
        parts = filename.replace('_green_rating.jsonl', '').split('_')
        if data['perturbation'] == 'add_typos':
            model_name = '_'.join(parts[3:])  # Skip add, typos, and prob
        elif data['perturbation'] == 'remove_sentences':
            model_name = '_'.join(parts[3:])  # Skip remove, sentences, and pct
        else:
            model_name = '_'.join(parts[1:])  # Skip perturbation name
        if model_name:
            break

    # Plot 1: Add Typos
    if len(typos_data) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))

        probs = [d['prob'] for d in typos_data]
        means = [d['mean'] for d in typos_data]
        cis = [d['ci'] for d in typos_data]

        ax.plot(probs, means, marker='o', linewidth=2, markersize=8, color='#ff7f0e')
        ax.fill_between(probs,
                        [m - c for m, c in zip(means, cis)],
                        [m + c for m, c in zip(means, cis)],
                        alpha=0.3, color='#ff7f0e')

        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax.set_xlabel('Typo Probability', fontsize=12)
        ax.set_ylabel('Score Degradation (Original - Perturbed)', fontsize=12)
        ax.set_title(f'GREEN Score Degradation vs Typo Probability',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(probs)
        ax.set_xticklabels([f'{p:.1f}' for p in probs])
        ax.set_ylim(0, 0.25)

        plt.tight_layout()
        output_path = os.path.join(output_dir, f'green_typo_severity_{model_name}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved plot: {os.path.basename(output_path)}")

    # Plot 2: Remove Sentences
    if len(remove_data) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))

        pcts = [d['pct'] for d in remove_data]
        means = [d['mean'] for d in remove_data]
        cis = [d['ci'] for d in remove_data]

        ax.plot(pcts, means, marker='o', linewidth=2, markersize=8, color='#ff7f0e')
        ax.fill_between(pcts,
                        [m - c for m, c in zip(means, cis)],
                        [m + c for m, c in zip(means, cis)],
                        alpha=0.3, color='#ff7f0e')

        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax.set_xlabel('Percentage of Sentences Removed (%)', fontsize=12)
        ax.set_ylabel('Score Degradation (Original - Perturbed)', fontsize=12)
        ax.set_title(f'GREEN Score Degradation vs Sentences Removed',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(pcts)
        ax.set_xticklabels([f'{p}%' for p in pcts])
        ax.set_ylim(0, 0.25)

        plt.tight_layout()
        output_path = os.path.join(output_dir, f'green_remove_severity_{model_name}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved plot: {os.path.basename(output_path)}")


def plot_degradation_distribution(perturbation_results, output_path):
    """Plot distribution of score degradations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    perturbations = list(perturbation_results.keys())

    for idx, perturbation in enumerate(perturbations[:4]):
        if idx >= len(axes):
            break

        ax = axes[idx]
        orig_scores = perturbation_results[perturbation]['original']
        pert_scores = perturbation_results[perturbation]['perturbed']
        degradations = orig_scores - pert_scores

        ax.hist(degradations, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
        ax.axvline(np.mean(degradations), color='green', linestyle='--', linewidth=2,
                  label=f'Mean: {np.mean(degradations):.3f}')

        ax.set_xlabel('Score Degradation (Original - Perturbed)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f"{perturbation_results[perturbation]['label']}", fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    for idx in range(len(perturbations), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('GREEN Score - Degradation Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved plot: {os.path.basename(output_path)}")


def generate_summary_report(perturbation_results, model_name, output_path):
    """Generate text summary report with Wilcoxon signed-rank test results."""
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"GREEN Evaluation Results - Wilcoxon Signed-Rank Test\n")
        f.write(f"Model: {model_name}\n")
        f.write("=" * 80 + "\n\n")

        for perturbation, data in perturbation_results.items():
            f.write(f"\n{'='*80}\n")
            f.write(f"Perturbation: {data['label']}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Number of samples: {data['stats']['n_samples']}\n\n")

            stats_data = data['stats']

            if stats_data['wilcoxon_p_value'] is not None:
                significance = "***" if stats_data['wilcoxon_p_value'] < 0.001 else \
                             "**" if stats_data['wilcoxon_p_value'] < 0.01 else \
                             "*" if stats_data['wilcoxon_p_value'] < 0.05 else "ns"

                f.write(f"GREEN Score: ")
                f.write(f"W={stats_data['wilcoxon_statistic']:.2f}, ")
                f.write(f"p={stats_data['wilcoxon_p_value']:.4e} {significance}\n")

            f.write("\n")

    print(f"  Saved summary: {os.path.basename(output_path)}")


def main():
    print("\n" + "="*80)
    print("GREEN Results Analysis")
    print("="*80)

    # Find all GREEN result files
    perturbation_results = {}
    model_name = None

    for perturbation_dir in os.listdir(output_base):
        perturbation_path = os.path.join(output_base, perturbation_dir)
        if not os.path.isdir(perturbation_path):
            continue

        for filename in os.listdir(perturbation_path):
            if filename.endswith('_green_rating.jsonl') and not filename.endswith('_chexbert_rating.jsonl'):
                filepath = os.path.join(perturbation_path, filename)

                # Extract model name from filename
                if model_name is None:
                    parts = filename.replace('_green_rating.jsonl', '').split('_')
                    # Find model part (everything after perturbation-specific parts)
                    if perturbation_dir == 'add_typos':
                        model_name = '_'.join(parts[3:])  # Skip add, typos, and prob
                    elif perturbation_dir == 'remove_sentences':
                        model_name = '_'.join(parts[3:])  # Skip remove, sentences, and pct
                    else:
                        model_name = '_'.join(parts[1:])  # Skip perturbation name

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

                results = load_green_results(filepath)
                print(f"  Loaded {len(results)} entries")

                orig, pert, orig_individual, pert_individual = extract_green_scores(results)
                stats_data = compute_statistics(orig, pert)

                # Check if we have individual scores (new format)
                has_individual_scores = any(len(ind) > 0 for ind in orig_individual)
                if has_individual_scores:
                    print(f"  Found individual scores (n={len(orig_individual[0])} completions per sample)")

                key = f"{perturbation_dir}_{filename}"
                perturbation_results[key] = {
                    'perturbation': perturbation_dir,
                    'filename': filename,
                    'label': label,
                    'original': orig,
                    'perturbed': pert,
                    'stats': stats_data
                }

    if not perturbation_results:
        print("\nNo GREEN result files found!")
        return

    print(f"\n{'='*80}")
    print(f"Generating Analysis Outputs")
    print(f"{'='*80}")

    # Generate bar plot
    print("\nGenerating comparison bar plot...")
    plot_path = os.path.join(analysis_output_dir, f'green_barplot_{model_name}.png')
    plot_comparison(perturbation_results, plot_path)

    # Generate severity effect plots
    print("\nGenerating severity effect plots...")
    plot_severity_effect(perturbation_results, analysis_output_dir)

    # Generate summary report
    print("\nGenerating summary report...")
    report_path = os.path.join(analysis_output_dir, f'green_summary_report_{model_name}.txt')
    generate_summary_report(perturbation_results, model_name, report_path)

    print(f"\n{'='*80}")
    print(f"Analysis Complete!")
    print(f"{'='*80}")
    print(f"Output directory: {analysis_output_dir}")
    print()


if __name__ == "__main__":
    main()
