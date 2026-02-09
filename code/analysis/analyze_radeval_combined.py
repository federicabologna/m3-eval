"""
Combined analysis of GREEN and CheXbert evaluation results from RadEval experiments.

Loads separate GREEN and CheXbert rating files, then creates a combined plot
with GREEN (green) and CheXbert (purple).
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import stats

# Path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
output_base = os.path.join(project_root, 'output', 'radeval', 'experiment_results', 'baseline')
analysis_output_dir = os.path.join(project_root, 'output', 'radeval', 'analysis')
os.makedirs(analysis_output_dir, exist_ok=True)


def load_results(filepath):
    """Load results from JSONL file."""
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

    for entry in results:
        if 'original_rating' in entry and 'perturbed_rating' in entry:
            # Try new format first (score), fall back to old format (green_score)
            orig = entry['original_rating'].get('score') or entry['original_rating'].get('green_score')
            pert = entry['perturbed_rating'].get('score') or entry['perturbed_rating'].get('green_score')

            if orig is not None and pert is not None:
                original_scores.append(orig)
                perturbed_scores.append(pert)

    return np.array(original_scores), np.array(perturbed_scores)


def extract_chexbert_scores(results, metric='chexbert_all_weighted_f1'):
    """Extract original and perturbed CheXbert scores."""
    original_scores = []
    perturbed_scores = []

    for entry in results:
        if 'original_chexbert_rating' in entry and 'perturbed_chexbert_rating' in entry:
            orig = entry['original_chexbert_rating'].get(metric)
            pert = entry['perturbed_chexbert_rating'].get(metric)

            if orig is not None and pert is not None:
                original_scores.append(orig)
                perturbed_scores.append(pert)

    return np.array(original_scores), np.array(perturbed_scores)


def plot_combined_comparison(green_results, chexbert_results, output_path):
    """Create combined 2-row plot with GREEN (green) and CheXbert (purple)."""

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

    # Collect all GREEN original scores
    all_green_original = []
    for key, data in green_results.items():
        all_green_original.extend(data['original'].tolist())

    green_orig_mean = np.mean(all_green_original)
    green_orig_ci = 1.96 * np.std(all_green_original) / np.sqrt(len(all_green_original))

    green_means = [green_orig_mean]
    green_cis = [green_orig_ci]
    green_degradations = [None]
    green_p_values = [None]
    labels = ['Original']

    # Collect all CheXbert original scores
    metric = 'chexbert_all_weighted_f1'
    all_chexbert_original = []
    for key, data in chexbert_results.items():
        all_chexbert_original.extend(data['original'][metric].tolist())

    chexbert_orig_mean = np.mean(all_chexbert_original)
    chexbert_orig_ci = 1.96 * np.std(all_chexbert_original) / np.sqrt(len(all_chexbert_original))

    chexbert_means = [chexbert_orig_mean]
    chexbert_cis = [chexbert_orig_ci]
    chexbert_degradations = [None]
    chexbert_p_values = [None]

    # Collect perturbed scores for each perturbation (in order)
    for pert_key, pert_label in perturbation_order:
        # Find matching GREEN perturbation
        green_found = False
        green_deg = None
        green_p = None
        for key, data in green_results.items():
            perturbation_dir = data['perturbation']
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
                green_mean = np.mean(pert_scores)
                green_ci = 1.96 * np.std(pert_scores) / np.sqrt(len(pert_scores))

                # Calculate degradation and p-value
                green_deg = np.mean(orig_scores) - green_mean
                _, green_p = stats.wilcoxon(orig_scores, pert_scores, alternative='greater')

                green_means.append(green_mean)
                green_cis.append(green_ci)
                green_found = True
                break

        # Find matching CheXbert perturbation
        chexbert_found = False
        chexbert_deg = None
        chexbert_p = None
        for key, data in chexbert_results.items():
            perturbation_dir = data['perturbation']
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
                orig_scores = data['original'][metric]
                pert_scores = data['perturbed'][metric]
                chexbert_mean = np.mean(pert_scores)
                chexbert_ci = 1.96 * np.std(pert_scores) / np.sqrt(len(pert_scores))

                # Calculate degradation and p-value
                chexbert_deg = np.mean(orig_scores) - chexbert_mean
                _, chexbert_p = stats.wilcoxon(orig_scores, pert_scores, alternative='greater')

                chexbert_means.append(chexbert_mean)
                chexbert_cis.append(chexbert_ci)
                chexbert_found = True
                break

        # Only add label if both GREEN and CheXbert found
        if green_found and chexbert_found:
            labels.append(pert_label)
            green_degradations.append(green_deg)
            green_p_values.append(green_p)
            chexbert_degradations.append(chexbert_deg)
            chexbert_p_values.append(chexbert_p)
        else:
            # Remove added means/cis if not both found
            if green_found and not chexbert_found:
                green_means.pop()
                green_cis.pop()
            elif chexbert_found and not green_found:
                chexbert_means.pop()
                chexbert_cis.pop()

    # Create 2-row plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    x = np.arange(len(labels))

    # Top plot: GREEN scores (green color)
    ax1 = axes[0]
    green_color = '#2ca02c'  # Green
    green_orig_color = '#1f77b4'  # Blue for original
    colors_green = [green_orig_color] + [green_color] * (len(labels) - 1)

    ax1.bar(x, green_means, yerr=green_cis, color=colors_green, alpha=0.7,
            capsize=5, error_kw={'linewidth': 2})

    # Add degradation values and significance stars above bars
    for i in range(1, len(green_means)):  # Skip original (index 0)
        # Determine significance stars
        if green_p_values[i] < 0.001:
            stars = '***'
        elif green_p_values[i] < 0.01:
            stars = '**'
        elif green_p_values[i] < 0.05:
            stars = '*'
        else:
            stars = ''

        # Add text annotation (show perturbed - original)
        y_pos = green_means[i] + green_cis[i] + 0.02
        diff = -green_degradations[i]  # degradation is (orig - pert), so negate for (pert - orig)
        text = f'{diff:.3f}{stars}'
        ax1.text(x[i], y_pos, text, ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_ylabel('GREEN Score', fontsize=12)
    ax1.set_title('GREEN Score - Original vs Perturbed (Mean ± 95% CI)',
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(bottom=0, top=1.15)  # Increased top to accommodate annotations

    # Bottom plot: CheXbert weighted F1 (purple color)
    ax2 = axes[1]
    purple_color = '#9467bd'  # Purple
    chexbert_orig_color = '#1f77b4'  # Blue for original
    colors_chexbert = [chexbert_orig_color] + [purple_color] * (len(labels) - 1)

    ax2.bar(x, chexbert_means, yerr=chexbert_cis, color=colors_chexbert, alpha=0.7,
            capsize=5, error_kw={'linewidth': 2})

    # Add degradation values and significance stars above bars
    for i in range(1, len(chexbert_means)):  # Skip original (index 0)
        # Determine significance stars
        if chexbert_p_values[i] < 0.001:
            stars = '***'
        elif chexbert_p_values[i] < 0.01:
            stars = '**'
        elif chexbert_p_values[i] < 0.05:
            stars = '*'
        else:
            stars = ''

        # Add text annotation (show perturbed - original)
        y_pos = chexbert_means[i] + chexbert_cis[i] + 0.02
        diff = -chexbert_degradations[i]  # degradation is (orig - pert), so negate for (pert - orig)
        text = f'{diff:.3f}{stars}'
        ax2.text(x[i], y_pos, text, ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.set_ylabel('Weighted F1 (All 14)', fontsize=12)
    ax2.set_title('Weighted F1 (All 14) - Original vs Perturbed (Mean ± 95% CI)',
                  fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(bottom=0, top=1.15)  # Increased top to accommodate annotations

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved combined plot: {os.path.basename(output_path)}")


def combine_severity_plots(output_path):
    """Combine four severity plots into a 2x2 grid."""

    # Input files
    green_remove = os.path.join(analysis_output_dir, 'green_remove_severity_gpt-4_1-2025-04-14.png')
    green_typo = os.path.join(analysis_output_dir, 'green_typo_severity_gpt-4_1-2025-04-14.png')
    chexbert_remove = os.path.join(analysis_output_dir, 'chexbert_all_weighted_f1_remove_severity.png')
    chexbert_typo = os.path.join(analysis_output_dir, 'chexbert_all_weighted_f1_typo_severity.png')

    # Check if all input files exist
    files = [green_remove, green_typo, chexbert_remove, chexbert_typo]
    missing = [f for f in files if not os.path.exists(f)]

    if missing:
        print("\n  Warning: Missing input files for severity plots:")
        for f in missing:
            print(f"    - {os.path.basename(f)}")
        return

    # Load images
    img_green_remove = mpimg.imread(green_remove)
    img_green_typo = mpimg.imread(green_typo)
    img_chexbert_remove = mpimg.imread(chexbert_remove)
    img_chexbert_typo = mpimg.imread(chexbert_typo)

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Top left: GREEN remove sentences
    axes[0, 0].imshow(img_green_remove)
    axes[0, 0].axis('off')
    axes[0, 0].set_title('GREEN - Remove Sentences Severity',
                         fontsize=14, fontweight='bold', pad=10)

    # Top right: GREEN typos
    axes[0, 1].imshow(img_green_typo)
    axes[0, 1].axis('off')
    axes[0, 1].set_title('GREEN - Typo Severity',
                         fontsize=14, fontweight='bold', pad=10)

    # Bottom left: CheXbert remove sentences
    axes[1, 0].imshow(img_chexbert_remove)
    axes[1, 0].axis('off')
    axes[1, 0].set_title('CheXbert Weighted F1 - Remove Sentences Severity',
                         fontsize=14, fontweight='bold', pad=10)

    # Bottom right: CheXbert typos
    axes[1, 1].imshow(img_chexbert_typo)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('CheXbert Weighted F1 - Typo Severity',
                         fontsize=14, fontweight='bold', pad=10)

    # Overall title
    fig.suptitle('Perturbation Severity Effects on GREEN and CheXbert Metrics',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved combined severity plot: {os.path.basename(output_path)}")


def main():
    print("\n" + "="*80)
    print("Combined GREEN and CheXbert Results Analysis")
    print("="*80)

    # Load GREEN results
    print("\nLoading GREEN results...")
    green_results = {}

    for perturbation_dir in os.listdir(output_base):
        perturbation_path = os.path.join(output_base, perturbation_dir)
        if not os.path.isdir(perturbation_path):
            continue

        for filename in os.listdir(perturbation_path):
            if filename.endswith('_green_rating.jsonl'):
                filepath = os.path.join(perturbation_path, filename)

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

                print(f"  Loading: {perturbation_dir}/{filename}")
                results = load_results(filepath)
                orig, pert = extract_green_scores(results)

                key = f"{perturbation_dir}_{filename}"
                green_results[key] = {
                    'perturbation': perturbation_dir,
                    'filename': filename,
                    'label': label,
                    'original': orig,
                    'perturbed': pert
                }

    # Load CheXbert results
    print("\nLoading CheXbert results...")
    chexbert_results = {}
    chexbert_metric = 'chexbert_all_weighted_f1'

    for perturbation_dir in os.listdir(output_base):
        perturbation_path = os.path.join(output_base, perturbation_dir)
        if not os.path.isdir(perturbation_path):
            continue

        for filename in os.listdir(perturbation_path):
            if filename.endswith('_chexbert_rating.jsonl'):
                filepath = os.path.join(perturbation_path, filename)

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

                print(f"  Loading: {perturbation_dir}/{filename}")
                results = load_results(filepath)
                orig, pert = extract_chexbert_scores(results, chexbert_metric)

                key = f"{perturbation_dir}_{filename}"
                chexbert_results[key] = {
                    'perturbation': perturbation_dir,
                    'filename': filename,
                    'label': label,
                    'original': {chexbert_metric: orig},
                    'perturbed': {chexbert_metric: pert}
                }

    if not green_results or not chexbert_results:
        print("\nInsufficient data for combined plot!")
        print(f"  GREEN results: {len(green_results)}")
        print(f"  CheXbert results: {len(chexbert_results)}")
        return

    print(f"\nLoaded {len(green_results)} GREEN results")
    print(f"Loaded {len(chexbert_results)} CheXbert results")

    # Generate combined plot
    print(f"\n{'='*80}")
    print("Generating Combined Plot")
    print(f"{'='*80}")

    combined_path = os.path.join(analysis_output_dir, 'combined_green_chexbert.png')
    plot_combined_comparison(green_results, chexbert_results, combined_path)

    # Generate combined severity plots
    print("\nGenerating Combined Severity Plots...")
    severity_path = os.path.join(analysis_output_dir, 'combined_severity_plots.png')
    combine_severity_plots(severity_path)

    print(f"\n{'='*80}")
    print(f"Analysis Complete!")
    print(f"{'='*80}")
    print(f"Combined comparison: {combined_path}")
    print(f"Combined severity: {severity_path}")
    print()


if __name__ == "__main__":
    main()
