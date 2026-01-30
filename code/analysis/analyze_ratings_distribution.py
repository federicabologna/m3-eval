import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import scipy.stats as stats

# Use relative path from script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))  # Go up two levels from /code/analysis to project root
output_dir = os.path.join(project_root, 'output')

# Find all perturbation output files (only in subdirectories, skip original ratings in root)
output_files = []

# Check subdirectories only (perturbation-specific folders)
for subdir in os.listdir(output_dir):
    subdir_path = os.path.join(output_dir, subdir)
    if os.path.isdir(subdir_path):
        for f in os.listdir(subdir_path):
            if f.endswith('_rating.jsonl'):
                output_files.append((subdir, f))  # Include subdirectory name

# Dimensions to analyze
dimensions = ['correctness', 'relevance', 'safety']

for subdir, filename in sorted(output_files):
    # Build full path
    filepath = os.path.join(output_dir, subdir, filename)

    # Parse filename
    # The subdirectory name IS the perturbation type (more reliable)
    perturbation = subdir

    filename_base = filename.replace('_rating.jsonl', '')

    # Remove perturbation prefix from filename
    if filename_base.startswith(perturbation + '_'):
        remainder = filename_base[len(perturbation) + 1:]  # Skip perturbation and underscore
    else:
        remainder = filename_base

    # Extract parameter info for specific perturbations
    params = None
    if perturbation == 'add_typos':
        # Format: {prob}prob_{level}_{model}
        # Example: 03prob_coarse_Qwen3-1_7B
        if 'prob_' in remainder:
            prob_part = remainder.split('prob_')[0]
            params = f"prob={float(prob_part)/10}"
            remainder = remainder.split('prob_')[1]
    elif perturbation == 'remove_must_have':
        # Format: {num}removed_{level}_{model}
        # Example: 1removed_coarse_Qwen3-1_7B
        if 'removed_' in remainder:
            num_part = remainder.split('removed_')[0]
            params = f"removed={num_part}"
            remainder = remainder.split('removed_')[1]

    # Parse remainder: {level}_{model}
    parts = remainder.split('_')
    level = parts[0]  # 'coarse' or 'fine'
    model = '_'.join(parts[1:])  # Everything after level is model name

    print("="*80)
    print(f"FILE: {filename}")
    if params:
        print(f"Perturbation: {perturbation} ({params}) | Level: {level} | Model: {model}")
    else:
        print(f"Perturbation: {perturbation} | Level: {level} | Model: {model}")
    print("="*80)

    # Load results
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except:
                pass

    if len(results) == 0:
        print("No data to analyze.\n")
        continue

    print(f"Total entries: {len(results)}\n")

    # Collect scores for each dimension
    data = {dim: {'original': [], 'perturbed': []} for dim in dimensions}

    for result in results:
        orig_rating = result.get('original_rating', {})
        pert_rating = result.get('perturbed_rating', {})

        for dim in dimensions:
            if dim in orig_rating and dim in pert_rating:
                if isinstance(orig_rating[dim], dict) and isinstance(pert_rating[dim], dict):
                    orig_score = orig_rating[dim].get('score')
                    pert_score = pert_rating[dim].get('score')

                    if orig_score is not None and pert_score is not None:
                        data[dim]['original'].append(orig_score)
                        data[dim]['perturbed'].append(pert_score)

    # Create single plot with grouped bars
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data for plotting and stats
    means_original = []
    means_perturbed = []
    ci_original = []
    ci_perturbed = []
    p_values = []
    stats_results = {
        'filename': filename,
        'perturbation': perturbation,
        'level': level,
        'model': model,
        'n_samples': len(results),
        'dimensions': {}
    }

    # Add parameter info if present
    if params:
        stats_results['parameters'] = params

    for dim in dimensions:
        original_scores = data[dim]['original']
        perturbed_scores = data[dim]['perturbed']

        if len(original_scores) == 0:
            means_original.append(0)
            means_perturbed.append(0)
            ci_original.append(0)
            ci_perturbed.append(0)
            p_values.append(1.0)
            continue

        # Calculate means
        mean_orig = np.mean(original_scores)
        mean_pert = np.mean(perturbed_scores)

        # Calculate 95% confidence intervals
        ci_orig = stats.sem(original_scores) * stats.t.ppf((1 + 0.95) / 2., len(original_scores)-1)
        ci_pert = stats.sem(perturbed_scores) * stats.t.ppf((1 + 0.95) / 2., len(perturbed_scores)-1)

        means_original.append(mean_orig)
        means_perturbed.append(mean_pert)
        ci_original.append(ci_orig)
        ci_perturbed.append(ci_pert)

        # Mann-Whitney U test
        statistic, p_value = mannwhitneyu(original_scores, perturbed_scores, alternative='two-sided')
        p_values.append(p_value)

        # Determine significance level
        if p_value < 0.001:
            sig = "***"
        elif p_value < 0.01:
            sig = "**"
        elif p_value < 0.05:
            sig = "*"
        else:
            sig = "ns"

        # Store stats in dictionary
        stats_results['dimensions'][dim] = {
            'n_scores': len(original_scores),
            'original': {
                'mean': float(mean_orig),
                'std': float(np.std(original_scores)),
                'median': float(np.median(original_scores)),
                'ci_lower': float(mean_orig - ci_orig),
                'ci_upper': float(mean_orig + ci_orig)
            },
            'perturbed': {
                'mean': float(mean_pert),
                'std': float(np.std(perturbed_scores)),
                'median': float(np.median(perturbed_scores)),
                'ci_lower': float(mean_pert - ci_pert),
                'ci_upper': float(mean_pert + ci_pert)
            },
            'mann_whitney_u': {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significance': sig
            }
        }

        # Print statistics
        print(f"{dim.upper()}:")
        print(f"  Original: mean={mean_orig:.2f}, 95% CI=[{mean_orig-ci_orig:.2f}, {mean_orig+ci_orig:.2f}]")
        print(f"  Perturbed: mean={mean_pert:.2f}, 95% CI=[{mean_pert-ci_pert:.2f}, {mean_pert+ci_pert:.2f}]")
        print(f"  Mann-Whitney U statistic: {statistic:.2f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Significance: {sig}")
        print()

    # Plot grouped bars
    x = np.arange(len(dimensions))
    width = 0.35

    bars1 = ax.bar(x - width/2, means_original, width, yerr=ci_original,
                   label='Original', alpha=0.8, color='steelblue',
                   capsize=5, error_kw={'linewidth': 2})
    bars2 = ax.bar(x + width/2, means_perturbed, width, yerr=ci_perturbed,
                   label='Perturbed', alpha=0.8, color='coral',
                   capsize=5, error_kw={'linewidth': 2})

    # Add difference and significance labels on perturbed bars
    for i, (bar, mean_orig, mean_pert, ci_pert_val, p_val) in enumerate(zip(bars2, means_original, means_perturbed, ci_perturbed, p_values)):
        # Calculate difference (perturbed - original)
        diff = mean_pert - mean_orig

        # Get significance marker
        if p_val < 0.001:
            sig_marker = '***'
        elif p_val < 0.01:
            sig_marker = '**'
        elif p_val < 0.05:
            sig_marker = '*'
        else:
            sig_marker = 'ns'

        # Add annotation above perturbed bar
        height = bar.get_height()
        y_pos = height + ci_pert_val + 0.1
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
               f'{diff:.2f}{sig_marker}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Formatting
    ax.set_xlabel('Evaluation Dimension', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Rating', fontsize=12, fontweight='bold')

    # Build title with parameters if present
    if params:
        title = f'{perturbation.title()} ({params}) - {level.title()} Level - {model}'
    else:
        title = f'{perturbation.title()} Perturbation - {level.title()} Level - {model}'
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([dim.title() for dim in dimensions], fontsize=11)
    ax.set_ylim(0, 5.5)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # P-values are now shown on the bars with significance markers

    plt.tight_layout()

    # Create perturbation-specific subdirectory
    perturbation_dir = os.path.join(output_dir, perturbation)
    os.makedirs(perturbation_dir, exist_ok=True)

    # Save plot
    plot_filename = filename.replace('.jsonl', '_means.png')
    plot_path = os.path.join(perturbation_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {perturbation}/{plot_filename}")

    # Save statistics to JSON
    stats_filename = filename.replace('.jsonl', '_stats.json')
    stats_path = os.path.join(perturbation_dir, stats_filename)
    with open(stats_path, 'w') as f:
        json.dump(stats_results, f, indent=2)
    print(f"Stats saved: {perturbation}/{stats_filename}\n")

    plt.close()

print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)
