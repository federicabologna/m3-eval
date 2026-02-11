"""
Analyze error priming experiment results for GREEN ratings.

Compares GREEN scores with and without error warning in the prompt:
- Control: Standard GREEN evaluation (no error warning) from baseline
- Primed: GREEN with "NOTE: candidate report contains errors..." added

Generates:
- Bar chart comparing mean scores between conditions
- Statistical analysis report with paired t-tests

Works with both RadEval and RexErr datasets.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def load_baseline_results(baseline_dir, perturbations=None):
    """
    Load baseline GREEN ratings (control condition).

    Args:
        baseline_dir: Path to baseline experiment results
        perturbations: List of perturbation names to load (None = all)

    Returns:
        Dictionary mapping perturbation -> list of entries
    """
    results = defaultdict(list)
    baseline_path = Path(baseline_dir)

    if not baseline_path.exists():
        print(f"Warning: Baseline directory not found: {baseline_dir}")
        return results

    # Check if this is RexErr (flat structure) or RadEval (subdirectories)
    jsonl_files = list(baseline_path.glob('*.jsonl'))

    if jsonl_files:
        # Flat structure (RexErr)
        for result_file in jsonl_files:
            with open(result_file, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        results['rexerr'].append(entry)
    else:
        # Subdirectory structure (RadEval)
        for perturbation_dir in baseline_path.iterdir():
            if not perturbation_dir.is_dir():
                continue

            perturbation_name = perturbation_dir.name

            # Filter by requested perturbations
            if perturbations and perturbation_name not in perturbations:
                continue

            for result_file in perturbation_dir.glob('*_green_rating.jsonl'):
                with open(result_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            entry = json.loads(line)
                            results[perturbation_name].append(entry)

    return results


def load_primed_results(primed_dir, perturbations=None):
    """
    Load error priming GREEN ratings (primed condition).

    Args:
        primed_dir: Path to error priming experiment results
        perturbations: List of perturbation names to load (None = all)

    Returns:
        Dictionary mapping perturbation -> list of entries
    """
    results = defaultdict(list)
    primed_path = Path(primed_dir)

    if not primed_path.exists():
        print(f"Warning: Primed directory not found: {primed_dir}")
        return results

    # Check if this is RexErr (flat structure) or RadEval (subdirectories)
    jsonl_files = list(primed_path.glob('*.jsonl'))

    if jsonl_files:
        # Flat structure (RexErr)
        for result_file in jsonl_files:
            with open(result_file, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        results['rexerr'].append(entry)
    else:
        # Subdirectory structure (RadEval)
        for perturbation_dir in primed_path.iterdir():
            if not perturbation_dir.is_dir():
                continue

            perturbation_name = perturbation_dir.name

            # Filter by requested perturbations
            if perturbations and perturbation_name not in perturbations:
                continue

            for result_file in perturbation_dir.glob('*_error_priming_primed_*.jsonl'):
                with open(result_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            entry = json.loads(line)
                            results[perturbation_name].append(entry)

    return results


def extract_green_score(entry, field_name='green_rating'):
    """Extract GREEN score from entry."""
    rating = entry.get(field_name)
    if rating is None:
        return None

    if isinstance(rating, dict):
        return rating.get('score')
    elif isinstance(rating, (int, float)):
        return float(rating)
    else:
        return None


def create_comparison_dataframe(baseline_results, primed_results):
    """
    Create dataframe comparing control vs primed scores.

    Returns:
        DataFrame with columns: id, perturbation, control_score, primed_score, delta
    """
    rows = []

    for perturbation in baseline_results.keys():
        if perturbation not in primed_results:
            print(f"Warning: No primed results for {perturbation}")
            continue

        # Create lookup by ID
        baseline_by_id = {entry['id']: entry for entry in baseline_results[perturbation]}
        primed_by_id = {entry['id']: entry for entry in primed_results[perturbation]}

        # Match entries by ID
        common_ids = set(baseline_by_id.keys()) & set(primed_by_id.keys())

        for item_id in common_ids:
            baseline_entry = baseline_by_id[item_id]
            primed_entry = primed_by_id[item_id]

            # Extract scores (handle different field names)
            control_score = (
                extract_green_score(baseline_entry, 'green_rating') or
                extract_green_score(baseline_entry, 'perturbed_rating')
            )

            primed_score = extract_green_score(primed_entry, 'green_rating_primed')

            if control_score is not None and primed_score is not None:
                rows.append({
                    'id': item_id,
                    'perturbation': perturbation,
                    'control_score': control_score,
                    'primed_score': primed_score,
                    'delta': primed_score - control_score
                })

    return pd.DataFrame(rows)


def plot_score_comparison(df, output_dir, dataset_name='Dataset'):
    """Plot side-by-side comparison of control vs primed scores."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    perturbations = df['perturbation'].unique()
    n_perturbations = len(perturbations)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(n_perturbations)
    width = 0.35

    # Calculate means and standard errors
    control_means = [df[df['perturbation'] == p]['control_score'].mean() for p in perturbations]
    primed_means = [df[df['perturbation'] == p]['primed_score'].mean() for p in perturbations]

    control_stds = [df[df['perturbation'] == p]['control_score'].sem() for p in perturbations]
    primed_stds = [df[df['perturbation'] == p]['primed_score'].sem() for p in perturbations]

    # Create bars
    bars1 = ax.bar(x - width/2, control_means, width,
                   yerr=control_stds, capsize=5,
                   label='Control (no warning)',
                   color='steelblue', alpha=0.8)

    bars2 = ax.bar(x + width/2, primed_means, width,
                   yerr=primed_stds, capsize=5,
                   label='Primed (with error warning)',
                   color='coral', alpha=0.8)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Perturbation Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('GREEN Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Error Priming Effect on GREEN Scores - {dataset_name}',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace('_', ' ').title() for p in perturbations],
                       rotation=45, ha='right')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(output_dir / f'error_priming_comparison_{dataset_name.lower()}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved plot: error_priming_comparison_{dataset_name.lower()}.png")


def generate_statistical_report(df, output_dir, dataset_name='Dataset'):
    """Generate statistical analysis report."""
    output_dir = Path(output_dir)
    report_path = output_dir / f'error_priming_stats_{dataset_name.lower()}.txt'

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"ERROR PRIMING STATISTICAL ANALYSIS - {dataset_name.upper()}\n")
        f.write("=" * 80 + "\n\n")

        for perturbation in df['perturbation'].unique():
            pert_df = df[df['perturbation'] == perturbation]

            f.write(f"\n{perturbation.upper().replace('_', ' ')}\n")
            f.write("-" * 80 + "\n\n")

            # Descriptive statistics
            f.write("Descriptive Statistics:\n")
            f.write(f"  Sample size: {len(pert_df)}\n\n")

            f.write("  Control (no error warning):\n")
            f.write(f"    Mean:   {pert_df['control_score'].mean():.4f}\n")
            f.write(f"    Std:    {pert_df['control_score'].std():.4f}\n")
            f.write(f"    Median: {pert_df['control_score'].median():.4f}\n")
            f.write(f"    Range:  [{pert_df['control_score'].min():.4f}, {pert_df['control_score'].max():.4f}]\n\n")

            f.write("  Primed (with error warning):\n")
            f.write(f"    Mean:   {pert_df['primed_score'].mean():.4f}\n")
            f.write(f"    Std:    {pert_df['primed_score'].std():.4f}\n")
            f.write(f"    Median: {pert_df['primed_score'].median():.4f}\n")
            f.write(f"    Range:  [{pert_df['primed_score'].min():.4f}, {pert_df['primed_score'].max():.4f}]\n\n")

            f.write("  Delta (Primed - Control):\n")
            f.write(f"    Mean:   {pert_df['delta'].mean():.4f}\n")
            f.write(f"    Std:    {pert_df['delta'].std():.4f}\n")
            f.write(f"    Median: {pert_df['delta'].median():.4f}\n")
            f.write(f"    Range:  [{pert_df['delta'].min():.4f}, {pert_df['delta'].max():.4f}]\n\n")

            # Paired t-test
            t_stat, p_value = stats.ttest_rel(pert_df['primed_score'], pert_df['control_score'])

            f.write("Paired t-test (Primed vs Control):\n")
            f.write(f"  t-statistic: {t_stat:.4f}\n")
            f.write(f"  p-value:     {p_value:.4e}\n")

            if p_value < 0.001:
                sig = "*** (p < 0.001)"
            elif p_value < 0.01:
                sig = "** (p < 0.01)"
            elif p_value < 0.05:
                sig = "* (p < 0.05)"
            else:
                sig = "n.s. (p >= 0.05)"

            f.write(f"  Significance: {sig}\n")

            # Effect size (Cohen's d)
            mean_diff = pert_df['primed_score'].mean() - pert_df['control_score'].mean()
            pooled_std = np.sqrt((pert_df['primed_score'].std()**2 + pert_df['control_score'].std()**2) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

            f.write(f"\n  Effect size (Cohen's d): {cohens_d:.4f}\n")

            if abs(cohens_d) < 0.2:
                effect_desc = "negligible"
            elif abs(cohens_d) < 0.5:
                effect_desc = "small"
            elif abs(cohens_d) < 0.8:
                effect_desc = "medium"
            else:
                effect_desc = "large"

            f.write(f"  Effect magnitude: {effect_desc}\n")

            # Correlation
            corr = pert_df['control_score'].corr(pert_df['primed_score'])
            f.write(f"\n  Correlation (control vs primed): {corr:.4f}\n")

            # Direction analysis
            increased = (pert_df['delta'] > 0).sum()
            decreased = (pert_df['delta'] < 0).sum()
            unchanged = (pert_df['delta'] == 0).sum()

            f.write(f"\nDirection of change:\n")
            f.write(f"  Increased: {increased} ({increased/len(pert_df)*100:.1f}%)\n")
            f.write(f"  Decreased: {decreased} ({decreased/len(pert_df)*100:.1f}%)\n")
            f.write(f"  Unchanged: {unchanged} ({unchanged/len(pert_df)*100:.1f}%)\n")

            f.write("\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("INTERPRETATION\n")
        f.write("=" * 80 + "\n\n")
        f.write("A negative delta indicates that adding the error warning DECREASED the GREEN score,\n")
        f.write("meaning the model became more sensitive to errors (detected more discrepancies).\n\n")
        f.write("A positive delta indicates that adding the error warning INCREASED the GREEN score,\n")
        f.write("meaning the model became less sensitive (unexpected behavior).\n\n")

    print(f"Saved report: {report_path}")


def analyze_dataset(dataset_name, baseline_dir, primed_dir, output_dir, perturbations=None):
    """Run complete analysis for a dataset."""
    print(f"\n{'='*80}")
    print(f"ANALYZING {dataset_name.upper()}")
    print(f"{'='*80}")
    print(f"Baseline: {baseline_dir}")
    print(f"Primed:   {primed_dir}")
    print(f"Output:   {output_dir}\n")

    # Load results
    print("Loading baseline results (control)...")
    baseline_results = load_baseline_results(baseline_dir, perturbations)

    print("Loading primed results...")
    primed_results = load_primed_results(primed_dir, perturbations)

    if not baseline_results or not primed_results:
        print(f"Warning: No results found for {dataset_name}")
        return

    print(f"\nFound {len(baseline_results)} perturbation types:")
    for pert in baseline_results.keys():
        n_baseline = len(baseline_results[pert])
        n_primed = len(primed_results.get(pert, []))
        print(f"  - {pert}: {n_baseline} baseline, {n_primed} primed")

    # Create comparison dataframe
    print("\nCreating comparison dataframe...")
    df = create_comparison_dataframe(baseline_results, primed_results)

    if df.empty:
        print(f"Warning: No matching data for {dataset_name}")
        return

    print(f"Matched {len(df)} examples across conditions")

    # Save dataframe
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    df.to_csv(output_path / f'error_priming_data_{dataset_name.lower()}.csv', index=False)
    print(f"Saved data: error_priming_data_{dataset_name.lower()}.csv")

    # Generate visualization
    print("\nGenerating visualization...")
    plot_score_comparison(df, output_dir, dataset_name)

    # Generate statistical report
    print("\nGenerating statistical report...")
    generate_statistical_report(df, output_dir, dataset_name)

    print(f"\n✓ Analysis complete for {dataset_name}")


def main():
    """Main analysis function."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    print("=" * 80)
    print("ERROR PRIMING ANALYSIS")
    print("=" * 80)

    # Analyze RadEval
    radeval_baseline = project_root / 'output' / 'radeval' / 'experiment_results' / 'baseline'
    radeval_primed = project_root / 'output' / 'radeval' / 'experiment_results' / 'error_priming'
    radeval_output = project_root / 'output' / 'radeval' / 'analysis'

    # Focus on LLM perturbations
    radeval_perturbations = ['inject_false_prediction', 'inject_contradiction', 'inject_false_negation']

    if radeval_baseline.exists():
        analyze_dataset('RadEval', radeval_baseline, radeval_primed, radeval_output,
                       perturbations=radeval_perturbations)

    # Analyze RexErr
    rexerr_baseline = project_root / 'output' / 'rexerr' / 'experiment_results' / 'baseline'
    rexerr_primed = project_root / 'output' / 'rexerr' / 'experiment_results' / 'error_priming'
    rexerr_output = project_root / 'output' / 'rexerr' / 'analysis'

    if rexerr_baseline.exists():
        analyze_dataset('RexErr', rexerr_baseline, rexerr_primed, rexerr_output)

    print("\n" + "=" * 80)
    print("ALL ANALYSES COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
