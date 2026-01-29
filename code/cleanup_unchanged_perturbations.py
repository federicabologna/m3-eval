"""
Remove unchanged entries from swap perturbation files and their corresponding result files.

This script:
1. Identifies entries where no perturbation was applied (prediction == perturbed_prediction)
2. Removes them from perturbation files
3. Removes them from all corresponding result files
"""

import json
import os
from pathlib import Path

# Paths
project_root = Path(__file__).parent.parent
perturbations_dir = project_root / 'output' / 'radeval' / 'perturbations'
results_base = project_root / 'output' / 'radeval' / 'experiment_results' / 'baseline'


def clean_perturbation_file(perturbation_name, text_field='prediction'):
    """
    Remove unchanged entries from a perturbation file.

    Returns:
        Set of IDs that were removed (unchanged entries)
    """
    input_file = perturbations_dir / f'{perturbation_name}.jsonl'
    temp_file = perturbations_dir / f'{perturbation_name}_temp.jsonl'

    if not input_file.exists():
        print(f"Warning: {input_file} not found, skipping")
        return set()

    removed_ids = set()
    kept_count = 0

    print(f"\n{'='*80}")
    print(f"Processing: {perturbation_name}.jsonl")
    print(f"{'='*80}")

    # Read and filter
    with open(input_file, 'r') as f_in, open(temp_file, 'w') as f_out:
        for line in f_in:
            entry = json.loads(line)

            # Check if unchanged
            original = entry[text_field]
            perturbed = entry[f'perturbed_{text_field}']

            if original == perturbed:
                removed_ids.add(entry['id'])
            else:
                # Keep this entry
                json.dump(entry, f_out)
                f_out.write('\n')
                kept_count += 1

    # Replace original with filtered version
    os.replace(temp_file, input_file)

    print(f"  Kept: {kept_count} entries")
    print(f"  Removed: {len(removed_ids)} unchanged entries")
    print(f"  IDs removed: {sorted(list(removed_ids))[:10]}..." if len(removed_ids) > 10 else f"  IDs removed: {sorted(list(removed_ids))}")

    return removed_ids


def clean_result_files(perturbation_name, removed_ids):
    """
    Remove entries with removed_ids from all result files for this perturbation.
    """
    results_dir = results_base / perturbation_name

    if not results_dir.exists():
        print(f"  No results directory found: {results_dir}")
        return

    print(f"\n  Cleaning result files in: {perturbation_name}/")

    for result_file in results_dir.glob('*.jsonl'):
        temp_file = result_file.parent / f'{result_file.stem}_temp.jsonl'

        original_count = 0
        kept_count = 0
        removed_count = 0

        # Read and filter
        with open(result_file, 'r') as f_in, open(temp_file, 'w') as f_out:
            for line in f_in:
                entry = json.loads(line)
                original_count += 1

                if entry['id'] in removed_ids:
                    removed_count += 1
                else:
                    json.dump(entry, f_out)
                    f_out.write('\n')
                    kept_count += 1

        # Replace original with filtered version
        os.replace(temp_file, result_file)

        print(f"    {result_file.name}:")
        print(f"      Original: {original_count} entries")
        print(f"      Kept: {kept_count} entries")
        print(f"      Removed: {removed_count} entries")


def main():
    print("\n" + "="*80)
    print("CLEANUP UNCHANGED PERTURBATIONS")
    print("="*80)

    # Process swap_organs
    removed_ids = clean_perturbation_file('swap_organs')
    if removed_ids:
        clean_result_files('swap_organs', removed_ids)

    # Process swap_qualifiers
    removed_ids = clean_perturbation_file('swap_qualifiers')
    if removed_ids:
        clean_result_files('swap_qualifiers', removed_ids)

    print("\n" + "="*80)
    print("CLEANUP COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
