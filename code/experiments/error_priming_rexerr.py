"""
Error Priming Experiment for RexErr Dataset.

Tests if warning about errors in the prompt affects GREEN ratings.

Compare two conditions:
- Control: Standard GREEN evaluation (no error warning)
- Primed: GREEN evaluation with "NOTE: The candidate report contains errors..." added to prompt

RexErr already contains pre-generated perturbations (prediction field has errors).
We evaluate how the error warning affects GREEN's ability to detect these errors.
"""

import os
import random
import time

from helpers.radeval_experiment_utils import (
    load_radeval_data,
    get_processed_ids,
    save_result,
    clean_model_name
)
from helpers.green_eval import get_green_rating


def run_error_priming_rexerr(args):
    """Run error priming experiment on RexErr dataset."""

    # Set random seed
    random.seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    print(f"Using GREEN model: {args.model or 'gpt-4o'}")

    # Setup paths
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', 'rexerr_acceptable_dataset.jsonl')

    # Determine output directory
    if args.output_dir is None:
        output_dir = os.path.join(project_root, 'output', 'rexerr')
    else:
        output_dir = args.output_dir

    print(f"\nDataset: RexErr")
    print(f"Output directory: {output_dir}")
    print(f"Data path: {data_path}")

    # Load data
    print("\nLoading RexErr data...")
    all_data = load_radeval_data(data_path)
    print(f"Loaded {len(all_data)} examples")

    # Apply start/end index filtering if specified
    if args.start_idx is not None or args.end_idx is not None:
        start = args.start_idx if args.start_idx is not None else 0
        end = args.end_idx if args.end_idx is not None else len(all_data)
        data = all_data[start:end]
        print(f"Using subset: indices {start} to {end} ({len(data)} examples)")
    else:
        data = all_data

    # RexErr fields
    text_field = 'prediction'  # Perturbed report (with errors)
    reference_field = 'reference'  # Original report (ground truth)

    print(f"\n{'='*80}")
    print("ERROR PRIMING EXPERIMENT - REXERR")
    print(f"{'='*80}")
    print("Computing primed GREEN ratings with error warning")
    print("  Testing on both ORIGINAL and PERTURBED reports:")
    print("  1. Original reports + error warning (control for false positives)")
    print("  2. Perturbed reports + error warning (test sensitivity)")
    print("  Baseline (no warning) ratings already exist in baseline experiment")

    # Create experiment directory
    experiment_dir = os.path.join(output_dir, 'experiment_results', 'error_priming')
    os.makedirs(experiment_dir, exist_ok=True)

    model_name_clean = clean_model_name(args.model) if args.model else "gpt-4o"

    # First, process original reports ONCE
    print(f"\n{'='*80}")
    print("PROCESSING ORIGINAL REPORTS (ONCE)")
    print(f"{'='*80}")

    # Create original ratings directory
    original_ratings_dir = os.path.join(output_dir, 'original_ratings')
    os.makedirs(original_ratings_dir, exist_ok=True)

    output_filename = f"original_{model_name_clean}_error_priming_green_rating.jsonl"
    output_path = os.path.join(original_ratings_dir, output_filename)

    # Check which entries have already been processed
    processed_ids = get_processed_ids(output_path)
    remaining_data = [item for item in data if item['id'] not in processed_ids]

    if len(remaining_data) == 0:
        print(f"✓ All {len(data)} original reports already processed")
    else:
        print(f"Processing {len(remaining_data)} remaining original reports")

        for idx, item in enumerate(remaining_data, 1):
            reference = item[reference_field]
            original_text = reference  # RexErr: reference is the original
            item_id = item['id']

            print(f"  [{idx}/{len(remaining_data)}] {item_id}...", end=" ")

            start_time = time.time()

            # Compute GREEN rating WITH error priming on ORIGINAL text
            rating = get_green_rating(
                original_text, reference,
                model_name=args.model or "gpt-4o",
                num_runs=5,
                error_priming=True
            )

            elapsed_time = time.time() - start_time
            print(f"{elapsed_time:.1f}s")

            # Build result
            result = item.copy()
            result['green_rating_original_primed'] = rating
            result['report_type'] = 'original'
            result['error_priming'] = True
            result['random_seed'] = args.seed

            # Save to file
            save_result(output_path, result)

        print(f"\n✓ Completed original reports")
        print(f"✓ Results saved to: {output_path}")

    # Now process perturbed reports
    print(f"\n{'='*80}")
    print("PROCESSING PERTURBED REPORTS")
    print(f"{'='*80}")

    output_filename = f"{model_name_clean}_error_priming_green_rating.jsonl"
    output_path = os.path.join(experiment_dir, output_filename)

    # Check which entries have already been processed
    processed_ids = get_processed_ids(output_path)
    remaining_data = [item for item in data if item['id'] not in processed_ids]

    if len(remaining_data) == 0:
        print(f"✓ All {len(data)} entries already processed")
    else:
        print(f"Processing {len(remaining_data)} remaining entries (out of {len(data)})")

        # Process each entry
        for idx, item in enumerate(remaining_data, 1):
            reference = item[reference_field]
            perturbed_text = item[text_field]  # Perturbed (with errors)
            item_id = item['id']

            print(f"  [{idx}/{len(remaining_data)}] {item_id}...", end=" ")

            start_time = time.time()

            # Compute GREEN rating WITH error priming
            rating = get_green_rating(
                perturbed_text, reference,
                model_name=args.model or "gpt-4o",
                num_runs=5,
                error_priming=True  # Primed condition
            )

            elapsed_time = time.time() - start_time
            print(f"{elapsed_time:.1f}s")

            # Build result
            result = item.copy()
            result['green_rating_perturbed_primed'] = rating
            result['report_type'] = 'perturbed'
            result['error_priming'] = True
            result['random_seed'] = args.seed

            # Save to file
            save_result(output_path, result)

        print(f"\n✓ Completed perturbed reports")
        print(f"✓ Results saved to: {output_path}")

    print(f"\n{'='*80}")
    print("ERROR PRIMING EXPERIMENT COMPLETED")
    print(f"{'='*80}")
    print(f"Baseline (control) results: output/rexerr/experiment_results/baseline/rexerr_evaluation_{model_name_clean}_green.jsonl")
    print(f"Primed results: {output_path}")
    print("\nNext step: Compare GREEN scores between baseline (control) and primed conditions")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run error priming experiment on RexErr')

    parser.add_argument('--model', type=str, default=None,
                       help='Model to use for GREEN evaluation (default: gpt-4o)')

    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: output/rexerr)')

    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')

    parser.add_argument('--start-idx', type=int, default=None,
                       help='Start index for data subset (default: 0)')

    parser.add_argument('--end-idx', type=int, default=None,
                       help='End index for data subset (default: all data)')

    args = parser.parse_args()

    run_error_priming_rexerr(args)
