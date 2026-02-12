"""
Error Priming Experiment for RadEval Dataset.

Tests if warning about errors in the prompt affects GREEN ratings.

Compare two conditions:
- Control: Standard GREEN evaluation (no error warning)
- Primed: GREEN evaluation with "NOTE: The candidate report contains errors..." added to prompt

Uses existing perturbations from baseline experiments:
- inject_false_prediction
- inject_contradiction
- inject_false_negation
"""

import json
import os
import random
import time
from pathlib import Path

from helpers.radeval_experiment_utils import (
    setup_radeval_paths,
    load_radeval_data,
    get_processed_ids,
    save_result,
    clean_model_name
)
from helpers.green_eval import get_green_rating


def run_error_priming_radeval(args):
    """Run error priming experiment on RadEval dataset."""

    # Set random seed
    random.seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    print(f"Using GREEN model: {args.model or 'gpt-4o'}")

    # Setup paths
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', 'radeval_expert_dataset.jsonl')

    # Determine output directory
    if args.output_dir is None:
        output_dir = os.path.join(project_root, 'output', 'radeval')
    else:
        output_dir = args.output_dir

    # Setup paths
    paths = setup_radeval_paths(output_dir, data_path)
    output_dir = paths['output_dir']

    print(f"\nDataset: RadEval")
    print(f"Output directory: {output_dir}")
    print(f"Data path: {data_path}")

    # Load data
    print("\nLoading RadEval data...")
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

    text_field = 'prediction'
    reference_field = 'reference'

    # Define perturbations to test
    perturbations = ['inject_false_prediction', 'inject_contradiction', 'inject_false_negation']

    if args.perturbation:
        if args.perturbation not in perturbations:
            raise ValueError(f"Invalid perturbation: {args.perturbation}. Choose from: {perturbations}")
        perturbations_to_run = [args.perturbation]
    else:
        perturbations_to_run = perturbations

    print(f"\n{'='*80}")
    print("ERROR PRIMING EXPERIMENT - RADEVAL")
    print(f"{'='*80}")
    print(f"Perturbations: {', '.join(perturbations_to_run)}")
    print("Computing primed GREEN ratings with error warning")
    print("  Testing on both ORIGINAL and PERTURBED reports:")
    print("  1. Original reports + error warning (control for false positives)")
    print("  2. Perturbed reports + error warning (test sensitivity)")
    print("  Baseline (no warning) ratings already exist in baseline experiments")

    # Create experiment directory
    experiment_dir = os.path.join(output_dir, 'experiment_results', 'error_priming')
    os.makedirs(experiment_dir, exist_ok=True)

    model_name_clean = clean_model_name(args.model) if args.model else "gpt-4o"

    # Process each perturbation
    baseline_dir = os.path.join(output_dir, 'experiment_results', 'baseline')

    # First, process original reports ONCE (same across all perturbations)
    print(f"\n{'='*80}")
    print("PROCESSING ORIGINAL REPORTS (ONCE FOR ALL PERTURBATIONS)")
    print(f"{'='*80}")

    # Load original reports from the first perturbation's baseline
    first_perturbation = perturbations_to_run[0]
    first_perturbation_dir = os.path.join(baseline_dir, first_perturbation)
    baseline_files = [f for f in os.listdir(first_perturbation_dir) if f.endswith('_green_rating.jsonl')]

    if baseline_files:
        baseline_file = os.path.join(first_perturbation_dir, baseline_files[0])

        # Load original data
        original_data = []
        with open(baseline_file, 'r') as f:
            for line in f:
                if line.strip():
                    original_data.append(json.loads(line))

        print(f"Loaded {len(original_data)} original examples")

        # Create original ratings directory
        original_ratings_dir = os.path.join(output_dir, 'original_ratings')
        os.makedirs(original_ratings_dir, exist_ok=True)

        output_filename = f"original_{model_name_clean}_error_priming_green_rating.jsonl"
        output_path = os.path.join(original_ratings_dir, output_filename)

        # Check which entries have already been processed
        processed_ids = get_processed_ids(output_path)
        remaining_data = [item for item in original_data if item['id'] not in processed_ids]

        if len(remaining_data) == 0:
            print(f"✓ All {len(original_data)} original reports already processed")
        else:
            print(f"Processing {len(remaining_data)} remaining original reports")

            for idx, item in enumerate(remaining_data, 1):
                reference = item[reference_field]
                original_text = item[text_field]  # Original (no errors)
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

            print(f"✓ Completed original reports")
            print(f"✓ Results saved to: {output_path}")

    # Now process perturbed reports for each perturbation
    print(f"\n{'='*80}")
    print("PROCESSING PERTURBED REPORTS (PER PERTURBATION)")
    print(f"{'='*80}")

    for perturbation_name in perturbations_to_run:
        print(f"\n{'='*80}")
        print(f"PERTURBATION: {perturbation_name.upper()}")
        print(f"{'='*80}")

        # Load pre-generated perturbations from baseline
        perturbation_baseline_dir = os.path.join(baseline_dir, perturbation_name)

        # Find the baseline GREEN rating file
        baseline_files = [f for f in os.listdir(perturbation_baseline_dir) if f.endswith('_green_rating.jsonl')]
        if not baseline_files:
            print(f"  Warning: No baseline GREEN ratings found for {perturbation_name}")
            continue

        baseline_file = os.path.join(perturbation_baseline_dir, baseline_files[0])
        print(f"  Loading data from: {baseline_files[0]}")

        # Load data
        loaded_data = []
        with open(baseline_file, 'r') as f:
            for line in f:
                if line.strip():
                    loaded_data.append(json.loads(line))

        print(f"  Loaded {len(loaded_data)} examples")

        # Create perturbation-specific subdirectory (matching baseline structure)
        perturbation_output_dir = os.path.join(experiment_dir, perturbation_name)
        os.makedirs(perturbation_output_dir, exist_ok=True)

        # Process only perturbed reports (original done once above)
        print(f"\n  Processing PERTURBED reports with error warning")

        output_filename = f"{perturbation_name}_{model_name_clean}_error_priming_green_rating.jsonl"
        output_path = os.path.join(perturbation_output_dir, output_filename)

        # Check which entries have already been processed
        processed_ids = get_processed_ids(output_path)
        remaining_data = [item for item in loaded_data if item['id'] not in processed_ids]

        if len(remaining_data) == 0:
            print(f"  ✓ All {len(loaded_data)} entries already processed")
        else:
            print(f"  Processing {len(remaining_data)} remaining entries")

            # Process each entry
            for idx, item in enumerate(remaining_data, 1):
                reference = item[reference_field]
                perturbed_text = item[f'perturbed_{text_field}']  # Perturbed (with errors)
                item_id = item['id']

                print(f"    [{idx}/{len(remaining_data)}] {item_id}...", end=" ")

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

            print(f"  ✓ Completed perturbed reports for {perturbation_name}")

    print(f"\n{'='*80}")
    print("ERROR PRIMING EXPERIMENT COMPLETED")
    print(f"{'='*80}")
    print(f"Primed results saved to: {experiment_dir}/{{perturbation}}/")
    print(f"Baseline (control) results in: {baseline_dir}/{{perturbation}}/")
    print("\nStructure matches baseline and CQA error_detection experiments")
    print("\nNext step: Compare GREEN scores between baseline (control) and primed conditions")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run error priming experiment on RadEval')

    parser.add_argument('--model', type=str, default=None,
                       help='Model to use for GREEN evaluation (default: gpt-4o)')

    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: output/radeval)')

    parser.add_argument('--perturbation', type=str, default=None,
                       choices=['inject_false_prediction', 'inject_contradiction', 'inject_false_negation'],
                       help='Specific perturbation to test (default: all three)')

    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')

    parser.add_argument('--start-idx', type=int, default=None,
                       help='Start index for data subset (default: 0)')

    parser.add_argument('--end-idx', type=int, default=None,
                       help='End index for data subset (default: all data)')

    args = parser.parse_args()

    run_error_priming_radeval(args)
